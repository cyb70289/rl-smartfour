"""AlphaZero MCTS with batched leaf evaluation, device-aware.

Runs on the net's device by default (or an explicit `device`): encodings are
moved to the device before each net forward, while tree bookkeeping stays in
CPU Python. Every node stores values from its OWN perspective (the player to
move at that node): the net evaluates the current-player encoding, priors are
masked to legal actions, and backpropagation flips the value sign at each
level.
"""

import math
import torch

from .config import MCTSConfig
from .encode import action_mask, action_to_xyz, encode, legal_actions
from .game import apply_move, is_terminal, terminal_value

NEG_INF = float("-inf")


def _infer_net_device(net) -> torch.device:
    """Device of the net's first parameter; CPU for parameterless test doubles."""
    params = getattr(net, "parameters", None)
    if params is None:
        return torch.device("cpu")
    try:
        return next(params()).device
    except StopIteration:
        return torch.device("cpu")


class Node:
    __slots__ = ("state", "legal", "prior", "children", "visits", "value_sum", "terminal")

    def __init__(self, state, legal, prior=None):
        self.state = state
        self.legal = legal
        self.prior = prior          # parent's prior for this node's action
        self.children = None        # dict[action, Node] once expanded
        self.visits = 0
        self.value_sum = 0.0
        self.terminal = None        # None | float from this node's perspective

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, net, mcts_cfg: MCTSConfig, device=None):
        self.net = net
        self.cfg = mcts_cfg
        self.device = device if device is not None else _infer_net_device(net)
        if hasattr(self.net, "eval"):
            self.net.eval()
        self.last_root_policy_visits = {}

    # ------------------------------------------------------------- public API

    def root_policy(self, state, root_noise: bool = True, temperature: float = 1.0):
        """Run a search from `state`.

        Returns (pi, chosen, root): pi is the (125,) visit-based policy,
        chosen the sampled/argmax action (None if terminal), root the tree.
        """
        root = self._build_root(state, root_noise)
        if root is None:
            return torch.zeros(125), None, None
        self._run(root)
        pi, chosen = self._visit_policy(root, temperature)
        self.last_root_policy_visits = {a: c.visits for a, c in root.children.items()}
        return pi, chosen, root

    # ------------------------------------------------------------- internals

    def _build_root(self, state, root_noise: bool):
        legal = legal_actions(state)
        if not legal:
            return None
        tensor = encode(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(tensor)
        node = Node(state, legal)
        self._expand(node, logits[0], value[0, 0].item())
        if root_noise:
            self._apply_dirichlet(node)
        return node

    def _run(self, root: Node):
        pending_unique = {}   # id(leaf) -> leaf, for one-shot expansion
        pending_order = []    # (path, leaf) per simulation, for backprop

        def drain():
            leaves = list(pending_unique.values())
            tensors = torch.stack([encode(leaf.state) for leaf in leaves]).to(self.device)
            with torch.no_grad():
                logits, values = self.net(tensors)
            value_of = {id(leaf): v.item() for leaf, v in zip(leaves, values)}
            expanded = set()
            for path, leaf in pending_order:
                if id(leaf) not in expanded:
                    expanded.add(id(leaf))
                    self._expand(leaf, logits[leaves.index(leaf)], value_of[id(leaf)])
                self._backprop(path, leaf, value_of[id(leaf)])

        sims = 0
        while sims < self.cfg.simulations:
            path, leaf, blocked = self._select(root, pending_unique)
            if blocked:
                # Every child of a node is already queued: expand them now so
                # the search can go deeper instead of piling duplicate visits
                # onto one child.
                if pending_order:
                    drain()
                    pending_unique.clear()
                    pending_order.clear()
                continue  # retry this simulation against the expanded tree
            sims += 1
            if leaf.terminal is not None:
                self._backprop(path, leaf, leaf.terminal)
            else:
                pending_unique.setdefault(id(leaf), leaf)
                pending_order.append((path, leaf))
                if len(pending_unique) >= self.cfg.batch_eval_size:
                    drain()
                    pending_unique.clear()
                    pending_order.clear()
        if pending_order:
            drain()

    def _select(self, root: Node, pending: dict):
        """Descend by UCB to an unexpanded leaf, skipping queued (pending) leaves.

        Returns (path, leaf, blocked): blocked=True when every child of the
        reached node is already queued, so the search cannot go deeper until
        the pending batch is evaluated.
        """
        path = []
        node = root
        while node.children is not None:
            best_score = NEG_INF
            best = []  # actions tied for the max score
            parent_visits = node.visits
            for a, child in node.children.items():
                if id(child) in pending:
                    continue  # already queued for expansion this batch
                q = -child.value  # child's perspective is the opponent's
                score = q + self.cfg.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
                if score > best_score:
                    best_score, best = score, [a]
                elif score == best_score:
                    best.append(a)
            if not best:
                return path, node, True  # blocked: all children pending
            # Random tie-break among equal scores (seeded via torch RNG).
            best_a = best[int(torch.randint(len(best), (1,)).item())]
            path.append((node, best_a))
            node = node.children[best_a]
        return path, node, False

    def _expand(self, node: Node, logits, value: float):
        if node.terminal is not None:
            return
        if not node.legal:
            node.terminal = 0.0  # no legal moves: treat as a draw
            return
        masked = logits.clone()
        mask = action_mask(node.state).to(logits.device)
        masked = torch.where(mask.bool(), logits, torch.full_like(logits, NEG_INF))
        prior = torch.softmax(masked, dim=0)
        node.children = {}
        for a in node.legal:
            x, z, _y = action_to_xyz(a)
            child_state = apply_move(node.state, x, z)
            child = Node(child_state, legal_actions(child_state), prior=float(prior[a]))
            if is_terminal(child_state):
                child.terminal = terminal_value(child_state)
            node.children[a] = child

    def _backprop(self, path, node: Node, value: float):
        # node is the leaf; walk the path upward, flipping perspective each
        # level, and update the root too.
        v = value
        while path:
            parent, _a = path.pop()
            node.visits += 1
            node.value_sum += v
            v = -v
            node = parent
        node.visits += 1
        node.value_sum += v

    def _apply_dirichlet(self, node: Node):
        n = len(node.legal)
        noise = torch.distributions.Dirichlet(
            torch.full((n,), self.cfg.dirichlet_alpha)
        ).sample()
        eps = self.cfg.dirichlet_epsilon
        for i, a in enumerate(node.legal):
            node.children[a].prior = (1 - eps) * node.children[a].prior + eps * float(noise[i])

    def _visit_policy(self, root: Node, temperature: float):
        legal = root.legal
        counts = torch.tensor([root.children[a].visits for a in legal], dtype=torch.float32)
        pi = torch.zeros(125)
        if counts.sum().item() == 0:
            return pi, None
        if temperature == 0.0:
            max_v = float(counts.max())
            cands = [i for i, c in enumerate(counts.tolist()) if c == max_v]
            idx = cands[int(torch.randint(len(cands), (1,)).item())]
            a = legal[idx]
            pi[a] = 1.0
            return pi, a
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
        pi[torch.tensor(legal)] = probs
        idx = int(torch.multinomial(probs, 1).item())
        return pi, legal[idx]
