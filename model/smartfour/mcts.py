"""AlphaZero MCTS with batched leaf evaluation, CPU-friendly.

Every node stores values from its OWN perspective (the player to move at that
node): the net evaluates the current-player encoding, priors are masked to
legal actions, and backpropagation flips the value sign at each level.
"""

import math
import torch

from .config import MCTSConfig
from .diagnostics import masked_entropy_bits, state_hash, state_key, visit_entropy_bits
from .encode import action_mask, action_to_xyz, encode, legal_actions
from .game import apply_move, is_terminal, terminal_value
NEG_INF = float("-inf")


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
    def __init__(self, net, mcts_cfg: MCTSConfig, device="cpu"):
        self.net = net
        self.cfg = mcts_cfg
        self.device = device
        if hasattr(self.net, "eval"):
            self.net.eval()
        self.last_root_policy_visits = {}
        self.last_stats: dict = {}  # per-search diagnostics, JSON-safe

    # ------------------------------------------------------------- public API

    def root_policy(self, state, root_noise: bool = True, temperature: float = 1.0):
        """Run a search from `state`.

        Returns (pi, chosen, root): pi is the (125,) visit-based policy,
        chosen the sampled/argmax action (None if terminal), root the tree.
        Populates self.last_stats with per-search diagnostics.
        """
        stats = {
            "sims": self.cfg.simulations,
            "depth_total": 0,
            "max_depth": 0,
            "sims_done": 0,
            "leaf_states": set(),
            "terminal_hits": 0,
            "nodes": 0,
            "net_forwards": 0,
            "batch_sizes": [],
            "blocked_drains": 0,
            "node_hashes": set(),
            "root_value": 0.0,
            "root_policy_entropy": 0.0,
        }
        root, root_value, policy_entropy = self._build_root(state, root_noise, stats)
        if root is None:
            self.last_stats = {
                "sims": 0, "depth_mean": 0.0, "max_depth": 0, "sims_done": 0,
                "leaf_distinct": 0, "terminal_hits": 0, "nodes": 0,
                "net_forwards": 0, "batch_size_mean": 0.0, "blocked_drains": 0,
                "n_states": 0, "node_hashes": [],
                "root_value": 0.0, "root_policy_entropy": 0.0,
                "root_entropy": 0.0, "root_width": 0, "chosen_prob": 0.0,
            }
            return torch.zeros(125), None, None
        stats["root_value"] = root_value
        stats["root_policy_entropy"] = policy_entropy
        self._run(root, stats)
        pi, chosen = self._visit_policy(root, temperature)
        legal = root.legal
        counts = torch.tensor(
            [root.children[a].visits for a in legal], dtype=torch.float32
        )
        total = counts.sum().item()
        root_width = int((counts > 0).sum().item()) if total else 0
        root_entropy = visit_entropy_bits(counts) if total else 0.0
        chosen_prob = (
            float(counts[legal.index(chosen)] / total)
            if total and chosen is not None
            else 0.0
        )
        self.last_stats = {
            "sims": stats["sims"],
            "depth_mean": stats["depth_total"] / max(stats["sims_done"], 1),
            "max_depth": stats["max_depth"],
            "sims_done": stats["sims_done"],
            "leaf_distinct": len(stats["leaf_states"]),
            "terminal_hits": stats["terminal_hits"],
            "nodes": stats["nodes"],
            "net_forwards": stats["net_forwards"],
            "batch_size_mean": (sum(stats["batch_sizes"]) / len(stats["batch_sizes"]))
            if stats["batch_sizes"] else 0.0,
            "blocked_drains": stats["blocked_drains"],
            "n_states": len(stats["node_hashes"]),
            "node_hashes": sorted(stats["node_hashes"]),
            "root_value": stats["root_value"],
            "root_policy_entropy": stats["root_policy_entropy"],
            "root_entropy": root_entropy,
            "root_width": root_width,
            "chosen_prob": chosen_prob,
        }
        self.last_root_policy_visits = {a: c.visits for a, c in root.children.items()}
        return pi, chosen, root


    # ------------------------------------------------------------- internals

    def _build_root(self, state, root_noise: bool, stats: dict):
        legal = legal_actions(state)
        if not legal:
            return None, 0.0, 0.0
        tensor = encode(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(tensor)
        policy_entropy = masked_entropy_bits(logits[0], action_mask(state))
        node = Node(state, legal)
        self._expand(node, logits[0], value[0, 0].item(), stats)
        if root_noise:
            self._apply_dirichlet(node)
        return node, value[0, 0].item(), policy_entropy

    def _run(self, root: Node, stats: dict):
        pending_unique = {}   # id(leaf) -> leaf, for one-shot expansion
        pending_order = []    # (path, leaf) per simulation, for backprop

        def drain():
            leaves = list(pending_unique.values())
            tensors = torch.stack([encode(leaf.state) for leaf in leaves])
            with torch.no_grad():
                logits, values = self.net(tensors)
            value_of = {id(leaf): v.item() for leaf, v in zip(leaves, values)}
            leaf_idx = {id(l): i for i, l in enumerate(leaves)}
            stats["net_forwards"] += 1
            stats["batch_sizes"].append(len(leaves))
            expanded = set()
            for path, leaf in pending_order:
                if id(leaf) not in expanded:
                    expanded.add(id(leaf))
                    self._expand(leaf, logits[leaf_idx[id(leaf)]], value_of[id(leaf)], stats)
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
                    stats["blocked_drains"] += 1
                continue  # retry this simulation against the expanded tree
            sims += 1
            stats["sims_done"] = sims
            stats["depth_total"] += len(path)
            stats["max_depth"] = max(stats["max_depth"], len(path))
            stats["leaf_states"].add(state_key(leaf.state))
            if leaf.terminal is not None:
                stats["terminal_hits"] += 1
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

    def _expand(self, node: Node, logits, value: float, stats: dict | None = None):
        if node.terminal is not None:
            return
        if not node.legal:
            node.terminal = 0.0  # no legal moves: treat as a draw
            return
        if stats is not None:
            stats["nodes"] += 1
            stats["node_hashes"].add(state_hash(node.state))
        masked = logits.clone()
        mask = action_mask(node.state)
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
