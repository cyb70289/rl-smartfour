"""AlphaZero MCTS with virtual-loss batched leaf evaluation.

Every node stores values from its OWN perspective (the player to move at that
node): the net evaluates the current-player encoding, priors are masked to
legal actions, and backpropagation flips the value sign at each level.

Batching: each selection pass descends up to `batch_eval_size` leaves,
charging a temporary virtual loss along every pending path so PUCT steers
consecutive descents to different lines, then evaluates all pending leaves
in ONE net forward and reconciles (penalties removed, values backpropagated).
Same PUCT math, priors, and net as the original sequential searcher; only
traversal order differs. One searcher serves training and inference.

Evaluation seam: `evaluator` maps a list of GameStates to (priors, values),
where priors is an (N, 125) float tensor already softmaxed over each state's
legal actions and values an (N,) float tensor. The default LocalEvaluator
wraps a torch net on `device`; training workers pass a RemoteEvaluator
backed by the central inference server.
"""

import math

import torch

from .config import MCTSConfig
from .diagnostics import state_hash, state_key, visit_entropy_bits
from .encode import action_to_xyz, encode, legal_actions
from .game import apply_move, is_terminal, terminal_value

NEG_INF = float("-inf")


class LocalEvaluator:
    """In-process net evaluation: (states) -> (masked priors, values)."""

    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device
        if hasattr(net, "eval"):
            net.eval()

    def __call__(self, states):
        n = len(states)
        with torch.no_grad():
            tensor = torch.stack([encode(s) for s in states]).to(self.device)
            logits, values = self.net(tensor)
            # One batched legality mask on device: channel 10+h set at
            # (x,z) iff column height is h; the flattened (5,5) plane index
            # equals the action index y*25 + x*5 + z.
            mask = (tensor[:, 10:15] > 0.5).reshape(n, 125)
            logits = logits.masked_fill(
                ~mask, torch.tensor(NEG_INF, device=logits.device)
            )
            priors = torch.softmax(logits, dim=1).cpu()
            vals = values.squeeze(1).cpu()
        return priors, vals


class Node:
    __slots__ = ("state", "legal", "prior", "children", "visits", "value_sum", "terminal", "pending")

    def __init__(self, state, legal, prior=None):
        self.state = state
        self.legal = legal            # sorted action indices, root frame
        self.prior = prior            # parent's prior for this node's action
        self.children = None          # dict[action, Node] once expanded
        self.visits = 0
        self.value_sum = 0.0
        self.terminal = None          # None | float from this node's perspective
        self.pending = 0              # virtual-loss: visits pending this pass

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class _ChildStub:
    """Unmaterialized child: only the parent's prior. Becomes a Node on
    first descent (see MCTS._materialize)."""
    __slots__ = ("prior", "visits", "value_sum", "terminal", "pending")

    def __init__(self, prior: float):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.terminal = None
        self.pending = 0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, net, mcts_cfg: MCTSConfig, device="cpu", evaluator=None):
        self.cfg = mcts_cfg
        self.evaluator = evaluator if evaluator is not None else LocalEvaluator(net, device)
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
            "node_hashes": set(),
            "root_value": 0.0,
            "root_policy_entropy": 0.0,
        }
        root, root_value, policy_entropy = self._build_root(state, root_noise, stats)
        if root is None:
            self.last_stats = {
                "sims": 0, "depth_mean": 0.0, "max_depth": 0, "sims_done": 0,
                "leaf_distinct": 0, "terminal_hits": 0, "nodes": 0,
                "net_forwards": 0, "batch_size_mean": 0.0,
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
        priors, values = self.evaluator([state])
        node = Node(state, legal)
        self._expand(node, priors[0], float(values[0]), stats)
        root_value = float(values[0])
        policy_entropy = self._policy_entropy(priors[0], legal)
        if root_noise:
            self._apply_dirichlet(node)
        return node, root_value, policy_entropy

    @staticmethod
    def _policy_entropy(prior: torch.Tensor, legal) -> float:
        p = prior[torch.tensor(legal, dtype=torch.long)]
        return float(-(p * torch.log2(p + 1e-12)).sum())

    # Virtual-loss value charged to every node on a pending path, from that
    # node's own perspective (a pending visit looks like a certain loss).
    _VL = 1.0

    def _run(self, root: Node, stats: dict):
        """Virtual-loss search: each pass descends up to `target` leaves,
        charging a temporary loss along each pending path so PUCT steers the
        next descent to a different line, then evaluates all pending leaves
        in one net forward and reconciles (penalties removed, values
        backpropagated). `target` = max(batch_eval_size, 8) bounded by
        remaining sims — small searches stay small, big searches fill
        device-sized batches.
        """
        target = max(int(self.cfg.batch_eval_size), 8)
        sims = 0
        while sims < self.cfg.simulations:
            k = min(target, self.cfg.simulations - sims)
            batch = []          # (path, leaf) pairs, pending this pass
            for _ in range(k):
                path, leaf = self._descend_vl(root)
                if leaf is None:
                    break       # whole tree is pending/terminal-blocked
                sims += 1
                stats["sims_done"] = sims
                stats["depth_total"] += len(path)
                stats["max_depth"] = max(stats["max_depth"], len(path))
                stats["leaf_states"].add(state_key(leaf.state))
                if leaf.terminal is not None:
                    # Terminal: value known, reconcile immediately.
                    stats["terminal_hits"] += 1
                    self._reconcile(path, leaf, leaf.terminal)
                else:
                    batch.append((path, leaf))
            if batch:
                self._evaluate_batch(batch, stats)
                stats["net_forwards"] += 1
                stats["batch_sizes"].append(len(batch))

    def _descend_vl(self, root: Node):
        """One PUCT descent with virtual loss on pending paths. Returns
        (path, leaf); (None, None) when no unblocked leaf exists. Penalties
        are charged only on a completed descent — an aborted one (every
        child of some node pending) must leave no trace."""
        path = []
        node = root
        while node.children is not None:
            best_score = NEG_INF
            best = []
            parent_visits = node.visits
            for a, child in node.children.items():
                q = -child.value
                if child.pending > 0:
                    q -= self._VL * child.pending
                score = q + self.cfg.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
                if score > best_score:
                    best_score, best = score, [a]
                elif score == best_score:
                    best.append(a)
            if not best:
                return None, None  # every child pending: pass short
            best_a = best[int(torch.randint(len(best), (1,)).item())]
            path.append((node, best_a))
            node = self._materialize(node, best_a)
        # Completed descent: charge the virtual loss along the path now.
        for parent, a in path:
            child = parent.children[a]
            child.pending += 1
        return path, node

    def _reconcile(self, path, leaf: Node, value: float):
        """Remove virtual-loss penalties along `path` and backprop `value`
        (from leaf's perspective)."""
        for parent, a in path:
            child = parent.children[a]
            child.pending -= 1
        self._backprop(path, leaf, value)

    def _evaluate_batch(self, batch, stats: dict):
        leaves = [leaf for _path, leaf in batch]
        priors, values = self.evaluator([leaf.state for leaf in leaves])
        expanded = set()
        for (path, leaf), prior, value in zip(batch, priors, values):
            v = float(value)
            if id(leaf) not in expanded:
                expanded.add(id(leaf))
                self._expand(leaf, prior, v, stats)
            self._reconcile(path, leaf, v)

    def _materialize(self, node: Node, a: int) -> Node:
        """Build the real child for action `a` from its stub."""
        child = node.children[a]
        if isinstance(child, _ChildStub):
            x, z, _y = action_to_xyz(a)
            child_state = apply_move(node.state, x, z)
            real = Node(child_state, legal_actions(child_state), prior=child.prior)
            if is_terminal(child_state):
                real.terminal = terminal_value(child_state)
            node.children[a] = real
            return real
        return child

    def _expand(self, node: Node, prior, value: float, stats: dict | None = None):
        if node.terminal is not None:
            return
        if not node.legal:
            node.terminal = 0.0  # no legal moves: treat as a draw
            return
        if stats is not None:
            stats["nodes"] += 1
            stats["node_hashes"].add(state_hash(node.state))
        # Lazy children: stubs carry only the parent's prior; the child's
        # state/legal/terminal are materialized on first descent. Unvisited
        # children never contribute more than prior/visits to UCB, so search
        # semantics are identical to eager expansion.
        node.children = {}
        for a in node.legal:
            node.children[a] = _ChildStub(float(prior[a]))

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
