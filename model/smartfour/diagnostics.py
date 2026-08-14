"""Training diagnostics: state hashing, entropy, per-iteration aggregation.

Lightweight, always-on instrumentation used by self-play, MCTS, and the
trainer to detect degenerate training (policy collapse, low state diversity,
replay-buffer duplication). Everything returned is JSON-serializable so it
can be appended to a diagnostics JSONL file.

State identity deliberately ignores `pieces_left`: for a given grid + player
to move, the game tree (and thus the search behavior) is identical. Two
states that differ only in remaining piece counts are the same position.
"""

import hashlib
import math
import statistics

import torch

from .game import BLACK, DRAW, WHITE

# Ply histogram edges (upper exclusive bounds). Games can end between ply 7
# (fastest possible 4-in-a-row) and ply 64 (draw cap).
PLY_BINS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 30, 40, 65)


def state_key(state) -> tuple:
    """Hashable canonical form of a GameState (fast, in-process use only)."""
    return (
        state.current,
        tuple(tuple(tuple(col) for col in plane) for plane in state.grid),
    )


def state_hash(state) -> str:
    """Stable cross-process md5 hex digest of a GameState (grid + player)."""
    h = hashlib.md5()
    h.update(bytes((state.current,)))
    for plane in state.grid:
        for col in plane:
            for p in col:
                h.update(b"\x00" if p is None else (b"\x01" if p == WHITE else b"\x02"))
    return h.hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    """Stable md5 hex digest of a (C, 5, 5) state tensor's raw bytes."""
    return hashlib.md5(t.numpy().tobytes()).hexdigest()


def masked_entropy_bits(logits: torch.Tensor, mask: torch.Tensor) -> float:
    """Entropy in bits of the softmax over logits restricted to mask == 1."""
    masked = torch.where(
        mask.bool(), logits, torch.full_like(logits, float("-inf"))
    )
    p = torch.softmax(masked, dim=0)
    return float(-(p * torch.log2(p + 1e-12)).sum())


def visit_entropy_bits(counts: torch.Tensor) -> float:
    """Entropy in bits of a visit-count distribution (normalized in place)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def ply_histogram(plies: list) -> dict:
    """Counts of game lengths falling into each PLY_BINS bucket."""
    hist = {str(edge): 0 for edge in PLY_BINS}
    for ply in plies:
        for edge in PLY_BINS:
            if ply < edge:
                hist[str(edge)] += 1
                break
    return hist


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def aggregate_games(games: list) -> dict:
    """Aggregate per-game stats dicts (from play_game) into iteration-level
    summary metrics. Missing/None entries are skipped."""
    games = [g for g in games if g]
    n = len(games)
    if n == 0:
        return {"n_games": 0}

    plies = [g["plies"] for g in games]
    winners = [g["winner"] for g in games]
    root_values = [v for g in games for v in g["root_values"]]
    root_entropies = [v for g in games for v in g["root_entropies"]]
    policy_entropies = [v for g in games for v in g["net_policy_entropies"]]
    root_widths = [v for g in games for v in g["root_widths"]]
    chosen_probs = [v for g in games for v in g["chosen_probs"]]
    depths = [v for g in games for v in g["depths"]]

    # State diversity: how much overlap exists across games in this batch.
    total_states = 0
    union = set()
    per_game = []
    for g in games:
        hs = set(g["state_hashes"])
        per_game.append(len(hs))
        total_states += len(hs)
        union |= hs
    redundancy = 1.0 - len(union) / total_states if total_states else 0.0

    return {
        "n_games": n,
        "plies": {
            "mean": _mean(plies),
            "median": statistics.median(plies) if plies else float("nan"),
            "min": min(plies) if plies else 0,
            "max": max(plies) if plies else 0,
            "hist": ply_histogram(plies),
        },
        "winners": {
            "white": winners.count("white"),
            "black": winners.count("black"),
            "draw": winners.count("draw"),
        },
        "root_value": {
            "mean": _mean(root_values),
            "abs_mean": _mean([abs(v) for v in root_values]),
            "frac_gt_0_9": _mean([1.0 if abs(v) > 0.9 else 0.0 for v in root_values]),
        },
        "value_alignment": _mean([g["value_align"] for g in games]),
        "value_calibration": _mean([g["value_cal"] for g in games]),
        "root_entropy": _mean(root_entropies),
        "net_policy_entropy": _mean(policy_entropies),
        "root_width": _mean(root_widths),
        "chosen_prob": _mean(chosen_probs),
        "depth": {"mean": _mean(depths), "max": max(g["max_depth"] for g in games)},
        "leaves_per_search": _mean([g["leaf_distinct_mean"] for g in games]),
        "terminal_hits_per_game": _mean([g["terminal_hits"] for g in games]),
        "nodes_per_search": _mean([g["nodes_mean"] for g in games]),
        "net_forwards_per_search": _mean([g["net_forwards_mean"] for g in games]),
        "blocked_drains": sum(g["blocked_drains"] for g in games),
        "states_per_game": _mean(per_game),
        "states_total": total_states,
        "states_distinct": len(union),
        "cross_game_redundancy": redundancy,
    }


def buffer_stats(states: list, pis: list, sample_cap: int = 4000) -> dict:
    """Duplication and policy-shape stats over (a sample of) the replay buffer.

    Exact-duplicate detection hashes the raw state tensor bytes; the policy
    shape stats report how peaked the stored MCTS targets are (a one-hot
    target carries no information).
    """
    n = len(states)
    if n == 0:
        return {
            "n": 0, "distinct_frac": 0.0, "dup_frac": 0.0,
            "pi_entropy": 0.0, "pi_one_hot_frac": 0.0, "pi_max_mass": 0.0,
        }
    idx = torch.randperm(n)[:sample_cap].tolist()
    seen = set()
    dup = 0
    for i in idx:
        h = tensor_hash(states[i])
        if h in seen:
            dup += 1
        else:
            seen.add(h)
    entropies = []
    one_hot = 0
    max_masses = []
    for i in idx[:1000]:
        pi = pis[i]
        max_masses.append(float(pi.max()))
        if float(pi.max()) >= 0.99:
            one_hot += 1
        p = pi[pi > 0]
        entropies.append(float(-(p * torch.log2(p)).sum()))
    m = len(idx)
    return {
        "n": n,
        "distinct_frac": len(seen) / m,
        "dup_frac": dup / m,
        "pi_entropy": _mean(entropies),
        "pi_one_hot_frac": one_hot / min(m, 1000),
        "pi_max_mass": _mean(max_masses),
    }


def format_lines(iteration: int, agg: dict, buf: dict, novel_frac: float,
                 losses: dict) -> list:
    """Human-readable tqdm.write lines for one iteration's diagnostics.

    agg: output of aggregate_games(); buf: output of buffer_stats();
    novel_frac: fraction of this iteration's stored samples whose state was
    already in the replay buffer; losses: {mean, policy, value} or None.
    """

    def f(x, fmt=".2f"):
        return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:{fmt}}"

    if agg.get("n_games", 0) == 0:
        return [f"[diag it {iteration}] no games played"]
    p = agg["plies"]
    w = agg["winners"]
    hist = " ".join(f"{k}:{v}" for k, v in p["hist"].items() if v)
    lines = []
    lines.append(
        f"[diag it {iteration}] plies avg={f(p['mean'])} med={f(p['median'], '.0f')} "
        f"min={p['min']} max={p['max']} | W={w['white']} B={w['black']} D={w['draw']}"
    )
    if hist:
        lines.append(f"[diag it {iteration}] ply hist (upper-exclusive) {hist}")
    lines.append(
        f"[diag it {iteration}] mcts depth avg={f(agg['depth']['mean'], '.1f')} "
        f"max={agg['depth']['max']} | leaves/search={f(agg['leaves_per_search'], '.1f')} "
        f"| nodes/search={f(agg['nodes_per_search'], '.1f')} "
        f"| terminal hits/game={f(agg['terminal_hits_per_game'], '.1f')} "
        f"| blocked drains={agg['blocked_drains']}"
    )
    lines.append(
        f"[diag it {iteration}] mcts root width={f(agg['root_width'], '.1f')}/25 "
        f"| visit entropy={f(agg['root_entropy'], '.2f')} bits "
        f"| chosen prob={f(agg['chosen_prob'], '.2f')} "
        f"| net policy entropy={f(agg['net_policy_entropy'], '.2f')} bits"
    )
    v = agg["root_value"]
    lines.append(
        f"[diag it {iteration}] net root value mean={f(v['mean'])} "
        f"|v|={f(v['abs_mean'])} |v|>0.9={f(v['frac_gt_0_9'] * 100, '.1f')}% "
        f"| v*z align={f(agg['value_alignment'])} "
        f"| |v-z|={f(agg['value_calibration'])}"
    )
    lines.append(
        f"[diag it {iteration}] states/game={f(agg['states_per_game'], '.0f')} "
        f"| cross-game dup={f(agg['cross_game_redundancy'] * 100, '.1f')}% "
        f"| novel vs buffer={f(novel_frac * 100, '.1f')}% "
        f"| buffer n={buf['n']} distinct={f(buf['distinct_frac'] * 100, '.1f')}% "
        f"dup={f(buf['dup_frac'] * 100, '.1f')}%"
    )
    lines.append(
        f"[diag it {iteration}] buffer pi entropy={f(buf['pi_entropy'], '.2f')} bits "
        f"| one-hot={f(buf['pi_one_hot_frac'] * 100, '.1f')}% "
        f"| pi max mass={f(buf['pi_max_mass'], '.2f')}"
    )
    if losses is not None:
        lines.append(
            f"[diag it {iteration}] loss={f(losses['mean'], '.4f')} "
            f"pol={f(losses['policy'], '.4f')} val={f(losses['value'], '.4f')}"
        )
    return lines
