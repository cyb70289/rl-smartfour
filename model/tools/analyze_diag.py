"""Summarize checkpoints/diagnostics.jsonl into a per-iteration table.

Usage:
  .venv/bin/python tools/analyze_diag.py [path/to/diagnostics.jsonl ...]
"""

import json
import sys
from pathlib import Path


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    paths = sys.argv[1:] or ["checkpoints/diagnostics.jsonl"]
    all_rows = []
    for p in paths:
        if Path(p).exists():
            all_rows.extend(load(p))
    if not all_rows:
        print("no diagnostics rows found")
        return
    print(f"{'iter':>4} {'plies':>7} {'W':>3} {'B':>3} {'D':>2} "
          f"{'depth':>5} {'rootW':>5} {'visEnt':>6} {'chosen':>6} "
          f"{'polEnt':>6} {'vMean':>6} {'|v|>09':>6} {'align':>6} "
          f"{'cal':>5} {'states':>6} {'xdup%':>6} {'novel%':>6} "
          f"{'bufDist%':>8} {'bufDup%':>7} {'piEnt':>6} {'1hot%':>6} "
          f"{'polL':>6} {'valL':>6}")
    for d in all_rows:
        it = d["iteration"]
        a = d["agg"]
        b = d["buffer"]
        pl = a["plies"]
        w = a["winners"]
        v = a["root_value"]
        print(
            f"{it:>4} {pl['mean']:>7.2f} {w['white']:>3} {w['black']:>3} "
            f"{w['draw']:>2} {a['depth']['mean']:>5.2f} {a['root_width']:>5.1f} "
            f"{a['root_entropy']:>6.2f} {a['chosen_prob']:>6.2f} "
            f"{a['net_policy_entropy']:>6.2f} {v['mean']:>6.2f} "
            f"{v['frac_gt_0_9'] * 100:>5.1f}% {a['value_alignment']:>6.2f} "
            f"{a['value_calibration']:>5.2f} {a['states_per_game']:>6.0f} "
            f"{a['cross_game_redundancy'] * 100:>5.1f}% "
            f"{d['novel_frac'] * 100:>5.1f}% "
            f"{b['distinct_frac'] * 100:>7.1f}% {b['dup_frac'] * 100:>6.1f}% "
            f"{b['pi_entropy']:>6.2f} {b['pi_one_hot_frac'] * 100:>5.1f}%"
        )
    # ply histograms, last 5 iterations
    print("\nply histograms (upper-exclusive bins):")
    for d in all_rows:
        hist = d["agg"]["plies"]["hist"]
        nonzero = {k: v for k, v in hist.items() if v}
        print(f"  iter {d['iteration']}: {nonzero}")


if __name__ == "__main__":
    main()
