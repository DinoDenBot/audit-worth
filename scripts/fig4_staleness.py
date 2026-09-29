"""Figure 4 and Section VII-D: how long a round-80 payment-seeking 16-record panel stays valid.

For every analysed group (4-33), seed (2-5) and provider, the payment-seeking
16-record panel is elicited on the round-80 task vectors with no competing
stake and evaluated on the task vectors of rounds 80-120 of the deterministic
training replay. Its squared error is compared with the mean squared error of
100 fixed random 16-record panels. Reported per evaluation round:
  * mean ratio: ratio averaged over the inventory-seed units of a group, then a
    95% t interval over the 30 groups (units whose random error is 0, i.e.
    inventories of at most 16 records, are excluded);
  * mean log ratio (group-averaged, floor 1e-12);
  * over all inventory-seed units: median ratio, share better than random
    (ratio < 1), share more than twice as bad (ratio > 2), pooled ratio
    (sum of panel errors / sum of random errors), and the number of units whose
    panel error is below 1e-6;
  * the round-80 panel evaluated on round-80 candidates generated with a
    different seed (same model; all ordered seed pairs s != s').
Writes results/fig4/staleness-k16.csv and staleness-k16-other-seed.csv in the
format of the paper's figure data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import statistics
from collections import defaultdict

import numpy as np

from auditworth.check import report
from auditworth.data import DECAY_ROUNDS, FEMNIST30_GROUPS, FEMNIST30_SEEDS, femnist30_decay_state
from auditworth.paths import RESULTS
from auditworth.staleness import log_ratio, panel_errors, random_panels, select, task_vectors
from auditworth.stats import t_interval

K = 16
NAME = "fig4_staleness"


def tci(values):
    lo, hi = t_interval(values)
    return {"mean": float(np.mean(values)), "lo": lo, "hi": hi}


def main() -> bool:
    units = defaultdict(list)                       # round -> [(group, t2, r2)]
    other = defaultdict(list)                       # group -> other-seed ratios at round 80
    for g in FEMNIST30_GROUPS:
        states = {(r, s): femnist30_decay_state(g, r, s) for r in DECAY_ROUNDS for s in FEMNIST30_SEEDS}
        F = {key: [task_vectors(pr) for pr in st] for key, st in states.items()}
        for s in FEMNIST30_SEEDS:
            for p in range(4):
                F80 = F[(80, s)][p]
                n = len(F80)
                T = select(F80 @ F80.T, min(K, n))    # payment-seeking panel at round 80
                R = random_panels(g, 1, s, K, p, n)
                for r in DECAY_ROUNDS:
                    t2, r2 = panel_errors(F[(r, s)][p], T, R)
                    units[r].append((g, t2, r2))
                for s2 in FEMNIST30_SEEDS:
                    if s2 != s:
                        t2, r2 = panel_errors(F[(80, s2)][p], T, R)
                        other[g].append(t2 / r2 if r2 > 0 else float("nan"))

    by_round, rows = {}, []
    for r in DECAY_ROUNDS:
        per_ratio, per_log = defaultdict(list), defaultdict(list)
        for g, t2, r2 in units[r]:
            per_ratio[g].append(t2 / r2 if r2 > 0 else float("nan"))
            per_log[g].append(log_ratio(t2, r2))
        gs = sorted(per_ratio)
        valid = [(t2, r2) for _, t2, r2 in units[r] if r2 > 0]
        rat = sorted(t2 / r2 for t2, r2 in valid)
        x = {
            "ratio_mean": tci([np.nanmean(per_ratio[g]) for g in gs]),
            "log_ratio_mean": tci([np.mean(per_log[g]) for g in gs]),
            "median_ratio": statistics.median(rat),
            "share_better": sum(v < 1 for v in rat) / len(rat),
            "share_worse_2x": sum(v > 2 for v in rat) / len(rat),
            "pooled_ratio": sum(t for t, _ in valid) / sum(q for _, q in valid),
            "units": len(rat),
            "near_zero_units": sum(t < 1e-6 for t, _ in valid),
        }
        by_round[str(r)] = x
        a = x["ratio_mean"]
        rows.append(f"{r},{a['mean']:.6g},{a['lo']:.6g},{a['hi']:.6g},{x['median_ratio']:.6g},"
                    f"{x['share_better']:.6g},{x['share_worse_2x']:.6g}\n")
    other_seed = tci([np.nanmean(other[g]) for g in sorted(other)])

    out = RESULTS / "fig4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "staleness-k16.csv").write_text("round,mean,lo,hi,median,better,worse2x\n" + "".join(rows))
    (out / "staleness-k16-other-seed.csv").write_text(
        "round,mean,lo,hi\n" f"80,{other_seed['mean']:.6g},{other_seed['lo']:.6g},{other_seed['hi']:.6g}\n")

    later = [by_round[str(r)] for r in DECAY_ROUNDS if r != 80]
    values = {
        "by_round": by_round,
        "other_seed_ratio": other_seed,
        "after80": {
            "ratio_mean_avg": float(np.mean([x["ratio_mean"]["mean"] for x in later])),
            "median_min": min(x["median_ratio"] for x in later),
            "median_max": max(x["median_ratio"] for x in later),
            "better_min": min(x["share_better"] for x in later),
            "better_max": max(x["share_better"] for x in later),
            "worse2x_min": min(x["share_worse_2x"] for x in later),
            "worse2x_max": max(x["share_worse_2x"] for x in later),
            "pooled_min": min(x["pooled_ratio"] for x in later),
            "pooled_max": max(x["pooled_ratio"] for x in later),
            "log_ratio_avg": float(np.mean([x["log_ratio_mean"]["mean"] for x in later])),
        },
        "all_rounds": {
            "near_zero_min": min(x["near_zero_units"] for x in by_round.values()),
            "near_zero_max": max(x["near_zero_units"] for x in by_round.values()),
            "units_mean": float(np.mean([x["units"] for x in by_round.values()])),
        },
    }
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
