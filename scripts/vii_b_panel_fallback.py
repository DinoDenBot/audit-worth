"""Section VII-B, "Without a competing stake: the panel fallback": payment-seeking panels versus one direct read and random panels.

30-group FEMNIST study, round 80, one-epoch contract, seeds 2-5, eight-coordinate
task vector. With no competing stake a best-responding panel minimizes its
distance to the inventory mean. Errors are reported relative to one directly
read record (pooled over inventories within a group, then averaged over the 30
groups with 95% t intervals); "direct-read equivalents" are the reciprocals.
The panel-to-random comparison uses 100 random panels per inventory.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
from collections import defaultdict

import numpy as np

from auditworth.check import report
from auditworth.leverage import FLOOR, KS, ci, panel_vs_random, summarize, sweep

NAME = "vii_b_panel_fallback"


def main() -> bool:
    units, rows = sweep(KS, (0.0,), reports=False)
    values = {"panel_error_vs_direct_read": {}, "direct_read_equivalents": {}}
    for k in KS:
        s = summarize(units, rows, k, 0.0, methods=("pan", "rand"))
        pan, rand = s["pooled_ratio"]["pan"], s["pooled_ratio"]["rand"]
        values["panel_error_vs_direct_read"][f"k{k}"] = {"mean": pan[0], "lo": pan[1], "hi": pan[2]}
        values["direct_read_equivalents"][f"k{k}"] = {"panel": 1 / pan[0], "panel_lo": 1 / pan[2],
                                                      "panel_hi": 1 / pan[1], "random": 1 / rand[0]}
    values["n_nondegenerate_inventories"] = sum(V >= FLOOR for _, V, _ in units.values())

    # Payment-seeking versus random 16-panels, three aggregations (per group, then over groups).
    per = defaultdict(lambda: defaultdict(list))
    for g, s, p, n, t2, r2 in panel_vs_random(16):
        per["ratio"][g].append(t2 / r2 if r2 > 0 else float("nan"))
        per["log"][g].append(math.log(max(t2, FLOOR) / max(r2, FLOOR)))
        per["t2"][g].append(t2)
        per["r2"][g].append(r2)
    gs = sorted(per["ratio"])
    pooled = ci([np.sum(per["t2"][g]) / np.sum(per["r2"][g]) for g in gs])
    mean_ratio = ci([np.nanmean(per["ratio"][g]) for g in gs])
    mean_log = ci([np.mean(per["log"][g]) for g in gs])
    values["panel_vs_random_k16"] = {"pooled_ratio": pooled[0], "mean_ratio": mean_ratio[0],
                                     "mean_log_ratio": mean_log[0], "exp_mean_log_ratio": math.exp(mean_log[0]),
                                     "intervals": {"pooled_ratio": pooled[1:], "mean_ratio": mean_ratio[1:],
                                                   "mean_log_ratio": mean_log[1:]}}
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
