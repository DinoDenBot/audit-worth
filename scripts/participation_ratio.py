"""Appendix B: the frozen participation-ratio prediction failed in the opposite direction (Spearman -0.29 [-0.47, -0.09]).

For each inventory of the 30-group FEMNIST study (round 80, one-epoch contract,
seeds 2-5, all 480 inventories) the participation ratio of the task-vector
covariance is correlated (Spearman) with the log ratio of the payment-seeking
16-panel's squared error to the mean of 100 random 16-panels. The interval is a
group-cluster bootstrap (2000 resamples of the 30 groups, generator seed 20260929,
percentile 2.5/97.5).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from auditworth.check import report
from auditworth.leverage import FLOOR, geometry, inventories, panel_vs_random

NAME = "participation_ratio"


def main() -> bool:
    pr = {(g, s, p): geometry(F)["PR"] for g, s, p, F in inventories()}
    units = [(g, pr[(g, s, p)], math.log(max(t2, FLOOR) / max(r2, FLOOR)))
             for g, s, p, n, t2, r2 in panel_vs_random(16)]
    rho = float(spearmanr([u[1] for u in units], [u[2] for u in units]).statistic)
    byg = defaultdict(list)
    for u in units:
        byg[u[0]].append(u)
    gids = sorted(byg)
    rng = np.random.default_rng(20260929)
    boots = []
    for _ in range(2000):
        us = [u for g in rng.choice(gids, len(gids), replace=True) for u in byg[g]]
        boots.append(spearmanr([u[1] for u in us], [u[2] for u in us]).statistic)
    return report(NAME, {"spearman": {"rho": rho, "lo": float(np.percentile(boots, 2.5)),
                                      "hi": float(np.percentile(boots, 97.5))}, "units": len(units)})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
