"""Section V (panels of genuine records): the dimension bound is 9.5 to 37 times (median 24) the random-panel error.

For each nondegenerate inventory of the 30-group FEMNIST study (round 80,
one-epoch contract, seeds 2-5) the worst-case bound on the squared representation
radius, 2(s+1)B/k^2 with s the rank of the centered task vectors and B the largest
squared task-vector norm, is divided by the expected squared error of a random
k-panel, V(n-k)/(k(n-1)), at k = 16.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.leverage import geometry, inventories, random_panel_error

NAME = "v_dimension_bound"
K = 16


def main() -> bool:
    ratios = []
    for _, _, _, F in inventories():
        geo = geometry(F)
        if geo["V"] <= 0:
            continue
        bound = 2 * (geo["rank"] + 1) * geo["B_task"] / K**2
        ratios.append(bound / random_panel_error(geo["V"], geo["n"], K))
    r = np.array(ratios)
    return report(NAME, {"bound_over_random": {"min": float(r.min()), "median": float(np.median(r)),
                                               "max": float(r.max())}, "n_inventories": len(r)})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
