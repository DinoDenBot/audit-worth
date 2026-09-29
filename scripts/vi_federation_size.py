"""Section VI, "Federation size": halving the regret of a random choice in the FEMNIST replay needs N*eps_dec of about 256.

With exact (truthful) reports the choice's regret depends on N and eps_dec only
through N*eps_dec. Using the replicated federations (N = 4, 16, 64 clients;
eps_dec in {0.25, ..., 16}), we find the smallest N*eps_dec on this grid at which
the mean regret over groups is at most half that of a uniformly random choice.
For reference we also locate the crossing on a continuous N*eps_dec scale by
bisection of the exact four-client law at eps_dec = N*eps_dec / 4.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.optimize import brentq

from auditworth.check import report
from auditworth.decision import (CHOICE_COPIES, EPS, GROUPS, SEEDS, choice_by_group, choice_probs, federation)

NAME = "vi_federation_size"


def main() -> bool:
    random_regret = float(np.mean(list(choice_by_group("uniform").values())))
    half = random_regret / 2
    curve = {}
    for c in CHOICE_COPIES:
        for eps in EPS:
            curve.setdefault(4 * c * eps, float(np.mean(list(choice_by_group("reports", c, eps).values()))))
    grid = sorted(curve)
    halving = next(x for x in grid if curve[x] <= half)

    targets = []
    for g in GROUPS:
        for s in SEEDS:
            mu = federation(g, s)[1]
            targets.append((g, np.concatenate([[0.0], mu.mean(0)])))

    def mean_regret(n_eps: float) -> float:
        by = {}
        for g, t in targets:  # four clients at eps = n_eps / 4: theta = eps * N / 2 = n_eps / 2
            by.setdefault(g, []).append(float(t.max() - choice_probs(t, n_eps / 2) @ t))
        return float(np.mean([np.mean(v) for v in by.values()]))

    crossing = brentq(lambda x: mean_regret(x) - half, 1.0, 4096.0)
    values = {"random_regret": random_regret,
              "regret_by_n_eps": {f"{x:g}": curve[x] for x in grid},
              "halving_n_eps_grid": halving, "halving_n_eps_continuous": crossing}
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
