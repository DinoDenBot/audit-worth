"""Section IV: payment change for degrading a report to the 16-draw reference, and the cost of preparing a report.

Under the recommended mechanism a report carries the four candidates' accuracy
differences from keep-current (B = 4). The expected payment falls by
kappa V_i / 16 with slope kappa = beta tau(eps_pay)/(4B), eps_pay = 1, at the
median variance V_i of the nondegenerate inventories of the 30-group FEMNIST
study (round 80, one-epoch local training, seeds 2-5). A client that reports the
mean of 16 random records instead of evaluating every record loses exactly this
amount, so full evaluation is worth its cost only if the payment scale exceeds
16 / (kappa V_i / beta) times the computation saved.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

import numpy as np

from auditworth import data
from auditworth.check import report

NAME = "payment_slope"
B = 4.0
TAU = math.tanh(0.5)


def main() -> bool:
    V = []
    for g in data.FEMNIST30_GROUPS:
        for s in data.FEMNIST30_SEEDS:
            for prov in data.femnist30_state(g, 80, s):
                F = (prov["bits"][1:] - prov["bits"][0]).T
                V.append(float(((F - F.mean(0)) ** 2).sum(1).mean()))
    V = np.array(V)
    Vmed = float(np.median(V[V > 1e-12]))
    per_beta = TAU / (4 * B) * Vmed / 16
    return report(NAME, {"median_V": Vmed, "payment_change_per_beta": per_beta,
                         "beta_over_cost_saved": 1 / per_beta,
                         "n_nondegenerate_inventories": int((V > 1e-12).sum())})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
