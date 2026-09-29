"""Section IV (leverage and its price): payment change for distorting a report to the accuracy of 16 direct reads.

The expected payment falls by kappa V_i / 16 with slope kappa = beta tau(eps_pay)/(4B).
With the eight-coordinate task vector (B = 8), eps_pay = 1 and the median inventory
variance V_i of the 30-group FEMNIST study (round 80, one-epoch contract, seeds 2-5,
nondegenerate inventories), this gives the payment change per unit of beta.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.leverage import B, FLOOR, TAU, inventories, variance

NAME = "iv_payment_slope"


def main() -> bool:
    V = np.array([variance(F) for _, _, _, F in inventories()])
    Vmed = float(np.median(V[V >= FLOOR]))
    return report(NAME, {"median_V": Vmed, "payment_change_per_beta": TAU / (4 * B) * Vmed / 16,
                         "n_nondegenerate_inventories": int((V >= FLOOR).sum())})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
