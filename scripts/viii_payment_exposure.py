"""Section VIII: expected net payment to a truthful FEMNIST client at beta/U = 300 (median and 90th percentile, in units of U).

With a fixed participation fee equal to the payment's expected value at a zero
report, a truthful client's expected net transfer is kappa ||mu_i||^2 with
kappa = beta tau(eps_pay) / (4B). With beta = 300 U, B = 4 (decision
coordinates only) and eps_pay = 1, this is 300 tanh(1/2) ||mu_i||^2 / 16 per
unit of U, over all 480 client inventories (30 groups x 4 seeds x 4 clients).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.decision import B, GROUPS, SEEDS, TAU_PAY, federation

NAME = "viii_payment_exposure"
RHO = 300


def main() -> bool:
    sq = np.array([float((m ** 2).sum()) for g in GROUPS for s in SEEDS for m in federation(g, s)[1]])
    net = RHO * TAU_PAY / (4 * B) * sq
    values = {"femnist": {"median_net_over_U": float(np.median(net)),
                          "p90_net_over_U": float(np.quantile(net, 0.9)), "n_inventories": int(len(net))}}
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
