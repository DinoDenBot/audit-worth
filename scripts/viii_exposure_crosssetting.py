"""Section VIII: expected net payment to a truthful client at beta/U = 300 in UCI HAR, Stack Overflow, and scaled FEMNIST.

With a participation fee equal to the payment's expected value at a zero report, the
expected net transfer to a truthful client is kappa * ||mu_i||^2 with
kappa = beta * tau(eps_pay) / (4B), here B = 3 (three accuracy differences), eps_pay = 1,
beta = 300 U. Reported relative to the stake U: median and 90th percentile over all
provider inventories. The update-scaling FEMNIST values are computed for completeness
(one row per scale); the paper does not report them.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.crosssetting import SCALES, TAU_PAY, cross_setting, separation_value

NAME = "viii_exposure_crosssetting"
B = 3.0
KAPPA_OVER_U = 300 * TAU_PAY / (4 * B)


def summarize(mu_sq):
    x = KAPPA_OVER_U * np.asarray(mu_sq)
    return dict(median_net_over_U=float(np.median(x)), p90_net_over_U=float(np.quantile(x, .9)), n=len(x))


def main() -> bool:
    _, _, mus = cross_setting()
    values = {}
    for setting, short in (("har", "har"), ("stackoverflow", "so")):
        values[short] = summarize([float((m ** 2).sum()) for k, mu in mus.items() if k[0] == setting for m in mu])
    _, states = separation_value()
    values["femnist_scaled"] = {f"x{s}": summarize([float((m ** 2).sum()) for k, (mu, _) in states.items()
                                                    if k[1] == s for m in mu]) for s in SCALES}
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
