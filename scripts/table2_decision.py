"""Table II: reports as accurate as 16 independent direct draws under the decision stake (left) and regret of the choice from reports or one audit per client (right).

Four clients, 30 groups, round 80, eps_pay = 1. Left: share of client
reports (feasible best responses, others truthful) whose squared error is at
most V/16, averaged within each group and then over groups. Right: expected
target regret of the exponential-mechanism choice from exact reports and from
one directly read record per client (400 draws). Caption: regret of a
uniformly random choice with its 95% t interval over groups.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

import numpy as np

from auditworth.check import report
from auditworth.decision import choice_by_group, unilateral_rows, within16_by_group
from auditworth.stats import t_interval

NAME = "table2_decision"
TABLE_EPS = (0.25, 1.0, 4.0, 16.0)
TABLE_RHOS = (30, 300, 3000)


def main() -> bool:
    rows = unilateral_rows()
    left = {}
    for eps in TABLE_EPS:
        for rho in TABLE_RHOS:
            by = within16_by_group((g, V, e) for g, _s, _i, ee, rr, V, e, _v, _b in rows if ee == eps and rr == rho)
            left[f"eps={eps:g}.rho={rho}"] = float(np.mean(list(by.values())))
    right = {"reports": {}, "one_audit": {}}
    for eps in TABLE_EPS + (math.inf,):
        label = "nonprivate" if math.isinf(eps) else f"eps={eps:g}"
        right["reports"][label] = float(np.mean(list(choice_by_group("reports", 1, eps).values())))
        right["one_audit"][label] = float(np.mean(list(choice_by_group("audits", 1, eps, 1).values())))
    uni = list(choice_by_group("uniform").values())
    lo, hi = t_interval(uni)
    values = {"within16": left, "regret": right,
              "random_choice": {"mean": float(np.mean(uni)), "lo": lo, "hi": hi}}
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
