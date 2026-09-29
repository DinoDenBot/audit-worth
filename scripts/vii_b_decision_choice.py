"""Section VII-B, "Does report accuracy improve the choice?": regret of the choice from exact reports versus 1, 4 or 16 audits per client.

Truthful reports versus the mean of r in {1, 4, 16} records drawn with
replacement from each client's inventory (400 seeded draws per federation),
fed to the same exponential mechanism, at eps_dec in {0.25, ..., 16} and without
privacy (argmax, ties split), in the four-client federations and in replicated
federations of 16 and 64 clients (each replica draws its own audits). Paired
differences (audits minus reports) use 95% t intervals over the 30 groups.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

import numpy as np

from auditworth.check import report
from auditworth.decision import AUDITS, EPS, choice_by_group
from auditworth.stats import t_interval


NAME = "vii_b_decision_choice"


def _mean(d: dict) -> float:
    return float(np.mean(list(d.values())))


def main() -> bool:
    small = []
    for eps in EPS:
        if eps > 4:
            continue
        rep = choice_by_group("reports", 1, eps)
        for r in AUDITS:
            aud = choice_by_group("audits", 1, eps, r)
            small.append(abs(np.mean([aud[g] - rep[g] for g in rep])))
    max_small = float(max(small))

    rep16, aud16 = choice_by_group("reports", 1, 16.0), choice_by_group("audits", 1, 16.0, 1)
    diff = [aud16[g] - rep16[g] for g in rep16]
    lo, hi = t_interval(diff)

    values = {
        "n4_eps_le_4": {"max_abs_diff": max_small, "at_most_0.0001": float(max_small <= 0.0001)},
        "n4_eps16": {"reports": _mean(rep16), "one_audit": _mean(aud16),
                     "diff": {"mean": float(np.mean(diff)), "lo": lo, "hi": hi}},
        "n4_nonprivate": {"reports": _mean(choice_by_group("reports", 1, math.inf)),
                          "one_audit": _mean(choice_by_group("audits", 1, math.inf, 1)),
                          "sixteen_audits": _mean(choice_by_group("audits", 1, math.inf, 16))},
        "n64_eps4": {"reports": _mean(choice_by_group("reports", 16, 4.0)),
                     "one_audit": _mean(choice_by_group("audits", 16, 4.0, 1))},
    }
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
