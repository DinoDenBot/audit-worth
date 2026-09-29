"""Section VII-B: the decision stake on UCI HAR and Stack Overflow.

Provider p values adoption of candidate p at U payment units (the candidates are not the
providers' own updates, so the stake is assigned by index). Reports carry the three
accuracy differences from candidate 0 (B = 3), eps_pay = 1, and each provider's report
maximizes -||m - mu_p||^2 + lambda * P(candidate p), lambda = 4B / (tau(eps_pay) * beta/U),
over the feasible set. All four providers best-respond simultaneously, iterated from
truthful reports for up to 30 sweeps; the last iterate feeds the private choice. The gain
is the regret of the choice from one direct read per provider minus the regret from the
best-response reports. Runtime: several minutes (numerical best responses).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.crosssetting import (cross_setting, cross_stratum, grouped_gain, probs, simultaneous,
                                     stake_lambda)

NAME = "settings_stake"
EPS = (1.0, 4.0, 16.0)
RHOS = (30, 100, 300, 1000, 3000)


def main() -> bool:
    _, decisions, mus = cross_setting()
    a1 = {(d["setting"], d["group"], d["seed"], d["state"], d["eps"]): d["audit1"] for d in decisions}
    rows = []
    for key in sorted(mus):
        mu = mus[key]
        truth = np.r_[0.0, mu.mean(0)]
        for eps in EPS:
            rt = float(truth.max() - probs(truth, eps) @ truth)
            for rho in RHOS:
                reps, conv, _ = simultaneous(mu, 2 * eps, stake_lambda(rho))
                rb = float(truth.max() - probs(np.r_[0.0, reps.mean(0)], eps) @ truth)
                rows.append(dict(setting=key[0], group=key[1], seed=key[2], state=key[3], eps=eps, rho=rho,
                                 truthful=rt, br=rb, audit1=a1[key + (eps,)], conv=conv))
    values = {}
    for setting, short in (("har", "har"), ("stackoverflow", "so")):
        out = {}
        for eps in EPS:
            shifts = []
            for rho in RHOS:
                cell = [r for r in rows if r["setting"] == setting and r["eps"] == eps and r["rho"] == rho]
                g, ci, _ = grouped_gain(cell, cross_stratum(setting), a="audit1", b="br")
                rt = float(np.mean([r["truthful"] for r in cell]))
                rb = float(np.mean([r["br"] for r in cell]))
                shifts.append(abs(rb - rt))
                out[f"eps{eps:g}_rho{rho}"] = dict(regret_truthful=rt, regret_br=rb, gain_br_over_audit1=g,
                                                   gain_lo=ci[0], gain_hi=ci[1],
                                                   nonconverged_share=float(np.mean([not r["conv"] for r in cell])))
            # largest shift of mean regret from its truthful value over beta/U in 30..3000
            out[f"eps{eps:g}_max_regret_shift"] = max(shifts)
        values[short] = out
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
