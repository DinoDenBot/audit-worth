"""Section VII-B, "Separation within FEMNIST": own-update decision stake in the update-scaling contract.

Each provider values adoption of its own update at U payment units (B = 3, eps_pay = 1,
feasible reports). Unilateral best responses (others truthful) give each report's squared
distortion relative to one directly read record; simultaneous best responses (up to 30
sweeps from truthful reports) give the regret of the private choice. Runtime: several
minutes (numerical best responses).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.crosssetting import (SCALES, best_response, grouped_gain, probs, separation_value, simultaneous,
                                     stake_lambda)

NAME = "vii_b_separation_stake"
EPS = (1.0, 4.0, 16.0)
RHOS = (30, 100, 300, 1000, 3000)


def main() -> bool:
    vrows, states = separation_value()
    a1 = {(r["group"], r["scale"], r["seed"], r["state"], r["eps"]): r["audit1"] for r in vrows}
    srows, urows = [], []
    for key in sorted(states, key=lambda k: (k[0], SCALES.index(k[1]), k[2], k[3])):
        mu, V = states[key]
        truth = np.r_[0.0, mu.mean(0)]
        for eps in EPS:
            theta = 2 * eps
            rt = float(truth.max() - probs(truth, eps) @ truth)
            for rho in RHOS:
                lam = stake_lambda(rho)
                for p in range(4):
                    m = best_response(p, mu, mu[p], theta, lam, p, 3)
                    urows.append(dict(scale=key[1], eps=eps, rho=rho, V=V[p], e=float(((m - mu[p]) ** 2).sum())))
                reps, conv, _ = simultaneous(mu, theta, lam)
                rb = float(truth.max() - probs(np.r_[0.0, reps.mean(0)], eps) @ truth)
                srows.append(dict(group=key[0], scale=key[1], eps=eps, rho=rho, truthful=rt, br=rb,
                                  audit1=a1[key + (eps,)], conv=conv))
    values = {}
    for s in SCALES:
        out = {}
        for eps in EPS:
            shifts = []
            for rho in RHOS:
                rr = [r for r in srows if r["scale"] == s and r["eps"] == eps and r["rho"] == rho]
                uu = [u for u in urows if u["scale"] == s and u["eps"] == eps and u["rho"] == rho and u["V"] > 1e-12]
                g, ci, _ = grouped_gain(rr, lambda r: r["group"], a="audit1", b="br")
                rt = float(np.mean([r["truthful"] for r in rr]))
                rb = float(np.mean([r["br"] for r in rr]))
                shifts.append(abs(rb - rt))
                out[f"eps{eps:g}_rho{rho}"] = dict(
                    regret_truthful=rt, regret_br=rb, gain_br=g, gain_lo=ci[0], gain_hi=ci[1],
                    nonconverged_share=float(np.mean([not r["conv"] for r in rr])),
                    pooled_distortion_over_V=float(sum(u["e"] for u in uu) / sum(u["V"] for u in uu)),
                    within16_share=float(np.mean([u["e"] <= u["V"] / 16 for u in uu])))
            out[f"eps{eps:g}_max_regret_shift"] = max(shifts)
        values[f"x{s}"] = out
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
