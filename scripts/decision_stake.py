"""Section VII-B, 'What payment does the decision stake require?', and Appendix D (replicated federations; box responses outside M_dec).

Feasible best responses of FEMNIST clients that value adoption of their own
update, over eps_dec in {0.25, ..., 16} and beta/U in {10, ..., 10000}:
Corollary 6 check; payment ratios that keep every report as accurate as 16 independent
direct draws; iterated best responses of all four clients (regret shift and
non-convergence within 30 sweeps); regret of the truthful choice at eps_dec = 16
and in replicated federations; distortion (pooled error relative to one direct
audit) at 4, 16 and 64 replicated clients; and the share of best responses over
the box [-1, 1]^4 that leave M_dec.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.decision import (EPS, FLOOR, RHOS, clone_rows, dynamics_rows, group_means,
                                 pooled_error_by_group, unilateral_rows, within16_by_group)
from auditworth.stats import t_interval

NAME = "decision_stake"
SHIFT_TOL = 0.0005


def _mean(by_group: dict) -> float:
    return float(np.mean(list(by_group.values())))


def main() -> bool:
    units = unilateral_rows()
    valid = [r for r in units if r[5] > FLOOR]
    ratios = [r[6] / r[8] for r in valid]
    cor6 = {"violations": int(sum(r[6] > r[8] + 1e-9 for r in valid)),
            "median_e_over_bound": float(np.median(ratios)), "max_e_over_bound": float(np.max(ratios))}

    # smallest payment ratio on the grid from which on every report is within 16 direct reads
    share = {(eps, rho): _mean(within16_by_group((g, V, e) for g, _s, _i, ee, rr, V, e, _v, _b in units
                                                 if ee == eps and rr == rho))
             for eps in EPS for rho in RHOS}
    required = {eps: min(rho for rho in RHOS if all(share[(eps, r)] == 1.0 for r in RHOS if r >= rho))
                for eps in EPS}
    payments = {"required_eps=0.25": required[0.25],
                "required_max_over_eps_le_4": max(required[e] for e in EPS if e <= 4),
                "required_eps=16": required[16.0]}

    # simultaneous best responses: mean regret shift per cell (over groups)
    feds = dynamics_rows()
    shift = {}
    for eps in EPS:
        for rho in RHOS:
            cell = [r for r in feds if r[2] == eps and r[3] == rho]
            shift[(eps, rho)] = _mean(group_means((r[0], r[5] - r[4]) for r in cell))
    exception = {k for k in shift if k[1] <= 100 and k[0] >= 8}
    outside = max(abs(v) for k, v in shift.items() if k not in exception)
    largest = max(shift, key=lambda k: abs(shift[k]))
    simultaneous = {"max_abs_shift_outside_exception": outside,
                    "within_0.0005_outside_exception": float(outside <= SHIFT_TOL),
                    "largest_abs_shift": abs(shift[largest]), "largest_shift_signed": shift[largest],
                    "largest_shift_cell": f"eps={largest[0]:g},rho={largest[1]}",
                    "nonconverged_share": float(np.mean([not r[6] for r in feds]))}

    truthful16 = group_means((r[0], r[4]) for r in feds if r[2] == 16.0 and r[3] == RHOS[0])
    lo, hi = t_interval(list(truthful16.values()))
    regret16 = {"mean": _mean(truthful16), "lo": lo, "hi": hi}

    clones = clone_rows()

    def clone_regret(c, eps):
        return _mean(group_means((r[0], r[8]) for r in clones if r[2] == c and r[3] == eps and r[4] == RHOS[0]
                                 and r[5] == 0))

    replicated = {"N=16.eps=16": clone_regret(4, 16.0), "N=64.eps=4": clone_regret(16, 4.0),
                  "N=64.eps=16": clone_regret(16, 16.0)}

    distortion = {}
    for eps in (4.0, 16.0):
        distortion[f"eps={eps:g}.N=4"] = _mean(pooled_error_by_group(
            (r[0], r[5], r[6]) for r in units if r[3] == eps and r[4] == 300))
        for c in (4, 16):
            distortion[f"eps={eps:g}.N={4 * c}"] = _mean(pooled_error_by_group(
                (r[0], r[6], r[7]) for r in clones if r[2] == c and r[3] == eps and r[4] == 300))

    out = [r for r in valid if r[7] > 1e-9]
    box = {"outside_M_share": len(out) / len(valid),
           "share_of_outside_in_eps4_rho10": (sum(r[3] == 4.0 and r[4] == 10 for r in out) / len(out))
           if out else float("nan")}

    values = {"cor6": cor6, "payments": payments, "simultaneous": simultaneous, "regret_eps16": regret16,
              "replicated_regret": replicated, "distortion_rho300": distortion, "box": box,
              "n_unit_rows": len(units), "n_valid_unit_rows": len(valid)}
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
