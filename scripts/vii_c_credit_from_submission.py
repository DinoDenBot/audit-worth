"""Section VII-C and Figure 3: credit computed from the submission (report versus panel as the payment ratio falls).

30-group FEMNIST study, round 80, one-epoch contract, seeds 2-5, eight-coordinate
task vector (B = 8, eps_pay = 1). Each client also values its own Shapley
coordinate with stake ratio nu = b/kappa on the grid {0} U {10^(x/4): x=-12..4},
i.e. payment ratio beta/b = 4B/(tau nu). Reports best-respond over the bounded
16-coalition score polytope; panels (k = 16) best-respond by greedy search over
genuine records. Errors are pooled squared errors relative to one direct read,
averaged over groups with 95% t intervals. Writes results/fig3_leverage_curve.csv
(the plotted curves) in addition to the checked values.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

import numpy as np

from auditworth.check import report
from auditworth.leverage import B, FLOOR, LAMBDAS, TAU, beta_over_b, summarize, sweep
from auditworth.paths import RESULTS

NAME = "vii_c_credit_from_submission"
K = 16


def main() -> bool:
    units, rows = sweep((K,), LAMBDAS, reports=True)
    curve = [(lam, summarize(units, rows, K, lam)) for lam in LAMBDAS]

    # Figure 3 data (one row per positive stake ratio, highest payment ratio first).
    RESULTS.mkdir(exist_ok=True)
    lines = ["beta_over_b,rep,rep_lo,rep_hi,pan,pan_lo,pan_hi,rand"]
    fig = {}
    for i, (lam, s) in enumerate(c for c in curve if c[0] > 0):
        r, p, q = s["pooled_ratio"]["rep"], s["pooled_ratio"]["pan"], s["pooled_ratio"]["rand"]
        vals = [beta_over_b(lam), *r, *p, q[0]]
        lines.append(",".join(f"{v:.6g}" for v in vals))
        fig[f"row{i:02d}"] = dict(zip(("beta_over_b", "rep", "rep_lo", "rep_hi", "pan", "pan_lo", "pan_hi",
                                       "rand"), (float(f"{v:.6g}") for v in vals)))
    (RESULTS / "fig3_leverage_curve.csv").write_text("\n".join(lines) + "\n")

    # Low-stake interior: payment ratio needed for the report to stay as accurate as 16 direct reads,
    # nu_16 = 2 sqrt(V/16), beta/b = 2B sqrt(16/V) / tau; credit inflation nu_16/2 (over inventories).
    live = [(V, sd) for _, V, sd in units.values() if V >= FLOOR]
    nu16 = np.array([2 * math.sqrt(V / 16) for V, _ in live])
    bb16 = 4 * B / (TAU * nu16)
    positive = [c for c in curve if c[0] > 0]
    # Panels beat reports on pooled error from this grid point downwards in beta/b.
    cross = next(i for i, (lam, s) in enumerate(positive)
                 if s["pooled_ratio"]["pan"][0] < s["pooled_ratio"]["rep"][0])
    # Panels become worse than random panels from this grid point downwards.
    worse = next(i for i, (lam, s) in enumerate(positive)
                 if s["pooled_ratio"]["pan"][0] > s["pooled_ratio"]["rand"][0])
    top = dict(curve)[LAMBDAS[-1]]
    values = {
        "threshold_16_direct_reads": {
            "beta_over_b_median": float(np.median(bb16)),
            "beta_over_b_q25": float(np.quantile(bb16, 0.25)),
            "beta_over_b_q75": float(np.quantile(bb16, 0.75)),
            "credit_inflation_median": float(np.median(nu16)) / 2},
        "nu10": {"report_error": top["pooled_ratio"]["rep"][0], "box_report_error": top["pooled_ratio"]["box"][0],
                 "report_credit_gain": top["credit_gain"]["rep"][0],
                 "box_credit_gain": top["credit_gain"]["box"][0],
                 "panel_credit_gain": top["credit_gain"]["pan"][0]},
        "crossing": {"panels_better_from_beta_over_b": beta_over_b(positive[cross][0]),
                     "share_panel_better_there": positive[cross][1]["share_panel_better"][0],
                     "reports_better_above_beta_over_b": beta_over_b(positive[cross - 1][0])},
        "panel_worse_than_random": {"still_better_at_beta_over_b": beta_over_b(positive[worse - 1][0]),
                                    "worse_at_beta_over_b": beta_over_b(positive[worse][0])},
        "one_audit_credit_sd_median": float(np.median([sd for _, sd in live])),
        "fig3": fig,
    }
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
