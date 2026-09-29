"""Section VII-B, "Two other task settings": panel leverage and report-vs-audit choice gain on UCI HAR and Stack Overflow.

With no competing stake, a payment-seeking 16-record panel is compared with random
16-record panels (pooled error relative to one directly read record; its inverse is
the number of direct reads matched). The private four-action choice made from exact
reports is compared with the same choice made from one or 16 directly read records
per provider. Intervals are 95% t intervals over the 20 HAR seeds or the eight Stack
Overflow federations.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.crosssetting import (CROSS_EPS, cross_setting, cross_stratum, distinct_equivalent, grouped_gain,
                                     pooled_ratio)

NAME = "vii_b_settings_leverage"


def main() -> bool:
    units, decisions, _ = cross_setting(with_panels=True)
    values = {}
    for setting, short in (("har", "har"), ("stackoverflow", "so")):
        sub = [u for u in units if u["setting"] == setting]
        valid = [u for u in sub if u["V"] >= 1e-12]  # constant inventories are excluded from ratios
        ds = [d for d in decisions if d["setting"] == setting]
        out = dict(n_units=len(sub), n_zero_variance=len(sub) - len(valid), n_states=len(ds) // len(CROSS_EPS),
                   panel_equivalent_reads=1 / pooled_ratio(valid),
                   panel_equivalent_distinct_reads=distinct_equivalent(valid),
                   random_equivalent_reads=1 / pooled_ratio(valid, "random_error"),
                   panel_better_share=float(np.mean([u["panel_error"] < u["random_error"] for u in valid])))
        for eps in CROSS_EPS:
            ee = [d for d in ds if d["eps"] == eps]
            key = "inf" if np.isinf(eps) else f"{eps:g}"
            cell = {m: float(np.mean([d[m] for d in ee])) for m in ("exact", "audit1", "audit16", "uniform", "range")}
            for a in ("audit1", "audit16"):
                g, ci, _ = grouped_gain(ee, cross_stratum(setting), a=a)
                cell[f"{a}_minus_exact"] = g
                cell[f"{a}_minus_exact_lo"], cell[f"{a}_minus_exact_hi"] = ci
            out[f"eps{key}"] = cell
        out["candidate_range"] = out["eps4"]["range"]
        values[short] = out
    values["total_units"] = len(units)
    values["total_decision_rows"] = len(decisions)
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
