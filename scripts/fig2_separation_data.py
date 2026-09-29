"""Figure 2: data for the separation figure (decision gain of exact reports vs candidate range, eps_dec = 1 and 4).

Each point is a four-client replay setting: x is the mean accuracy range between the best
and worst action on the target, y is the regret of the private choice from one directly
read record per client minus the regret from exact reports, with 95% t intervals over
groups (FEMNIST, Stack Overflow) or seeds (UCI HAR). The 30-group FEMNIST point uses the
round-80 choice comparison (five actions, equal client weights). Writes
results/fig2/separation-*.csv in the format the figure reads.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.crosssetting import (cross_setting, cross_stratum, femnist30_choice, grouped_gain, separation_value,
                                     strong_setting)
from auditworth.paths import RESULTS
from auditworth.stats import t_interval

NAME = "fig2_separation_data"
GROUPS = {"femnist30": ("femnist30",), "femnistscales": ("femnistx05", "femnistx1", "femnistx2"),
          "har": ("har",), "so": ("so",), "sostrong": ("sostrong",)}


def rows():
    out = []
    ch = femnist30_choice()
    rng30 = float(np.mean([r["range"] for r in ch]))
    for eps in (1.0, 4.0):
        # group means over seeds, then audits minus reports per group
        diffs = []
        for g in sorted({r["group"] for r in ch}):
            rr = [r for r in ch if r["group"] == g]
            diffs.append(np.mean([r[("audits", eps, 1)] for r in rr]) - np.mean([r[("reports", eps)] for r in rr]))
        out.append(("femnist30", eps, rng30, float(np.mean(diffs)), *t_interval(diffs)))
    vrows, _ = separation_value()
    for s, name in (("0.5", "femnistx05"), ("1", "femnistx1"), ("2", "femnistx2")):
        for eps in (1.0, 4.0):
            rr = [r for r in vrows if r["scale"] == s and r["eps"] == eps]
            g, ci, _ = grouped_gain(rr, lambda r: r["group"])
            out.append((name, eps, float(np.mean([r["range"] for r in rr])), g, *ci))
    _, decisions, _ = cross_setting()
    for setting, name in (("har", "har"), ("stackoverflow", "so")):
        for eps in (1.0, 4.0):
            ee = [d for d in decisions if d["setting"] == setting and d["eps"] == eps]
            g, ci, _ = grouped_gain(ee, cross_stratum(setting))
            out.append((name, eps, float(np.mean([d["range"] for d in ee])), g, *ci))
    _, sdec = strong_setting()
    for eps in (1.0, 4.0):
        sub = [d for d in sdec if d["target"] == "providers" and d["eps"] == eps]
        g, ci, _ = grouped_gain(sub, lambda r: r["group"])
        out.append(("sostrong", eps, float(np.mean([d["range"] for d in sub])), g, *ci))
    return out


def main() -> bool:
    data = rows()
    outdir = RESULTS / "fig2"
    outdir.mkdir(parents=True, exist_ok=True)
    values = {}
    for eps in (1.0, 4.0):
        with (outdir / f"separation-eps{int(eps)}.csv").open("w") as f:
            f.write("setting,range,gain,lo,hi\n")
            for r in data:
                if r[1] == eps:
                    f.write(f"{r[0]},{r[2]:.6g},{r[3]:.6g},{r[4]:.6g},{r[5]:.6g}\n")
                    values[f"eps{int(eps)}.{r[0]}"] = dict(range=r[2], gain=r[3], lo=r[4], hi=r[5])
        for gname, members in GROUPS.items():
            with (outdir / f"separation-{gname}-eps{int(eps)}.csv").open("w") as f:
                f.write("range,gain,lo,hi\n")
                for r in data:
                    if r[1] == eps and r[0] in members:
                        f.write(f"{r[2]:.6g},{r[3]:.6g},{r[4]:.6g},{r[5]:.6g}\n")
    values["femnist30_candidate_range"] = values["eps4.femnist30"]["range"]
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
