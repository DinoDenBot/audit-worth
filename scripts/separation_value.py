"""Section VII-C: candidate ranges and the gain of exact reports as FEMNIST candidate updates are scaled apart.

The update-scaling FEMNIST contract (four 64-writer groups, round-160 CNN, seeds 4-11,
four states) offers the four contributors' own one-epoch updates scaled by 0.5, 1, or 2.
The target is the four providers' complete inventories; the gain is the regret of the
private choice from one directly read record per provider minus that from exact reports,
with 95% t intervals over the four groups. The secondary target is the declared
64-writer target accuracy.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.crosssetting import SCALES, SEP_EPS, grouped_gain, separation_value
from auditworth.stats import t_interval

NAME = "separation_value"


def key(eps):
    return "inf" if np.isinf(eps) else f"{eps:g}"


def main() -> bool:
    rows, states = separation_value()
    by_group = lambda r: r["group"]
    values = {"n_states": len(states)}
    for s in SCALES:
        out = {}
        for eps in SEP_EPS:
            rr = [r for r in rows if r["scale"] == s and r["eps"] == eps]
            g, ci, per = grouped_gain(rr, by_group)
            gs, cis, _ = grouped_gain(rr, by_group, a="audit1_sec", b="exact_sec")
            out[f"eps{key(eps)}"] = dict(
                exact=float(np.mean([r["exact"] for r in rr])), audit1=float(np.mean([r["audit1"] for r in rr])),
                audit16=float(np.mean([r["audit16"] for r in rr])), uniform=float(np.mean([r["uniform"] for r in rr])),
                gain=g, gain_lo=ci[0], gain_hi=ci[1], gain_by_group=per,
                gain_secondary=gs, gain_secondary_lo=cis[0], gain_secondary_hi=cis[1])
        out["range"] = float(np.mean([r["range"] for r in rows if r["scale"] == s and r["eps"] == 4.0]))
        values[f"x{s}"] = out
    for eps in SEP_EPS:
        d = {}
        for scale in ("0.5", "2"):
            _, _, per = grouped_gain([r for r in rows if r["scale"] == scale and r["eps"] == eps], lambda r: r["group"])
            d[scale] = per
        diffs = [b - a for a, b in zip(d["0.5"], d["2"])]
        ci = t_interval(diffs)
        values[f"diff_x2_minus_x0.5_eps{key(eps)}"] = dict(mean=float(np.mean(diffs)), lo=ci[0], hi=ci[1],
                                                          by_group=diffs,
                                                          groups_positive=int(sum(x > 0 for x in diffs)))
    values["max_gain_eps_le_1"] = max(values[f"x{s}"][f"eps{e}"]["gain"] for s in SCALES for e in ("0.5", "1"))
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
