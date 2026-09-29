"""Section VII-B, "A stronger text model": Stack Overflow replay with a 120-round sparse character-feature model.

Model quality (equal-user working-inventory accuracy per group, with the earlier MLP and
the best constant predictor) is read from data/stackoverflow_strong/quality.json, since
it comes from training rather than from the score caches. Panel leverage and the
report-vs-one-audit choice gain are recomputed from the score caches as in the other
text replay; intervals are 95% t intervals over the eight federations. Also evaluated:
the prespecified model-quality and candidate-variation gates.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np

from auditworth.check import report
from auditworth.crosssetting import STRONG_EPS, distinct_equivalent, grouped_gain, pooled_ratio, strong_setting
from auditworth.paths import DATA
from auditworth.stats import t_interval

NAME = "vii_b_strong_textmodel"


def summary(values):
    lo, hi = t_interval(values)
    return dict(mean=float(np.mean(values)), lo=lo, hi=hi)


def main() -> bool:
    q = json.loads((DATA / "stackoverflow_strong" / "quality.json").read_text())["groups"]
    new = [g["sparse_model_accuracy"] for g in q]
    old = [g["mlp_accuracy"] for g in q]
    const = [g["best_constant_accuracy"] for g in q]
    quality = dict(model_accuracy=summary(new), mlp_accuracy=summary(old), constant_accuracy=summary(const),
                   model_minus_mlp=summary([a - b for a, b in zip(new, old)]),
                   groups_above_constant_by_0_05=int(sum(a - c >= .05 for a, c in zip(new, const))))

    units, decisions = strong_setting(with_panels=True)
    values = dict(quality=quality, n_units=len(units), n_decision_rows=len(decisions),
                  n_zero_variance=sum(u["V"] < 1e-12 for u in units),
                  panel_equivalent_reads=1 / pooled_ratio(units),
                  panel_equivalent_distinct_reads=distinct_equivalent([u for u in units if u["V"] >= 1e-12]),
                  random_equivalent_reads=1 / pooled_ratio(units, "random_error"))
    for target in ("providers", "full"):
        for eps in STRONG_EPS:
            sub = [d for d in decisions if d["target"] == target and d["eps"] == eps]
            g, ci, _ = grouped_gain(sub, lambda r: r["group"])
            cell = {m: float(np.mean([d[m] for d in sub])) for m in ("exact", "audit1", "audit16", "uniform", "range")}
            cell.update(gain=g, gain_lo=ci[0], gain_hi=ci[1], gain_ci_crosses_zero=int(ci[0] < 0 < ci[1]))
            values[f"{target}_eps{'inf' if np.isinf(eps) else f'{eps:g}'}"] = cell
    values["gate_model_quality"] = int(quality["model_minus_mlp"]["mean"] >= .05
                                       and quality["model_accuracy"]["mean"] - quality["constant_accuracy"]["mean"] >= .12
                                       and quality["groups_above_constant_by_0_05"] >= 7)
    values["gate_candidate_variation"] = int(values["providers_eps4"]["range"] >= .02)
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
