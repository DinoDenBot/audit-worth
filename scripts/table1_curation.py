"""Table I: credit gain, flip rate, and target effect (95% interval) of credit-seeking curation in ten studies.

Recomputes each row from the per-state curation rows in data/curation_studies/,
and the exact-Shapley credit gain of the 30-group study quoted in the caption.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auditworth.check import report
from auditworth.curation import EXACT_SHAPLEY_ARM, STUDIES, summarize

NAME = "table1_curation"


def main() -> bool:
    values = {"rows": {}}
    for stem, label, model, _, _ in STUDIES:
        s = summarize(stem)
        values["rows"][stem] = {
            "credit_gain": s["credit_gain"], "flip_rate": s["flip_rate"],
            "target_effect": s["target_effect"],
            "target_effect_ci_low": s["ci95"][0], "target_effect_ci_high": s["ci95"][1],
        }
        print(f"{label:26s} {model:28s} gain {s['credit_gain']:.4f}  flip {s['flip_rate']:.3f}  "
              f"effect {s['target_effect']:.5f} [{s['ci95'][0]:.5f}, {s['ci95'][1]:.5f}]")
    exact = summarize(EXACT_SHAPLEY_ARM)
    values["caption_exact_shapley_credit_gain_30groups"] = exact["credit_gain"]
    print(f"30-group study, exact-Shapley arm: credit gain {exact['credit_gain']:.4f}")
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
