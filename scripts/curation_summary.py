"""Section VII-A and Section I: credit rose in all ten primary contrasts; panel-over-target excess grew in all eight evaluable ones.

Also the largest credit gain ("up to 0.2") and the 30-group study's credit gain.
"""
import math
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auditworth.check import report
from auditworth.curation import STUDIES, summarize

NAME = "curation_summary"


def main() -> bool:
    summaries = {stem: summarize(stem) for stem, *_ in STUDIES}
    gains = [s["credit_gain"] for s in summaries.values()]
    # Excess of panel over target credit is evaluable where the study's credit
    # rule could also be computed on a target reference.
    evaluable = [s for s in summaries.values()
                 if not (math.isnan(s["excess_committed"]) or math.isnan(s["excess_curated"]))]
    values = {
        "primary_contrasts": len(gains),
        "credit_gain_positive": sum(g > 0 for g in gains),
        "excess_evaluable": len(evaluable),
        "excess_grew": sum(s["excess_curated"] > s["excess_committed"] for s in evaluable),
        "max_credit_gain": max(gains),
        "femnist30_credit_gain": summaries["femnist_cnn_30groups_keep_current"]["credit_gain"],
    }
    for k, v in values.items():
        print(f"{k:24s} {v}")
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
