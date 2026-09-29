"""Table I and Section VI: mean top gap (best minus second-best action) and range of each replay setting, and the size of theta*Delta at eps_dec = 4.

For every analysed state the target advantage of each action over the reference
is the equal-weight mean, over the four clients' complete evaluation sets, of the
per-record correctness differences. Range = best minus worst action; top gap =
best minus second best (the Delta of Proposition 12). With four equal-weight
clients the private choice uses theta = eps_dec / (2 pi_max) = 2 eps_dec.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth import data
from auditworth.check import report

NAME = "top_gap"


def gaps(adv):
    t = np.sort(np.r_[0.0, adv])[::-1]
    return t[0] - t[1], t[0] - t[-1]


def summarize(advs):
    g = np.array([gaps(a) for a in advs])
    return {"top_gap": float(g[:, 0].mean()), "range": float(g[:, 1].mean())}


def main() -> bool:
    out = {}
    advs = []
    for g in data.FEMNIST30_GROUPS:
        for s in data.FEMNIST30_SEEDS:
            st = data.femnist30_state(g, 80, s)
            advs.append(np.mean([((p["bits"][1:] - p["bits"][0]).T).mean(0) for p in st], 0))
    out["femnist30"] = summarize(advs)
    for sc in ("0.5", "1", "2"):
        advs = []
        for g in (1, 2, 3, 4):
            for z in range(4, 12):
                for a in range(4):
                    d = data.femnist_scaled_state(g, sc, z, a)
                    advs.append(np.mean([((d[f"candidate_bits_{p}"][1:].astype(float) - d[f"candidate_bits_{p}"][0]).T).mean(0)
                                         for p in range(4)], 0))
        out[f"femnist_scaled_{sc}"] = summarize(advs)
    out["har"] = summarize([np.mean([((b[1:] - b[0]).T).mean(0) for b, _ in data.har_state(seed, a)], 0)
                            for seed in range(8, 28) for a in range(4)])
    out["stackoverflow"] = summarize([np.mean([((b[1:] - b[0]).T).mean(0) for b, _ in data.stackoverflow_state(g, seed, a)], 0)
                                      for g in range(1, 9) for seed in range(4, 20) for a in range(4)])
    out["max_theta_delta_eps4"] = 2 * 4.0 * max(v["top_gap"] for v in out.values())
    return report(NAME, out)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
