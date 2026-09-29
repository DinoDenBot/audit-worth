"""Section VI (Proposition 13) and Section VII-C: second-order prediction of the gain of exact reports over one audit per client at small privacy budgets, against the simulated gain.

For each state, the private choice from exact advantages t is compared with the
same choice from one uniformly drawn record per client (equal weights), with
common random draws across budgets. The prediction is
-(theta^2/2) Cov_A(w, t) with w_a the variance of action a's audit error
relative to the mean error over actions, theta = 2 eps_dec (four clients).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth import data
from auditworth.check import report

NAME = "small_budget"
DRAWS = 3000
EPS = (0.25, 0.5, 1.0)


def settings():
    yield "femnist30", ([(p["bits"][1:] - p["bits"][0]).T for p in data.femnist30_state(g, 80, s)]
                        for g in data.FEMNIST30_GROUPS for s in data.FEMNIST30_SEEDS)
    yield "har", ([(b[1:] - b[0]).T for b, _ in data.har_state(seed, a)] for seed in range(8, 28) for a in range(4))
    yield "stackoverflow", ([(b[1:] - b[0]).T for b, _ in data.stackoverflow_state(g, seed, a)]
                            for g in range(1, 9) for seed in range(4, 20) for a in range(4))


def main() -> bool:
    rng = np.random.default_rng(7)
    out = {}
    for name, gen in settings():
        sim = np.zeros(len(EPS)); pred = 0.0; n = 0
        for F in gen:
            t = np.r_[0.0, np.mean([f.mean(0) for f in F], 0)]
            est = np.zeros((DRAWS, len(t) - 1)); w = np.zeros(len(t))
            for f in F:
                est += f[rng.integers(len(f), size=DRAWS)] / len(F)
                Z = np.column_stack([np.zeros(len(f)), f])
                w += (Z - Z.mean(1, keepdims=True)).var(0) / len(F) ** 2
            S = np.column_stack([np.zeros(DRAWS), est])
            for j, e in enumerate(EPS):
                th = 2 * e
                pe = np.exp(th * (t - t.max())); pe /= pe.sum()
                z = th * S; z -= z.max(1, keepdims=True); pa = np.exp(z); pa /= pa.sum(1, keepdims=True)
                sim[j] += pe @ t - (pa @ t).mean()
            pred += -0.5 * np.mean((w - w.mean()) * (t - t.mean())); n += 1
        sim /= n; pred /= n
        out[name] = {f"eps{e:g}": {"simulated": float(sim[j]), "predicted": float(pred * (2 * e) ** 2)} for j, e in enumerate(EPS)}
    ratios = [v["simulated"] / v["predicted"] for k in ("har", "stackoverflow") for v in out[k].values()]
    out["max_relative_deviation_har_so"] = float(max(abs(r - 1) for r in ratios))
    out["femnist_max_abs_gain_eps_le_1"] = float(max(abs(v["simulated"]) for v in out["femnist30"].values()))
    return report(NAME, out)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
