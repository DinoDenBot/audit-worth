"""Appendix E: richer elicitation kernels do not keep round-80 panels accurate at rounds 120 and 160.

For every analysed group (4-33), seed (2-5) and provider, three 16-record panels
are elicited at round 80 (one-epoch contract) by the same bounded search:
  T  the task kernel F F^T (the payment-seeking panel);
  X  F F^T + alpha K_rich, K_rich = (1{y = y'} + exp(-||x - x'||^2/(2h^2)))/2 on
     7x7 average-pooled images, alpha = max_z ||f(z)||^2 at round 80;
  M  F F^T + 4 alpha K_model, K_model a Gaussian kernel on the round-80 model's
     128-dimensional penultimate activations (the variant chosen on the four
     debugging groups before the test).
Each panel is evaluated on the task vectors of rounds 120 and 160 at every seed
(2-5); the outcome is the panel's distance ||m_E - mu||, and the reported
contrasts are X - T and M - T, averaged over providers and evaluation states
within a group, with a 95% t interval over the 30 groups. "No better" means the
interval's upper end is not below zero.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from auditworth.check import report
from auditworth.data import FEMNIST30_GROUPS, FEMNIST30_SEEDS, femnist30_state
from auditworth.staleness import (kernel_inputs, label_image_kernel, model_output_kernel, select,
                                  task_vectors)
from auditworth.stats import t_interval

K = 16
MODEL_WEIGHT = 4
NAME = "staleness_kernels"


def tci(values):
    lo, hi = t_interval(values)
    return {"mean": float(np.mean(values)), "lo": lo, "hi": hi}


def main() -> bool:
    x_minus_t, m_minus_t = [], []
    for g in FEMNIST30_GROUPS:
        states = {(r, s): femnist30_state(g, r, s) for r in (80, 120, 160) for s in FEMNIST30_SEEDS}
        inp = kernel_inputs(g)
        for p in range(4):
            assert np.array_equal(inp[p]["rows"], states[(80, FEMNIST30_SEEDS[0])][p]["rows"])
        K_rich = [label_image_kernel(i) for i in inp]
        K_model = [model_output_kernel(i) for i in inp]
        dx, dm = [], []
        for s in FEMNIST30_SEEDS:
            panels = []
            for p, pr in enumerate(states[(80, s)]):
                F = task_vectors(pr)
                kk = min(K, len(F))
                Gt = F @ F.T
                alpha = float((F**2).sum(1).max())
                panels.append((select(Gt, kk), select(Gt + alpha * K_rich[p], kk),
                               select(Gt + MODEL_WEIGHT * alpha * K_model[p], kk)))
            for r in (120, 160):
                for s2 in FEMNIST30_SEEDS:
                    for p, pr in enumerate(states[(r, s2)]):
                        F = task_vectors(pr)
                        mu = F.mean(0)
                        T, X, M = (float(np.linalg.norm(F[E].mean(0) - mu)) for E in panels[p])
                        dx.append(X - T)
                        dm.append(M - T)
        x_minus_t.append(np.mean(dx))
        m_minus_t.append(np.mean(dm))
    xt, mt = tci(x_minus_t), tci(m_minus_t)
    values = {
        "label_image_minus_task": xt,
        "model_output_minus_task": mt,
        "label_image_not_better": int(xt["hi"] >= 0),
        "model_output_not_better": int(mt["hi"] >= 0),
    }
    return report(NAME, values)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
