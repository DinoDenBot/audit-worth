"""Panel elicitation and staleness measurements for the 30-group FEMNIST study.

A provider's task vector on record z is f(z) = (u_1 - u_0, ..., u_4 - u_0,
psi_0, ..., psi_3)(z): the four candidate-minus-keep-current correctness
differences and the four contributors' per-record Shapley values (B = 8).
A payment-seeking panel of k genuine records minimizes the kernel distance
||m_E - mu||^2 between the panel mean and the inventory mean; it is found by
the bounded search ``select`` (greedy forward selection, then best-improving
single swaps).
"""
from __future__ import annotations

import math

import numpy as np

from .data import DATA

N_RANDOM = 100          # random panels per provider
FLOOR = 1e-12           # guards log(0) when a panel matches the inventory exactly


def task_vectors(prov: dict) -> np.ndarray:
    """(n, 8) task vectors of one provider's inventory."""
    return np.concatenate([(prov["bits"][1:] - prov["bits"][0]).T, prov["psi"].T], axis=1)


def select(G: np.ndarray, k: int) -> np.ndarray:
    """Greedy forward selection then best-improving swaps minimizing ||m_E - mu||^2 under Gram G."""
    n = len(G)
    gbar = G.mean(axis=1)
    chosen: list[int] = []
    for size in range(1, k + 1):
        best, best_val = None, None
        S = G[np.ix_(chosen, chosen)].sum() if chosen else 0.0
        T = gbar[chosen].sum() if chosen else 0.0
        r = G[:, chosen].sum(axis=1) if chosen else np.zeros(n)
        for i in range(n):
            if i in chosen:
                continue
            val = (S + 2 * r[i] + G[i, i]) / size**2 - 2 * (T + gbar[i]) / size
            if best_val is None or val < best_val - 1e-12:
                best, best_val = i, val
        chosen.append(best)
    E = np.array(sorted(chosen))
    while True:
        inset = np.zeros(n, bool)
        inset[E] = True
        out_idx = E
        in_idx = np.flatnonzero(~inset)
        if len(in_idx) == 0:
            break
        S = G[np.ix_(E, E)].sum()
        T = gbar[E].sum()
        r = G[:, E].sum(axis=1)
        diag = np.diag(G)
        Snew = (S - 2 * r[out_idx][:, None] + diag[out_idx][:, None]
                + 2 * (r[in_idx][None, :] - G[np.ix_(out_idx, in_idx)]) + diag[in_idx][None, :])
        Tnew = T - gbar[out_idx][:, None] + gbar[in_idx][None, :]
        J = Snew / k**2 - 2 * Tnew / k
        cur = S / k**2 - 2 * T / k
        a, b = np.unravel_index(np.argmin(J), J.shape)
        if J[a, b] < cur - 1e-12:
            E = np.array(sorted(set(E.tolist()) - {int(out_idx[a])} | {int(in_idx[b])}))
        else:
            break
    return E


def random_panels(group: int, epochs: int, seed: int, k: int, p: int, n: int) -> list[np.ndarray]:
    """The fixed random comparison panels (drawn without replacement)."""
    rng = np.random.default_rng([group, epochs, seed, k, p])
    kk = min(k, n)
    return [np.sort(rng.choice(n, kk, replace=False)) for _ in range(N_RANDOM)]


def panel_errors(F: np.ndarray, T: np.ndarray, R: list[np.ndarray]) -> tuple[float, float]:
    """Squared error of panel T and mean squared error of the random panels R on task vectors F."""
    mu = F.mean(0)
    t2 = float(np.linalg.norm(F[T].mean(0) - mu)) ** 2
    r2 = float(np.mean([np.linalg.norm(F[E].mean(0) - mu) ** 2 for E in R]))
    return t2, r2


def log_ratio(t2: float, r2: float) -> float:
    return math.log(max(t2, FLOOR) / max(r2, FLOOR))


# ---- kernels that add information beyond the task vector -------------------------------------

def kernel_inputs(group: int) -> list[dict]:
    """Per-provider labels, 7x7 pooled images and round-80 penultimate activations (source-row order)."""
    with np.load(DATA / "femnist30" / "kernel_inputs" / f"g{group:02d}.npz") as d:
        return [{"rows": d[f"rows_{p}"], "y": d[f"y_{p}"], "pooled": d[f"pooled_{p}"],
                 "h": d[f"h_{p}"].astype(float)} for p in range(4)]


def label_image_kernel(inp: dict) -> np.ndarray:
    """(1{y = y'} + exp(-||x - x'||^2 / (2 h^2))) / 2 with median-heuristic bandwidth h."""
    x, y = inp["pooled"], inp["y"]
    sq = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    h = np.sqrt(np.median(sq[np.triu_indices(len(x), 1)]))
    rbf = np.exp(-sq / (2 * h * h))
    return ((y[:, None] == y[None, :]).astype(float) + rbf) / 2


def model_output_kernel(inp: dict) -> np.ndarray:
    """Gaussian kernel on penultimate activations, median-heuristic bandwidth, max diagonal 1."""
    v = inp["h"].reshape(len(inp["h"]), -1)
    sq = ((v[:, None, :] - v[None, :, :]) ** 2).sum(-1)
    off = sq[np.triu_indices(len(v), 1)]
    h2 = np.median(off) if np.median(off) > 0 else (off[off > 0].min() if (off > 0).any() else 1.0)
    K = np.exp(-sq / (2 * h2))
    return K / np.diag(K).max()
