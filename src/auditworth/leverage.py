"""Report leverage, panel search and credit stakes on the 30-group FEMNIST study.

Every provider inventory is represented by its eight-coordinate task vectors:
the four contributor updates' correctness differences relative to keep-current
and the four contributors' per-record Shapley values. A report is scored against
the inventory mean ``mu``; a panel submits the mean of ``k`` genuine records.

The helpers here compute
  * payment-seeking panels (greedy forward selection followed by best-improving
    swaps that minimize the distance of the panel mean to a target),
  * best-responding reports when the client also values its own Shapley
    coordinate with stake ratio ``nu`` (called ``lam`` below), restricted to the
    bounded 16-coalition score polytope ``m = A q, q in [0, 1]^16``,
  * the unrestricted coordinate-box report used for comparison, and
  * per-group summaries with 95% t intervals over the 30 groups.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import linprog, minimize

from .data import FEMNIST30_GROUPS, FEMNIST30_SEEDS, femnist30_state
from .stats import t_interval

ROUND = 80
B = 8.0                      # bound on the squared norm of a task vector
TAU = math.tanh(0.5)         # tau(eps_pay) at eps_pay = 1
FLOOR = 1e-12                # inventories with variance below this are degenerate
KS = (8, 16, 32)
# Stake-ratio grid nu = b / kappa: zero and 10^(x/4) for x = -12..4, recorded to six significant digits.
LAMBDAS = (0.0,) + tuple(float(f"{10 ** (x / 4):.6g}") for x in range(-12, 5))
N_RANDOM = 100               # random panels drawn per inventory


def beta_over_b(lam: float) -> float:
    """Payment ratio beta/b corresponding to stake ratio nu = b/kappa, kappa = beta tau / (4B)."""
    return 4 * B / (TAU * lam)


def task_vectors(prov: dict) -> np.ndarray:
    """(n, 8) task vectors: correctness differences of the four updates, then the four Shapley values."""
    return np.concatenate([(prov["bits"][1:] - prov["bits"][0]).T, prov["psi"].T], axis=1)


@lru_cache(maxsize=None)
def inventories() -> tuple:
    """All (group, seed, provider, F) at round 80 of the one-epoch contract, groups 4-33, seeds 2-5."""
    out = []
    for g in FEMNIST30_GROUPS:
        for s in FEMNIST30_SEEDS:
            for p, prov in enumerate(femnist30_state(g, ROUND, s, 1)):
                out.append((g, s, p, task_vectors(prov)))
    return tuple(out)


def variance(F: np.ndarray) -> float:
    """V_i: mean squared distance of a task vector to the inventory mean."""
    return float(((F - F.mean(0)) ** 2).sum(1).mean())


def random_panel_error(V: float, n: int, k: int) -> float:
    """Expected squared error of a k-panel drawn without replacement."""
    return V * (n - k) / (k * (n - 1)) if n > 1 else 0.0


# ---------------------------------------------------------------- panel search

def select_to_target(F: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """Greedy forward selection then best-improving single swaps minimizing ||m_E - c||^2.

    With c = mu + (nu/2) w this is the payment-plus-credit maximizing panel search.
    """
    n = len(F)
    chosen: list[int] = []
    total = np.zeros(F.shape[1])
    for size in range(1, k + 1):
        val = (((total[None, :] + F) / size - c[None, :]) ** 2).sum(1)
        val[chosen] = np.inf
        best = int(np.flatnonzero(val <= val.min() + 1e-12)[0])
        chosen.append(best)
        total = total + F[best]
    E = np.array(sorted(chosen))
    while True:
        inset = np.zeros(n, bool)
        inset[E] = True
        out_idx, in_idx = E, np.flatnonzero(~inset)
        if len(in_idx) == 0:
            break
        total = F[E].sum(0)
        new = (total[None, None, :] - F[out_idx][:, None, :] + F[in_idx][None, :, :]) / k - c
        J = (new**2).sum(-1)
        cur = float(((total / k - c) ** 2).sum())
        a, b = np.unravel_index(np.argmin(J), J.shape)
        if J[a, b] < cur - 1e-12:
            E = np.array(sorted(set(E.tolist()) - {int(out_idx[a])} | {int(in_idx[b])}))
        else:
            break
    return E


def select_gram(G: np.ndarray, k: int) -> np.ndarray:
    """The same payment-seeking search written with the Gram matrix G = F F^T (target: inventory mean).

    Used for the comparison of payment-seeking with random panels; ties are
    broken towards the lowest record index.
    """
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
    diag = np.diag(G)
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


@lru_cache(maxsize=None)
def panel_vs_random(k: int = 16) -> tuple:
    """Per inventory: (group, seed, provider, n, squared error of the payment-seeking panel,
    mean squared error of N_RANDOM random panels drawn without replacement).

    Random panels use the generator seeded with [group, 1, seed, k, provider].
    """
    out = []
    for g, s, p, F in inventories():
        n = len(F)
        kk = min(k, n)
        mu = F.mean(0)
        T = select_gram(F @ F.T, kk)
        rng = np.random.default_rng([g, 1, s, k, p])
        R = [np.sort(rng.choice(n, kk, replace=False)) for _ in range(N_RANDOM)]
        t2 = float(np.linalg.norm(F[T].mean(0) - mu)) ** 2
        r2 = float(np.mean([float(np.linalg.norm(F[E].mean(0) - mu)) ** 2 for E in R]))
        out.append((g, s, p, n, t2, r2))
    return tuple(out)


# ------------------------------------------------------ reports under a credit stake

def coalition_matrix() -> np.ndarray:
    """A (8 x 16): maps the 16 coalition scores q to the task vector mean m = A q.

    Rows 0-3: accuracy of contributor j alone minus the empty coalition (keep-current).
    Rows 4-7: Shapley value of contributor j under the four-player coalition game.
    """
    A = np.zeros((8, 16))
    for j in range(4):
        A[j, 1 << j] += 1
        A[j, 0] -= 1
        for mask in range(16):
            if mask & (1 << j):
                continue
            weight = 1 / (4 * math.comb(3, bin(mask).count("1")))
            A[4 + j, mask | (1 << j)] += weight
            A[4 + j, mask] -= weight
    return A


def truthful_coalition_scores(A: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """A bounded coalition-score vector q in [0,1]^16 with A q = mu (feasibility of the truth)."""
    res = linprog(np.zeros(16), A_eq=A, b_eq=mu, bounds=[(0, 1)] * 16, method="highs")
    if not res.success or not np.allclose(A @ res.x, mu, atol=1e-8):
        raise RuntimeError(f"inventory mean has no bounded coalition scores: {res.message}")
    return res.x


def feasible_report(A: np.ndarray, q0: np.ndarray, mu: np.ndarray, lam: float, p: int) -> np.ndarray:
    """Best-responding report m = A q maximizing -||m - mu||^2 + nu <w, m>, q in [0,1]^16, w = e_{4+p}."""
    if lam == 0:
        return A @ q0
    w = np.zeros(8)
    w[4 + p] = 1

    def fun(q):
        m = A @ q
        return float(((m - mu) ** 2).sum() - lam * (w @ m)), A.T @ (2 * (m - mu) - lam * w)

    res = minimize(fun, q0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * 16,
                   options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    q = np.clip(res.x, 0, 1)
    residual = float(np.max(np.abs(q - np.clip(q - fun(q)[1], 0, 1))))  # projected gradient
    if residual > 2e-5:
        raise RuntimeError(f"report optimizer did not converge (residual {residual}) at nu={lam}")
    return A @ q


def box_report(mu: np.ndarray, lam: float, p: int) -> np.ndarray:
    """The same best response over the coordinate box [-1, 1]^8 (not a valid report domain)."""
    w = np.zeros(8)
    w[4 + p] = 1
    return np.clip(mu + lam / 2 * w, -1.0, 1.0)


@lru_cache(maxsize=None)
def sweep(ks: tuple, lambdas: tuple, reports: bool = True) -> tuple:
    """Responses of every inventory for each panel size and stake ratio.

    Returns (units, rows): units[(g, s, p)] = (n, V, sd of own Shapley coordinate);
    rows is a list of dicts with the report's and panel's squared error and credit
    gain, the box report's error, and the random-panel error.
    """
    A = coalition_matrix()
    units, rows = {}, []
    for g, s, p, F in inventories():
        n = len(F)
        mu = F.mean(0)
        V = variance(F)
        w = np.zeros(8)
        w[4 + p] = 1
        units[(g, s, p)] = (n, V, float(F[:, 4 + p].std()))
        rep = {}
        if reports:
            q0 = truthful_coalition_scores(A, mu)
            for lam in lambdas:
                m = feasible_report(A, q0, mu, lam, p)
                mb = box_report(mu, lam, p)
                rep[lam] = (float(((m - mu) ** 2).sum()), float(m[4 + p] - mu[4 + p]),
                            float(((mb - mu) ** 2).sum()), float(mb[4 + p] - mu[4 + p]))
        for k in ks:
            kk = min(k, n)
            for lam in lambdas:
                E = select_to_target(F, mu + lam / 2 * w, kk)
                mE = F[E].mean(0)
                row = {"group": g, "seed": s, "provider": p, "k": k, "lambda": lam,
                       "e_pan": float(((mE - mu) ** 2).sum()), "gain_pan": float(w @ (mE - mu)),
                       "e_rand": random_panel_error(V, n, kk)}
                if reports:
                    row.update(zip(("e_rep", "gain_rep", "e_box", "gain_box"), rep[lam]))
                rows.append(row)
    return units, rows


def ci(values) -> list[float]:
    """[mean, lower, upper] with a 95% t interval over groups."""
    a = np.asarray(values, float)
    return [float(a.mean()), *t_interval(a)]


def summarize(units: dict, rows: list, k: int, lam: float, methods=("rep", "box", "pan", "rand")) -> dict:
    """Per-group aggregation over nondegenerate inventories (V >= FLOOR), then 95% t interval over groups.

    pooled_ratio: sum of squared errors over sum of V_i (error relative to one direct read);
    mean_log_ratio: mean of log(error / V_i); credit_gain: mean own-credit gain;
    share_panel_better: share of inventories whose panel error is below the report's.
    """
    byg = defaultdict(list)
    for r in rows:
        if r["k"] != k or r["lambda"] != lam:
            continue
        V = units[(r["group"], r["seed"], r["provider"])][1]
        if V < FLOOR:
            continue
        byg[r["group"]].append((V, r))
    methods = [m for m in methods if f"e_{m}" in next(iter(byg.values()))[0][1]]
    pooled = {m: [] for m in methods}
    logr = {m: [] for m in methods}
    gain = {m: [] for m in methods if m != "rand"}
    share = []
    for g, items in sorted(byg.items()):
        sV = sum(V for V, _ in items)
        for m in methods:
            pooled[m].append(sum(r[f"e_{m}"] for _, r in items) / sV)
            logr[m].append(float(np.mean([math.log(max(r[f"e_{m}"], FLOOR) / V) for V, r in items])))
        for m in gain:
            gain[m].append(float(np.mean([r[f"gain_{m}"] for _, r in items])))
        if "rep" in methods:
            share.append(float(np.mean([r["e_pan"] < r["e_rep"] for _, r in items])))
    out = {"pooled_ratio": {m: ci(v) for m, v in pooled.items()},
           "mean_log_ratio": {m: ci(v) for m, v in logr.items()},
           "credit_gain": {m: ci(v) for m, v in gain.items()}}
    if share:
        out["share_panel_better"] = ci(share)
    return out


def geometry(F: np.ndarray) -> dict:
    """Spectrum of the task-vector covariance: total variance, participation ratio, rank."""
    C = np.cov(F.T, bias=True)
    lam = np.sort(np.clip(np.linalg.eigvalsh(C), 0, None))[::-1]
    total = lam.sum()
    return {"V": float(total), "PR": float(total**2 / (lam**2).sum()) if total > 0 else 0.0,
            "rank": int((lam > 1e-9 * max(lam.max(), 1e-300)).sum()),
            "B_task": float((F**2).sum(1).max()), "n": len(F)}
