"""The decision stake and the private model choice in the 30-group FEMNIST study.

Setting (Section VII-B, Appendix D): round 80, confirmation seeds 2-5, groups
4-33; four contributor clients with equal weight 1/4 and five actions
(keep-current and each contributor's own update). Credit is computed from the
audits, so a report carries only the decision coordinates
f(z) = (u_1 - u_0, ..., u_4 - u_0)(z), with B = 4 and eps_pay = 1.

The decision is the exponential mechanism P(a) ~ exp(eps_dec s_a / (2 pi_max))
on the scores s_a = sum_p pi_p m_p[a], s_0 = 0. Client p values adoption of its
own update at U payment units; its best report maximizes
-||m - mu_p||^2 + lambda P_own(m), lambda = 4B / (tau(eps_pay) beta/U), over the
feasible set M_dec = {m : max_j m_j^+ + max_j m_j^- <= 1}.

Set AUDITWORTH_QUICK=1 to restrict every replay to two groups (smoke test; the
checks against the paper are then expected to fail). AUDITWORTH_WORKERS sets
the number of worker processes (default: all CPUs).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from .data import FEMNIST30_GROUPS, FEMNIST30_SEEDS, femnist30_state

QUICK = os.environ.get("AUDITWORTH_QUICK", "") == "1"
GROUPS = FEMNIST30_GROUPS[:2] if QUICK else FEMNIST30_GROUPS
SEEDS = FEMNIST30_SEEDS
ROUND = 80
EPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
RHOS = (10, 30, 100, 300, 1000, 3000, 10000)  # payment ratios beta/U
CLONES = (4, 16)  # replicated federations: N = 4c clients
B = 4.0
TAU_PAY = math.tanh(0.5)  # tau(eps_pay) = tanh(eps_pay / 2) at eps_pay = 1
FLOOR = 1e-12  # inventories with constant evaluation vectors (V < FLOOR) are excluded from ratios
SWEEPS = 30
MOVE_TOL = 1e-6

# choice from reports versus audits
CHOICE_EPS = EPS + (math.inf,)
CHOICE_COPIES = (1, 4, 16)
AUDITS = (1, 4, 16)
DRAWS = 400


def _workers() -> int:
    return int(os.environ.get("AUDITWORTH_WORKERS", os.cpu_count() or 1))


def _map_groups(fn):
    """Apply a per-group replay to every analysed group; rows are returned in group order."""
    with ProcessPoolExecutor(max_workers=min(_workers(), len(GROUPS))) as ex:
        return [row for rows in ex.map(fn, GROUPS) for row in rows]


# ---------------------------------------------------------------- inputs

def federation(group: int, seed: int):
    """Per-record decision coordinates F_p (n_p, 4), inventory means mu (4, 4), variances V_p."""
    st = femnist30_state(group, ROUND, seed)
    F = [(pr["bits"][1:] - pr["bits"][0]).T for pr in st]
    mu = np.stack([f.mean(0) for f in F])
    V = [float(((f - f.mean(0)) ** 2).sum(1).mean()) for f in F]
    return F, mu, V


# ---------------------------------------------------------------- decision law

def probs(scores: np.ndarray, theta: float) -> np.ndarray:
    z = theta * scores
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def scores_of(reports: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reports are (clients, 4) decision coordinates; keep-current has score 0."""
    return np.concatenate([[0.0], weights @ reports])


def regret(p: np.ndarray, target: np.ndarray) -> float:
    return float(target.max() - p @ target)


def violation(m: np.ndarray) -> float:
    """max m^+ + max m^- - 1; positive outside M_dec."""
    return float(max(m.max(), 0.0) + max(-m.min(), 0.0) - 1.0)


def _starts(mu: np.ndarray, own: int) -> list[np.ndarray]:
    """Seven starting points: mu, and mu moved toward the own coordinate and away from the others."""
    starts = [mu.copy()]
    for d in (0.25, 0.5, 1.0):
        up = mu.copy()
        up[own - 1] += d
        starts.append(np.clip(up, -1, 1))
        dn = mu - d
        dn[own - 1] = mu[own - 1] + d
        starts.append(np.clip(dn, -1, 1))
    return starts


def _objective(i, reports, weights, mu, theta, lam, own):
    others = reports.copy()

    def neg(m: np.ndarray):
        others[i] = m
        p = probs(scores_of(others, weights), theta)
        po = p[own]
        # dP_own/ds_a = P_own (1{a=own} - p_a); ds_a/dm[a-1] = w_i
        g_s = po * ((np.arange(5) == own).astype(float) - p) * theta * weights[i]
        grad = -2 * (m - mu) + lam * g_s[1:]
        return -(-np.sum((m - mu) ** 2) + lam * po), -grad

    return neg


def feasible_best_response(i, reports, weights, mu, theta, lam, own) -> np.ndarray:
    """Best report of client i over M_dec (others fixed), by SLSQP on (m, t):
    m_j <= t, -m_j <= 1 - t, 0 <= t <= 1, m in [-1, 1]^4; best of seven starts."""
    obj = _objective(i, reports, weights, mu, theta, lam, own)

    def neg(x):
        v, g = obj(x[:4])
        return v, np.concatenate([g, [0.0]])

    E = np.eye(5)
    cons = [{"type": "ineq", "fun": lambda x, j=j: x[4] - x[j], "jac": lambda x, j=j: E[4] - E[j]}
            for j in range(4)]
    cons += [{"type": "ineq", "fun": lambda x, j=j: 1 - x[4] + x[j], "jac": lambda x, j=j: -E[4] + E[j]}
             for j in range(4)]
    bounds = [(-1, 1)] * 4 + [(0, 1)]

    def t_of(m):
        return float(np.clip(max(m.max(), 0.0), 0, 1))

    best, best_val = mu.copy(), neg(np.concatenate([mu, [t_of(mu)]]))[0]
    for x0 in _starts(mu, own):
        res = minimize(neg, np.concatenate([x0, [t_of(x0)]]), jac=True, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
        m = np.clip(res.x[:4], -1, 1)
        if violation(m) <= 1e-7 and res.fun < best_val - 1e-12:
            best, best_val = m.copy(), res.fun
    return best


def box_best_response(i, reports, weights, mu, theta, lam, own) -> np.ndarray:
    """The same objective over the box [-1, 1]^4 (L-BFGS-B, seven starts); used only to
    count how often an optimizer over the box leaves M_dec (Appendix D)."""
    neg = _objective(i, reports, weights, mu, theta, lam, own)
    best, best_val = mu.copy(), neg(mu)[0]
    for x0 in _starts(mu, own):
        res = minimize(neg, x0, jac=True, method="L-BFGS-B", bounds=[(-1, 1)] * 4)
        if res.fun < best_val - 1e-12:
            best, best_val = res.x.copy(), res.fun
    return best


def _lam(rho: float) -> float:
    return 4 * B / (TAU_PAY * rho)


# ---------------------------------------------------------------- stake replays (per group)

def _unilateral_group(group: int) -> list[tuple]:
    """Each client best-responds with the others truthful (four-client federation)."""
    rows = []
    w4 = np.full(4, 0.25)
    for s in SEEDS:
        _, mu, V = federation(group, s)
        for eps in EPS:
            theta = eps / (2 * 0.25)
            for rho in RHOS:
                lam = _lam(rho)
                for i in range(4):
                    mb = box_best_response(i, mu, w4, mu[i], theta, lam, i + 1)
                    mf = feasible_best_response(i, mu, w4, mu[i], theta, lam, i + 1)
                    # Corollary 6: e <= min(4B U tau(eps_dec)/(beta tau(1)), (B eps_dec U L/(beta tau(1)))^2), L = 1
                    bound = min(4 * B * math.tanh(eps / 2) / (TAU_PAY * rho), (B * eps / (TAU_PAY * rho)) ** 2)
                    rows.append((group, s, i, eps, rho, V[i], float(((mf - mu[i]) ** 2).sum()),
                                 violation(mb), bound))
    return rows


def _dynamics_group(group: int) -> list[tuple]:
    """Iterated feasible best responses of all four clients from truthful reports."""
    rows = []
    w4 = np.full(4, 0.25)
    for s in SEEDS:
        _, mu, _ = federation(group, s)
        target = scores_of(mu, w4)
        for eps in EPS:
            theta = eps / (2 * 0.25)
            r_true = regret(probs(target, theta), target)
            for rho in RHOS:
                lam = _lam(rho)
                reps = mu.copy()
                converged = False
                for _sweep in range(1, SWEEPS + 1):
                    move = 0.0
                    for i in range(4):
                        m = feasible_best_response(i, reps, w4, mu[i], theta, lam, i + 1)
                        move = max(move, float(np.abs(m - reps[i]).max()))
                        reps[i] = m
                    if move < MOVE_TOL:
                        converged = True
                        break
                r_br = regret(probs(scores_of(reps, w4), theta), target)
                rows.append((group, s, eps, rho, r_true, r_br, converged))
    return rows


def _clone_group(group: int) -> list[tuple]:
    """Replicated federations: each inventory copied c times (N = 4c, weights 1/N); one copy deviates."""
    rows = []
    for s in SEEDS:
        _, mu, V = federation(group, s)
        target = scores_of(mu, np.full(4, 0.25))
        for c in CLONES:
            n = 4 * c
            wc = np.full(n, 1.0 / n)
            muc = np.repeat(mu, c, axis=0)
            for eps in EPS:
                theta = eps / (2 / n)
                r_true = regret(probs(scores_of(muc, wc), theta), target)
                for rho in RHOS:
                    lam = _lam(rho)
                    for i in range(4):
                        m = feasible_best_response(i * c, muc, wc, mu[i], theta, lam, i + 1)
                        rows.append((group, s, c, eps, rho, i, V[i], float(((m - mu[i]) ** 2).sum()), r_true))
    return rows


@lru_cache(maxsize=None)
def unilateral_rows() -> list[tuple]:
    """(group, seed, client, eps, rho, V, e_feasible, box_violation, cor6_bound) for all units."""
    return _map_groups(_unilateral_group)


@lru_cache(maxsize=None)
def dynamics_rows() -> list[tuple]:
    """(group, seed, eps, rho, regret_truthful, regret_best_responses, converged)."""
    return _map_groups(_dynamics_group)


@lru_cache(maxsize=None)
def clone_rows() -> list[tuple]:
    """(group, seed, copies, eps, rho, client, V, e_feasible, regret_truthful)."""
    return _map_groups(_clone_group)


# ---------------------------------------------------------------- aggregation over groups

def group_means(pairs) -> dict:
    """pairs: iterable of (group, value) -> {group: mean value}."""
    acc: dict = {}
    for g, v in pairs:
        acc.setdefault(g, []).append(v)
    return {g: float(np.mean(v)) for g, v in acc.items()}


def pooled_error_by_group(units) -> dict:
    """Per group: sum e / sum V over its non-degenerate units (error relative to one direct audit)."""
    num: dict = {}
    den: dict = {}
    for g, V, e in units:
        if V < FLOOR:
            continue
        num[g] = num.get(g, 0.0) + e
        den[g] = den.get(g, 0.0) + V
    return {g: num[g] / den[g] for g in num}


def within16_by_group(units) -> dict:
    """Per group: share of non-degenerate units with e <= V/16 (as accurate as 16 direct reads)."""
    return group_means((g, float(e <= V / 16)) for g, V, e in units if V >= FLOOR)


# ---------------------------------------------------------------- choice from reports versus audits

def choice_probs(scores: np.ndarray, theta: float) -> np.ndarray:
    """Exponential mechanism over the last axis; theta = inf gives the argmax with ties split."""
    if math.isinf(theta):
        m = scores.max(axis=-1, keepdims=True)
        w = (scores >= m - 1e-12).astype(float)
    else:
        w = np.exp(theta * (scores - scores.max(axis=-1, keepdims=True)))
    return w / w.sum(axis=-1, keepdims=True)


def _choice_group(group: int) -> list[tuple]:
    """Regret of the choice from exact reports, from r audits per client (400 draws), and uniform."""
    rows = []
    for s in SEEDS:
        F, mu, _ = federation(group, s)
        target = np.concatenate([[0.0], mu.mean(0)])
        best = target.max()
        rows.append((group, s, "uniform", None, None, None, float(best - target.mean())))
        rng = np.random.default_rng([group, s, 4040])
        for c in CHOICE_COPIES:
            n = 4 * c
            for eps in CHOICE_EPS:
                theta = math.inf if math.isinf(eps) else eps * n / 2
                rows.append((group, s, "reports", c, eps, None, float(best - choice_probs(target, theta) @ target)))
                for r in AUDITS:
                    # mean of r records drawn with replacement, independently for each replica
                    means = np.empty((DRAWS, n, 4))
                    for k in range(n):
                        f = F[k % 4]
                        idx = rng.integers(0, len(f), size=(DRAWS, r))
                        means[:, k, :] = f[idx].mean(axis=1)
                    sc = np.concatenate([np.zeros((DRAWS, 1)), means.mean(axis=1)], axis=1)
                    reg = float(np.mean(best - choice_probs(sc, theta) @ target))
                    rows.append((group, s, "audits", c, eps, r, reg))
    return rows


@lru_cache(maxsize=None)
def choice_rows() -> list[tuple]:
    """(group, seed, arm, copies, eps, audits, regret); arm in {uniform, reports, audits}."""
    return _map_groups(_choice_group)


def choice_by_group(arm: str, copies=None, eps=None, audits=None) -> dict:
    return group_means((row[0], row[6]) for row in choice_rows()
                       if row[2] == arm and row[3] == copies and row[4] == eps and row[5] == audits)
