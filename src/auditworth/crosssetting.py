"""Shared computations for the HAR, Stack Overflow, and FEMNIST separation replays.

These replays compare the private model choice made from exact (truthful) reports
with the same choice made from directly read audit records, compute
payment-seeking 16-record panels, and simulate best responses to the decision
stake over the feasible report set.

Conventions (Section VII-B and the study-design appendix):
- Four evaluation providers per state and four candidate actions without a
  keep-current action. A provider's task vector per record is the three
  correctness differences from candidate 0 plus (for the leverage measurement)
  its own Shapley coordinate.
- The private choice is the exponential mechanism with logits 2*eps*score over
  scores (0, mean of the four providers' reported differences); eps = inf is
  the argmax with ties split.
- Audit arms draw r records uniformly with replacement per provider, averaged
  over 256 seeded draws per state.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from .data import (FEMNIST30_GROUPS, FEMNIST30_SEEDS, femnist30_state, femnist_scaled_state, har_state,
                   stackoverflow_state, stackoverflow_strong_state)
from .stats import t_interval

TAU_PAY = math.tanh(0.5)  # tau(eps_pay) at eps_pay = 1
N_DRAWS = 256
K_PANEL = 16

HAR_SEEDS = tuple(range(8, 28))
SO_GROUPS = tuple(range(1, 9))
SO_SEEDS = tuple(range(4, 20))
STATES = (0, 1, 2, 3)
SCALES = ("0.5", "1", "2")
SCALED_GROUPS = (1, 2, 3, 4)
SCALED_SEEDS = tuple(range(4, 12))


# ---------------------------------------------------------------------------
# Basic pieces
# ---------------------------------------------------------------------------

def probs(scores, eps: float) -> np.ndarray:
    """Exponential-mechanism choice probabilities (logits 2*eps*score); argmax with ties split at eps=inf."""
    scores = np.asarray(scores, float)
    if math.isinf(eps):
        w = (scores >= scores.max(axis=-1, keepdims=True) - 1e-12).astype(float)
        return w / w.sum(axis=-1, keepdims=True)
    z = 2 * eps * scores
    w = np.exp(z - z.max(axis=-1, keepdims=True))
    return w / w.sum(axis=-1, keepdims=True)


def panel(F: np.ndarray, k: int = K_PANEL) -> np.ndarray:
    """Payment-seeking k-record panel: greedy forward selection toward the inventory mean,
    then best-improving single swaps (ties broken by record index)."""
    n, dim = F.shape
    target = F.mean(axis=0)
    chosen: list[int] = []
    total = np.zeros(dim)
    for size in range(1, k + 1):
        error = (((total[None, :] + F) / size - target) ** 2).sum(axis=1)
        error[chosen] = np.inf
        ix = int(np.flatnonzero(error <= error.min() + 1e-12)[0])
        chosen.append(ix)
        total += F[ix]
    selected = np.array(sorted(chosen), dtype=int)
    for _ in range(1000):
        unused = np.setdiff1d(np.arange(n), selected, assume_unique=True)
        if not len(unused):
            return selected
        total = F[selected].sum(axis=0)
        diff = (total[None, None, :] - F[selected, None, :] + F[None, unused, :]) / k - target
        errors = (diff * diff).sum(axis=2)
        old = float(((total / k - target) ** 2).sum())
        a, b = np.unravel_index(np.argmin(errors), errors.shape)
        if errors[a, b] >= old - 1e-12:
            return selected
        selected = np.array(sorted(set(selected.tolist()) - {int(selected[a])} | {int(unused[b])}), dtype=int)
    raise RuntimeError("panel search did not converge")


def panel_unit(F: np.ndarray) -> dict:
    """Inventory variance V, payment-seeking panel error, and the random-panel error
    (exact without-replacement formula) for one provider inventory."""
    mu = F.mean(axis=0)
    V = float(((F - mu) ** 2).sum(axis=1).mean())
    ix = panel(F)
    return dict(n=len(F), V=V, panel_error=float(((F[ix].mean(axis=0) - mu) ** 2).sum()),
                random_error=float(V * (len(F) - K_PANEL) / (K_PANEL * (len(F) - 1))), mu=mu)


def audit_scores(F_list, rng, audits=(1, 16)) -> dict:
    """Scores (0, mean audited differences) from r with-replacement audits per provider, N_DRAWS draws."""
    out = {}
    for r in audits:
        est = np.zeros((N_DRAWS, 3))
        for F in F_list:
            ix = rng.integers(len(F), size=(N_DRAWS, r))
            est += F[ix, :3].mean(axis=1) / 4
        out[r] = np.column_stack((np.zeros(N_DRAWS), est))
    return out


def regret(target: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(target.max() - p @ target))


def pooled_ratio(units, key="panel_error") -> float:
    return sum(u[key] for u in units) / sum(u["V"] for u in units)


def grouped_gain(rows, stratum, a="audit1", b="exact"):
    """Mean over strata of the per-stratum mean of (a - b), and its 95% t interval over strata."""
    by = defaultdict(list)
    for r in rows:
        by[stratum(r)].append(r[a] - r[b])
    means = [float(np.mean(by[k])) for k in sorted(by)]
    return float(np.mean(means)), t_interval(means), means


# ---------------------------------------------------------------------------
# Decision-stake best responses over the feasible set
# ---------------------------------------------------------------------------

def _stake_probs(scores, theta):
    z = theta * scores
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def best_response(i, reports, mu, theta, lam, own, dim=3):
    """Provider i's feasible best report: maximize -||m - mu||^2 + lam * P(own action is chosen)
    over M = {m : max_j m_j^+ + max_j m_j^- <= 1}, by SLSQP from four starting points.
    Scores are (0, mean of the four reports); lam = 4B / (tau(eps_pay) * beta/U)."""
    others = reports.copy()
    n = len(reports)
    w = 1.0 / n
    A = dim + 1

    def neg(x):
        m = x[:dim]
        others[i] = m
        p = _stake_probs(np.r_[0.0, others.mean(0)], theta)
        po = p[own]
        g_s = po * ((np.arange(A) == own).astype(float) - p) * theta * w
        grad = -2 * (m - mu) + lam * g_s[1:]
        return -(-np.sum((m - mu) ** 2) + lam * po), np.r_[-grad, 0.0]

    # Auxiliary variable x[dim] = c with m_j <= c and c - m_j <= 1 encodes the feasible set.
    E = np.eye(dim + 1)
    cons = [{"type": "ineq", "fun": lambda x, j=j: x[dim] - x[j], "jac": lambda x, j=j: E[dim] - E[j]}
            for j in range(dim)]
    cons += [{"type": "ineq", "fun": lambda x, j=j: 1 - x[dim] + x[j], "jac": lambda x, j=j: -E[dim] + E[j]}
             for j in range(dim)]
    bounds = [(-1, 1)] * dim + [(0, 1)]

    def viol(m):
        return max(m.max(), 0) + max(-m.min(), 0) - 1

    starts = [mu.copy()]
    for d in (0.25, 0.5, 1.0):
        x = mu - d
        if own > 0:
            x[own - 1] = mu[own - 1] + d
        starts.append(np.clip(x, -1, 1))
    best, bv = mu.copy(), neg(np.r_[mu, max(mu.max(), 0)])[0]
    for x0 in starts:
        res = minimize(neg, np.r_[x0, float(np.clip(max(x0.max(), 0), 0, 1))], jac=True, method="SLSQP",
                       bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
        m = np.clip(res.x[:dim], -1, 1)
        if viol(m) <= 1e-7 and res.fun < bv - 1e-12:
            best, bv = m.copy(), res.fun
    return best


def stake_lambda(rho: float, B: float = 3.0) -> float:
    return 4 * B / (TAU_PAY * rho)


def simultaneous(mu: np.ndarray, theta: float, lam: float, max_sweeps: int = 30):
    """Iterate best responses of all four providers from truthful reports (provider p values candidate p)."""
    reps = mu.copy()
    for sweep in range(1, max_sweeps + 1):
        move = 0.0
        for p in range(4):
            m = best_response(p, reps, mu[p], theta, lam, p, 3)
            move = max(move, float(np.abs(m - reps[p]).max()))
            reps[p] = m
        if move < 1e-6:
            return reps, True, sweep
    return reps, False, max_sweeps


# ---------------------------------------------------------------------------
# HAR and Stack Overflow (trained MLP)
# ---------------------------------------------------------------------------

def cross_states():
    """Yield (setting, group, seed, state, [F_p]) with F_p = (three differences, own Shapley)."""
    for seed in HAR_SEEDS:
        for state in STATES:
            yield "har", 0, seed, state, [np.column_stack(((b[1:] - b[0]).T, psi)) for b, psi in har_state(seed, state)]
    for group in SO_GROUPS:
        for seed in SO_SEEDS:
            for state in STATES:
                yield ("stackoverflow", group, seed, state,
                       [np.column_stack(((b[1:] - b[0]).T, psi)) for b, psi in stackoverflow_state(group, seed, state)])


CROSS_EPS = (0.5, 1.0, 4.0, 16.0, math.inf)


@lru_cache(maxsize=None)
def cross_setting(with_panels: bool = False):
    """Per-unit panel measurements (optional) and per-state decision rows for HAR and Stack Overflow."""
    units, decisions, mus = [], [], {}
    for setting, group, seed, state, F_list in cross_states():
        key = (setting, group, seed, state)
        mus[key] = np.stack([F.mean(axis=0)[:3] for F in F_list])
        if with_panels:
            for p, F in enumerate(F_list):
                u = panel_unit(F)
                u.update(setting=setting, group=group, seed=seed, state=state, provider=p)
                units.append(u)
        truth = np.r_[0.0, mus[key].mean(axis=0)]
        rng = np.random.default_rng([279929, 0 if setting == "har" else 1, group, seed, state])
        draws = audit_scores(F_list, rng)
        for eps in CROSS_EPS:
            decisions.append(dict(setting=setting, group=group, seed=seed, state=state, eps=eps,
                                  exact=regret(truth, probs(truth, eps)),
                                  audit1=regret(truth, probs(draws[1], eps)),
                                  audit16=regret(truth, probs(draws[16], eps)),
                                  uniform=float(truth.max() - truth.mean()),
                                  range=float(truth.max() - truth.min())))
    return units, decisions, mus


def cross_stratum(setting):
    """Intervals are over the 20 HAR seeds or the eight Stack Overflow federations."""
    return (lambda r: r["seed"]) if setting == "har" else (lambda r: r["group"])


# ---------------------------------------------------------------------------
# FEMNIST update-scaling contract
# ---------------------------------------------------------------------------

SEP_EPS = (0.5, 1.0, 4.0, 16.0, math.inf)


def scaled_states():
    for g in SCALED_GROUPS:
        for s in SCALES:
            for z in SCALED_SEEDS:
                for a in STATES:
                    d = femnist_scaled_state(g, s, z, a)
                    F = [(d[f"candidate_bits_{p}"][1:].astype(float) - d[f"candidate_bits_{p}"][0]).T
                         for p in range(4)]
                    yield g, s, z, a, F, d["target_acc"].astype(float)


@lru_cache(maxsize=None)
def separation_value():
    """Decision value rows of the update-scaling contract, plus per-state provider means and variances."""
    rows, states = [], {}
    for g, s, z, a, F, tacc in scaled_states():
        mu = np.stack([f.mean(0) for f in F])
        V = [float(((f - f.mean(0)) ** 2).sum(1).mean()) for f in F]
        truth = np.r_[0.0, mu.mean(0)]
        tsec = tacc - tacc[0]  # secondary (declared 64-writer) target
        rng = np.random.default_rng([290929, g, int(float(s) * 2), z, a])
        draws = audit_scores(F, rng)
        states[(g, s, z, a)] = (mu, V)
        for eps in SEP_EPS:
            rows.append(dict(group=g, scale=s, seed=z, state=a, eps=eps,
                             exact=regret(truth, probs(truth, eps)),
                             audit1=regret(truth, probs(draws[1], eps)),
                             audit16=regret(truth, probs(draws[16], eps)),
                             uniform=float(truth.max() - truth.mean()),
                             range=float(truth.max() - truth.min()),
                             exact_sec=regret(tsec, probs(truth, eps)),
                             audit1_sec=regret(tsec, probs(draws[1], eps))))
    return rows, states


# ---------------------------------------------------------------------------
# Stack Overflow with the stronger sparse-feature model
# ---------------------------------------------------------------------------

STRONG_EPS = (1.0, 4.0, 16.0, math.inf)


@lru_cache(maxsize=None)
def strong_setting(with_panels: bool = False):
    units, decisions = [], []
    for group in SO_GROUPS:
        for seed in SO_SEEDS:
            for state in STATES:
                z = stackoverflow_strong_state(group, seed, state)
                F_list = []
                for p in range(4):
                    bits = z[f"bits_{p}"].astype(float)
                    F = np.column_stack(((bits[1:] - bits[0]).T, z[f"psi_{p}"].astype(float)))
                    F_list.append(F)
                    if with_panels:
                        u = panel_unit(F)
                        u.update(group=group, seed=seed, state=state, provider=p)
                        units.append(u)
                exact = np.r_[0.0, np.mean([F[:, :3].mean(axis=0) for F in F_list], axis=0)]
                rng = np.random.default_rng([290929, group, seed, state])
                draws = audit_scores(F_list, rng)
                for tname, target in (("providers", z["provider_accuracy"]), ("full", z["full_accuracy"])):
                    for eps in STRONG_EPS:
                        decisions.append(dict(group=group, seed=seed, state=state, target=tname, eps=eps,
                                              exact=regret(target, probs(exact, eps)),
                                              audit1=regret(target, probs(draws[1], eps)),
                                              audit16=regret(target, probs(draws[16], eps)),
                                              uniform=float(target.max() - target.mean()),
                                              range=float(np.ptp(target))))
    return units, decisions


# ---------------------------------------------------------------------------
# 30-group FEMNIST: choice from exact reports vs one audit per client (four clients)
# ---------------------------------------------------------------------------

CHOICE_EPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
CHOICE_RS = (1, 4, 16)
CHOICE_DRAWS = 400


def _choice_probs(scores, theta):
    if math.isinf(theta):
        w = (scores >= scores.max(axis=-1, keepdims=True) - 1e-12).astype(float)
    else:
        w = np.exp(theta * (scores - scores.max(axis=-1, keepdims=True)))
    return w / w.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=None)
def femnist30_choice():
    """Round-80 one-epoch contract, five actions (keep-current and four updates), four clients.

    Returns per-(group, seed) regrets from exact reports and from r audits per client
    (400 seeded with-replacement draws), and the candidate range on the equal-weight target.
    The random stream matches the original study, which drew the four-client cells first
    for each seed; only those cells are computed here.
    """
    rows = []
    for g in FEMNIST30_GROUPS:
        for s in FEMNIST30_SEEDS:
            F = [(pr["bits"][1:] - pr["bits"][0]).T for pr in femnist30_state(g, 80, s)]
            mu = np.stack([f.mean(0) for f in F])
            target = np.concatenate([[0.0], mu.mean(0)])
            best = target.max()
            rng = np.random.default_rng([g, s, 4040])
            row = dict(group=g, seed=s, range=float(target.max() - target.min()),
                       uniform=float(best - target.mean()))
            for eps in CHOICE_EPS:
                theta = math.inf if math.isinf(eps) else eps * 4 / 2
                row[("reports", eps)] = float(best - _choice_probs(target, theta) @ target)
                for r in CHOICE_RS:
                    means = np.empty((CHOICE_DRAWS, 4, 4))
                    for k in range(4):
                        idx = rng.integers(0, len(F[k]), size=(CHOICE_DRAWS, r))
                        means[:, k, :] = F[k][idx].mean(axis=1)
                    sc = np.concatenate([np.zeros((CHOICE_DRAWS, 1)), means.mean(axis=1)], axis=1)
                    row[("audits", eps, r)] = float(np.mean(best - _choice_probs(sc, theta) @ target))
            rows.append(row)
    return rows


def distinct_equivalent(units) -> float:
    """Number of distinct (without-replacement) direct reads per inventory whose pooled expected
    squared error, sum_j V_j max(n_j - r, 0) / (r (n_j - 1)), equals the pooled panel error;
    solved for continuous r by bisection. Inventories with n_j <= r are read in full."""
    error = sum(u["panel_error"] for u in units)

    def distinct_error(r):
        return sum(u["V"] * max(u["n"] - r, 0) / (r * (u["n"] - 1)) for u in units)

    lo, hi = 1.0, float(max(u["n"] for u in units))
    for _ in range(60):
        mid = (lo + hi) / 2
        if distinct_error(mid) > error:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
