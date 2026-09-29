"""Credit-seeking curation studies (Table I and Section VII-A).

Each file ``data/curation_studies/<study>.csv`` holds one row per training
state and curating client of a study's primary contrast: the client's panel
committed before the state was known versus the panel it curated after
seeing the state. Columns:

    group, seed, state, client  -- federation, training seed, training state, curating client
    weight                      -- the study's averaging weight of the row
    choice_committed/curated    -- candidate adopted under each panel
    flip                        -- 1 if the adopted candidate changed (a fraction when
                                   averaged over random inspection orders)
    regret_committed/curated    -- target regret of the adopted candidate
    credit_committed/curated    -- the client's panel credit under the study's credit rule
    credit_target               -- the same credit rule on the target reference (empty if
                                   not computable)

Summaries are weight-averaged over rows. The target effect's 95% interval
follows each study's own analysis: a bootstrap over seeds, a two-stage
bootstrap over groups then seeds within groups (with the study's generator
seed), or a t interval over groups.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict

import numpy as np

from .paths import DATA
from .stats import t_interval

DIR = DATA / "curation_studies"

# Table I rows in order: (file stem, study label, model and decision, interval method, generator seed)
STUDIES = [
    ("femnist_linear_1group", "FEMNIST, 1 group", "linear, 4 updates", "seed_bootstrap", 270928),
    ("femnist_linear_12groups", "FEMNIST, 12 groups", "linear, 4 updates", "two_stage", 270931),
    ("har_linear", "UCI HAR", "linear, 4 updates", "two_stage", 270932),
    ("wisdm_linear", "WISDM", "linear, 4 updates", "two_stage", 270932),
    ("stackoverflow_linear_4groups", "Stack Overflow", "linear, 4 updates", "two_stage", 270927),
    ("femnist_cnn_5groups", "FEMNIST, 5 groups", "CNN, 4 updates", "two_stage", 271160),
    ("femnist_cnn_4groups_doubled", "FEMNIST, 4 groups", "CNN, doubled updates", "two_stage", 270932),
    ("femnist_cnn_4groups_doubled_bounded", "FEMNIST, 4 groups", "CNN, doubled, bounded",
     "two_stage_loop", 270931),
    ("stackoverflow_mlp_8groups_bounded", "Stack Overflow, 8 groups", "MLP, natural, bounded",
     "two_stage", 280929),
    ("femnist_cnn_30groups_keep_current", "FEMNIST, 30 groups", "CNN, natural, keep-current",
     "t_groups", None),
]
# Exact-Shapley arm of the 30-group study (credit rule differs from its primary proxy).
EXACT_SHAPLEY_ARM = "femnist_cnn_30groups_keep_current_exact_shapley"
DRAWS = 10000

_INT = ("group", "seed", "state", "client", "choice_committed", "choice_curated")


def load(stem: str) -> list[dict]:
    rows = []
    with (DIR / f"{stem}.csv").open() as f:
        for raw in csv.DictReader(f):
            rows.append({k: (int(v) if k in _INT else (math.nan if v == "" else float(v)))
                         for k, v in raw.items()})
    return rows


def wmean(rows: list[dict], value) -> float:
    pairs = [(r["weight"], value(r)) for r in rows]
    pairs = [(w, v) for w, v in pairs if not math.isnan(v)]
    total = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / total if pairs else math.nan


def effect_matrix(rows: list[dict]) -> np.ndarray:
    """Group x seed matrix of target effects (curated minus committed regret).

    Within a (group, seed) cell the effect is the weight-averaged regret
    difference, i.e. the equal-state mean of the client-weighted difference.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[(r["group"], r["seed"])].append(r)
    groups = sorted({g for g, _ in cells})
    seeds = sorted({s for _, s in cells})
    return np.array([[wmean(cells[(g, s)], lambda r: r["regret_curated"] - r["regret_committed"])
                      for s in seeds] for g in groups])


def two_stage_bootstrap(mat: np.ndarray, seed: int) -> list[float]:
    """Resample groups, then seeds within each resampled group (vectorized draws)."""
    g, s = mat.shape
    rng = np.random.default_rng(seed)
    gi = rng.integers(0, g, size=(DRAWS, g))
    si = rng.integers(0, s, size=(DRAWS, g, s))
    draws = mat[gi[..., None], si].mean(axis=2).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def two_stage_bootstrap_loop(mat: np.ndarray, seed: int) -> list[float]:
    """Same estimator, with draws taken in the per-replicate order that study used."""
    g, s = mat.shape
    rng = np.random.default_rng(seed)
    draws = np.empty(DRAWS)
    for b in range(DRAWS):
        gix = rng.integers(0, g, g)
        draws[b] = np.mean([np.mean(mat[k, rng.integers(0, s, s)]) for k in gix])
    return np.quantile(draws, [0.025, 0.975]).tolist()


def seed_bootstrap(values: np.ndarray, seed: int) -> list[float]:
    """Bootstrap over seeds (single group)."""
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(values), size=(DRAWS, len(values)))
    return np.quantile(values[ix].mean(axis=1), [0.025, 0.975]).tolist()


def interval(rows: list[dict], method: str, seed) -> list[float]:
    mat = effect_matrix(rows)
    if method == "seed_bootstrap":
        assert mat.shape[0] == 1
        return seed_bootstrap(mat[0], seed)
    if method == "two_stage":
        return two_stage_bootstrap(mat, seed)
    if method == "two_stage_loop":
        return two_stage_bootstrap_loop(mat, seed)
    if method == "t_groups":
        return t_interval(mat.mean(axis=1))
    raise ValueError(method)


def summarize(stem: str) -> dict:
    rows = load(stem)
    out = {
        "rows": len(rows),
        "credit_gain": wmean(rows, lambda r: r["credit_curated"] - r["credit_committed"]),
        "flip_rate": wmean(rows, lambda r: r["flip"]),
        "target_effect": wmean(rows, lambda r: r["regret_curated"] - r["regret_committed"]),
        # Excess of panel credit over target credit, committed and curated.
        "excess_committed": wmean(rows, lambda r: r["credit_committed"] - r["credit_target"]),
        "excess_curated": wmean(rows, lambda r: r["credit_curated"] - r["credit_target"]),
    }
    spec = next((s for s in STUDIES if s[0] == stem), None)
    if spec is not None:
        out["ci95"] = interval(rows, spec[3], spec[4])
    return out
