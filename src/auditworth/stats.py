"""Interval helpers used throughout (95% t intervals over independent strata)."""
from __future__ import annotations

import numpy as np
from scipy.stats import t


def t_interval(values, level: float = 0.95) -> list[float]:
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return [float("nan"), float("nan")]
    half = float(t.ppf(0.5 + level / 2, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return [float(a.mean() - half), float(a.mean() + half)]
