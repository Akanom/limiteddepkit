"""Small shared helpers for opt-in execution-path acceleration."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from scipy.special import logsumexp

PerformanceEngine = Literal["reference", "accelerated"]


def normalize_performance_engine(value: str) -> PerformanceEngine:
    """Validate a reference/accelerated engine selector."""
    if not isinstance(value, str):
        raise TypeError("engine must be 'reference' or 'accelerated'.")
    normalized = value.strip().lower()
    if normalized not in {"reference", "accelerated"}:
        raise ValueError("engine must be 'reference' or 'accelerated'.")
    return cast(PerformanceEngine, normalized)


def rowwise_logsumexp(values: np.ndarray) -> np.ndarray:
    """Evaluate SciPy's stable log-sum-exp once across equal-width rows.

    This helper changes only call granularity. SciPy still controls the stable
    reduction algorithm, and row order is unchanged.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("values must be a non-empty two-dimensional array.")
    return np.asarray(logsumexp(array, axis=1), dtype=float)
