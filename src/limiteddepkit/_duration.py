"""Shared contracts for parametric and grouped duration estimators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaincc, gammaln
from scipy.stats import norm

from .ordinal import _as_2d_array


def validate_prediction_design(
    X: Any,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, pd.Index]:
    """Validate a duration prediction design and preserve row labels."""
    design, names = _as_2d_array(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(
            f"X must contain {len(feature_names)} regressors; received {design.shape[1]}."
        )
    if isinstance(X, pd.DataFrame) and tuple(names) != feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
    return design, index


def validate_entry(
    entry: Any | None,
    duration: np.ndarray,
    *,
    discrete: bool = False,
) -> np.ndarray:
    """Return validated delayed-entry times/periods."""
    if entry is None:
        return np.zeros(duration.shape, dtype=int if discrete else float)
    values = np.asarray(entry)
    if values.ndim != 1 or values.size != duration.size:
        raise ValueError("entry must contain one value per observation.")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("entry must contain numeric values.") from exc
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
        raise ValueError("entry must contain finite non-negative values.")
    if discrete and np.any(numeric != np.floor(numeric)):
        raise ValueError("entry must contain integer periods for a discrete-time model.")
    if np.any(numeric >= duration):
        raise ValueError("Every entry value must be strictly smaller than duration.")
    return numeric.astype(int if discrete else float)


def validate_frequency_weights(
    weights: Any | None,
    nobs: int,
    *,
    n_params: int,
) -> np.ndarray:
    """Validate exact non-negative integer row-replication weights."""
    if weights is None:
        numeric = np.ones(nobs, dtype=float)
    else:
        values = np.asarray(weights)
        if values.ndim != 1 or values.size != nobs:
            raise ValueError("frequency_weights must contain one value per observation.")
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("frequency_weights must contain numeric values.") from exc
        if (
            not np.isfinite(numeric).all()
            or np.any(numeric < 0.0)
            or np.any(numeric != np.floor(numeric))
        ):
            raise ValueError(
                "frequency_weights must contain finite non-negative integers."
            )
    if np.sum(numeric) <= n_params:
        raise ValueError(
            "The sum of frequency_weights must exceed the number of parameters."
        )
    return numeric


def validate_covariance_request(
    covariance_type: str,
    clusters: Any | None,
    nobs: int,
    *,
    active: np.ndarray | None = None,
) -> tuple[str, np.ndarray | None]:
    """Normalize observed, robust, or cluster covariance requests."""
    normalized = str(covariance_type).strip().lower().replace("_", "-")
    aliases = {
        "observed": "observed-information",
        "observed-information": "observed-information",
        "robust": "sandwich",
        "sandwich": "sandwich",
        "hc0": "sandwich",
        "cluster": "cluster-sandwich",
        "clustered": "cluster-sandwich",
        "cluster-sandwich": "cluster-sandwich",
    }
    if normalized not in aliases:
        raise ValueError(
            "covariance_type must be 'observed', 'robust', or 'cluster'."
        )
    kind = aliases[normalized]
    if kind != "cluster-sandwich":
        if clusters is not None:
            raise ValueError("clusters can be supplied only with covariance_type='cluster'.")
        return kind, None
    if clusters is None:
        raise ValueError("clusters is required with covariance_type='cluster'.")
    labels = np.asarray(clusters, dtype=object)
    if labels.ndim != 1 or labels.size != nobs:
        raise ValueError("clusters must contain one label per observation.")
    active_rows = (
        np.ones(nobs, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    )
    if active_rows.shape != (nobs,):  # pragma: no cover - internal contract
        raise RuntimeError("active must contain one flag per observation.")
    if pd.isna(labels[active_rows]).any():
        raise ValueError("clusters must not be missing on positive-weight rows.")
    try:
        active_codes, levels = pd.factorize(labels[active_rows], sort=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("clusters must contain hashable scalar labels.") from exc
    if len(levels) < 2:
        raise ValueError("Cluster covariance requires at least two distinct clusters.")
    codes = np.full(nobs, -1, dtype=np.int64)
    codes[active_rows] = np.asarray(active_codes, dtype=np.int64)
    return kind, codes


def numerical_score_matrix(
    contribution_function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
) -> np.ndarray:
    """Numerically differentiate per-observation negative log likelihoods."""
    point = np.asarray(parameters, dtype=float)
    baseline = np.asarray(contribution_function(point), dtype=float)
    if baseline.ndim != 1 or not np.isfinite(baseline).all():
        raise RuntimeError("Likelihood contributions are not finite at the fitted solution.")
    scores = np.empty((baseline.size, point.size), dtype=float)
    steps = 1e-5 * (1.0 + np.abs(point))
    for column, step in enumerate(steps):
        shift = np.zeros_like(point)
        shift[column] = step
        upper = np.asarray(contribution_function(point + shift), dtype=float)
        lower = np.asarray(contribution_function(point - shift), dtype=float)
        scores[:, column] = -(upper - lower) / (2.0 * step)
    if not np.isfinite(scores).all():
        raise RuntimeError("Per-observation score contributions are not finite.")
    return scores


def scaled_frequency_score_norm(
    weighted_scores: np.ndarray,
    frequency_weights: np.ndarray,
) -> float:
    """Scale a summed score using exact row-replication semantics."""
    scores = np.asarray(weighted_scores, dtype=float)
    weights = np.asarray(frequency_weights, dtype=float)
    active = weights > 0.0
    base_scores = scores[active] / weights[active, None]
    total = scores[active].sum(axis=0)
    squared_scale = (base_scores**2).T @ weights[active]
    denominator = np.maximum(np.sqrt(squared_scale), np.finfo(float).tiny)
    return float(np.max(np.abs(total) / denominator))


def stationarity_limit(tolerance: float) -> float:
    """Return a strict inference gate independent of a deliberately coarse fit."""
    return max(min(100.0 * float(tolerance), 1e-4), 1e-5)


def log_gammaincc(shape: float, values: np.ndarray) -> np.ndarray:
    """Evaluate ``log(Q(shape, x))`` without upper-tail underflow.

    SciPy's regularized upper incomplete gamma is accurate over its finite
    range but necessarily underflows before its logarithm does.  For
    ``x >= shape + 1`` this uses the standard continued fraction for the upper
    incomplete gamma directly in log space.
    """
    a = float(shape)
    x = np.asarray(values, dtype=float)
    if not np.isfinite(a) or a <= 0.0 or np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("shape must be positive and values finite and non-negative.")
    result = np.empty_like(x, dtype=float)
    use_direct = x < a + 1.0
    if np.any(use_direct):
        result[use_direct] = np.log(gammaincc(a, x[use_direct]))

    tiny = np.finfo(float).tiny / np.finfo(float).eps
    for index in np.argwhere(~use_direct):
        location = tuple(index)
        value = float(x[location])
        b = value + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / max(abs(b), tiny)
        if b < 0.0:
            d = -d
        fraction = d
        for iteration in range(1, 10_001):
            numerator = -iteration * (iteration - a)
            b += 2.0
            d = numerator * d + b
            if abs(d) < tiny:
                d = tiny if d >= 0.0 else -tiny
            c = b + numerator / c
            if abs(c) < tiny:
                c = tiny if c >= 0.0 else -tiny
            d = 1.0 / d
            delta = d * c
            fraction *= delta
            if abs(delta - 1.0) <= 1e-14:
                break
        else:  # pragma: no cover - conservative numerical failure guard
            raise RuntimeError("Upper incomplete-gamma continued fraction did not converge.")
        if fraction <= 0.0 or not np.isfinite(fraction):
            raise RuntimeError("Upper incomplete-gamma continued fraction is invalid.")
        result[location] = (
            -value + a * np.log(value) - gammaln(a) + np.log(fraction)
        )
    return result


def covariance_from_information_and_scores(
    information: np.ndarray,
    *,
    covariance_type: str,
    contribution_function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    cluster_codes: np.ndarray | None,
    frequency_weights: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    """Construct observed-information or finite-sample sandwich covariance."""
    symmetric = (np.asarray(information, dtype=float) + np.asarray(information).T) / 2.0
    if not np.isfinite(symmetric).all():
        raise RuntimeError("The observed-information matrix contains non-finite values.")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    largest = float(eigenvalues[-1])
    if largest <= 0.0 or eigenvalues[0] <= max(1e-10 * largest, 1e-10):
        raise RuntimeError("The observed-information matrix is singular or indefinite.")
    bread = np.linalg.inv(symmetric)
    if covariance_type == "observed-information":
        return (bread + bread.T) / 2.0, None

    scores = numerical_score_matrix(contribution_function, parameters)
    weights = np.asarray(frequency_weights, dtype=float)
    active = weights > 0.0
    correction_nobs = int(np.sum(weights))
    _, nparams = scores.shape
    if covariance_type == "sandwich":
        # ``scores`` already contain w_i. Literal row replication contributes
        # w_i * s_i s_i', not w_i**2 * s_i s_i'.
        replicated_scores = scores[active] / np.sqrt(weights[active])[:, None]
        meat = replicated_scores.T @ replicated_scores
        correction = correction_nobs / (correction_nobs - nparams)
        covariance = correction * bread @ meat @ bread
        return (covariance + covariance.T) / 2.0, None

    if cluster_codes is None:  # pragma: no cover - validated by caller
        raise RuntimeError("Cluster codes are unavailable.")
    active_codes = cluster_codes[active]
    unique = np.unique(active_codes)
    cluster_scores = np.vstack(
        [scores[active & (cluster_codes == code)].sum(axis=0) for code in unique]
    )
    meat = cluster_scores.T @ cluster_scores
    nclusters = len(unique)
    correction = nclusters / (nclusters - 1.0)
    correction *= (correction_nobs - 1.0) / (correction_nobs - nparams)
    covariance = correction * bread @ meat @ bread
    return (covariance + covariance.T) / 2.0, nclusters


def validate_probability(value: float, *, name: str = "probability") -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one.")
    return probability


def prepare_prediction_times(
    times: float | Sequence[float] | np.ndarray,
    *,
    discrete: bool = False,
    allow_zero: bool = True,
) -> tuple[np.ndarray, bool]:
    """Normalize a scalar or unique prediction-time grid."""
    scalar = np.isscalar(times)
    try:
        values = np.asarray([times] if scalar else times, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must contain numeric values.") from exc
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("times must be a non-empty finite scalar or one-dimensional grid.")
    minimum_ok = values >= 0.0 if allow_zero else values > 0.0
    if not np.all(minimum_ok):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"times must contain only {qualifier} values.")
    if discrete and np.any(values != np.floor(values)):
        raise ValueError("times must contain integer periods for a discrete-time model.")
    if len(np.unique(values)) != len(values):
        raise ValueError("times must not contain duplicate values.")
    return values.astype(int if discrete else float), bool(scalar)


def format_time_predictions(
    values: np.ndarray,
    *,
    index: pd.Index,
    times: np.ndarray,
    scalar: bool,
    name: str,
) -> pd.Series | pd.DataFrame:
    """Return a labelled Series for one time or a DataFrame for a grid."""
    array = np.asarray(values, dtype=float)
    if scalar:
        return pd.Series(array[:, 0], index=index, name=name)
    columns = pd.Index(times, name="time")
    return pd.DataFrame(array, index=index, columns=columns)


class DurationResultMixin:
    """Common stable diagnostics for duration-model results."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    loglike: float
    nobs: int
    covariance_type: str
    frequency_weight_sum: float

    @property
    def all_params(self) -> pd.Series:
        values = self.params.copy()
        if hasattr(self, "shape_param"):
            label = "log_k" if type(self).__name__.startswith("Gamma") else "log_alpha"
            values = pd.concat(
                [values, pd.Series({label: np.log(float(self.shape_param))})]
            )
        return values.rename("estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def df_resid(self) -> int:
        return int(self.frequency_weight_sum) - self.n_params

    @property
    def effective_nobs(self) -> int:
        """Return the literal row count implied by frequency weights."""
        return int(self.frequency_weight_sum)

    @property
    def aic(self) -> float:
        return -2.0 * self.loglike + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglike + np.log(self.effective_nobs) * self.n_params

    @property
    def backend(self) -> str:
        return "native-duration-mle"

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = float(norm.ppf(0.5 + level / 2.0))
        estimates = self.all_params
        return pd.DataFrame(
            {
                "lower": estimates - critical * self.standard_errors,
                "upper": estimates + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)
