"""Shared validation and covariance helpers for count regressions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ordinal import _as_2d_array


def validate_count_design(
    X: Any,
) -> tuple[np.ndarray, tuple[str, ...], pd.Index]:
    """Return a finite, identified design and its public row/column schema."""
    design, raw_names = _as_2d_array(X)
    names = tuple(raw_names)
    if len(set(names)) != len(names):
        raise ValueError("X feature names must be unique after conversion to strings.")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("X must have full column rank.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(design.shape[0])
    return design, names, index


def validate_prediction_design(
    X: Any,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, pd.Index]:
    """Validate new data against a fitted count-model schema."""
    design, raw_names = _as_2d_array(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(
            f"X must contain {len(feature_names)} regressors; received {design.shape[1]}."
        )
    if isinstance(X, pd.DataFrame) and tuple(raw_names) != feature_names:
        raise ValueError("X columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(design.shape[0])
    return design, index


def validate_count_response(y: Any, nobs: int, index: pd.Index) -> np.ndarray:
    """Validate an aligned, non-negative integer response."""
    if isinstance(y, pd.Series) and not y.index.equals(index):
        raise ValueError("y index must match the X DataFrame index exactly.")
    raw = np.asarray(y)
    if raw.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    try:
        counts = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("y must contain numeric counts.") from error
    if counts.size != nobs:
        raise ValueError("X and y must contain the same number of observations.")
    if not np.isfinite(counts).all():
        raise ValueError("y contains missing or non-finite values.")
    if np.any(counts < 0.0) or np.any(counts != np.floor(counts)):
        raise ValueError("y must contain non-negative integer counts.")
    if not np.any(counts > 0.0):
        raise ValueError("At least one positive count is required.")
    return counts


def _aligned_vector(
    values: Any,
    *,
    name: str,
    nobs: int,
    index: pd.Index,
    allow_scalar: bool = False,
) -> np.ndarray:
    if isinstance(values, pd.Series) and not values.index.equals(index):
        raise ValueError(f"{name} index must match the X DataFrame index exactly.")
    raw = np.asarray(values, dtype=float)
    if raw.ndim == 0 and allow_scalar:
        raw = np.full(nobs, float(raw))
    if raw.ndim != 1 or raw.size != nobs:
        raise ValueError(f"{name} must contain one value per observation.")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} contains missing or non-finite values.")
    return raw


def combined_offset(
    *,
    offset: Any | None,
    exposure: Any | None,
    nobs: int,
    index: pd.Index,
) -> np.ndarray:
    """Combine an additive offset and the log of a positive exposure."""
    combined = np.zeros(nobs, dtype=float)
    if offset is not None:
        combined += _aligned_vector(
            offset,
            name="offset",
            nobs=nobs,
            index=index,
            allow_scalar=True,
        )
    if exposure is not None:
        exposure_array = _aligned_vector(
            exposure,
            name="exposure",
            nobs=nobs,
            index=index,
            allow_scalar=True,
        )
        if np.any(exposure_array <= 0.0):
            raise ValueError("exposure must contain strictly positive values.")
        combined += np.log(exposure_array)
    return combined


def validate_weights(
    *,
    freq_weights: Any | None,
    analytic_weights: Any | None,
    nobs: int,
    index: pd.Index,
    n_params: int,
) -> tuple[np.ndarray, str, float, int]:
    """Validate mutually exclusive frequency or analytic weights.

    Frequency weights are deliberately restricted to non-negative integers so
    their likelihood and covariance have the exact row-replication meaning.
    Analytic weights scale estimating equations and therefore define a
    pseudo-likelihood rather than a replicated sampling likelihood.
    """
    if freq_weights is not None and analytic_weights is not None:
        raise ValueError("Specify only one of freq_weights and analytic_weights.")

    if freq_weights is not None:
        weights = _aligned_vector(
            freq_weights,
            name="freq_weights",
            nobs=nobs,
            index=index,
        )
        if np.any(weights < 0.0) or np.any(weights != np.floor(weights)):
            raise ValueError("freq_weights must contain non-negative integers.")
        weighted_nobs = float(np.sum(weights))
        if weighted_nobs <= n_params:
            raise ValueError(
                "The sum of freq_weights must exceed the number of parameters."
            )
        return weights, "frequency", weighted_nobs, int(weighted_nobs)

    if analytic_weights is not None:
        weights = _aligned_vector(
            analytic_weights,
            name="analytic_weights",
            nobs=nobs,
            index=index,
        )
        if np.any(weights <= 0.0):
            raise ValueError("analytic_weights must contain strictly positive values.")
        if nobs <= n_params:
            raise ValueError("Inference requires more observations than parameters.")
        return weights, "analytic", float(np.sum(weights)), nobs

    if nobs <= n_params:
        raise ValueError("Inference requires more observations than parameters.")
    return np.ones(nobs, dtype=float), "none", float(nobs), nobs


def validate_covariance(
    *,
    cov_type: str,
    clusters: Any | None,
    nobs: int,
    index: pd.Index,
    active: np.ndarray,
) -> tuple[str, np.ndarray | None, int | None]:
    """Normalize covariance configuration and validate cluster labels."""
    if not isinstance(cov_type, str):
        raise TypeError("cov_type must be a string.")
    normalized = cov_type.strip().lower()
    aliases = {
        "nonrobust": "nonrobust",
        "observed-information": "nonrobust",
        "hc0": "HC0",
        "hc1": "HC1",
        "cluster": "cluster",
    }
    if normalized not in aliases:
        raise ValueError("cov_type must be one of 'nonrobust', 'HC0', 'HC1', or 'cluster'.")
    canonical = aliases[normalized]
    if canonical != "cluster":
        if clusters is not None:
            raise ValueError("clusters may be supplied only when cov_type='cluster'.")
        return canonical, None, None
    if clusters is None:
        raise ValueError("clusters is required when cov_type='cluster'.")
    if isinstance(clusters, pd.Series) and not clusters.index.equals(index):
        raise ValueError("clusters index must match the X DataFrame index exactly.")
    cluster_array = np.asarray(clusters, dtype=object)
    if cluster_array.ndim != 1 or cluster_array.size != nobs:
        raise ValueError("clusters must contain one label per observation.")
    if pd.isna(cluster_array[active]).any():
        raise ValueError("clusters contains missing values among positive-weight rows.")
    n_clusters = int(pd.unique(cluster_array[active]).size)
    if n_clusters < 2:
        raise ValueError("Cluster covariance requires at least two active clusters.")
    return canonical, cluster_array, n_clusters


def scaled_score_norm(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    weight_type: str,
) -> float:
    """Return a scale-invariant maximum score imbalance.

    Each summed score component is divided by its empirical root-sum-square
    contribution. Frequency weights use literal row-replication scaling;
    analytic weights scale each estimating-equation contribution.
    """
    total_score = scores.T @ weights
    if weight_type == "frequency":
        active = weights > 0.0
        squared_scale = (scores[active] ** 2).T @ weights[active]
    else:
        squared_scale = ((scores * weights[:, None]) ** 2).sum(axis=0)
    denominator = np.maximum(np.sqrt(squared_scale), np.finfo(float).tiny)
    return float(np.max(np.abs(total_score) / denominator))


def covariance_from_scores(
    information: np.ndarray,
    scores: np.ndarray,
    *,
    weights: np.ndarray,
    weight_type: str,
    cov_type: str,
    clusters: np.ndarray | None,
    correction_nobs: int,
    n_params: int,
    use_correction: bool,
) -> np.ndarray:
    """Return information or sandwich covariance for independent score rows."""
    bread = np.linalg.inv(information)
    if cov_type == "nonrobust":
        return bread

    if cov_type == "cluster":
        if clusters is None:  # pragma: no cover - protected by validation
            raise RuntimeError("Internal cluster-covariance configuration error.")
        active = weights > 0.0
        weighted_scores = scores[active] * weights[active, None]
        codes, labels = pd.factorize(clusters[active], sort=False)
        cluster_scores = np.zeros((len(labels), scores.shape[1]), dtype=float)
        np.add.at(cluster_scores, codes, weighted_scores)
        meat = cluster_scores.T @ cluster_scores
        n_clusters = len(labels)
        if use_correction:
            meat *= (n_clusters / (n_clusters - 1.0)) * (
                (correction_nobs - 1.0) / (correction_nobs - n_params)
            )
    else:
        if weight_type == "frequency":
            active = weights > 0.0
            weighted_scores = scores[active] * np.sqrt(weights[active])[:, None]
        else:
            weighted_scores = scores * weights[:, None]
        meat = weighted_scores.T @ weighted_scores
        if cov_type == "HC1" and use_correction:
            meat *= correction_nobs / (correction_nobs - n_params)

    covariance = bread @ meat @ bread
    return 0.5 * (covariance + covariance.T)


def validate_optimizer_options(maxiter: int, tolerance: float) -> None:
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
