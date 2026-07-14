"""Probability calibration diagnostics for limited outcomes.

This module is dependency-light: it uses only the package's mandatory
scientific-Python stack.  Calibration is diagnostic rather than corrective;
the fitted intercept and slope describe held-out predictions and must not be
estimated on the same observations used to fit the prediction model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

__all__ = [
    "BinaryCalibrationResult",
    "BrierDecomposition",
    "OrdinalCalibrationResult",
    "binary_brier_decomposition",
    "binary_calibration_intercept_slope",
    "binary_reliability_table",
    "ordinal_cumulative_calibration",
]


@dataclass(frozen=True)
class BinaryCalibrationResult:
    """Logistic recalibration of a binary probability forecast.

    The fitted model is ``logit(P(Y=1)) = intercept + slope * logit(p)``.
    Perfect calibration corresponds to an intercept of zero and slope of one.
    """

    intercept: float
    slope: float
    covariance: np.ndarray
    converged: bool
    nobs: int
    log_loss_before: float
    log_loss_after: float

    @property
    def standard_errors(self) -> np.ndarray:
        """Return standard errors for ``(intercept, slope)``."""

        return np.sqrt(np.diag(self.covariance))


@dataclass(frozen=True)
class BrierDecomposition:
    """Grouped Murphy decomposition of the binary Brier score.

    With binned probabilities, ``reliability - resolution + uncertainty`` is
    a grouped reconstruction rather than necessarily the raw score.  The
    explicit residual records the within-bin remainder, so the identity
    ``score = reliability - resolution + uncertainty + residual`` always
    holds instead of silently overstating an approximate decomposition.
    """

    score: float
    reliability: float
    resolution: float
    uncertainty: float
    residual: float
    n_bins: int

    @property
    def reconstructed_score(self) -> float:
        """Return the grouped reconstruction before the residual remainder."""

        return self.reliability - self.resolution + self.uncertainty


@dataclass(frozen=True)
class OrdinalCalibrationResult:
    """Calibration diagnostics for every non-trivial ordinal threshold."""

    table: pd.DataFrame
    labels: tuple[Any, ...]

    @property
    def valid_thresholds(self) -> int:
        """Return the number of thresholds with identified recalibration."""

        return int(self.table["valid"].sum())

    @property
    def mean_absolute_intercept(self) -> float:
        """Return the macro mean absolute calibration intercept."""

        values = self.table.loc[self.table["valid"], "intercept"]
        return float(values.abs().mean()) if len(values) else float("nan")

    @property
    def mean_absolute_slope_deviation(self) -> float:
        """Return the macro mean of ``abs(slope - 1)``."""

        values = self.table.loc[self.table["valid"], "slope"]
        return float((values - 1.0).abs().mean()) if len(values) else float("nan")


def _binary_inputs(y_true: Any, probability: Any) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y_true)
    predicted = np.asarray(probability)
    if target.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and probability must be one-dimensional.")
    if len(target) == 0:
        raise ValueError("Calibration requires at least one observation.")
    if len(target) != len(predicted):
        raise ValueError("y_true and probability must have the same length.")
    try:
        target = target.astype(float)
        predicted = predicted.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and probability must be numeric.") from exc
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predicted)):
        raise ValueError("y_true and probability must contain only finite values.")
    if not np.all(np.isin(target, (0.0, 1.0))):
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("probability must contain values between 0 and 1.")
    return target.astype(int), predicted


def _eps(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 0.5:
        raise ValueError("eps must be finite and strictly between 0 and 0.5.")
    return value


def _log_loss(target: np.ndarray, probability: np.ndarray, eps: float) -> float:
    clipped = np.clip(probability, eps, 1.0 - eps)
    return float(
        -np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    )


def binary_calibration_intercept_slope(
    y_true: Any,
    probability: Any,
    *,
    eps: float = 1e-12,
    maxiter: int = 1_000,
) -> BinaryCalibrationResult:
    """Estimate binary calibration intercept and slope by logistic MLE.

    Both outcome classes and variation in the predicted log odds are required.
    Endpoint probabilities are clipped only to form finite log odds.  A failed
    or unidentified recalibration is rejected rather than returned as a valid
    diagnostic.
    """

    target, predicted = _binary_inputs(y_true, probability)
    if len(np.unique(target)) != 2:
        raise ValueError("Binary calibration requires both outcome classes.")
    eps = _eps(eps)
    if isinstance(maxiter, bool) or int(maxiter) != maxiter or int(maxiter) < 1:
        raise ValueError("maxiter must be a positive integer.")

    clipped = np.clip(predicted, eps, 1.0 - eps)
    log_odds = np.log(clipped) - np.log1p(-clipped)
    if np.ptp(log_odds) <= np.sqrt(np.finfo(float).eps):
        raise ValueError("Calibration slope is unidentified for constant predictions.")
    design = np.column_stack((np.ones(len(target)), log_odds))

    def objective(parameters: np.ndarray) -> float:
        linear = design @ parameters
        return float(np.sum(np.logaddexp(0.0, linear) - target * linear))

    def gradient(parameters: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ parameters) - target)

    fitted = minimize(
        objective,
        x0=np.array([0.0, 1.0]),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"Binary recalibration did not converge: {fitted.message}")

    recalibrated = expit(design @ fitted.x)
    variance = recalibrated * (1.0 - recalibrated)
    information = design.T @ (variance[:, None] * design)
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= np.finfo(float).eps * max(1.0, eigenvalues[-1]):
        raise RuntimeError(
            "Binary recalibration information is singular; predictions may separate outcomes."
        )
    covariance = np.linalg.inv(information)
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError("Binary recalibration covariance is not finite.")

    return BinaryCalibrationResult(
        intercept=float(fitted.x[0]),
        slope=float(fitted.x[1]),
        covariance=covariance,
        converged=True,
        nobs=len(target),
        log_loss_before=_log_loss(target, predicted, eps),
        log_loss_after=_log_loss(target, recalibrated, eps),
    )


def _bin_edges(
    probability: np.ndarray,
    n_bins: int,
    strategy: Literal["uniform", "quantile"],
) -> np.ndarray:
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or int(n_bins) < 1:
        raise ValueError("n_bins must be a positive integer.")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, int(n_bins) + 1)
    if strategy == "quantile":
        edges = np.quantile(probability, np.linspace(0.0, 1.0, int(n_bins) + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        return np.array([0.0, 1.0]) if len(edges) < 2 else edges
    raise ValueError("strategy must be 'uniform' or 'quantile'.")


def binary_reliability_table(
    y_true: Any,
    probability: Any,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> pd.DataFrame:
    """Return non-empty probability bins and their observed event rates."""

    target, predicted = _binary_inputs(y_true, probability)
    edges = _bin_edges(predicted, n_bins, strategy)
    bin_index = np.searchsorted(edges[1:-1], predicted, side="right")
    records: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        selected = bin_index == index
        count = int(np.sum(selected))
        if count == 0:
            continue
        mean_probability = float(np.mean(predicted[selected]))
        event_rate = float(np.mean(target[selected]))
        records.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "fraction": count / len(target),
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "calibration_error": mean_probability - event_rate,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "bin",
            "lower",
            "upper",
            "count",
            "fraction",
            "mean_probability",
            "event_rate",
            "calibration_error",
        ],
    )


def binary_brier_decomposition(
    y_true: Any,
    probability: Any,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> BrierDecomposition:
    """Return reliability, resolution and uncertainty for grouped forecasts."""

    target, predicted = _binary_inputs(y_true, probability)
    table = binary_reliability_table(
        target,
        predicted,
        n_bins=n_bins,
        strategy=strategy,
    )
    weights = table["fraction"].to_numpy(dtype=float)
    forecast = table["mean_probability"].to_numpy(dtype=float)
    observed = table["event_rate"].to_numpy(dtype=float)
    prevalence = float(np.mean(target))
    reliability = float(np.sum(weights * (forecast - observed) ** 2))
    resolution = float(np.sum(weights * (observed - prevalence) ** 2))
    uncertainty = prevalence * (1.0 - prevalence)
    score = float(np.mean((predicted - target) ** 2))
    reconstruction = reliability - resolution + uncertainty
    return BrierDecomposition(
        score=score,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        residual=score - reconstruction,
        n_bins=len(table),
    )


def _ordinal_inputs(
    y_true: Any,
    probabilities: Any,
    labels: Any | None,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    target = np.asarray(y_true)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array.")
    try:
        matrix = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be numeric.") from exc
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("probabilities must have at least two class columns.")
    if matrix.shape[0] != len(target):
        raise ValueError("y_true and probabilities must have the same number of rows.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("probabilities must be between 0 and 1.")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError("Rows of probabilities must sum to 1.")

    frame_labels = tuple(probabilities.columns) if isinstance(probabilities, pd.DataFrame) else None
    resolved = tuple(labels) if labels is not None else frame_labels
    if resolved is None:
        resolved = tuple(range(matrix.shape[1]))
    if frame_labels is not None and labels is not None and resolved != frame_labels:
        raise ValueError("labels must match probability DataFrame columns in order.")
    index = pd.Index(resolved)
    if len(index) != matrix.shape[1]:
        raise ValueError("labels must contain one entry per probability column.")
    if not index.is_unique or index.hasnans:
        raise ValueError("labels must be unique and cannot contain missing values.")
    if np.any(pd.isna(target)):
        raise ValueError("y_true cannot contain missing values.")
    codes = index.get_indexer(target)
    if np.any(codes < 0):
        raise ValueError("Every observed category must occur in labels.")
    return codes, matrix, tuple(resolved)


def ordinal_cumulative_calibration(
    y_true: Any,
    probabilities: Any,
    *,
    labels: Any | None = None,
    eps: float = 1e-12,
) -> OrdinalCalibrationResult:
    """Diagnose calibration of each cumulative ordinal probability.

    For boundary ``k``, the binary event is ``Y <= k`` and its forecast is
    the sum of probabilities through category ``k``.  An unidentifiable
    boundary is retained with ``valid=False`` and an explanatory reason so a
    sparse category cannot disappear silently from a calibration report.
    """

    codes, matrix, resolved = _ordinal_inputs(y_true, probabilities, labels)
    cumulative = np.cumsum(matrix, axis=1)[:, :-1]
    records: list[dict[str, Any]] = []
    for boundary in range(matrix.shape[1] - 1):
        target = (codes <= boundary).astype(int)
        forecast = cumulative[:, boundary]
        record: dict[str, Any] = {
            "boundary": boundary,
            "threshold": resolved[boundary],
            "observed_cumulative_rate": float(np.mean(target)),
            "mean_cumulative_probability": float(np.mean(forecast)),
            "brier_score": float(np.mean((forecast - target) ** 2)),
            "intercept": float("nan"),
            "slope": float("nan"),
            "intercept_se": float("nan"),
            "slope_se": float("nan"),
            "valid": False,
            "reason": "",
        }
        try:
            fitted = binary_calibration_intercept_slope(target, forecast, eps=eps)
        except (RuntimeError, ValueError) as exc:
            record["reason"] = str(exc)
        else:
            standard_errors = fitted.standard_errors
            record.update(
                {
                    "intercept": fitted.intercept,
                    "slope": fitted.slope,
                    "intercept_se": float(standard_errors[0]),
                    "slope_se": float(standard_errors[1]),
                    "valid": True,
                }
            )
        records.append(record)
    return OrdinalCalibrationResult(table=pd.DataFrame(records), labels=resolved)
