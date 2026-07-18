"""Discrete-time duration model (survival with period data) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from ._duration import (
    DurationResultMixin,
    covariance_from_information_and_scores,
    format_time_predictions,
    prepare_prediction_times,
    scaled_frequency_score_norm,
    stationarity_limit,
    validate_covariance_request,
    validate_entry,
    validate_frequency_weights,
    validate_prediction_design,
    validate_probability,
)
from .ordinal import _as_2d_array


@dataclass(frozen=True)
class GeometricDurationResult(DurationResultMixin):
    """Fitted constant-hazard geometric duration result."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    converged: bool
    loglike: float
    nobs: int
    n_events: int
    feature_names: tuple[str, ...]
    covariance_type: str
    n_clusters: int | None
    frequency_weight_sum: float
    n_delayed_entry: int
    scaled_score_norm: float
    optimizer_result: Any

    def _hazard(self, X: Any) -> tuple[np.ndarray, pd.Index]:
        design, index = validate_prediction_design(X, self.feature_names)
        return expit(design @ self.params.to_numpy(dtype=float)), index

    def predict_hazard(
        self, X: Any, times: Any | None = None
    ) -> pd.Series | pd.DataFrame:
        """Predict the constant per-period hazard."""
        hazard, index = self._hazard(X)
        if times is None:
            return pd.Series(hazard, index=index, name="hazard")
        grid, scalar = prepare_prediction_times(times, discrete=True)
        values = np.broadcast_to(hazard[:, None], (len(hazard), len(grid)))
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="hazard"
        )

    def predict_survival(
        self,
        X: Any,
        times: Any | None = None,
        *,
        period: Any | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Predict survival through one period or a period grid."""
        if times is not None and period is not None:
            raise ValueError("Specify only one of times and the compatibility alias period.")
        requested = period if times is None else times
        if requested is None:
            raise ValueError("times is required.")
        hazard, index = self._hazard(X)
        grid, scalar = prepare_prediction_times(requested, discrete=True)
        values = (1.0 - hazard[:, None]) ** grid[None, :]
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="survival"
        )

    def predict_cumulative_hazard(
        self, X: Any, times: Any
    ) -> pd.Series | pd.DataFrame:
        """Predict discrete cumulative hazard ``-log(S_t)``."""
        hazard, index = self._hazard(X)
        grid, scalar = prepare_prediction_times(times, discrete=True)
        values = -grid[None, :] * np.log1p(-hazard[:, None])
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="cumulative_hazard"
        )

    def predict_quantile(self, X: Any, probability: float) -> pd.Series:
        """Predict the first integer period reaching an event CDF probability."""
        quantile = validate_probability(probability)
        hazard, index = self._hazard(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.maximum(
                np.ceil(np.log1p(-quantile) / np.log1p(-hazard)),
                1.0,
            )
        maximum_integer = np.iinfo(np.int64).max
        if np.isfinite(values).all() and np.all(values <= maximum_integer):
            values = values.astype(np.int64)
        else:
            values = np.where(
                np.isfinite(values) & (values <= maximum_integer), values, np.inf
            )
        return pd.Series(values, index=index, name=f"quantile_{quantile:g}")

    def predict_mean(self, X: Any) -> pd.Series:
        """Predict the geometric mean duration, ``1 / hazard``."""
        hazard, index = self._hazard(X)
        return pd.Series(1.0 / hazard, index=index, name="mean_duration")

    def predict(self, X: Any) -> pd.Series:
        """Alias for :meth:`predict_mean`."""
        return self.predict_mean(X).rename("predicted")


class GeometricDuration:
    """Discrete-time duration model for period-recorded survival data.

    This grouped geometric-duration model uses a logit hazard that is constant
    over periods for a subject: ``h_i = logistic(X_i beta)``. An event at
    period ``t`` contributes ``(t-1) log(1-h_i) + log(h_i)``; censoring through
    period ``t`` contributes ``t log(1-h_i)``.
    """

    def fit(
        self,
        X: Any,
        duration: Any,
        event: Any,
        *,
        entry_period: Any | None = None,
        frequency_weights: Any | None = None,
        covariance_type: str = "observed",
        clusters: Any | None = None,
        maxiter: int = 300,
        tolerance: float = 1e-8,
    ) -> GeometricDurationResult:
        """Fit discrete-time duration model.

        Parameters
        ----------
        X : array-like
            Features (same for all periods for individual)
        duration : array-like
            Duration (period number, 1-indexed)
        event : array-like
            Event indicator (1 if event observed, 0 if censored at this period)
        """
        design, feature_names = _as_2d_array(X)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X feature names must be unique after conversion to strings.")
        duration_values = np.asarray(duration)
        event_values = np.asarray(event)
        if duration_values.ndim != 1:
            raise ValueError("duration must be one-dimensional.")
        if event_values.ndim != 1:
            raise ValueError("event must be one-dimensional.")
        if pd.isna(duration_values).any() or pd.isna(event_values).any():
            raise ValueError("duration/event contain missing values.")
        durations_float = np.asarray(duration_values, dtype=float)
        events_float = np.asarray(event_values, dtype=float)
        if not np.isfinite(durations_float).all() or not np.isfinite(events_float).all():
            raise ValueError("duration/event contain non-finite values.")
        if np.any(durations_float != np.floor(durations_float)):
            raise ValueError("All durations must be positive integers.")
        if np.any(events_float != np.floor(events_float)):
            raise ValueError("event must be binary (0 or 1).")
        durations = durations_float.astype(int)
        events = events_float.astype(int)

        if design.shape[0] != durations.size:
            raise ValueError("X and duration must have same number of observations.")
        if durations.size != events.size:
            raise ValueError("duration and event must have same length.")
        if np.any(durations <= 0):
            raise ValueError("All durations must be positive integers.")
        if not np.isin(events, [0, 1]).all():
            raise ValueError("event must be binary (0 or 1).")
        entries = validate_entry(entry_period, durations, discrete=True)
        exposure_periods = durations - entries
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        frequency = validate_frequency_weights(
            frequency_weights,
            len(durations),
            n_params=design.shape[1],
        )
        active = frequency > 0.0
        n_events = int(frequency @ events)
        n_failures = int(frequency @ (exposure_periods - events))
        if n_events == 0 or n_failures == 0:
            raise ValueError(
                "Both event and event-free exposure periods with positive frequency "
                "weight are required."
            )
        if np.linalg.matrix_rank(design[active]) < design.shape[1]:
            raise ValueError("X must have full column rank on positive-weight rows.")
        covariance_label, cluster_codes = validate_covariance_request(
            covariance_type, clusters, len(durations), active=active
        )

        def contributions(parameters: np.ndarray) -> np.ndarray:
            eta = design[active] @ parameters
            log_hazard = -np.logaddexp(0.0, -eta)
            log_survival = -np.logaddexp(0.0, eta)
            failures = exposure_periods[active] - events[active]
            values = np.zeros(len(durations), dtype=float)
            values[active] = -frequency[active] * (
                events[active] * log_hazard + failures * log_survival
            )
            return values

        def negative_loglike(parameters: np.ndarray) -> float:
            return float(np.sum(contributions(parameters)))

        def gradient(parameters: np.ndarray) -> np.ndarray:
            hazard = expit(design[active] @ parameters)
            return design[active].T @ (
                frequency[active]
                * (exposure_periods[active] * hazard - events[active])
            )

        initial = np.zeros(design.shape[1], dtype=float)
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="BFGS",
            jac=gradient,
            options={"maxiter": int(maxiter), "gtol": float(tolerance)},
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        fitted_hazard = expit(design[active] @ parameters)
        information_weights = (
            frequency[active]
            * exposure_periods[active]
            * fitted_hazard
            * (1.0 - fitted_hazard)
        )
        information = design[active].T @ (
            information_weights[:, None] * design[active]
        )
        weighted_scores = np.zeros((len(durations), design.shape[1]), dtype=float)
        weighted_scores[active] = (
            frequency[active, None]
            * design[active]
            * (events[active] - exposure_periods[active] * fitted_hazard)[:, None]
        )
        relative_score_norm = scaled_frequency_score_norm(
            weighted_scores, frequency
        )
        converged = bool(
            np.isfinite(optimizer_result.fun)
            and np.isfinite(parameters).all()
            and relative_score_norm <= stationarity_limit(tolerance)
        )
        n_clusters: int | None = None
        try:
            covariance, n_clusters = covariance_from_information_and_scores(
                information,
                covariance_type=covariance_label,
                contribution_function=contributions,
                parameters=parameters,
                cluster_codes=cluster_codes,
                frequency_weights=frequency,
            )
            inference_valid = converged
        except RuntimeError:
            covariance = np.full_like(information, np.nan)
            inference_valid = False
        if inference_valid:
            standard_errors = np.sqrt(np.diag(covariance))
            zstats = parameters / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full_like(information, np.nan)
            standard_errors = np.full(parameters.shape, np.nan)
            zstats = np.full(parameters.shape, np.nan)
            pvalues = np.full(parameters.shape, np.nan)

        params_series = pd.Series(parameters, index=feature_names, name="coef")
        covariance_frame = pd.DataFrame(covariance, index=feature_names, columns=feature_names)
        standard_errors_series = pd.Series(standard_errors, index=feature_names, name="std_err")
        zstats_series = pd.Series(zstats, index=feature_names, name="z")
        pvalues_series = pd.Series(pvalues, index=feature_names, name="p_value")

        return GeometricDurationResult(
            params=params_series,
            covariance=covariance_frame,
            standard_errors=standard_errors_series,
            zstats=zstats_series,
            pvalues=pvalues_series,
            inference_valid=inference_valid,
            converged=converged,
            loglike=-float(optimizer_result.fun),
            nobs=int(design.shape[0]),
            n_events=n_events,
            feature_names=tuple(feature_names),
            covariance_type=covariance_label,
            n_clusters=n_clusters,
            frequency_weight_sum=float(np.sum(frequency)),
            n_delayed_entry=int(np.count_nonzero(entries > 0)),
            scaled_score_norm=relative_score_norm,
            optimizer_result=optimizer_result,
        )


# The likelihood is geometric because its logit hazard is constant within a
# spell.  The explicit name prevents callers from mistaking it for a general
# person-period model with time-varying baseline hazards.  Historical names are
# retained as compatibility aliases.
DiscreteTimeDuration = GeometricDuration
DiscreteTimeDurationResult = GeometricDurationResult


__all__ = [
    "DiscreteTimeDuration",
    "DiscreteTimeDurationResult",
    "GeometricDuration",
    "GeometricDurationResult",
]
