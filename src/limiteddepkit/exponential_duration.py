"""Exponential duration model (survival analysis) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
class ExponentialDurationResult(DurationResultMixin):
    """Fitted exponential duration result."""

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

    def _scale(self, X: Any) -> tuple[np.ndarray, pd.Index]:
        design, index = validate_prediction_design(X, self.feature_names)
        return np.exp(design @ self.params.to_numpy(dtype=float)), index

    def predict_mean(self, X: Any) -> pd.Series:
        """Predict the conditional mean duration."""
        scale, index = self._scale(X)
        return pd.Series(scale, index=index, name="mean_duration")

    def predict(self, X: Any) -> pd.Series:
        """Alias for :meth:`predict_mean`."""
        return self.predict_mean(X).rename("predicted")

    def predict_survival(self, X: Any, times: Any) -> pd.Series | pd.DataFrame:
        """Predict the survival function at one time or a time grid."""
        scale, index = self._scale(X)
        grid, scalar = prepare_prediction_times(times)
        values = np.exp(-grid[None, :] / scale[:, None])
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="survival"
        )

    def predict_cumulative_hazard(
        self, X: Any, times: Any
    ) -> pd.Series | pd.DataFrame:
        """Predict cumulative hazard at one time or a time grid."""
        scale, index = self._scale(X)
        grid, scalar = prepare_prediction_times(times)
        values = grid[None, :] / scale[:, None]
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="cumulative_hazard"
        )

    def predict_hazard(
        self, X: Any, times: Any | None = None
    ) -> pd.Series | pd.DataFrame:
        """Predict the constant event rate, optionally repeated over a time grid."""
        scale, index = self._scale(X)
        rate = 1.0 / scale
        if times is None:
            return pd.Series(rate, index=index, name="hazard")
        grid, scalar = prepare_prediction_times(times)
        values = np.broadcast_to(rate[:, None], (len(rate), len(grid)))
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="hazard"
        )

    def predict_quantile(self, X: Any, probability: float) -> pd.Series:
        """Predict the duration quantile whose event CDF equals ``probability``."""
        quantile = validate_probability(probability)
        scale, index = self._scale(X)
        values = -scale * np.log1p(-quantile)
        return pd.Series(values, index=index, name=f"quantile_{quantile:g}")


class ExponentialDuration:
    """Exponential duration model for survival data with right censoring.

    Model: Duration ~ Exponential(lambda) where lambda = exp(-X*beta)
    Handles right censoring where event time is only observed up to censoring time.
    """

    def fit(
        self,
        X: Any,
        duration: Any,
        event: Any,
        *,
        entry: Any | None = None,
        frequency_weights: Any | None = None,
        covariance_type: str = "observed",
        clusters: Any | None = None,
        maxiter: int = 300,
        tolerance: float = 1e-8,
    ) -> ExponentialDurationResult:
        """Fit exponential duration model.

        Parameters
        ----------
        X : array-like
            Features
        duration : array-like
            Observed duration (min of event time and censoring time)
        event : array-like
            Event indicator (1 if event observed, 0 if censored)
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
        durations = np.asarray(duration_values, dtype=float)
        events_float = np.asarray(event_values, dtype=float)
        if not np.isfinite(durations).all() or not np.isfinite(events_float).all():
            raise ValueError("duration/event contain missing or non-finite values.")
        if np.any(events_float != np.floor(events_float)):
            raise ValueError("event must be binary (0 or 1).")
        events = events_float.astype(int)

        if design.shape[0] != durations.size:
            raise ValueError("X and duration must have same number of observations.")
        if durations.size != events.size:
            raise ValueError("duration and event must have same length.")
        if np.any(durations <= 0):
            raise ValueError("All durations must be positive.")
        if not np.isin(events, [0, 1]).all():
            raise ValueError("event must be binary (0 or 1).")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        entries = validate_entry(entry, durations)
        weights = validate_frequency_weights(
            frequency_weights,
            len(durations),
            n_params=design.shape[1],
        )
        active = weights > 0.0
        n_events = int(weights @ events)
        if n_events == 0:
            raise ValueError(
                "At least one observed event with positive frequency weight is required."
            )
        if np.linalg.matrix_rank(design[active]) < design.shape[1]:
            raise ValueError("X must have full column rank on positive-weight rows.")
        covariance_label, cluster_codes = validate_covariance_request(
            covariance_type, clusters, len(durations), active=active
        )
        exposure = durations - entries

        def contributions(parameters: np.ndarray) -> np.ndarray:
            eta = design[active] @ parameters
            if np.min(eta) < -709.0:
                return np.where(active, 1e300, 0.0)
            values = np.zeros(len(durations), dtype=float)
            values[active] = weights[active] * (
                events[active] * eta + np.exp(-eta) * exposure[active]
            )
            return values

        def negative_loglike(parameters: np.ndarray) -> float:
            value = float(np.sum(contributions(parameters)))
            return value if np.isfinite(value) else 1e300

        def gradient(parameters: np.ndarray) -> np.ndarray:
            eta = design[active] @ parameters
            if np.min(eta) < -709.0:
                return np.full(parameters.shape, 1e300)
            return design[active].T @ (
                weights[active]
                * (events[active] - np.exp(-eta) * exposure[active])
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
        fitted_exposure = np.exp(-(design[active] @ parameters)) * exposure[active]
        information = design[active].T @ (
            (weights[active] * fitted_exposure)[:, None] * design[active]
        )
        weighted_scores = np.zeros((len(durations), design.shape[1]), dtype=float)
        weighted_scores[active] = (
            weights[active, None]
            * design[active]
            * (fitted_exposure - events[active])[:, None]
        )
        relative_score_norm = scaled_frequency_score_norm(weighted_scores, weights)
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
                frequency_weights=weights,
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

        return ExponentialDurationResult(
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
            frequency_weight_sum=float(np.sum(weights)),
            n_delayed_entry=int(np.count_nonzero(entries > 0.0)),
            scaled_score_norm=relative_score_norm,
            optimizer_result=optimizer_result,
        )
