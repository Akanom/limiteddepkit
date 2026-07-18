"""Weibull duration model (survival analysis) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma as gamma_function
from scipy.stats import norm

from ._duration import (
    DurationResultMixin,
    covariance_from_information_and_scores,
    format_time_predictions,
    numerical_score_matrix,
    prepare_prediction_times,
    scaled_frequency_score_norm,
    stationarity_limit,
    validate_covariance_request,
    validate_entry,
    validate_frequency_weights,
    validate_prediction_design,
    validate_probability,
)
from .ordinal import _as_2d_array, _numerical_hessian


@dataclass(frozen=True)
class WeibullDurationResult(DurationResultMixin):
    """Fitted Weibull duration result."""

    params: pd.Series
    shape_param: float
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

    @property
    def log_shape_param(self) -> float:
        return float(np.log(self.shape_param))

    def shape_conf_int(self, level: float = 0.95) -> tuple[float, float]:
        """Return a positive Wald interval for the Weibull shape."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        se = float(self.standard_errors.loc["log_alpha"])
        return (
            float(np.exp(self.log_shape_param - critical * se)),
            float(np.exp(self.log_shape_param + critical * se)),
        )

    def _scale(self, X: Any) -> tuple[np.ndarray, pd.Index]:
        design, index = validate_prediction_design(X, self.feature_names)
        return np.exp(design @ self.params.to_numpy(dtype=float)), index

    def predict_mean(self, X: Any) -> pd.Series:
        """Predict the conditional mean duration."""
        scale, index = self._scale(X)
        mean = scale * gamma_function(1.0 + 1.0 / self.shape_param)
        return pd.Series(mean, index=index, name="mean_duration")

    def predict(self, X: Any) -> pd.Series:
        """Alias for :meth:`predict_mean`."""
        return self.predict_mean(X).rename("predicted")

    def predict_cumulative_hazard(
        self, X: Any, times: Any
    ) -> pd.Series | pd.DataFrame:
        """Predict cumulative hazard at one time or a time grid."""
        scale, index = self._scale(X)
        grid, scalar = prepare_prediction_times(times)
        values = (grid[None, :] / scale[:, None]) ** self.shape_param
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="cumulative_hazard"
        )

    def predict_survival(self, X: Any, times: Any) -> pd.Series | pd.DataFrame:
        """Predict survival at one time or a time grid."""
        scale, index = self._scale(X)
        grid, scalar = prepare_prediction_times(times)
        cumulative = (grid[None, :] / scale[:, None]) ** self.shape_param
        return format_time_predictions(
            np.exp(-cumulative),
            index=index,
            times=grid,
            scalar=scalar,
            name="survival",
        )

    def predict_hazard(self, X: Any, times: Any) -> pd.Series | pd.DataFrame:
        """Predict hazard at one positive time or a time grid."""
        scale, index = self._scale(X)
        grid, scalar = prepare_prediction_times(times, allow_zero=False)
        ratio = grid[None, :] / scale[:, None]
        values = self.shape_param / scale[:, None] * ratio ** (self.shape_param - 1.0)
        return format_time_predictions(
            values, index=index, times=grid, scalar=scalar, name="hazard"
        )

    def predict_quantile(self, X: Any, probability: float) -> pd.Series:
        """Predict the duration quantile whose event CDF equals ``probability``."""
        quantile = validate_probability(probability)
        scale, index = self._scale(X)
        values = scale * (-np.log1p(-quantile)) ** (1.0 / self.shape_param)
        return pd.Series(values, index=index, name=f"quantile_{quantile:g}")


class WeibullDuration:
    """Weibull duration model for survival data with right censoring.

    Model: Duration ~ Weibull(shape=alpha, scale) where scale = exp(X*beta)
    Generalizes exponential model (exponential is special case when alpha=1).
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
    ) -> WeibullDurationResult:
        """Fit Weibull duration model.

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
        if "log_alpha" in feature_names:
            raise ValueError("X feature name 'log_alpha' is reserved for Weibull shape.")
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
        n_features = design.shape[1]
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        entries = validate_entry(entry, durations)
        weights = validate_frequency_weights(
            frequency_weights,
            len(durations),
            n_params=n_features + 1,
        )
        active = weights > 0.0
        n_events = int(weights @ events)
        if n_events == 0:
            raise ValueError(
                "At least one observed event with positive frequency weight is required."
            )
        if np.linalg.matrix_rank(design[active]) < n_features:
            raise ValueError("X must have full column rank on positive-weight rows.")
        covariance_label, cluster_codes = validate_covariance_request(
            covariance_type, clusters, len(durations), active=active
        )

        active_design = design[active]
        active_events = events[active]
        active_entries = entries[active]
        active_weights = weights[active]
        log_durations = np.log(durations[active])
        log_entries = np.full(len(active_entries), -np.inf, dtype=float)
        positive_entry = active_entries > 0.0
        log_entries[positive_entry] = np.log(active_entries[positive_entry])

        def contributions(parameters: np.ndarray) -> np.ndarray:
            beta = parameters[:n_features]
            log_alpha = parameters[n_features]
            alpha = np.exp(log_alpha)
            eta = active_design @ beta
            log_scaled = log_durations - eta
            log_cumulative = alpha * log_scaled
            log_entry_cumulative = alpha * (log_entries - eta)
            if np.max(log_cumulative) > 709.0:
                return np.where(active, 1e300, 0.0)
            cumulative = np.exp(log_cumulative)
            entry_cumulative = np.zeros(len(active_entries), dtype=float)
            entry_cumulative[positive_entry] = np.exp(
                np.clip(log_entry_cumulative[positive_entry], -745.0, 709.0)
            )
            log_density_part = active_events * (
                log_alpha + (alpha - 1.0) * log_durations - alpha * eta
            )
            values = np.zeros(len(durations), dtype=float)
            values[active] = active_weights * (
                cumulative - entry_cumulative - log_density_part
            )
            return values

        def negative_loglike(parameters: np.ndarray) -> float:
            value = float(np.sum(contributions(parameters)))
            return value if np.isfinite(value) else 1e300

        def gradient(parameters: np.ndarray) -> np.ndarray:
            beta = parameters[:n_features]
            log_alpha = parameters[n_features]
            alpha = np.exp(log_alpha)
            eta = active_design @ beta
            log_scaled = log_durations - eta
            log_cumulative = alpha * log_scaled
            log_entry_scaled = log_entries - eta
            log_entry_cumulative = alpha * log_entry_scaled
            if np.max(log_cumulative) > 709.0:
                return np.full(parameters.shape, 1e300)
            cumulative = np.exp(log_cumulative)
            entry_cumulative = np.zeros(len(active_entries), dtype=float)
            entry_cumulative[positive_entry] = np.exp(
                np.clip(log_entry_cumulative[positive_entry], -745.0, 709.0)
            )
            entry_shape_term = np.zeros(len(active_entries), dtype=float)
            entry_shape_term[positive_entry] = (
                entry_cumulative[positive_entry]
                * alpha
                * log_entry_scaled[positive_entry]
            )
            difference = cumulative - entry_cumulative
            beta_gradient = alpha * (
                active_design.T
                @ (active_weights * (active_events - difference))
            )
            shape_gradient = np.sum(
                active_weights
                * (
                    cumulative * alpha * log_scaled
                    - entry_shape_term
                    - active_events * (1.0 + alpha * log_scaled)
                )
            )
            return np.append(beta_gradient, shape_gradient)

        # Initial values
        initial = np.zeros(n_features + 1, dtype=float)
        initial[:n_features] = np.linalg.lstsq(
            active_design, log_durations, rcond=None
        )[0]
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(None, None)] * n_features + [(-10.0, 10.0)],
            options={
                "maxiter": int(maxiter),
                "ftol": float(min(tolerance**2, 1e-14)),
                "gtol": float(tolerance),
                "maxls": 50,
            },
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        beta = parameters[:n_features]
        alpha = np.exp(parameters[n_features])

        hessian = _numerical_hessian(negative_loglike, parameters)
        hessian = 0.5 * (hessian + hessian.T)
        weighted_scores = numerical_score_matrix(contributions, parameters)
        relative_score_norm = scaled_frequency_score_norm(weighted_scores, weights)
        converged = bool(
            np.isfinite(optimizer_result.fun)
            and np.isfinite(parameters).all()
            and relative_score_norm <= stationarity_limit(tolerance)
            and -10.0 < parameters[n_features] < 10.0
        )
        structurally_valid = bool(
            converged
            and -10.0 < parameters[n_features] < 10.0
            and np.isfinite(hessian).all()
            and np.linalg.eigvalsh(hessian).min() > 0.0
        )
        n_clusters: int | None = None
        if structurally_valid:
            try:
                covariance, n_clusters = covariance_from_information_and_scores(
                    hessian,
                    covariance_type=covariance_label,
                    contribution_function=contributions,
                    parameters=parameters,
                    cluster_codes=cluster_codes,
                    frequency_weights=weights,
                )
                inference_valid = True
            except RuntimeError:
                covariance = np.full_like(hessian, np.nan)
                inference_valid = False
        else:
            covariance = np.full_like(hessian, np.nan)
            inference_valid = False
        if inference_valid:
            standard_errors = np.sqrt(np.diag(covariance))
            zstats = parameters / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full_like(hessian, np.nan)
            standard_errors = np.full(parameters.shape, np.nan)
            zstats = np.full(parameters.shape, np.nan)
            pvalues = np.full(parameters.shape, np.nan)

        param_labels = list(feature_names) + ["log_alpha"]
        params_series = pd.Series(beta, index=feature_names, name="coef")
        covariance_frame = pd.DataFrame(covariance, index=param_labels, columns=param_labels)
        standard_errors_series = pd.Series(standard_errors, index=param_labels, name="std_err")
        zstats_series = pd.Series(zstats, index=param_labels, name="z")
        pvalues_series = pd.Series(pvalues, index=param_labels, name="p_value")

        return WeibullDurationResult(
            params=params_series,
            shape_param=float(alpha),
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
            n_delayed_entry=int(np.count_nonzero((entries > 0.0) & active)),
            scaled_score_norm=relative_score_norm,
            optimizer_result=optimizer_result,
        )
