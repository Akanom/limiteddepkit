"""Weibull duration model (survival analysis) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .ordinal import _as_2d_array, _numerical_hessian


@dataclass(frozen=True)
class WeibullDurationResult:
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

    def predict(self, X: Any) -> pd.Series:
        from scipy.special import gamma

        design, _ = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X must contain {len(self.feature_names)} regressors; "
                f"received {design.shape[1]}."
            )
        linear_pred = design @ self.params.to_numpy(dtype=float)
        scale = np.exp(linear_pred)
        mean_duration = scale * gamma(1.0 + 1.0 / self.shape_param)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.Series(mean_duration, index=index, name="predicted")


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
        maxiter: int = 300,
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
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("X must have full column rank.")
        n_features = design.shape[1]
        n_events = int(events.sum())
        if n_events == 0:
            raise ValueError("At least one observed event is required.")
        if design.shape[0] <= n_features + 1:
            raise ValueError("The number of observations must exceed the parameters.")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        log_durations = np.log(durations)

        def negative_loglike(parameters: np.ndarray) -> float:
            beta = parameters[:n_features]
            log_alpha = parameters[n_features]
            alpha = np.exp(log_alpha)
            eta = design @ beta
            log_scaled = log_durations - eta
            log_cumulative_hazard = alpha * log_scaled
            if np.max(log_cumulative_hazard) > 709.0:
                return 1e300
            cumulative_hazard = np.exp(log_cumulative_hazard)
            loglike = events * (
                log_alpha + (alpha - 1.0) * log_durations - alpha * eta
            ) - cumulative_hazard
            value = -float(np.sum(loglike))
            return value if np.isfinite(value) else 1e300

        def gradient(parameters: np.ndarray) -> np.ndarray:
            beta = parameters[:n_features]
            log_alpha = parameters[n_features]
            alpha = np.exp(log_alpha)
            eta = design @ beta
            log_scaled = log_durations - eta
            log_cumulative_hazard = alpha * log_scaled
            if np.max(log_cumulative_hazard) > 709.0:
                return np.full(parameters.shape, 1e300)
            cumulative_hazard = np.exp(log_cumulative_hazard)
            beta_gradient = alpha * (design.T @ (events - cumulative_hazard))
            shape_gradient = np.sum(
                cumulative_hazard * alpha * log_scaled
                - events * (1.0 + alpha * log_scaled)
            )
            return np.append(beta_gradient, shape_gradient)

        # Initial values
        initial = np.zeros(n_features + 1, dtype=float)
        initial[:n_features] = np.linalg.lstsq(
            design, log_durations, rcond=None
        )[0]
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(None, None)] * n_features + [(-10.0, 10.0)],
            options={"maxiter": maxiter},
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        beta = parameters[:n_features]
        alpha = np.exp(parameters[n_features])

        hessian = _numerical_hessian(negative_loglike, parameters)
        hessian = 0.5 * (hessian + hessian.T)
        score_norm = float(np.linalg.norm(gradient(parameters), ord=np.inf))
        converged = bool(optimizer_result.success or score_norm <= 1e-5)
        inference_valid = bool(
            converged
            and -10.0 < parameters[n_features] < 10.0
            and np.isfinite(hessian).all()
            and np.linalg.eigvalsh(hessian).min() > 0.0
        )
        if inference_valid:
            covariance = np.linalg.inv(hessian)
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
            optimizer_result=optimizer_result,
        )
