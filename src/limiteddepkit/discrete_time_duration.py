"""Discrete-time duration model (survival with period data) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from .ordinal import _as_2d_array


@dataclass(frozen=True)
class DiscreteTimeDurationResult:
    """Fitted discrete-time duration result."""

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
    optimizer_result: Any

    def predict_hazard(self, X: Any) -> pd.Series:
        """Predict the constant per-period event hazard."""
        design, _ = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X must contain {len(self.feature_names)} regressors; "
                f"received {design.shape[1]}."
            )
        linear_pred = design @ self.params.to_numpy(dtype=float)
        hazard = expit(linear_pred)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.Series(hazard, index=index, name="hazard")

    def predict_survival(self, X: Any, period: int) -> pd.Series:
        """Predict survival through an integer number of periods."""
        if not isinstance(period, (int, np.integer)) or period < 0:
            raise ValueError("period must be a non-negative integer.")
        hazard = self.predict_hazard(X)
        return ((1.0 - hazard) ** period).rename("survival")

    def predict(self, X: Any) -> pd.Series:
        """Predict the geometric mean duration, 1 / hazard."""
        hazard = self.predict_hazard(X)
        mean_duration = 1.0 / hazard
        return pd.Series(mean_duration, name="predicted")


class DiscreteTimeDuration:
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
        maxiter: int = 300,
    ) -> DiscreteTimeDurationResult:
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
        n_events = int(events.sum())
        n_failures = int(np.sum(durations - events))
        if n_events == 0 or n_failures == 0:
            raise ValueError("Both event and event-free exposure periods are required.")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("X must have full column rank.")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        def negative_loglike(parameters: np.ndarray) -> float:
            eta = design @ parameters
            log_hazard = -np.logaddexp(0.0, -eta)
            log_survival = -np.logaddexp(0.0, eta)
            failures = durations - events
            return -float(np.sum(events * log_hazard + failures * log_survival))

        def gradient(parameters: np.ndarray) -> np.ndarray:
            hazard = expit(design @ parameters)
            return design.T @ (durations * hazard - events)

        initial = np.zeros(design.shape[1], dtype=float)
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="BFGS",
            jac=gradient,
            options={"maxiter": maxiter},
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        fitted_hazard = expit(design @ parameters)
        weights = durations * fitted_hazard * (1.0 - fitted_hazard)
        information = design.T @ (weights[:, None] * design)
        score_norm = float(np.linalg.norm(gradient(parameters), ord=np.inf))
        converged = bool(optimizer_result.success or score_norm <= 1e-6)
        inference_valid = bool(
            converged
            and np.isfinite(information).all()
            and np.linalg.eigvalsh(information).min() > 0.0
        )
        if inference_valid:
            covariance = np.linalg.inv(information)
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

        return DiscreteTimeDurationResult(
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
            optimizer_result=optimizer_result,
        )
