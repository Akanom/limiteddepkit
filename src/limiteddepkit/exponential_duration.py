"""Exponential duration model (survival analysis) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .ordinal import _as_2d_array


@dataclass(frozen=True)
class ExponentialDurationResult:
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
    optimizer_result: Any

    def predict(self, X: Any) -> pd.Series:
        design, _ = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X must contain {len(self.feature_names)} regressors; "
                f"received {design.shape[1]}."
            )
        linear_pred = design @ self.params.to_numpy(dtype=float)
        mean_duration = np.exp(linear_pred)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.Series(mean_duration, index=index, name="predicted")


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
        maxiter: int = 300,
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
        n_events = int(events.sum())
        if n_events == 0:
            raise ValueError("At least one observed event is required.")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("X must have full column rank.")
        if design.shape[0] <= design.shape[1]:
            raise ValueError("The number of observations must exceed the regressors.")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        def negative_loglike(parameters: np.ndarray) -> float:
            eta = design @ parameters
            if np.min(eta) < -709.0:
                return 1e300
            rate_exposure = np.exp(-eta) * durations
            value = float(np.sum(events * eta + rate_exposure))
            return value if np.isfinite(value) else 1e300

        def gradient(parameters: np.ndarray) -> np.ndarray:
            eta = design @ parameters
            if np.min(eta) < -709.0:
                return np.full(parameters.shape, 1e300)
            return design.T @ (events - np.exp(-eta) * durations)

        initial = np.zeros(design.shape[1], dtype=float)
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="BFGS",
            jac=gradient,
            options={"maxiter": maxiter},
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        rate_exposure = np.exp(-(design @ parameters)) * durations
        information = design.T @ (rate_exposure[:, None] * design)
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
            optimizer_result=optimizer_result,
        )
