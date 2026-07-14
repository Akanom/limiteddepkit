"""Gamma duration model (survival analysis) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaincc, gammaln
from scipy.stats import norm

from .ordinal import _as_2d_array, _numerical_hessian


@dataclass(frozen=True)
class GammaDurationResult:
    """Fitted Gamma duration result."""

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
        """Return a positive Wald interval for the Gamma shape."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        se = float(self.standard_errors.loc["log_k"])
        return (
            float(np.exp(self.log_shape_param - critical * se)),
            float(np.exp(self.log_shape_param + critical * se)),
        )

    def predict(self, X: Any) -> pd.Series:
        design, _ = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X must contain {len(self.feature_names)} regressors; "
                f"received {design.shape[1]}."
            )
        linear_pred = design @ self.params.to_numpy(dtype=float)

        scale = np.exp(linear_pred)
        mean_duration = self.shape_param * scale
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.Series(mean_duration, index=index, name="predicted")


class GammaDuration:
    """Gamma duration model for survival data with right censoring.

    Model: Duration ~ Gamma(shape=k, scale) where scale = exp(X*beta)
    The exponential model is the special case k=1. Gamma and Weibull are
    distinct duration families; neither generally nests the other.
    Handles right censoring where event time is only observed up to censoring time.
    """

    def fit(
        self,
        X: Any,
        duration: Any,
        event: Any,
        *,
        maxiter: int = 300,
    ) -> GammaDurationResult:
        """Fit Gamma duration model.

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

        n_features = design.shape[1]
        n_events = int(events.sum())
        if n_events == 0:
            raise ValueError("At least one observed event is required.")
        if np.linalg.matrix_rank(design) < n_features:
            raise ValueError("X must have full column rank.")
        if design.shape[0] <= n_features + 1:
            raise ValueError("The number of observations must exceed the parameters.")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        log_durations = np.log(durations)
        tiny = np.finfo(float).tiny

        def negative_loglike(parameters: np.ndarray) -> float:
            beta = parameters[:n_features]
            log_k = parameters[n_features]
            k = np.exp(log_k)  # Ensure positive shape
            eta = design @ beta
            log_t_scaled = log_durations - eta
            # exp(709) is near the floating-point limit. Clipping only affects
            # parameter proposals whose likelihood is already effectively zero.
            t_scaled = np.exp(np.clip(log_t_scaled, -745.0, 709.0))

            # Event observations: log(f(t))
            # f(t) = t^(k-1) exp(-t/scale) / (scale^k Gamma(k))
            event_mask = events == 1
            log_f = (
                (k - 1.0) * log_durations[event_mask]
                - t_scaled[event_mask]
                - k * eta[event_mask]
                - gammaln(k)
            )

            # Censored observations: log(S(t))
            censored_mask = events == 0
            survival_prob = gammaincc(k, t_scaled[censored_mask])
            log_survival = np.log(np.clip(survival_prob, tiny, 1.0))

            value = -float(np.sum(log_f) + np.sum(log_survival))
            return value if np.isfinite(value) else 1e300

        # Initial values
        initial = np.zeros(n_features + 1, dtype=float)
        initial[:n_features] = np.linalg.lstsq(
            design, log_durations, rcond=None
        )[0]

        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * n_features + [(-10.0, 10.0)],
            options={"maxiter": maxiter},
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        beta = parameters[:n_features]
        k = np.exp(parameters[n_features])

        hessian = _numerical_hessian(negative_loglike, parameters)
        hessian = 0.5 * (hessian + hessian.T)
        inference_valid = bool(
            optimizer_result.success
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

        param_labels = list(feature_names) + ["log_k"]
        params_series = pd.Series(beta, index=feature_names, name="coef")
        covariance_frame = pd.DataFrame(covariance, index=param_labels, columns=param_labels)
        standard_errors_series = pd.Series(standard_errors, index=param_labels, name="std_err")
        zstats_series = pd.Series(zstats, index=param_labels, name="z")
        pvalues_series = pd.Series(pvalues, index=param_labels, name="p_value")

        return GammaDurationResult(
            params=params_series,
            shape_param=float(k),
            covariance=covariance_frame,
            standard_errors=standard_errors_series,
            zstats=zstats_series,
            pvalues=pvalues_series,
            inference_valid=inference_valid,
            converged=bool(optimizer_result.success),
            loglike=-float(optimizer_result.fun),
            nobs=int(design.shape[0]),
            n_events=n_events,
            feature_names=tuple(feature_names),
            optimizer_result=optimizer_result,
        )
