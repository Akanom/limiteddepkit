"""Gaussian interval regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

from ._continuous import (
    _check_optimizer_result,
    _ContinuousResultMixin,
    _inference_series,
    _mle_covariance,
    _validate_covariance_options,
    _validate_fit_design,
    _validate_optimizer_options,
    _validate_outcome,
)


def _log_interval_probability(
    standardized_lower: np.ndarray,
    standardized_upper: np.ndarray,
) -> np.ndarray:
    """Return stable log normal probabilities for finite, nonempty intervals."""
    lower = np.asarray(standardized_lower, dtype=float)
    upper = np.asarray(standardized_upper, dtype=float)
    if lower.shape != upper.shape or np.any(lower >= upper):
        raise ValueError("Standardized interval bounds must have matching shapes and lower < upper.")

    values = np.empty_like(lower)
    left_tail = upper <= 0.0
    right_tail = lower >= 0.0
    central = ~(left_tail | right_tail)

    if np.any(left_tail):
        log_upper_cdf = log_ndtr(upper[left_tail])
        log_lower_cdf = log_ndtr(lower[left_tail])
        values[left_tail] = log_upper_cdf + np.log(
            -np.expm1(log_lower_cdf - log_upper_cdf)
        )
    if np.any(right_tail):
        log_lower_survival = log_ndtr(-lower[right_tail])
        log_upper_survival = log_ndtr(-upper[right_tail])
        values[right_tail] = log_lower_survival + np.log(
            -np.expm1(log_upper_survival - log_lower_survival)
        )
    if np.any(central):
        probabilities = ndtr(upper[central]) - ndtr(lower[central])
        central_values = np.log(probabilities)
        underflow = ~np.isfinite(central_values)
        if np.any(underflow):
            midpoint = (lower[central][underflow] + upper[central][underflow]) / 2.0
            width = upper[central][underflow] - lower[central][underflow]
            central_values[underflow] = norm.logpdf(midpoint) + np.log(width)
        values[central] = central_values
    return values


@dataclass(frozen=True)
class IntervalRegressionResult(_ContinuousResultMixin):
    """Fitted Gaussian interval regression result."""

    params: pd.Series
    sigma: float
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    n_exact: int
    n_interval: int
    n_left_censored: int
    n_right_censored: int
    n_clusters: int | None
    _covariance_type: str
    score_norm: float
    optimizer_result: Any

    def predict(self, X: Any) -> pd.Series:
        """Predict the latent Gaussian mean ``E[y* | X] = X beta``."""
        return self.predict_latent(X)

    def predict_interval(self, X: Any, *, level: float = 0.95) -> pd.DataFrame:
        """Return a latent-outcome predictive interval, not a coefficient interval."""
        return self.predict_latent_interval(X, level=level)


class IntervalRegression:
    """Gaussian regression for exact, interval-, left-, and right-censored data."""

    def fit(
        self,
        X: Any,
        lower: Any,
        upper: Any,
        *,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
        covariance_type: str = "observed-information",
        clusters: Any = None,
    ) -> IntervalRegressionResult:
        maxiter, tolerance = _validate_optimizer_options(maxiter, tolerance)
        design, feature_names = _validate_fit_design(X)
        lower_bounds = _validate_outcome(
            lower,
            name="lower",
            nobs=design.shape[0],
            allow_infinite=True,
        )
        upper_bounds = _validate_outcome(
            upper,
            name="upper",
            nobs=design.shape[0],
            allow_infinite=True,
        )
        if np.any(lower_bounds > upper_bounds):
            raise ValueError("Every lower bound must be less than or equal to its upper bound.")
        if np.any(np.isposinf(lower_bounds)) or np.any(np.isneginf(upper_bounds)):
            raise ValueError("lower cannot be +inf and upper cannot be -inf.")
        if np.any(np.isneginf(lower_bounds) & np.isposinf(upper_bounds)):
            raise ValueError("An interval spanning (-inf, +inf) contains no outcome information.")

        finite_lower = np.isfinite(lower_bounds)
        finite_upper = np.isfinite(upper_bounds)
        exact = finite_lower & finite_upper & (lower_bounds == upper_bounds)
        interval = finite_lower & finite_upper & (lower_bounds < upper_bounds)
        left_censored = np.isneginf(lower_bounds) & finite_upper
        right_censored = finite_lower & np.isposinf(upper_bounds)
        classified = exact | interval | left_censored | right_censored
        if not np.all(classified):
            raise ValueError("Bounds do not define valid exact or censored observations.")
        covariance_type, cluster_codes, n_clusters = _validate_covariance_options(
            covariance_type,
            clusters,
            nobs=design.shape[0],
        )

        n_features = design.shape[1]

        def loglike_contributions(raw_parameters: np.ndarray) -> np.ndarray:
            beta = raw_parameters[:n_features]
            log_sigma = raw_parameters[-1]
            sigma = np.exp(log_sigma)
            mean = design @ beta
            contributions = np.empty(design.shape[0], dtype=float)
            if np.any(exact):
                standardized = (lower_bounds[exact] - mean[exact]) / sigma
                contributions[exact] = norm.logpdf(standardized) - log_sigma
            if np.any(interval):
                standardized_lower = (lower_bounds[interval] - mean[interval]) / sigma
                standardized_upper = (upper_bounds[interval] - mean[interval]) / sigma
                contributions[interval] = _log_interval_probability(
                    standardized_lower, standardized_upper
                )
            if np.any(left_censored):
                standardized_upper = (
                    upper_bounds[left_censored] - mean[left_censored]
                ) / sigma
                contributions[left_censored] = log_ndtr(standardized_upper)
            if np.any(right_censored):
                standardized_lower = (
                    lower_bounds[right_censored] - mean[right_censored]
                ) / sigma
                contributions[right_censored] = log_ndtr(-standardized_lower)
            if not np.isfinite(contributions).all():
                return np.full(design.shape[0], -np.inf)
            return contributions

        def objective(raw_parameters: np.ndarray) -> float:
            return -float(np.sum(loglike_contributions(raw_parameters)))

        finite_widths = upper_bounds[interval] - lower_bounds[interval]
        reference_width = (
            float(np.median(finite_widths)) if finite_widths.size else 1.0
        )
        pseudo_outcome = np.empty(design.shape[0], dtype=float)
        pseudo_outcome[exact] = lower_bounds[exact]
        pseudo_outcome[interval] = (
            lower_bounds[interval] + upper_bounds[interval]
        ) / 2.0
        pseudo_outcome[left_censored] = upper_bounds[left_censored] - reference_width
        pseudo_outcome[right_censored] = lower_bounds[right_censored] + reference_width
        initial_beta = np.linalg.lstsq(design, pseudo_outcome, rcond=None)[0]
        residuals = pseudo_outcome - design @ initial_beta
        sigma_guess = max(float(np.sqrt(np.mean(residuals**2))), reference_width, 0.5)
        initial = np.concatenate([initial_beta, [np.log(sigma_guess)]])
        optimizer_result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * n_features + [(-20.0, 20.0)],
            options={
                "maxiter": maxiter,
                "ftol": min(tolerance, 1e-12),
                "gtol": min(tolerance, 1e-6),
                "maxls": 50,
            },
        )
        score_norm = _check_optimizer_result(
            optimizer_result,
            model_name="Interval regression",
            tolerance=tolerance,
            nobs=design.shape[0],
        )
        raw_parameters = np.asarray(optimizer_result.x, dtype=float)
        beta = raw_parameters[:n_features]
        sigma = float(np.exp(raw_parameters[-1]))
        covariance = _mle_covariance(
            objective,
            loglike_contributions,
            raw_parameters,
            sigma,
            covariance_type=covariance_type,
            cluster_codes=cluster_codes,
        )
        covariance_frame, standard_errors, zstats, pvalues = _inference_series(
            beta, sigma, feature_names, covariance
        )

        return IntervalRegressionResult(
            params=pd.Series(beta, index=feature_names, name="estimate"),
            sigma=sigma,
            covariance=covariance_frame,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            converged=True,
            loglike=-float(optimizer_result.fun),
            nobs=int(design.shape[0]),
            feature_names=tuple(feature_names),
            n_exact=int(np.sum(exact)),
            n_interval=int(np.sum(interval)),
            n_left_censored=int(np.sum(left_censored)),
            n_right_censored=int(np.sum(right_censored)),
            n_clusters=n_clusters,
            _covariance_type=covariance_type,
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
