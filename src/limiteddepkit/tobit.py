"""Experimental left-censored Gaussian (Tobit) regression."""

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
    _observed_information_covariance,
    _validate_fit_design,
    _validate_optimizer_options,
    _validate_outcome,
    _validate_prediction_design,
)


@dataclass(frozen=True)
class TobitResult(_ContinuousResultMixin):
    """Fitted left-censored Gaussian regression result."""

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
    censoring_point: float
    n_censored: int
    score_norm: float
    optimizer_result: Any

    def predict(self, X: Any, *, which: str = "observed") -> pd.Series:
        """Predict the observed mean, latent mean, or censoring probability.

        ``which="observed"`` returns ``E[max(c, y*) | X]`` and is the default.
        It is not the clipped latent linear predictor.
        """
        design, index = _validate_prediction_design(X, self.feature_names)
        latent_mean = design @ self.params.to_numpy(dtype=float)
        standardized_point = (self.censoring_point - latent_mean) / self.sigma
        censoring_probability = ndtr(standardized_point)
        if which == "latent":
            values = latent_mean
        elif which == "censoring_probability":
            values = censoring_probability
        elif which == "observed":
            values = (
                self.censoring_point * censoring_probability
                + latent_mean * (1.0 - censoring_probability)
                + self.sigma * norm.pdf(standardized_point)
            )
        else:
            raise ValueError(
                "which must be 'observed', 'latent', or 'censoring_probability'."
            )
        return pd.Series(values, index=index, name=f"predicted_{which}")


class Tobit:
    """Left-censored Gaussian regression estimated by maximum likelihood."""

    def __init__(self, *, censoring_point: float = 0.0) -> None:
        if not np.isfinite(censoring_point):
            raise ValueError("censoring_point must be finite.")
        self.censoring_point = float(censoring_point)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> TobitResult:
        maxiter, tolerance = _validate_optimizer_options(maxiter, tolerance)
        design, feature_names = _validate_fit_design(X)
        outcomes = _validate_outcome(y, name="y", nobs=design.shape[0])
        if np.any(outcomes < self.censoring_point):
            raise ValueError(
                "Observed left-censored outcomes cannot be below censoring_point."
            )

        censored = outcomes == self.censoring_point
        uncensored = ~censored
        if not np.any(uncensored):
            raise ValueError("At least one uncensored outcome is required for identification.")

        n_features = design.shape[1]

        def objective(raw_parameters: np.ndarray) -> float:
            beta = raw_parameters[:n_features]
            log_sigma = raw_parameters[-1]
            sigma = np.exp(log_sigma)
            mean = design @ beta
            standardized = (outcomes[uncensored] - mean[uncensored]) / sigma
            censoring_index = (self.censoring_point - mean[censored]) / sigma
            loglike = np.sum(norm.logpdf(standardized) - log_sigma)
            loglike += np.sum(log_ndtr(censoring_index))
            return -float(loglike)

        def gradient(raw_parameters: np.ndarray) -> np.ndarray:
            beta = raw_parameters[:n_features]
            sigma = np.exp(raw_parameters[-1])
            mean = design @ beta
            standardized = (outcomes[uncensored] - mean[uncensored]) / sigma
            censoring_index = (self.censoring_point - mean[censored]) / sigma
            beta_gradient = -(design[uncensored].T @ standardized) / sigma
            log_sigma_gradient = float(np.sum(1.0 - standardized**2))
            if np.any(censored):
                log_mills = norm.logpdf(censoring_index) - log_ndtr(censoring_index)
                mills_ratio = np.exp(np.minimum(log_mills, 700.0))
                beta_gradient += (design[censored].T @ mills_ratio) / sigma
                log_sigma_gradient += float(np.sum(mills_ratio * censoring_index))
            return np.concatenate([beta_gradient, [log_sigma_gradient]])

        initial_beta = np.linalg.lstsq(design, outcomes, rcond=None)[0]
        initial_residuals = outcomes[uncensored] - design[uncensored] @ initial_beta
        sigma_guess = max(float(np.sqrt(np.mean(initial_residuals**2))), 0.5)
        initial = np.concatenate([initial_beta, [np.log(sigma_guess)]])
        optimizer_result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(None, None)] * n_features + [(-20.0, 20.0)],
            options={"maxiter": maxiter, "ftol": tolerance, "gtol": tolerance},
        )
        score_norm = _check_optimizer_result(
            optimizer_result,
            model_name="Tobit",
            tolerance=tolerance,
        )
        raw_parameters = np.asarray(optimizer_result.x, dtype=float)
        beta = raw_parameters[:n_features]
        sigma = float(np.exp(raw_parameters[-1]))
        covariance = _observed_information_covariance(objective, raw_parameters, sigma)
        covariance_frame, standard_errors, zstats, pvalues = _inference_series(
            beta, sigma, feature_names, covariance
        )

        return TobitResult(
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
            censoring_point=self.censoring_point,
            n_censored=int(np.sum(censored)),
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
