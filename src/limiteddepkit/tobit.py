"""One-sided censored Gaussian (Tobit) regression."""

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
    side: str
    n_censored: int
    n_clusters: int | None
    _covariance_type: str
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
        if self.side == "left":
            censoring_probability = ndtr(standardized_point)
        else:
            censoring_probability = ndtr(-standardized_point)
        if which == "latent":
            values = latent_mean
        elif which == "censoring_probability":
            values = censoring_probability
        elif which == "observed":
            tail_adjustment = self.sigma * norm.pdf(standardized_point)
            values = (
                self.censoring_point * censoring_probability
                + latent_mean * (1.0 - censoring_probability)
                + (tail_adjustment if self.side == "left" else -tail_adjustment)
            )
        else:
            raise ValueError(
                "which must be 'observed', 'latent', or 'censoring_probability'."
            )
        return pd.Series(values, index=index, name=f"predicted_{which}")


class Tobit:
    """One-sided censored Gaussian regression estimated by maximum likelihood.

    ``side="left"`` represents ``y = max(c, y*)`` and ``side="right"``
    represents ``y = min(c, y*)``. The default preserves the conventional
    left-censored Tobit specification.
    """

    def __init__(self, *, censoring_point: float = 0.0, side: str = "left") -> None:
        if not np.isfinite(censoring_point):
            raise ValueError("censoring_point must be finite.")
        if not isinstance(side, str) or side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        self.censoring_point = float(censoring_point)
        self.side = side

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
        covariance_type: str = "observed-information",
        clusters: Any = None,
    ) -> TobitResult:
        maxiter, tolerance = _validate_optimizer_options(maxiter, tolerance)
        design, feature_names = _validate_fit_design(X)
        outcomes = _validate_outcome(y, name="y", nobs=design.shape[0])
        invalid_support = (
            outcomes < self.censoring_point
            if self.side == "left"
            else outcomes > self.censoring_point
        )
        if np.any(invalid_support):
            relation = "below" if self.side == "left" else "above"
            raise ValueError(
                f"Observed {self.side}-censored outcomes cannot be {relation} "
                "censoring_point."
            )

        censored = outcomes == self.censoring_point
        uncensored = ~censored
        if not np.any(uncensored):
            raise ValueError("At least one uncensored outcome is required for identification.")
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
            standardized = (outcomes[uncensored] - mean[uncensored]) / sigma
            censoring_index = (self.censoring_point - mean[censored]) / sigma
            contributions[uncensored] = norm.logpdf(standardized) - log_sigma
            contributions[censored] = log_ndtr(
                censoring_index if self.side == "left" else -censoring_index
            )
            return contributions

        def objective(raw_parameters: np.ndarray) -> float:
            return -float(np.sum(loglike_contributions(raw_parameters)))

        def gradient(raw_parameters: np.ndarray) -> np.ndarray:
            beta = raw_parameters[:n_features]
            sigma = np.exp(raw_parameters[-1])
            mean = design @ beta
            standardized = (outcomes[uncensored] - mean[uncensored]) / sigma
            censoring_index = (self.censoring_point - mean[censored]) / sigma
            beta_gradient = -(design[uncensored].T @ standardized) / sigma
            log_sigma_gradient = float(np.sum(1.0 - standardized**2))
            if np.any(censored):
                tail_index = (
                    censoring_index if self.side == "left" else -censoring_index
                )
                log_mills = norm.logpdf(tail_index) - log_ndtr(tail_index)
                mills_ratio = np.exp(np.minimum(log_mills, 700.0))
                side_sign = 1.0 if self.side == "left" else -1.0
                beta_gradient += (
                    side_sign * (design[censored].T @ mills_ratio) / sigma
                )
                log_sigma_gradient += float(
                    side_sign * np.sum(mills_ratio * censoring_index)
                )
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
            options={
                "maxiter": maxiter,
                "ftol": min(tolerance, 1e-12),
                "gtol": min(tolerance, 1e-6),
                "maxls": 50,
            },
        )
        score_norm = _check_optimizer_result(
            optimizer_result,
            model_name="Tobit",
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
            side=self.side,
            n_censored=int(np.sum(censored)),
            n_clusters=n_clusters,
            _covariance_type=covariance_type,
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
