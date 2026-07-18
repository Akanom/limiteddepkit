"""One-sided truncated Gaussian regression."""

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
class TruncatedRegressionResult(_ContinuousResultMixin):
    """Fitted left-truncated Gaussian regression result."""

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
    truncation_point: float
    side: str
    n_clusters: int | None
    _covariance_type: str
    score_norm: float
    optimizer_result: Any

    def predict(self, X: Any, *, which: str = "conditional") -> pd.Series:
        """Predict the observed-sample mean, latent mean, or selection probability.

        The default is ``E[y* | y* > a, X]``, the mean among observations that
        survive left truncation. ``which="latent"`` returns ``X beta``.
        """
        design, index = _validate_prediction_design(X, self.feature_names)
        latent_mean = design @ self.params.to_numpy(dtype=float)
        truncation_index = (self.truncation_point - latent_mean) / self.sigma
        selection_probability = ndtr(
            -truncation_index if self.side == "left" else truncation_index
        )
        if which == "latent":
            values = latent_mean
        elif which == "selection_probability":
            values = selection_probability
        elif which == "conditional":
            tail_index = (
                -truncation_index if self.side == "left" else truncation_index
            )
            log_mills = norm.logpdf(tail_index) - log_ndtr(tail_index)
            mills_ratio = np.exp(np.minimum(log_mills, 700.0))
            direction = 1.0 if self.side == "left" else -1.0
            values = latent_mean + direction * self.sigma * mills_ratio
        else:
            raise ValueError(
                "which must be 'conditional', 'latent', or 'selection_probability'."
            )
        return pd.Series(values, index=index, name=f"predicted_{which}")


class TruncatedRegression:
    """One-sided truncated Gaussian regression estimated by maximum likelihood.

    ``side="left"`` fits observations retained when ``y* > a``;
    ``side="right"`` fits observations retained when ``y* < a``.
    """

    def __init__(self, *, truncation_point: float = 0.0, side: str = "left") -> None:
        if not np.isfinite(truncation_point):
            raise ValueError("truncation_point must be finite.")
        if not isinstance(side, str) or side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        self.truncation_point = float(truncation_point)
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
    ) -> TruncatedRegressionResult:
        maxiter, tolerance = _validate_optimizer_options(maxiter, tolerance)
        design, feature_names = _validate_fit_design(X)
        outcomes = _validate_outcome(y, name="y", nobs=design.shape[0])
        invalid_support = (
            outcomes <= self.truncation_point
            if self.side == "left"
            else outcomes >= self.truncation_point
        )
        if np.any(invalid_support):
            relation = "greater" if self.side == "left" else "less"
            raise ValueError(
                f"All outcomes must be strictly {relation} than truncation_point "
                f"{self.truncation_point}."
            )
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
            standardized = (outcomes - mean) / sigma
            truncation_index = (self.truncation_point - mean) / sigma
            tail_index = (
                -truncation_index if self.side == "left" else truncation_index
            )
            return norm.logpdf(standardized) - log_sigma - log_ndtr(tail_index)

        def objective(raw_parameters: np.ndarray) -> float:
            return -float(np.sum(loglike_contributions(raw_parameters)))

        def gradient(raw_parameters: np.ndarray) -> np.ndarray:
            beta = raw_parameters[:n_features]
            sigma = np.exp(raw_parameters[-1])
            mean = design @ beta
            standardized = (outcomes - mean) / sigma
            truncation_index = (self.truncation_point - mean) / sigma
            tail_index = (
                -truncation_index if self.side == "left" else truncation_index
            )
            log_mills = norm.logpdf(tail_index) - log_ndtr(tail_index)
            mills_ratio = np.exp(np.minimum(log_mills, 700.0))
            direction = 1.0 if self.side == "left" else -1.0
            beta_gradient = design.T @ (
                direction * mills_ratio - standardized
            ) / sigma
            log_sigma_gradient = np.sum(
                1.0
                - standardized**2
                + direction * mills_ratio * truncation_index
            )
            return np.concatenate([beta_gradient, [float(log_sigma_gradient)]])

        initial_beta = np.linalg.lstsq(design, outcomes, rcond=None)[0]
        initial_residuals = outcomes - design @ initial_beta
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
            model_name="Truncated regression",
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

        return TruncatedRegressionResult(
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
            truncation_point=self.truncation_point,
            side=self.side,
            n_clusters=n_clusters,
            _covariance_type=covariance_type,
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
