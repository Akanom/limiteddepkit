"""Negative binomial regression estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from .ordinal import _as_2d_array, _numerical_hessian


@dataclass(frozen=True)
class NegativeBinomialResult:
    """Fitted negative binomial result."""

    params: pd.Series
    log_alpha: float
    alpha: float
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        return pd.concat(
            [self.params.copy(), pd.Series({"log_alpha": self.log_alpha})]
        ).rename("estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.DataFrame(
            {
                "lower": self.params
                - critical * self.standard_errors.loc[list(self.feature_names)],
                "upper": self.params
                + critical * self.standard_errors.loc[list(self.feature_names)],
            }
        )

    def alpha_conf_int(self, level: float = 0.95) -> tuple[float, float]:
        """Return a positive Wald interval for the NB2 dispersion parameter."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        se = float(self.standard_errors.loc["log_alpha"])
        return (
            float(np.exp(self.log_alpha - critical * se)),
            float(np.exp(self.log_alpha + critical * se)),
        )

    def predict(self, X: Any) -> pd.Series:
        design, _ = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X must contain {len(self.feature_names)} regressors; "
                f"received {design.shape[1]}."
            )
        rates = np.exp(design @ self.params.to_numpy(dtype=float))
        return pd.Series(rates, name="predicted")


class NegativeBinomial:
    """Negative binomial regression estimated by maximum likelihood."""

    def fit(self, X: Any, y: Any, *, maxiter: int = 300) -> NegativeBinomialResult:
        design, feature_names = _as_2d_array(X)
        counts = np.asarray(y, dtype=float).reshape(-1)
        if counts.size != design.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if pd.isna(counts).any():
            raise ValueError("y contains missing values.")
        if not np.isfinite(counts).all():
            raise ValueError("y contains non-finite values.")
        if np.any(counts < 0):
            raise ValueError("y must contain non-negative counts.")
        if np.any(counts != np.floor(counts)):
            raise ValueError("y must contain integer counts.")
        if not np.any(counts > 0):
            raise ValueError("At least one positive count is required.")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("X must have full column rank.")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        def negative_loglike(parameters: np.ndarray) -> float:
            beta = parameters[:-1]
            log_alpha = parameters[-1]
            eta = design @ beta
            r = np.exp(-log_alpha)
            log_denom = np.logaddexp(0.0, log_alpha + eta)

            loglik = (
                gammaln(counts + r)
                - gammaln(counts + 1.0)
                - gammaln(r)
                - r * log_denom
                + counts * (log_alpha + eta - log_denom)
            )
            value = -float(np.sum(loglik))
            return value if np.isfinite(value) else 1e300

        initial = np.zeros(design.shape[1] + 1, dtype=float)
        initial[:-1] = np.linalg.lstsq(
            design, np.log(counts + 0.1), rcond=None
        )[0]
        sample_mean = float(np.mean(counts))
        moment_alpha = max(
            (float(np.var(counts, ddof=1)) - sample_mean) / sample_mean**2,
            0.1,
        )
        initial[-1] = np.log(moment_alpha)
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * design.shape[1] + [(-10.0, 10.0)],
            options={"maxiter": maxiter},
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        beta = parameters[:-1]
        log_alpha = parameters[-1]
        alpha = np.exp(log_alpha)

        hessian = _numerical_hessian(negative_loglike, parameters)
        hessian = 0.5 * (hessian + hessian.T)
        inference_valid = bool(
            optimizer_result.success
            and -10.0 < log_alpha < 10.0
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

        params = pd.Series(beta, index=feature_names, name="estimate")
        all_params_index = list(feature_names) + ["log_alpha"]
        covariance_frame = pd.DataFrame(covariance, index=all_params_index, columns=all_params_index)
        standard_errors_series = pd.Series(standard_errors, index=all_params_index, name="std_err")
        zstats_series = pd.Series(zstats, index=all_params_index, name="z")
        pvalues_series = pd.Series(pvalues, index=all_params_index, name="p_value")

        return NegativeBinomialResult(
            params=params,
            log_alpha=log_alpha,
            alpha=alpha,
            covariance=covariance_frame,
            standard_errors=standard_errors_series,
            zstats=zstats_series,
            pvalues=pvalues_series,
            inference_valid=inference_valid,
            converged=bool(optimizer_result.success),
            loglike=-float(optimizer_result.fun),
            nobs=int(design.shape[0]),
            feature_names=tuple(feature_names),
            optimizer_result=optimizer_result,
        )
