"""Count-data Poisson regression estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from .ordinal import _as_2d_array


@dataclass(frozen=True)
class PoissonResult:
    """Fitted Poisson regression result."""

    params: pd.Series
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
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return len(self.params)

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
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


class PoissonRegressor:
    """Poisson regression estimated by maximum likelihood."""

    def fit(self, X: Any, y: Any, *, maxiter: int = 300) -> PoissonResult:
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

        def negative_loglike(beta: np.ndarray) -> float:
            eta = design @ beta
            if np.max(eta) > 709.0:
                return 1e300
            mean = np.exp(eta)
            value = -float(np.sum(counts * eta - mean - gammaln(counts + 1.0)))
            return value if np.isfinite(value) else 1e300

        def gradient(beta: np.ndarray) -> np.ndarray:
            eta = design @ beta
            if np.max(eta) > 709.0:
                return np.full(beta.shape, 1e300)
            return design.T @ (np.exp(eta) - counts)

        initial = np.zeros(design.shape[1], dtype=float)
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="BFGS",
            jac=gradient,
            options={"maxiter": maxiter},
        )
        coefficients = np.asarray(optimizer_result.x, dtype=float)
        fitted_mean = np.exp(design @ coefficients)
        information = design.T @ (fitted_mean[:, None] * design)
        inference_valid = bool(
            optimizer_result.success
            and np.isfinite(information).all()
            and np.linalg.eigvalsh(information).min() > 0.0
        )
        if inference_valid:
            covariance = np.linalg.inv(information)
            standard_errors = np.sqrt(np.diag(covariance))
            zstats = coefficients / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full_like(information, np.nan)
            standard_errors = np.full(coefficients.shape, np.nan)
            zstats = np.full(coefficients.shape, np.nan)
            pvalues = np.full(coefficients.shape, np.nan)

        params = pd.Series(coefficients, index=feature_names, name="estimate")
        covariance_frame = pd.DataFrame(covariance, index=feature_names, columns=feature_names)
        standard_errors_series = pd.Series(standard_errors, index=feature_names, name="std_err")
        zstats_series = pd.Series(zstats, index=feature_names, name="z")
        pvalues_series = pd.Series(pvalues, index=feature_names, name="p_value")

        return PoissonResult(
            params=params,
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
