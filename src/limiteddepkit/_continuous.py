"""Shared contracts for experimental continuous limited-outcome models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ordinal import _as_2d_array, _numerical_hessian


def _validate_fit_design(X: Any) -> tuple[np.ndarray, list[str]]:
    design, feature_names = _as_2d_array(X)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("X feature names must be unique after conversion to strings.")
    if design.shape[0] <= design.shape[1] + 1:
        raise ValueError(
            "Continuous limited-outcome inference requires more observations "
            "than regression and dispersion parameters."
        )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("X is rank deficient; regression parameters are not identified.")
    return design, feature_names


def _validate_outcome(
    values: Any,
    *,
    name: str,
    nobs: int,
    allow_infinite: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    try:
        array = array.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if array.size != nobs:
        raise ValueError(f"X and {name} must contain the same number of observations.")
    if np.isnan(array).any():
        raise ValueError(f"{name} contains missing values.")
    if not allow_infinite and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_optimizer_options(maxiter: int, tolerance: float) -> tuple[int, float]:
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    return int(maxiter), float(tolerance)


def _validate_prediction_design(
    X: Any, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, pd.Index]:
    design, names = _as_2d_array(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(f"X has {design.shape[1]} columns; expected {len(feature_names)}.")
    if isinstance(X, pd.DataFrame) and tuple(names) != feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(design.shape[0])
    return design, index


def _score_norm(optimizer_result: Any) -> float:
    gradient = np.asarray(getattr(optimizer_result, "jac", np.nan), dtype=float)
    if gradient.size == 0 or not np.isfinite(gradient).all():
        return np.inf
    return float(np.max(np.abs(gradient)))


def _check_optimizer_result(
    optimizer_result: Any,
    *,
    model_name: str,
    tolerance: float,
) -> float:
    score_norm = _score_norm(optimizer_result)
    converged = bool(
        optimizer_result.success or score_norm <= max(10.0 * tolerance, 1e-6)
    )
    parameters = np.asarray(optimizer_result.x, dtype=float)
    if (
        not converged
        or not np.isfinite(optimizer_result.fun)
        or not np.isfinite(parameters).all()
    ):
        raise RuntimeError(f"{model_name} optimization failed: {optimizer_result.message}")
    return score_norm


def _observed_information_covariance(
    negative_loglike: Any,
    raw_parameters: np.ndarray,
    sigma: float,
) -> np.ndarray:
    information = _numerical_hessian(negative_loglike, raw_parameters)
    information = (information + information.T) / 2.0
    if not np.isfinite(information).all():
        raise RuntimeError("The observed-information matrix contains non-finite values.")
    eigenvalues = np.linalg.eigvalsh(information)
    largest = float(eigenvalues[-1])
    if largest <= 0.0 or eigenvalues[0] <= max(1e-10 * largest, 1e-10):
        raise RuntimeError(
            "The observed-information matrix is singular or not positive definite; "
            "parameter inference is not reliable."
        )
    raw_covariance = np.linalg.inv(information)
    jacobian = np.eye(raw_parameters.size, dtype=float)
    jacobian[-1, -1] = sigma
    covariance = jacobian @ raw_covariance @ jacobian.T
    return (covariance + covariance.T) / 2.0


def _inference_series(
    beta: np.ndarray,
    sigma: float,
    feature_names: list[str],
    covariance: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    labels = [*feature_names, "sigma"]
    estimates = np.concatenate([np.asarray(beta, dtype=float), [float(sigma)]])
    standard_errors = np.sqrt(np.diag(covariance))
    zstats = estimates / standard_errors
    pvalues = 2.0 * norm.sf(np.abs(zstats))
    return (
        pd.DataFrame(covariance, index=labels, columns=labels),
        pd.Series(standard_errors, index=labels, name="std_err"),
        pd.Series(zstats, index=labels, name="z"),
        pd.Series(pvalues, index=labels, name="p_value"),
    )


class _ContinuousResultMixin:
    """Shared result API for Gaussian continuous limited-outcome models."""

    params: pd.Series
    sigma: float
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    loglike: float
    nobs: int

    @property
    def all_params(self) -> pd.Series:
        dispersion = pd.Series({"sigma": self.sigma}, name=self.params.name)
        return pd.concat([self.params, dispersion])

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def df_resid(self) -> int:
        return self.nobs - self.n_params

    @property
    def aic(self) -> float:
        return -2.0 * self.loglike + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglike + np.log(self.nobs) * self.n_params

    @property
    def inference_valid(self) -> bool:
        return True

    @property
    def covariance_type(self) -> str:
        return "observed-information"

    @property
    def backend(self) -> str:
        return "experimental-native-mle"

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = float(norm.ppf(0.5 + level / 2.0))
        estimates = self.all_params
        intervals = pd.DataFrame(
            {
                "lower": estimates - critical * self.standard_errors,
                "upper": estimates + critical * self.standard_errors,
            }
        )
        log_sigma_standard_error = float(self.standard_errors["sigma"] / self.sigma)
        intervals.loc["sigma", "lower"] = self.sigma * np.exp(
            -critical * log_sigma_standard_error
        )
        intervals.loc["sigma", "upper"] = self.sigma * np.exp(
            critical * log_sigma_standard_error
        )
        return intervals

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()
