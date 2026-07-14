"""Gaussian sample-selection (Heckman) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

from .ordinal import _as_2d_array, _numerical_hessian


@dataclass(frozen=True)
class SampleSelectionResult:
    """Full-information maximum-likelihood Heckman result."""

    params_outcome: pd.Series
    params_selection: pd.Series
    sigma: float
    rho: float
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    converged: bool
    loglike: float
    nobs_observed: int
    nobs_total: int
    feature_names_x: tuple[str, ...]
    feature_names_z: tuple[str, ...]
    optimizer_result: Any

    @property
    def log_sigma(self) -> float:
        return float(np.log(self.sigma))

    @property
    def atanh_rho(self) -> float:
        return float(np.arctanh(self.rho))

    def sigma_conf_int(self, level: float = 0.95) -> tuple[float, float]:
        """Return a positive Wald interval for the outcome scale."""
        critical = _critical_value(level)
        se = float(self.standard_errors.loc["log_sigma"])
        return (
            float(np.exp(self.log_sigma - critical * se)),
            float(np.exp(self.log_sigma + critical * se)),
        )

    def rho_conf_int(self, level: float = 0.95) -> tuple[float, float]:
        """Return a Wald interval transformed to the correlation scale."""
        critical = _critical_value(level)
        se = float(self.standard_errors.loc["atanh_rho"])
        return (
            float(np.tanh(self.atanh_rho - critical * se)),
            float(np.tanh(self.atanh_rho + critical * se)),
        )

    def predict_selection(self, Z: Any) -> pd.Series:
        """Predict the probability that the outcome is observed."""
        design, _ = _as_2d_array(Z)
        if design.shape[1] != len(self.feature_names_z):
            raise ValueError(
                f"Z must contain {len(self.feature_names_z)} regressors; "
                f"received {design.shape[1]}."
            )
        probability = ndtr(design @ self.params_selection.to_numpy(dtype=float))
        index = Z.index if isinstance(Z, pd.DataFrame) else None
        return pd.Series(probability, index=index, name="selection_probability")

    def predict(self, X: Any) -> pd.Series:
        """Predict the structural (unconditional) outcome mean ``X beta``."""
        design, _ = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names_x):
            raise ValueError(
                f"X must contain {len(self.feature_names_x)} regressors; "
                f"received {design.shape[1]}."
            )
        mean = design @ self.params_outcome.to_numpy(dtype=float)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.Series(mean, index=index, name="predicted")

    def predict_observed(self, X: Any, Z: Any) -> pd.Series:
        """Predict ``E[y | selected=1, X, Z]`` using the inverse Mills ratio."""
        outcome_mean = self.predict(X)
        selection_design, _ = _as_2d_array(Z)
        if selection_design.shape[0] != len(outcome_mean):
            raise ValueError("X and Z must contain the same number of observations.")
        if selection_design.shape[1] != len(self.feature_names_z):
            raise ValueError(
                f"Z must contain {len(self.feature_names_z)} regressors; "
                f"received {selection_design.shape[1]}."
            )
        selection_index = selection_design @ self.params_selection.to_numpy(dtype=float)
        mills = np.exp(np.clip(norm.logpdf(selection_index) - log_ndtr(selection_index), -745, 709))
        return (outcome_mean + self.rho * self.sigma * mills).rename(
            "predicted_observed"
        )


def _critical_value(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one.")
    return float(norm.ppf(0.5 + level / 2.0))


class SampleSelection:
    """Gaussian Heckman sample-selection model estimated by full MLE.

    The outcome equation is ``y = X beta + epsilon`` and the outcome is
    observed when ``Z gamma + u > 0``. The disturbances are jointly normal,
    ``sd(epsilon)=sigma``, ``sd(u)=1``, and ``corr(epsilon, u)=rho``.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        Z: Any,
        selection: Any | None = None,
        *,
        maxiter: int = 500,
    ) -> SampleSelectionResult:
        """Fit the model on the full selected and unselected sample.

        If ``selection`` is omitted, finite/non-missing values of ``y`` define
        selected observations. When it is supplied, ``y`` is only read where
        ``selection == 1`` and may be missing elsewhere.
        """
        X_design, X_names = _as_2d_array(X)
        Z_design, Z_names = _as_2d_array(Z)
        y_values = np.asarray(y)
        if y_values.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if X_design.shape[0] != y_values.size:
            raise ValueError("X and y must have the same number of observations.")
        if Z_design.shape[0] != y_values.size:
            raise ValueError("Z and y must have the same number of observations.")

        if selection is None:
            selected = ~pd.isna(y_values)
        else:
            selection_values = np.asarray(selection)
            if selection_values.ndim != 1 or selection_values.size != y_values.size:
                raise ValueError("selection must be one-dimensional and match y.")
            if pd.isna(selection_values).any():
                raise ValueError("selection contains missing values.")
            selection_float = np.asarray(selection_values, dtype=float)
            if not np.isfinite(selection_float).all() or not np.isin(
                selection_float, [0.0, 1.0]
            ).all():
                raise ValueError("selection must be binary (0 or 1).")
            selected = selection_float.astype(bool)

        if selected.all() or not selected.any():
            raise ValueError("Both selected and unselected observations are required.")
        selected_y = np.asarray(y_values[selected], dtype=float)
        if not np.isfinite(selected_y).all():
            raise ValueError("Selected outcomes must be finite and non-missing.")
        if np.linalg.matrix_rank(X_design[selected]) < X_design.shape[1]:
            raise ValueError("Selected-sample X must have full column rank.")
        if np.linalg.matrix_rank(Z_design) < Z_design.shape[1]:
            raise ValueError("Z must have full column rank.")
        n_parameters = X_design.shape[1] + Z_design.shape[1] + 2
        if y_values.size <= n_parameters or selected.sum() <= X_design.shape[1]:
            raise ValueError("There are too few observations to identify the model.")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        n_x = X_design.shape[1]
        n_z = Z_design.shape[1]
        not_selected = ~selected

        def negative_loglike(parameters: np.ndarray) -> float:
            outcome_beta = parameters[:n_x]
            selection_beta = parameters[n_x : n_x + n_z]
            log_sigma = parameters[n_x + n_z]
            rho_raw = parameters[n_x + n_z + 1]
            sigma = np.exp(log_sigma)
            rho = np.tanh(rho_raw)
            conditional_scale = np.sqrt(1.0 - rho**2)

            residual = (
                selected_y - X_design[selected] @ outcome_beta
            ) / sigma
            selection_index = Z_design @ selection_beta
            selected_index = (
                selection_index[selected] + rho * residual
            ) / conditional_scale
            selected_loglike = (
                norm.logpdf(residual)
                - log_sigma
                + log_ndtr(selected_index)
            )
            unselected_loglike = log_ndtr(-selection_index[not_selected])
            value = -float(
                np.sum(selected_loglike) + np.sum(unselected_loglike)
            )
            return value if np.isfinite(value) else 1e300

        outcome_initial = np.linalg.lstsq(
            X_design[selected], selected_y, rcond=None
        )[0]
        outcome_residual = selected_y - X_design[selected] @ outcome_initial
        sigma_initial = max(float(np.sqrt(np.mean(outcome_residual**2))), 1e-3)

        def selection_negative_loglike(gamma: np.ndarray) -> float:
            index = Z_design @ gamma
            return -float(
                np.sum(log_ndtr(index[selected]))
                + np.sum(log_ndtr(-index[not_selected]))
            )

        selection_fit = minimize(
            selection_negative_loglike,
            np.zeros(n_z),
            method="BFGS",
            options={"maxiter": maxiter},
        )
        selection_initial = np.asarray(selection_fit.x, dtype=float)
        initial = np.concatenate(
            [
                outcome_initial,
                selection_initial,
                [np.log(sigma_initial), 0.0],
            ]
        )

        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * (n_x + n_z) + [(-10.0, 10.0), (-5.0, 5.0)],
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        outcome_beta = parameters[:n_x]
        selection_beta = parameters[n_x : n_x + n_z]
        sigma = float(np.exp(parameters[n_x + n_z]))
        rho = float(np.tanh(parameters[n_x + n_z + 1]))

        hessian = _numerical_hessian(negative_loglike, parameters)
        hessian = 0.5 * (hessian + hessian.T)
        inference_valid = bool(
            optimizer_result.success
            and -10.0 < parameters[n_x + n_z] < 10.0
            and -5.0 < parameters[n_x + n_z + 1] < 5.0
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

        labels = (
            [f"outcome:{name}" for name in X_names]
            + [f"selection:{name}" for name in Z_names]
            + ["log_sigma", "atanh_rho"]
        )
        return SampleSelectionResult(
            params_outcome=pd.Series(outcome_beta, index=X_names, name="outcome"),
            params_selection=pd.Series(
                selection_beta, index=Z_names, name="selection"
            ),
            sigma=sigma,
            rho=rho,
            covariance=pd.DataFrame(covariance, index=labels, columns=labels),
            standard_errors=pd.Series(
                standard_errors, index=labels, name="std_err"
            ),
            zstats=pd.Series(zstats, index=labels, name="z"),
            pvalues=pd.Series(pvalues, index=labels, name="p_value"),
            inference_valid=inference_valid,
            converged=bool(optimizer_result.success),
            loglike=-float(optimizer_result.fun),
            nobs_observed=int(selected.sum()),
            nobs_total=int(y_values.size),
            feature_names_x=tuple(X_names),
            feature_names_z=tuple(Z_names),
            optimizer_result=optimizer_result,
        )
