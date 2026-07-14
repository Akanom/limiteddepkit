"""Deprecated legacy 2SLS estimator.

Linear instrumental-variables estimation is outside limiteddepkit's
limited-dependent-variable scope. This module is retained temporarily as a
migration path and is deliberately absent from the public namespaces.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ordinal import _as_2d_array


@dataclass(frozen=True)
class TreatmentEffectResult:
    """Fitted treatment effect result."""

    params_endog: pd.Series
    params_exog: pd.Series
    sigma: float
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    nobs: int
    feature_names_endog: tuple[str, ...]
    feature_names_exog: tuple[str, ...]
    optimizer_result: Any

    def predict(self, X_endog: Any, X_exog: Any) -> pd.Series:
        design_endog, _ = _as_2d_array(X_endog)
        design_exog, _ = _as_2d_array(X_exog)
        if design_endog.shape[0] != design_exog.shape[0]:
            raise ValueError("X_endog and X_exog must have the same number of rows.")
        if design_endog.shape[1] != len(self.feature_names_endog):
            raise ValueError(
                f"X_endog must contain {len(self.feature_names_endog)} columns."
            )
        if design_exog.shape[1] != len(self.feature_names_exog):
            raise ValueError(
                f"X_exog must contain {len(self.feature_names_exog)} columns."
            )
        if isinstance(X_endog, pd.DataFrame) and tuple(X_endog.columns.astype(str)) != self.feature_names_endog:
            raise ValueError("X_endog columns must match the fitted schema and order.")
        if isinstance(X_exog, pd.DataFrame) and tuple(X_exog.columns.astype(str)) != self.feature_names_exog:
            raise ValueError("X_exog columns must match the fitted schema and order.")

        prediction = (
            design_endog @ self.params_endog.to_numpy(dtype=float)
            + design_exog @ self.params_exog.to_numpy(dtype=float)
        )

        index = X_endog.index if isinstance(X_endog, pd.DataFrame) else None
        return pd.Series(prediction, index=index, name="predicted")


class TreatmentEffect:
    """Deprecated homoskedastic two-stage least-squares estimator.

    Model: Y = T*beta1 + X*beta2 + error
    where T is endogenous, treated with instruments Z.
    """

    def __init__(self) -> None:
        warnings.warn(
            "TreatmentEffect is ordinary linear 2SLS and is outside limiteddepkit's "
            "scope; migrate this workflow to a causal/IV package.",
            FutureWarning,
            stacklevel=2,
        )

    def fit(
        self,
        y: Any,
        X_endog: Any,
        X_exog: Any,
        Z: Any,
    ) -> TreatmentEffectResult:
        """Fit 2SLS treatment effect model.

        Parameters
        ----------
        y : array-like
            Outcome variable
        X_endog : array-like
            Endogenous variables (e.g., treatment)
        X_exog : array-like
            Exogenous variables
        Z : array-like
            Instruments (superset of exogenous variables)
        """
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        y_vals = np.asarray(y_array, dtype=float)
        X_endog_design, X_endog_names = _as_2d_array(X_endog)
        X_exog_design, X_exog_names = _as_2d_array(X_exog)
        Z_design, _ = _as_2d_array(Z)

        if X_endog_design.shape[0] != y_vals.size:
            raise ValueError("X_endog and y must have same number of observations.")
        if X_exog_design.shape[0] != y_vals.size:
            raise ValueError("X_exog and y must have same number of observations.")
        if Z_design.shape[0] != y_vals.size:
            raise ValueError("Z and y must have same number of observations.")
        if not np.isfinite(y_vals).all():
            raise ValueError("y contains missing or non-finite values.")

        n_obs = y_vals.size
        structural_design = np.hstack([X_endog_design, X_exog_design])
        n_parameters = structural_design.shape[1]
        if n_obs <= n_parameters:
            raise ValueError("The number of observations must exceed the regressors.")
        if np.linalg.matrix_rank(structural_design) < n_parameters:
            raise ValueError("The structural design matrix must have full column rank.")
        if np.linalg.matrix_rank(Z_design) < Z_design.shape[1]:
            raise ValueError("The instrument matrix must have full column rank.")
        if Z_design.shape[1] < n_parameters:
            raise ValueError("The model is underidentified: too few instruments.")
        exogenous_projection = Z_design @ np.linalg.pinv(Z_design) @ X_exog_design
        if not np.allclose(exogenous_projection, X_exog_design, rtol=1e-8, atol=1e-8):
            raise ValueError("Z must span every column of X_exog.")

        projected_design = Z_design @ np.linalg.pinv(Z_design) @ structural_design
        normal_matrix = structural_design.T @ projected_design
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
        if np.linalg.matrix_rank(normal_matrix) < n_parameters:
            raise ValueError("The excluded instruments do not identify the endogenous regressors.")
        beta = np.linalg.solve(normal_matrix, projected_design.T @ y_vals)

        n_endog = X_endog_design.shape[1]
        beta_endog = beta[:n_endog]
        beta_exog = beta[n_endog:]

        # Structural residuals, not residuals from a regression on fitted endog.
        y_pred = structural_design @ beta
        residuals = y_vals - y_pred
        residual_var = float(residuals @ residuals / (n_obs - n_parameters))
        sigma = np.sqrt(residual_var)
        var_covar = residual_var * np.linalg.inv(normal_matrix)

        standard_errors = np.sqrt(np.maximum(np.diag(var_covar), 0.0))
        zstats = beta / standard_errors
        pvalues = 2.0 * norm.sf(np.abs(zstats))

        param_labels = [f"endog:{name}" for name in X_endog_names] + [
            f"exog:{name}" for name in X_exog_names
        ]
        params_endog = pd.Series(beta_endog, index=X_endog_names, name="coef")
        params_exog = pd.Series(beta_exog, index=X_exog_names, name="coef")
        covariance_frame = pd.DataFrame(var_covar, index=param_labels, columns=param_labels)
        standard_errors_series = pd.Series(standard_errors, index=param_labels, name="std_err")
        zstats_series = pd.Series(zstats, index=param_labels, name="z")
        pvalues_series = pd.Series(pvalues, index=param_labels, name="p_value")

        return TreatmentEffectResult(
            params_endog=params_endog,
            params_exog=params_exog,
            sigma=float(sigma),
            covariance=covariance_frame,
            standard_errors=standard_errors_series,
            zstats=zstats_series,
            pvalues=pvalues_series,
            converged=True,  # OLS always converges
            nobs=int(n_obs),
            feature_names_endog=tuple(X_endog_names),
            feature_names_exog=tuple(X_exog_names),
            optimizer_result=None,
        )
