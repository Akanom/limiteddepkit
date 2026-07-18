"""Stable count-data Poisson regression estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy
from scipy.stats import norm

from ._count_common import (
    combined_offset,
    covariance_from_scores,
    scaled_score_norm,
    validate_count_design,
    validate_count_response,
    validate_covariance,
    validate_optimizer_options,
    validate_prediction_design,
    validate_weights,
)
from ._irls import damped_newton


@dataclass(frozen=True)
class PoissonResult:
    """Fitted Poisson maximum-likelihood or weighted estimating-equation result."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    converged: bool
    loglike: float
    nobs: int
    weighted_nobs: float
    feature_names: tuple[str, ...]
    cov_type: str
    weight_type: str
    n_clusters: int | None
    score_norm: float
    scaled_score_norm: float
    information_condition: float
    pearson_chi2: float
    deviance: float
    fitted_values: pd.Series
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        """Return the complete labelled parameter vector."""
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def df_resid(self) -> float:
        sample_size = self.weighted_nobs if self.weight_type == "frequency" else self.nobs
        return float(sample_size - self.n_params)

    @property
    def aic(self) -> float:
        """Return AIC, or NaN when analytic weights define a pseudo-likelihood."""
        if self.weight_type == "analytic":
            return np.nan
        return -2.0 * self.loglike + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        """Return likelihood BIC, or NaN for an analytic-weight pseudo-likelihood."""
        if self.weight_type == "analytic":
            return np.nan
        sample_size = self.weighted_nobs if self.weight_type == "frequency" else self.nobs
        return -2.0 * self.loglike + np.log(sample_size) * self.n_params

    @property
    def covariance_type(self) -> str:
        return self.cov_type

    @property
    def backend(self) -> str:
        return "native-mle"

    def vcov(self) -> pd.DataFrame:
        """Return a defensive covariance copy."""
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        """Return normal-approximation confidence intervals."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return the standard package coefficient table."""
        from .postestimation import summary_frame

        return summary_frame(self)

    def diagnostics(self) -> pd.Series:
        """Return compact convergence, fit, and covariance diagnostics."""
        return pd.Series(
            {
                "converged": self.converged,
                "inference_valid": self.inference_valid,
                "score_norm": self.score_norm,
                "scaled_score_norm": self.scaled_score_norm,
                "information_condition": self.information_condition,
                "pearson_chi2": self.pearson_chi2,
                "deviance": self.deviance,
                "covariance_type": self.covariance_type,
                "weight_type": self.weight_type,
                "weighted_nobs": self.weighted_nobs,
                "n_clusters": self.n_clusters,
            },
            name="diagnostic",
        )

    def predict(
        self,
        X: Any,
        *,
        offset: Any | None = None,
        exposure: Any | None = None,
    ) -> pd.Series:
        """Return expected counts, preserving and validating the new-data schema."""
        design, index = validate_prediction_design(X, self.feature_names)
        prediction_offset = combined_offset(
            offset=offset,
            exposure=exposure,
            nobs=design.shape[0],
            index=index,
        )
        rates = np.exp(
            design @ self.params.to_numpy(dtype=float) + prediction_offset
        )
        return pd.Series(rates, index=index, name="predicted")


class PoissonRegressor:
    """Poisson regression with offsets, exposure, weights, and robust covariance.

    ``freq_weights`` have exact integer row-replication semantics. In contrast,
    ``analytic_weights`` scale the estimating equations; AIC and BIC are not
    defined for that pseudo-likelihood and are returned as ``NaN``. Robust
    covariance choices are ``"HC0"``, ``"HC1"``, and ``"cluster"``.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        offset: Any | None = None,
        exposure: Any | None = None,
        freq_weights: Any | None = None,
        analytic_weights: Any | None = None,
        cov_type: str = "nonrobust",
        clusters: Any | None = None,
        use_correction: bool = True,
        maxiter: int = 300,
        tolerance: float = 1e-8,
    ) -> PoissonResult:
        """Estimate a Poisson mean model."""
        validate_optimizer_options(maxiter, tolerance)
        if not isinstance(use_correction, (bool, np.bool_)):
            raise TypeError("use_correction must be boolean.")

        design, feature_names, index = validate_count_design(X)
        counts = validate_count_response(y, design.shape[0], index)
        model_offset = combined_offset(
            offset=offset,
            exposure=exposure,
            nobs=design.shape[0],
            index=index,
        )
        weights, weight_type, weighted_nobs, correction_nobs = validate_weights(
            freq_weights=freq_weights,
            analytic_weights=analytic_weights,
            nobs=design.shape[0],
            index=index,
            n_params=design.shape[1],
        )
        active = weights > 0.0
        if not np.any(active & (counts > 0.0)):
            raise ValueError("At least one positive count must have positive weight.")
        if np.linalg.matrix_rank(design[active]) < design.shape[1]:
            raise ValueError("X must have full column rank among positive-weight rows.")
        canonical_cov_type, cluster_array, n_clusters = validate_covariance(
            cov_type=cov_type,
            clusters=clusters,
            nobs=design.shape[0],
            index=index,
            active=active,
        )

        def negative_loglike(beta: np.ndarray) -> float:
            eta = design[active] @ beta + model_offset[active]
            if np.max(eta) > 709.0:
                return 1e300
            mean = np.exp(eta)
            contributions = (
                counts[active] * eta
                - mean
                - gammaln(counts[active] + 1.0)
            )
            value = -float(weights[active] @ contributions)
            return value if np.isfinite(value) else 1e300

        def gradient(beta: np.ndarray) -> np.ndarray:
            eta = design[active] @ beta + model_offset[active]
            if np.max(eta) > 709.0:
                return np.full(beta.shape, 1e300)
            return design[active].T @ (
                weights[active] * (np.exp(eta) - counts[active])
            )

        def information_at(beta: np.ndarray) -> np.ndarray:
            eta = design[active] @ beta + model_offset[active]
            if np.max(eta) > 709.0:
                return np.full((beta.size, beta.size), np.nan)
            mean = np.exp(eta)
            return design[active].T @ (
                (weights[active] * mean)[:, None] * design[active]
            )

        initial = np.linalg.lstsq(
            design[active],
            np.log(counts[active] + 0.1) - model_offset[active],
            rcond=None,
        )[0]
        optimizer_result = damped_newton(
            negative_loglike,
            gradient,
            information_at,
            initial,
            maxiter=int(maxiter),
            tolerance=float(tolerance),
        )
        if not optimizer_result.success:
            optimizer_result = minimize(
                negative_loglike,
                np.asarray(optimizer_result.x, dtype=float),
                method="BFGS",
                jac=gradient,
                options={"maxiter": int(maxiter), "gtol": float(tolerance)},
            )

        coefficients = np.asarray(optimizer_result.x, dtype=float)
        linear_index = design @ coefficients + model_offset
        with np.errstate(over="ignore"):
            fitted_mean = np.exp(linear_index)
        active_mean = fitted_mean[active]
        information = design[active].T @ (
            (weights[active] * active_mean)[:, None] * design[active]
        )
        score_rows = np.zeros_like(design)
        score_rows[active] = design[active] * (
            counts[active] - active_mean
        )[:, None]
        score_norm = float(np.max(np.abs(score_rows.T @ weights)))
        relative_score_norm = scaled_score_norm(
            score_rows,
            weights,
            weight_type=weight_type,
        )
        convergence_threshold = max(
            min(100.0 * float(tolerance), 1e-4),
            1e-5,
        )
        converged = bool(
            np.isfinite(optimizer_result.fun)
            and np.isfinite(coefficients).all()
            and (
                relative_score_norm <= convergence_threshold
            )
        )
        information_valid = bool(
            converged
            and np.isfinite(information).all()
            and np.linalg.eigvalsh(information).min() > 0.0
        )
        information_condition = (
            float(np.linalg.cond(information)) if information_valid else np.inf
        )
        if information_valid:
            covariance = covariance_from_scores(
                information,
                score_rows,
                weights=weights,
                weight_type=weight_type,
                cov_type=canonical_cov_type,
                clusters=cluster_array,
                correction_nobs=correction_nobs,
                n_params=design.shape[1],
                use_correction=bool(use_correction),
            )
            inference_valid = bool(
                np.isfinite(covariance).all() and np.all(np.diag(covariance) >= 0.0)
            )
        else:
            covariance = np.full_like(information, np.nan)
            inference_valid = False

        if inference_valid:
            standard_errors = np.sqrt(np.diag(covariance))
            zstats = coefficients / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full_like(information, np.nan)
            standard_errors = np.full(coefficients.shape, np.nan)
            zstats = np.full(coefficients.shape, np.nan)
            pvalues = np.full(coefficients.shape, np.nan)

        params = pd.Series(coefficients, index=feature_names, name="estimate")
        covariance_frame = pd.DataFrame(
            covariance, index=feature_names, columns=feature_names
        )
        standard_errors_series = pd.Series(
            standard_errors, index=feature_names, name="std_err"
        )
        zstats_series = pd.Series(zstats, index=feature_names, name="z")
        pvalues_series = pd.Series(pvalues, index=feature_names, name="p_value")
        loglike = -float(negative_loglike(coefficients))
        pearson_chi2 = float(
            np.sum(
                weights[active]
                * (counts[active] - active_mean) ** 2
                / active_mean
            )
        )
        deviance_terms = np.where(
            counts[active] > 0.0,
            xlogy(counts[active], counts[active] / active_mean)
            - (counts[active] - active_mean),
            active_mean,
        )
        deviance = float(2.0 * weights[active] @ deviance_terms)

        return PoissonResult(
            params=params,
            covariance=covariance_frame,
            standard_errors=standard_errors_series,
            zstats=zstats_series,
            pvalues=pvalues_series,
            inference_valid=inference_valid,
            converged=converged,
            loglike=loglike,
            nobs=int(design.shape[0]),
            weighted_nobs=weighted_nobs,
            feature_names=feature_names,
            cov_type=canonical_cov_type,
            weight_type=weight_type,
            n_clusters=n_clusters,
            score_norm=score_norm,
            scaled_score_norm=relative_score_norm,
            information_condition=information_condition,
            pearson_chi2=pearson_chi2,
            deviance=deviance,
            fitted_values=pd.Series(fitted_mean, index=index, name="fitted_mean"),
            optimizer_result=optimizer_result,
        )
