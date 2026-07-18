"""Stable negative-binomial NB2 regression estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, xlogy
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
from .ordinal import _numerical_hessian


@dataclass(frozen=True)
class NegativeBinomialResult:
    """Fitted NB2 result with log-dispersion inference."""

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
        return pd.concat(
            [self.params.copy(), pd.Series({"log_alpha": self.log_alpha})]
        ).rename("estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

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
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        """Return Wald intervals for coefficients and log dispersion."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.DataFrame(
            {
                "lower": self.all_params - critical * self.standard_errors,
                "upper": self.all_params + critical * self.standard_errors,
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

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def diagnostics(self) -> pd.Series:
        """Return compact convergence, fit, dispersion, and covariance diagnostics."""
        return pd.Series(
            {
                "converged": self.converged,
                "inference_valid": self.inference_valid,
                "score_norm": self.score_norm,
                "scaled_score_norm": self.scaled_score_norm,
                "information_condition": self.information_condition,
                "alpha": self.alpha,
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


class NegativeBinomial:
    """NB2 regression with offsets, exposure, weights, and robust covariance.

    The conditional variance is ``mu + alpha * mu**2``. ``freq_weights``
    have exact integer row-replication semantics. ``analytic_weights`` scale
    the estimating equations, so AIC and BIC are deliberately undefined.
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
        maxiter: int = 500,
        tolerance: float = 1e-8,
    ) -> NegativeBinomialResult:
        """Estimate an NB2 mean and dispersion model."""
        validate_optimizer_options(maxiter, tolerance)
        if not isinstance(use_correction, (bool, np.bool_)):
            raise TypeError("use_correction must be boolean.")

        design, feature_names, index = validate_count_design(X)
        if "log_alpha" in feature_names:
            raise ValueError("X feature name 'log_alpha' is reserved for NB2 dispersion.")
        counts = validate_count_response(y, design.shape[0], index)
        model_offset = combined_offset(
            offset=offset,
            exposure=exposure,
            nobs=design.shape[0],
            index=index,
        )
        n_parameters = design.shape[1] + 1
        weights, weight_type, weighted_nobs, correction_nobs = validate_weights(
            freq_weights=freq_weights,
            analytic_weights=analytic_weights,
            nobs=design.shape[0],
            index=index,
            n_params=n_parameters,
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

        def score_rows(parameters: np.ndarray) -> np.ndarray:
            beta = parameters[:-1]
            log_alpha = parameters[-1]
            r = np.exp(-log_alpha)
            eta = design @ beta + model_offset
            log_mean_times_alpha = log_alpha + eta
            inverse_denominator = expit(-log_mean_times_alpha)
            q = expit(log_mean_times_alpha)
            scaled_residual = counts * inverse_denominator - r * q
            beta_score = design * scaled_residual[:, None]
            log_denominator = np.logaddexp(0.0, log_mean_times_alpha)
            dispersion_score = (
                -r * (digamma(counts + r) - digamma(r))
                + r * log_denominator
                - r * q
                + counts * inverse_denominator
            )
            output = np.column_stack([beta_score, dispersion_score])
            if weight_type == "frequency":
                output[~active] = 0.0
            return output

        def negative_loglike(parameters: np.ndarray) -> float:
            beta = parameters[:-1]
            log_alpha = parameters[-1]
            eta = design @ beta + model_offset
            r = np.exp(-log_alpha)
            log_denom = np.logaddexp(0.0, log_alpha + eta)
            loglik = (
                gammaln(counts + r)
                - gammaln(counts + 1.0)
                - gammaln(r)
                - r * log_denom
                + counts * (log_alpha + eta - log_denom)
            )
            value = -float(weights[active] @ loglik[active])
            return value if np.isfinite(value) else 1e300

        def gradient(parameters: np.ndarray) -> np.ndarray:
            with np.errstate(over="ignore", invalid="ignore"):
                score = score_rows(parameters)
            if not np.isfinite(score).all():
                return np.full(parameters.shape, 1e300)
            return -(score.T @ weights)

        initial = np.zeros(n_parameters, dtype=float)
        initial[:-1] = np.linalg.lstsq(
            design[active],
            np.log(counts[active] + 0.1) - model_offset[active],
            rcond=None,
        )[0]
        sample_mean = float(np.average(counts, weights=weights))
        sample_variance = float(np.average((counts - sample_mean) ** 2, weights=weights))
        moment_alpha = max((sample_variance - sample_mean) / sample_mean**2, 0.1)
        initial[-1] = np.log(moment_alpha)
        lower_log_alpha = -15.0
        upper_log_alpha = 10.0
        optimizer_result = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(None, None)] * design.shape[1]
            + [(lower_log_alpha, upper_log_alpha)],
            options={
                "maxiter": int(maxiter),
                "ftol": min(float(tolerance) * 1e-3, 1e-12),
                "gtol": float(tolerance),
            },
        )

        parameters = np.asarray(optimizer_result.x, dtype=float)
        beta = parameters[:-1]
        log_alpha = float(parameters[-1])
        alpha = float(np.exp(log_alpha))
        score = score_rows(parameters)
        score_norm = float(np.max(np.abs(score.T @ weights)))
        relative_score_norm = scaled_score_norm(
            score,
            weights,
            weight_type=weight_type,
        )
        convergence_threshold = max(
            min(100.0 * float(tolerance), 1e-4),
            1e-5,
        )
        converged = bool(
            np.isfinite(optimizer_result.fun)
            and np.isfinite(parameters).all()
            and relative_score_norm <= convergence_threshold
        )
        hessian = _numerical_hessian(negative_loglike, parameters)
        hessian = 0.5 * (hessian + hessian.T)
        information_valid = bool(
            converged
            and lower_log_alpha + 1e-7 < log_alpha < upper_log_alpha - 1e-7
            and np.isfinite(hessian).all()
            and np.linalg.eigvalsh(hessian).min() > 0.0
        )
        information_condition = float(np.linalg.cond(hessian)) if information_valid else np.inf
        if information_valid:
            covariance = covariance_from_scores(
                hessian,
                score,
                weights=weights,
                weight_type=weight_type,
                cov_type=canonical_cov_type,
                clusters=cluster_array,
                correction_nobs=correction_nobs,
                n_params=n_parameters,
                use_correction=bool(use_correction),
            )
            inference_valid = bool(
                np.isfinite(covariance).all() and np.all(np.diag(covariance) >= 0.0)
            )
        else:
            covariance = np.full_like(hessian, np.nan)
            inference_valid = False

        if inference_valid:
            standard_errors = np.sqrt(np.diag(covariance))
            zstats = parameters / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full_like(hessian, np.nan)
            standard_errors = np.full(parameters.shape, np.nan)
            zstats = np.full(parameters.shape, np.nan)
            pvalues = np.full(parameters.shape, np.nan)

        all_names = (*feature_names, "log_alpha")
        with np.errstate(over="ignore"):
            fitted_mean = np.exp(design @ beta + model_offset)
        active_mean = fitted_mean[active]
        variance = active_mean + alpha * active_mean**2
        pearson_chi2 = float(
            np.sum(
                weights[active]
                * (counts[active] - active_mean) ** 2
                / variance
            )
        )
        inverse_alpha = 1.0 / alpha
        deviance_terms = xlogy(
            counts[active],
            counts[active] / active_mean,
        ) - (counts[active] + inverse_alpha) * np.log(
            (counts[active] + inverse_alpha)
            / (active_mean + inverse_alpha)
        )
        deviance = float(2.0 * weights[active] @ deviance_terms)

        return NegativeBinomialResult(
            params=pd.Series(beta, index=feature_names, name="estimate"),
            log_alpha=log_alpha,
            alpha=alpha,
            covariance=pd.DataFrame(covariance, index=all_names, columns=all_names),
            standard_errors=pd.Series(standard_errors, index=all_names, name="std_err"),
            zstats=pd.Series(zstats, index=all_names, name="z"),
            pvalues=pd.Series(pvalues, index=all_names, name="p_value"),
            inference_valid=inference_valid,
            converged=converged,
            loglike=-float(negative_loglike(parameters)),
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


# Explicit family name while preserving the original public class.
NegativeBinomialNB2 = NegativeBinomial
