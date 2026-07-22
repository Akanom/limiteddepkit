"""Validated binary Probit maximum-likelihood estimator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

from ._irls import damped_newton
from .binary import (
    _binary_ame_inference,
    _binary_margins,
    _has_separation,
    _invert_information,
    _validate_fit_data,
    _validate_prediction_data,
)


def _inverse_mills_ratio(values: np.ndarray) -> np.ndarray:
    """Compute ``phi(values) / Phi(values)`` without taking a CDF ratio."""
    return np.exp(norm.logpdf(values) - log_ndtr(values))


@dataclass(frozen=True)
class BinaryProbitResult:
    """Fitted binary Probit result."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    constant_features: tuple[str, ...]
    score_norm: float
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return len(self.params)

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
    def scaled_score_norm(self) -> float:
        return self.score_norm / max(1, self.nobs)

    @property
    def covariance_type(self) -> str:
        return "observed-information"

    @property
    def backend(self) -> str:
        return "native-mle"

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

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def predict_proba(self, X: Any) -> pd.DataFrame:
        design, index = _validate_prediction_data(X, self.feature_names)
        probabilities = norm.cdf(design @ self.params.to_numpy(dtype=float))
        return pd.DataFrame({0: 1.0 - probabilities, 1: probabilities}, index=index)

    def predict(self, X: Any, *, threshold: float = 0.5) -> pd.Series:
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be finite and strictly between zero and one.")
        probabilities = self.predict_proba(X)[1]
        return (probabilities >= threshold).astype(int).rename("prediction")

    def marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return derivatives of ``P(y=1)`` with respect to continuous regressors."""
        design, index = _validate_prediction_data(X, self.feature_names)
        density = norm.pdf(design @ self.params.to_numpy(dtype=float))
        effect_features = [
            feature for feature in self.feature_names if feature not in self.constant_features
        ]
        effect_indices = [self.feature_names.index(feature) for feature in effect_features]
        effects = density[:, None] * self.params.to_numpy(dtype=float)[None, effect_indices]
        return pd.DataFrame(effects, index=index, columns=effect_features)

    def average_marginal_effects(self, X: Any) -> pd.Series:
        return self.marginal_effects(X).mean(axis=0).rename("estimate")

    def average_marginal_effects_inference(self, X: Any, *, level: float = 0.95) -> pd.DataFrame:
        """Return delta-method inference for average marginal effects."""
        return _binary_ame_inference(self, X, scale_function=norm.pdf, level=level)

    def margins(
        self,
        X: Any,
        *,
        at: str | Mapping[str, float] = "overall",
        kind: str = "probability",
    ) -> pd.Series:
        """Evaluate average or representative binary probabilities/effects."""
        return _binary_margins(self, X, at=at, kind=kind)

    def lincom(
        self,
        weights: Mapping[str, float],
        *,
        value: float = 0.0,
        level: float = 0.95,
    ) -> pd.Series:
        from .postestimation import lincom

        return lincom(self, weights, value=value, level=level)

    def wald_test(
        self,
        restrictions: Mapping[str, float] | Sequence[Mapping[str, float]],
        *,
        values: float | Sequence[float] = 0.0,
    ) -> pd.Series:
        from .postestimation import wald_test

        return wald_test(self, restrictions, values=values)


class BinaryProbit:
    """Binary Probit estimated by unpenalized maximum likelihood."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> BinaryProbitResult:
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        design, feature_names, outcomes = _validate_fit_data(X, y)
        if _has_separation(design, outcomes):
            raise ValueError(
                "The data exhibit complete or quasi-complete separation; a finite "
                "unpenalized maximum-likelihood estimate does not exist."
            )
        signs = 2.0 * outcomes - 1.0

        def negative_loglike(beta: np.ndarray) -> float:
            signed_indices = signs * (design @ beta)
            return float(-np.sum(log_ndtr(signed_indices)))

        def gradient(beta: np.ndarray) -> np.ndarray:
            signed_indices = signs * (design @ beta)
            return -(design.T @ (signs * _inverse_mills_ratio(signed_indices)))

        def information_at(beta: np.ndarray) -> np.ndarray:
            signed_indices = signs * (design @ beta)
            inverse_mills = _inverse_mills_ratio(signed_indices)
            weights = inverse_mills * (signed_indices + inverse_mills)
            return design.T @ (weights[:, None] * design)

        optimizer_result = damped_newton(
            negative_loglike,
            gradient,
            information_at,
            np.zeros(design.shape[1], dtype=float),
            maxiter=int(maxiter),
            tolerance=float(min(tolerance, 1e-7)),
        )
        if not optimizer_result.success:
            optimizer_result = minimize(
                negative_loglike,
                np.asarray(optimizer_result.x, dtype=float),
                jac=gradient,
                method="BFGS",
                options={"maxiter": int(maxiter), "gtol": min(tolerance, 1e-7)},
            )
        score_norm = float(np.max(np.abs(gradient(optimizer_result.x))))
        stationarity_limit = max(min(10.0 * tolerance, 1e-6), 1e-7)
        converged = bool(np.isfinite(score_norm) and score_norm <= stationarity_limit)
        if (
            not converged
            or not np.isfinite(optimizer_result.fun)
            or not np.isfinite(optimizer_result.x).all()
        ):
            raise RuntimeError(
                "Binary Probit optimization failed: " + str(optimizer_result.message)
            )

        coefficients = np.asarray(optimizer_result.x, dtype=float)
        information = information_at(coefficients)
        covariance = _invert_information(information)
        standard_errors = np.sqrt(np.diag(covariance))
        zstats = coefficients / standard_errors
        pvalues = 2.0 * norm.sf(np.abs(zstats))

        params = pd.Series(coefficients, index=feature_names, name="estimate")
        covariance_frame = pd.DataFrame(covariance, index=feature_names, columns=feature_names)
        standard_errors_series = pd.Series(standard_errors, index=feature_names, name="std_err")
        zstats_series = pd.Series(zstats, index=feature_names, name="z")
        pvalues_series = pd.Series(pvalues, index=feature_names, name="p_value")

        return BinaryProbitResult(
            params=params,
            covariance=covariance_frame,
            standard_errors=standard_errors_series,
            zstats=zstats_series,
            pvalues=pvalues_series,
            converged=converged,
            loglike=-float(optimizer_result.fun),
            nobs=int(design.shape[0]),
            feature_names=tuple(feature_names),
            constant_features=tuple(
                feature_names[index]
                for index in range(design.shape[1])
                if np.ptp(design[:, index]) <= 1e-12
            ),
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
