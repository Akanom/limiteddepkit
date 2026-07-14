"""Experimental Logit-hurdle, zero-truncated Poisson estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_expit
from scipy.stats import norm, poisson

from .binary import _has_separation, _invert_information
from .zero_inflated_poisson import (
    _prediction_design,
    _validate_counts,
    _validate_design,
    _validate_optimization_options,
)


def _log_positive_poisson_probability(
    linear_index: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    """Return ``log(1 - exp(-mean))`` stably, including tiny means."""
    output = np.empty_like(mean)
    small = linear_index < -20.0
    output[small] = linear_index[small] + np.log1p(
        -0.5 * mean[small] + mean[small] ** 2 / 6.0
    )
    output[~small] = np.log(-np.expm1(-mean[~small]))
    return output


def _zero_truncated_mean(mean: np.ndarray) -> np.ndarray:
    """Mean of a Poisson distribution conditional on a positive count."""
    output = np.empty_like(mean)
    small = mean < 1e-5
    output[small] = 1.0 + mean[small] / 2.0 + mean[small] ** 2 / 12.0
    output[~small] = mean[~small] / (-np.expm1(-mean[~small]))
    return output


def _zero_truncated_variance(mean: np.ndarray) -> np.ndarray:
    """Variance of a Poisson distribution conditional on a positive count."""
    output = np.empty_like(mean)
    small = mean < 1e-5
    output[small] = mean[small] / 2.0 + mean[small] ** 2 / 6.0
    truncated_mean = _zero_truncated_mean(mean[~small])
    output[~small] = truncated_mean * (
        1.0 + mean[~small] - truncated_mean
    )
    return output


@dataclass(frozen=True)
class HurdlePoissonResult:
    """Fitted experimental Logit-hurdle Poisson result."""

    params_hurdle: pd.Series
    params_poisson: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    inference_valid: bool
    loglike: float
    nobs: int
    n_positive: int
    feature_names: tuple[str, ...]
    hurdle_feature_names: tuple[str, ...]
    score_norm: float
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        hurdle = self.params_hurdle.copy()
        hurdle.index = [f"hurdle: {name}" for name in hurdle.index]
        count = self.params_poisson.copy()
        count.index = [f"count: {name}" for name in count.index]
        return pd.concat([hurdle, count]).rename("estimate")

    @property
    def params(self) -> pd.Series:
        return self.all_params

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
    def covariance_type(self) -> str:
        return "observed-information"

    @property
    def backend(self) -> str:
        return "experimental-native-mle"

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.DataFrame(
            {
                "lower": self.all_params - critical * self.standard_errors,
                "upper": self.all_params + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def _prediction_components(
        self,
        X: Any,
        X_hurdle: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        count_design, index = _prediction_design(X, self.feature_names, label="X")
        hurdle_source = X if X_hurdle is None else X_hurdle
        hurdle_design, hurdle_index = _prediction_design(
            hurdle_source,
            self.hurdle_feature_names,
            label="X_hurdle",
        )
        if hurdle_design.shape[0] != count_design.shape[0]:
            raise ValueError("X and X_hurdle must contain the same prediction rows.")
        if (
            isinstance(X, pd.DataFrame)
            and isinstance(hurdle_source, pd.DataFrame)
            and not hurdle_index.equals(index)
        ):
            raise ValueError("X and X_hurdle DataFrame indices must match.")
        mean = np.exp(count_design @ self.params_poisson.to_numpy(dtype=float))
        positive_probability = expit(
            hurdle_design @ self.params_hurdle.to_numpy(dtype=float)
        )
        return mean, positive_probability, index

    def predict_positive_probability(self, X_hurdle: Any) -> pd.Series:
        design, index = _prediction_design(
            X_hurdle,
            self.hurdle_feature_names,
            label="X_hurdle",
        )
        probability = expit(design @ self.params_hurdle.to_numpy(dtype=float))
        return pd.Series(probability, index=index, name="positive_probability")

    def predict_positive_mean(self, X: Any) -> pd.Series:
        design, index = _prediction_design(X, self.feature_names, label="X")
        mean = np.exp(design @ self.params_poisson.to_numpy(dtype=float))
        return pd.Series(
            _zero_truncated_mean(mean),
            index=index,
            name="positive_mean",
        )

    def predict_zero_probability(
        self,
        X: Any,
        *,
        X_hurdle: Any | None = None,
    ) -> pd.Series:
        _, positive_probability, index = self._prediction_components(X, X_hurdle)
        return pd.Series(
            1.0 - positive_probability,
            index=index,
            name="zero_probability",
        )

    def predict(
        self,
        X: Any,
        *,
        X_hurdle: Any | None = None,
    ) -> pd.Series:
        """Return ``P(y>0) * E[y | y>0]``."""
        mean, positive_probability, index = self._prediction_components(X, X_hurdle)
        prediction = positive_probability * _zero_truncated_mean(mean)
        return pd.Series(prediction, index=index, name="prediction")

    def predict_pmf(
        self,
        X: Any,
        *,
        max_count: int,
        X_hurdle: Any | None = None,
    ) -> pd.DataFrame:
        if (
            isinstance(max_count, bool)
            or not isinstance(max_count, (int, np.integer))
            or max_count < 0
        ):
            raise ValueError("max_count must be a non-negative integer.")
        mean, positive_probability, index = self._prediction_components(X, X_hurdle)
        values = np.arange(int(max_count) + 1)
        probabilities = np.zeros((len(mean), len(values)), dtype=float)
        probabilities[:, 0] = 1.0 - positive_probability
        if max_count > 0:
            normalization = -np.expm1(-mean)
            probabilities[:, 1:] = positive_probability[:, None] * poisson.pmf(
                values[None, 1:], mean[:, None]
            ) / normalization[:, None]
        return pd.DataFrame(probabilities, index=index, columns=values)


class HurdlePoisson:
    """Experimental Logit hurdle with a zero-truncated Poisson count model."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_hurdle: Any | None = None,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> HurdlePoissonResult:
        _validate_optimization_options(maxiter, tolerance)
        counts = _validate_counts(y, model_label="Hurdle")
        count_design, count_names = _validate_design(X, label="X", nobs=counts.size)
        hurdle_source = X if X_hurdle is None else X_hurdle
        hurdle_design, hurdle_names = _validate_design(
            hurdle_source,
            label="X_hurdle",
            nobs=counts.size,
        )
        positive = counts > 0.0
        positive_outcome = positive.astype(float)
        if _has_separation(hurdle_design, positive_outcome):
            raise ValueError(
                "The hurdle equation exhibits complete or quasi-complete separation."
            )
        positive_design = count_design[positive]
        positive_counts = counts[positive]
        if positive_counts.size <= count_design.shape[1]:
            raise ValueError(
                "The positive-count equation requires more positive observations than regressors."
            )
        if np.linalg.matrix_rank(positive_design) < count_design.shape[1]:
            raise ValueError(
                "X is rank deficient within positive counts; count parameters are not identified."
            )
        if not np.any(positive_counts > 1.0):
            raise ValueError(
                "A zero-truncated Poisson mean is not finitely identified when every "
                "positive count equals one."
            )
        n_hurdle = hurdle_design.shape[1]
        n_count = count_design.shape[1]
        if counts.size <= n_hurdle + n_count:
            raise ValueError("Hurdle inference requires more observations than total parameters.")

        def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return parameters[:n_hurdle], parameters[n_hurdle:]

        def negative_loglike(parameters: np.ndarray) -> float:
            hurdle_beta, count_beta = unpack(parameters)
            hurdle_index = hurdle_design @ hurdle_beta
            count_index = positive_design @ count_beta
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                mean = np.exp(count_index)
                contributions = np.where(
                    positive,
                    log_expit(hurdle_index),
                    log_expit(-hurdle_index),
                )
                contributions[positive] += (
                    positive_counts * count_index
                    - mean
                    - gammaln(positive_counts + 1.0)
                    - _log_positive_poisson_probability(count_index, mean)
                )
            if not np.isfinite(contributions).all():
                return np.inf
            return float(-np.sum(contributions))

        def gradient(parameters: np.ndarray) -> np.ndarray:
            hurdle_beta, count_beta = unpack(parameters)
            hurdle_probability = expit(hurdle_design @ hurdle_beta)
            count_index = positive_design @ count_beta
            with np.errstate(over="ignore", invalid="ignore"):
                mean = np.exp(count_index)
                truncated_mean = _zero_truncated_mean(mean)
                hurdle_gradient = hurdle_design.T @ (
                    hurdle_probability - positive_outcome
                )
                count_gradient = positive_design.T @ (
                    truncated_mean - positive_counts
                )
            return np.r_[hurdle_gradient, count_gradient]

        initial = np.zeros(n_hurdle + n_count)
        optimizer_result = minimize(
            negative_loglike,
            initial,
            jac=gradient,
            method="BFGS",
            options={"maxiter": int(maxiter), "gtol": tolerance},
        )
        if not np.isfinite(optimizer_result.fun) or not np.isfinite(optimizer_result.x).all():
            raise RuntimeError("Hurdle Poisson optimization produced a non-finite fit.")
        score_norm = float(np.max(np.abs(gradient(optimizer_result.x))))
        converged = bool(
            optimizer_result.success or score_norm <= max(10.0 * tolerance, 1e-7)
        )
        if not converged:
            raise RuntimeError(
                "Hurdle Poisson optimization failed: " + str(optimizer_result.message)
            )

        hurdle_beta, count_beta = unpack(np.asarray(optimizer_result.x, dtype=float))
        hurdle_probability = expit(hurdle_design @ hurdle_beta)
        count_mean = np.exp(positive_design @ count_beta)
        hurdle_information = hurdle_design.T @ (
            (hurdle_probability * (1.0 - hurdle_probability))[:, None]
            * hurdle_design
        )
        count_information = positive_design.T @ (
            _zero_truncated_variance(count_mean)[:, None] * positive_design
        )
        covariance_hurdle = _invert_information(hurdle_information)
        covariance_count = _invert_information(count_information)
        covariance = block_diag(covariance_hurdle, covariance_count)
        standard_errors = np.sqrt(np.diag(covariance))
        estimates = np.r_[hurdle_beta, count_beta]
        zstats = estimates / standard_errors
        pvalues = 2.0 * norm.sf(np.abs(zstats))
        labels = [f"hurdle: {name}" for name in hurdle_names] + [
            f"count: {name}" for name in count_names
        ]

        return HurdlePoissonResult(
            params_hurdle=pd.Series(hurdle_beta, index=hurdle_names, name="hurdle"),
            params_poisson=pd.Series(count_beta, index=count_names, name="count"),
            covariance=pd.DataFrame(covariance, index=labels, columns=labels),
            standard_errors=pd.Series(standard_errors, index=labels, name="std_err"),
            zstats=pd.Series(zstats, index=labels, name="z"),
            pvalues=pd.Series(pvalues, index=labels, name="p_value"),
            converged=converged,
            inference_valid=True,
            loglike=-float(optimizer_result.fun),
            nobs=int(counts.size),
            n_positive=int(positive.sum()),
            feature_names=tuple(count_names),
            hurdle_feature_names=tuple(hurdle_names),
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
