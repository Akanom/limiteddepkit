"""Experimental zero-inflated Poisson maximum-likelihood estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_expit
from scipy.stats import norm, poisson

from .binary import _invert_information
from .ordinal import _as_2d_array


def _validate_design(
    X: Any,
    *,
    label: str,
    nobs: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    design, names = _as_2d_array(X)
    if nobs is not None and design.shape[0] != nobs:
        raise ValueError(f"{label} must contain the same number of observations as y.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} feature names must be unique after conversion to strings.")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(f"{label} is rank deficient; its parameters are not identified.")
    return design, names


def _validate_counts(
    y: Any,
    *,
    require_zero_and_positive: bool = True,
    model_label: str = "Zero-inflated",
) -> np.ndarray:
    raw = np.asarray(y)
    if raw.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    try:
        counts = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("y must contain numeric counts.") from error
    if not np.isfinite(counts).all():
        raise ValueError("y contains missing or non-finite values.")
    if np.any(counts < 0.0) or np.any(counts != np.floor(counts)):
        raise ValueError("y must contain non-negative integer counts.")
    if require_zero_and_positive and (not np.any(counts == 0.0) or not np.any(counts > 0.0)):
        raise ValueError(f"{model_label} estimation requires both zero and positive counts.")
    return counts


def _prediction_design(
    X: Any,
    feature_names: tuple[str, ...],
    *,
    label: str,
) -> tuple[np.ndarray, pd.Index]:
    design, names = _as_2d_array(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(f"{label} has {design.shape[1]} columns; expected {len(feature_names)}.")
    if isinstance(X, pd.DataFrame) and tuple(names) != feature_names:
        raise ValueError(f"{label} columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(design.shape[0])
    return design, index


def _validate_optimization_options(maxiter: int, tolerance: float) -> None:
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")


@dataclass(frozen=True)
class ZeroInflatedPoissonResult:
    """Fitted experimental Logit-inflated Poisson result."""

    params_inflation: pd.Series
    params_poisson: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    inference_valid: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    inflation_feature_names: tuple[str, ...]
    score_norm: float
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        inflation = self.params_inflation.copy()
        inflation.index = [f"inflation: {name}" for name in inflation.index]
        count = self.params_poisson.copy()
        count.index = [f"count: {name}" for name in count.index]
        return pd.concat([inflation, count]).rename("estimate")

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
        X_inflation: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        count_design, index = _prediction_design(X, self.feature_names, label="X")
        inflation_source = X if X_inflation is None else X_inflation
        inflation_design, inflation_index = _prediction_design(
            inflation_source,
            self.inflation_feature_names,
            label="X_inflation",
        )
        if inflation_design.shape[0] != count_design.shape[0]:
            raise ValueError("X and X_inflation must contain the same prediction rows.")
        if (
            isinstance(X, pd.DataFrame)
            and isinstance(inflation_source, pd.DataFrame)
            and not inflation_index.equals(index)
        ):
            raise ValueError("X and X_inflation DataFrame indices must match.")
        count_mean = np.exp(count_design @ self.params_poisson.to_numpy(dtype=float))
        inflation_probability = expit(
            inflation_design @ self.params_inflation.to_numpy(dtype=float)
        )
        return count_mean, inflation_probability, index

    def predict_count_mean(self, X: Any) -> pd.Series:
        design, index = _prediction_design(X, self.feature_names, label="X")
        mean = np.exp(design @ self.params_poisson.to_numpy(dtype=float))
        return pd.Series(mean, index=index, name="count_mean")

    def predict_inflation_probability(self, X_inflation: Any) -> pd.Series:
        design, index = _prediction_design(
            X_inflation,
            self.inflation_feature_names,
            label="X_inflation",
        )
        probability = expit(design @ self.params_inflation.to_numpy(dtype=float))
        return pd.Series(probability, index=index, name="inflation_probability")

    def predict_zero_probability(
        self,
        X: Any,
        *,
        X_inflation: Any | None = None,
    ) -> pd.Series:
        mean, inflation, index = self._prediction_components(X, X_inflation)
        probability = inflation + (1.0 - inflation) * np.exp(-mean)
        return pd.Series(probability, index=index, name="zero_probability")

    def predict(
        self,
        X: Any,
        *,
        X_inflation: Any | None = None,
    ) -> pd.Series:
        """Return the unconditional expected count ``(1 - pi) * mu``."""
        mean, inflation, index = self._prediction_components(X, X_inflation)
        return pd.Series((1.0 - inflation) * mean, index=index, name="prediction")

    def predict_pmf(
        self,
        X: Any,
        *,
        max_count: int,
        X_inflation: Any | None = None,
    ) -> pd.DataFrame:
        if (
            isinstance(max_count, bool)
            or not isinstance(max_count, (int, np.integer))
            or max_count < 0
        ):
            raise ValueError("max_count must be a non-negative integer.")
        mean, inflation, index = self._prediction_components(X, X_inflation)
        values = np.arange(int(max_count) + 1)
        probabilities = (1.0 - inflation[:, None]) * poisson.pmf(
            values[None, :], mean[:, None]
        )
        probabilities[:, 0] += inflation
        return pd.DataFrame(probabilities, index=index, columns=values)


class ZeroInflatedPoisson:
    """Experimental Logit-inflated Poisson estimated by maximum likelihood."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_inflation: Any | None = None,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> ZeroInflatedPoissonResult:
        _validate_optimization_options(maxiter, tolerance)
        counts = _validate_counts(y)
        count_design, count_names = _validate_design(X, label="X", nobs=counts.size)
        inflation_source = X if X_inflation is None else X_inflation
        inflation_design, inflation_names = _validate_design(
            inflation_source,
            label="X_inflation",
            nobs=counts.size,
        )
        n_inflation = inflation_design.shape[1]
        n_count = count_design.shape[1]
        if counts.size <= n_inflation + n_count:
            raise ValueError(
                "Zero-inflated inference requires more observations than total parameters."
            )
        zero = counts == 0.0

        def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return parameters[:n_inflation], parameters[n_inflation:]

        def negative_loglike(parameters: np.ndarray) -> float:
            inflation_beta, count_beta = unpack(parameters)
            inflation_index = inflation_design @ inflation_beta
            count_index = count_design @ count_beta
            with np.errstate(over="ignore", invalid="ignore"):
                mean = np.exp(count_index)
                contributions = np.empty(counts.size, dtype=float)
                contributions[zero] = np.logaddexp(
                    log_expit(inflation_index[zero]),
                    log_expit(-inflation_index[zero]) - mean[zero],
                )
                positive = ~zero
                contributions[positive] = (
                    log_expit(-inflation_index[positive])
                    + counts[positive] * count_index[positive]
                    - mean[positive]
                    - gammaln(counts[positive] + 1.0)
                )
            if not np.isfinite(contributions).all():
                return np.inf
            return float(-np.sum(contributions))

        def gradient(parameters: np.ndarray) -> np.ndarray:
            inflation_beta, count_beta = unpack(parameters)
            inflation_index = inflation_design @ inflation_beta
            count_index = count_design @ count_beta
            with np.errstate(over="ignore", invalid="ignore"):
                mean = np.exp(count_index)
                inflation_probability = expit(inflation_index)
                inflation_score = -inflation_probability
                count_score = counts - mean
                posterior_inflation = expit(inflation_index[zero] + mean[zero])
                inflation_score[zero] = (
                    posterior_inflation - inflation_probability[zero]
                )
                log_poisson_zero_weight = log_expit(
                    -(inflation_index[zero] + mean[zero])
                )
                count_score[zero] = -np.exp(
                    count_index[zero] + log_poisson_zero_weight
                )
                score = np.r_[
                    inflation_design.T @ inflation_score,
                    count_design.T @ count_score,
                ]
            return -score

        def poisson_objective(beta: np.ndarray) -> float:
            index = count_design @ beta
            with np.errstate(over="ignore", invalid="ignore"):
                value = np.sum(np.exp(index) - counts * index)
            return float(value) if np.isfinite(value) else np.inf

        def poisson_gradient(beta: np.ndarray) -> np.ndarray:
            with np.errstate(over="ignore", invalid="ignore"):
                return count_design.T @ (np.exp(count_design @ beta) - counts)

        poisson_start_result = minimize(
            poisson_objective,
            np.zeros(n_count),
            jac=poisson_gradient,
            method="BFGS",
            options={"maxiter": int(maxiter), "gtol": tolerance},
        )
        count_start = (
            np.asarray(poisson_start_result.x, dtype=float)
            if np.isfinite(poisson_start_result.x).all()
            else np.zeros(n_count)
        )
        mean_start = np.exp(count_design @ count_start)
        poisson_zero_share = float(np.mean(np.exp(-mean_start)))
        observed_zero_share = float(np.mean(zero))
        inflation_share = np.clip(
            (observed_zero_share - poisson_zero_share)
            / max(1.0 - poisson_zero_share, 1e-8),
            0.01,
            0.8,
        )
        inflation_start = np.zeros(n_inflation)
        constant_columns = [
            column
            for column in range(n_inflation)
            if np.ptp(inflation_design[:, column]) <= 1e-12
        ]
        if constant_columns:
            column = constant_columns[0]
            inflation_start[column] = (
                np.log(inflation_share / (1.0 - inflation_share))
                / inflation_design[0, column]
            )
        starts = [
            np.r_[inflation_start, count_start],
            np.r_[np.zeros(n_inflation), count_start],
            np.zeros(n_inflation + n_count),
        ]
        candidates = [
            minimize(
                negative_loglike,
                start,
                jac=gradient,
                method="BFGS",
                options={"maxiter": int(maxiter), "gtol": tolerance},
            )
            for start in starts
        ]
        finite_candidates = [
            candidate
            for candidate in candidates
            if np.isfinite(candidate.fun) and np.isfinite(candidate.x).all()
        ]
        if not finite_candidates:
            raise RuntimeError("Zero-inflated Poisson optimization produced no finite fit.")
        optimizer_result = min(finite_candidates, key=lambda candidate: candidate.fun)
        score_norm = float(np.max(np.abs(gradient(optimizer_result.x))))
        converged = bool(
            optimizer_result.success or score_norm <= max(10.0 * tolerance, 1e-7)
        )
        if not converged:
            raise RuntimeError(
                "Zero-inflated Poisson optimization failed: "
                + str(optimizer_result.message)
            )

        inflation_beta, count_beta = unpack(np.asarray(optimizer_result.x, dtype=float))
        inflation_index = inflation_design @ inflation_beta
        count_index = count_design @ count_beta
        mean = np.exp(count_index)
        inflation_probability = expit(inflation_index)
        posterior_inflation = expit(inflation_index[zero] + mean[zero])

        information_inflation_weights = inflation_probability * (
            1.0 - inflation_probability
        )
        information_count_weights = mean.copy()
        cross_weights = np.zeros(counts.size)
        information_inflation_weights[zero] -= posterior_inflation * (
            1.0 - posterior_inflation
        )
        log_poisson_zero_weight = log_expit(-(inflation_index[zero] + mean[zero]))
        poisson_weighted_mean = np.exp(count_index[zero] + log_poisson_zero_weight)
        posterior_mean_squared_weight = np.exp(
            log_expit(inflation_index[zero] + mean[zero])
            + 2.0 * count_index[zero]
            + log_poisson_zero_weight
        )
        information_count_weights[zero] = (
            poisson_weighted_mean - posterior_mean_squared_weight
        )
        cross_weights[zero] = -posterior_inflation * poisson_weighted_mean

        information_inflation = inflation_design.T @ (
            information_inflation_weights[:, None] * inflation_design
        )
        information_count = count_design.T @ (
            information_count_weights[:, None] * count_design
        )
        information_cross = inflation_design.T @ (
            cross_weights[:, None] * count_design
        )
        information = np.block(
            [
                [information_inflation, information_cross],
                [information_cross.T, information_count],
            ]
        )
        covariance = _invert_information(information)
        standard_errors = np.sqrt(np.diag(covariance))
        estimates = np.r_[inflation_beta, count_beta]
        zstats = estimates / standard_errors
        pvalues = 2.0 * norm.sf(np.abs(zstats))
        labels = [f"inflation: {name}" for name in inflation_names] + [
            f"count: {name}" for name in count_names
        ]

        return ZeroInflatedPoissonResult(
            params_inflation=pd.Series(
                inflation_beta, index=inflation_names, name="inflation"
            ),
            params_poisson=pd.Series(count_beta, index=count_names, name="count"),
            covariance=pd.DataFrame(covariance, index=labels, columns=labels),
            standard_errors=pd.Series(standard_errors, index=labels, name="std_err"),
            zstats=pd.Series(zstats, index=labels, name="z"),
            pvalues=pd.Series(pvalues, index=labels, name="p_value"),
            converged=converged,
            inference_valid=True,
            loglike=-float(optimizer_result.fun),
            nobs=int(counts.size),
            feature_names=tuple(count_names),
            inflation_feature_names=tuple(inflation_names),
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
