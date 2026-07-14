"""Experimental small-sample and separation-resistant response estimators.

These estimators deliberately remain separate from the ordinary maximum-
likelihood APIs. Penalized likelihood values and approximate covariance
estimands are labelled explicitly and should not be compared to ordinary MLE
information criteria as though they were the same estimand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, log_expit
from scipy.stats import norm

from ..ordinal import (
    _as_2d_array,
    _category_probabilities,
    _numerical_hessian,
    _ordered_categories,
    _threshold_jacobian,
    _unpack_thresholds,
)


def _validate_optimizer_options(maxiter: int, tolerance: float) -> tuple[int, float]:
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    return int(maxiter), float(tolerance)


def _validate_penalty(penalty: float) -> float:
    numeric = float(penalty)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("penalty must be finite and strictly positive.")
    return numeric


def _constant_feature_names(design: np.ndarray, feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        feature_names[index]
        for index in range(design.shape[1])
        if np.ptp(design[:, index]) <= 1e-12
    )


def _validate_binary_fit_data(X: Any, y: Any) -> tuple[np.ndarray, list[str], np.ndarray]:
    design, feature_names = _as_2d_array(X)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("X feature names must be unique after conversion to strings.")
    outcomes = np.asarray(y)
    if outcomes.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if outcomes.size != design.shape[0]:
        raise ValueError("X and y must contain the same number of observations.")
    if pd.isna(outcomes).any():
        raise ValueError("y contains missing values.")
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("y must contain binary outcomes coded as 0/1.")
    outcomes = outcomes.astype(float)
    if np.unique(outcomes).size != 2:
        raise ValueError("y must contain observations from both binary outcome classes.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Small-sample inference requires more observations than regressors.")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("X is rank deficient; penalized coefficients are not identified uniquely.")
    return design, feature_names, outcomes


def _validate_prediction_data(
    X: Any, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, pd.Index]:
    design, names = _as_2d_array(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(f"X has {design.shape[1]} columns; expected {len(feature_names)}.")
    if isinstance(X, pd.DataFrame) and tuple(names) != feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(design.shape[0])
    return design, index


def _positive_definite_inverse(matrix: np.ndarray, *, label: str) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0.0:
        raise RuntimeError(f"{label} is non-finite or non-positive.")
    lower_limit = max(1e-12, 1e-12 * eigenvalues[-1])
    if eigenvalues[0] <= lower_limit:
        raise RuntimeError(f"{label} is singular or too ill-conditioned for inference.")
    inverse = np.linalg.inv(symmetric)
    return 0.5 * (inverse + inverse.T)


def _coefficient_inference(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    parameter_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or np.any(diagonal <= 0.0):
        raise RuntimeError("Approximate penalized-estimator covariance is not positive and finite.")
    standard_errors = np.sqrt(diagonal)
    zstats = coefficients / standard_errors
    pvalues = 2.0 * norm.sf(np.abs(zstats))
    names = list(parameter_names)
    return (
        pd.DataFrame(covariance, index=names, columns=names),
        pd.Series(standard_errors, index=names, name="standard_error"),
        pd.Series(zstats, index=names, name="z_stat"),
        pd.Series(pvalues, index=names, name="p_value"),
    )


@dataclass(frozen=True)
class _PenalizedBinaryLogitResult:
    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    inference_valid: bool
    loglike: float
    penalized_loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    constant_features: tuple[str, ...]
    score_norm: float
    n_iter: int
    optimizer_result: OptimizeResult
    backend: str
    covariance_type: str
    inference_note: str

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def df_resid(self) -> int:
        return self.nobs - self.n_params

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.standard_errors,
                "z": self.zstats,
                "p_value": self.pvalues,
            }
        )

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = float(norm.ppf(0.5 + level / 2.0))
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
        )

    def predict_proba(self, X: Any) -> pd.DataFrame:
        design, index = _validate_prediction_data(X, self.feature_names)
        probability_one = expit(design @ self.params.to_numpy(dtype=float))
        return pd.DataFrame({0: 1.0 - probability_one, 1: probability_one}, index=index)

    def predict(self, X: Any, *, threshold: float = 0.5) -> pd.Series:
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be finite and strictly between zero and one.")
        return (self.predict_proba(X)[1] >= threshold).astype(int).rename("prediction")


@dataclass(frozen=True)
class FirthBinaryLogitResult(_PenalizedBinaryLogitResult):
    """Mean-bias-reduced Binary Logit result.

    ``covariance`` is the inverse ordinary Fisher information evaluated at the
    bias-reduced estimate. It is an approximate Wald covariance, not a
    profile-penalized-likelihood interval or the inverse penalized Hessian.
    """

    jeffreys_penalty: float
    step_halvings: int


@dataclass(frozen=True)
class RidgeBinaryLogitResult(_PenalizedBinaryLogitResult):
    """Ridge Binary Logit result with approximate sandwich inference."""

    penalty: float
    penalize_intercept: bool
    penalty_mask: pd.Series
    effective_df: float


def _firth_components(
    design: np.ndarray, outcomes: np.ndarray, coefficients: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    linear_predictor = design @ coefficients
    probabilities = expit(linear_predictor)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    sign, log_determinant = np.linalg.slogdet(information)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise RuntimeError("Firth Fisher information became singular during iteration.")

    weighted_design = np.sqrt(weights)[:, None] * design
    leverage = np.sum(
        weighted_design * np.linalg.solve(information, weighted_design.T).T,
        axis=1,
    )
    leverage = np.clip(leverage, 0.0, 1.0)
    adjusted_score = design.T @ (
        outcomes - probabilities + leverage * (0.5 - probabilities)
    )
    loglike = float(
        np.sum(outcomes * linear_predictor - np.logaddexp(0.0, linear_predictor))
    )
    penalized_loglike = loglike + 0.5 * float(log_determinant)
    return penalized_loglike, loglike, adjusted_score, information, probabilities


class FirthBinaryLogit:
    """Binary Logit using Firth's mean-bias-reducing adjusted score.

    The fitted target is ``log L(beta) + 0.5 log|I(beta)|``. Unlike the
    ordinary MLE estimator, this class intentionally permits complete and
    quasi-complete separation.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
        max_step: float = 5.0,
        max_step_halvings: int = 30,
    ) -> FirthBinaryLogitResult:
        maxiter, tolerance = _validate_optimizer_options(maxiter, tolerance)
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be finite and positive.")
        if (
            isinstance(max_step_halvings, bool)
            or not isinstance(max_step_halvings, (int, np.integer))
            or max_step_halvings < 1
        ):
            raise ValueError("max_step_halvings must be a positive integer.")
        design, feature_names, outcomes = _validate_binary_fit_data(X, y)

        coefficients = np.zeros(design.shape[1], dtype=float)
        components = _firth_components(design, outcomes, coefficients)
        total_step_halvings = 0
        converged = False
        iterations = 0

        for iteration in range(1, maxiter + 1):
            penalized_loglike, _, adjusted_score, information, _ = components
            score_norm = float(np.max(np.abs(adjusted_score)))
            if score_norm <= tolerance:
                converged = True
                iterations = iteration - 1
                break

            scoring_step = np.linalg.solve(information, adjusted_score)
            largest_step = float(np.max(np.abs(scoring_step)))
            if largest_step > max_step:
                scoring_step *= max_step / largest_step

            step_scale = 1.0
            candidate_components = None
            for halving in range(max_step_halvings + 1):
                candidate = coefficients + step_scale * scoring_step
                try:
                    evaluated = _firth_components(design, outcomes, candidate)
                except (np.linalg.LinAlgError, RuntimeError):
                    evaluated = None
                if (
                    evaluated is not None
                    and np.isfinite(evaluated[0])
                    and evaluated[0] >= penalized_loglike - 1e-12
                ):
                    candidate_components = evaluated
                    break
                if halving < max_step_halvings:
                    step_scale *= 0.5
                    total_step_halvings += 1

            if candidate_components is None:
                raise RuntimeError(
                    "Firth Binary Logit step halving failed to improve the penalized likelihood."
                )

            parameter_change = float(np.max(np.abs(step_scale * scoring_step)))
            coefficients = coefficients + step_scale * scoring_step
            components = candidate_components
            iterations = iteration
            new_score_norm = float(np.max(np.abs(components[2])))
            if (
                parameter_change <= tolerance * (1.0 + np.max(np.abs(coefficients)))
                and new_score_norm <= max(10.0 * tolerance, 1e-8)
            ):
                converged = True
                break

        penalized_loglike, loglike, adjusted_score, information, _ = components
        score_norm = float(np.max(np.abs(adjusted_score)))
        converged = bool(converged or score_norm <= max(10.0 * tolerance, 1e-8))
        if not converged or not np.isfinite(coefficients).all():
            raise RuntimeError(
                "Firth Binary Logit did not converge; "
                f"adjusted score norm={score_norm:.6g} after {iterations} iterations."
            )

        covariance_values = _positive_definite_inverse(
            information, label="Firth ordinary Fisher information"
        )
        covariance, standard_errors, zstats, pvalues = _coefficient_inference(
            coefficients, covariance_values, feature_names
        )
        jeffreys_penalty = float(penalized_loglike - loglike)
        optimizer_result = OptimizeResult(
            x=coefficients.copy(),
            success=True,
            status=0,
            message="Firth adjusted-score Fisher scoring converged.",
            nit=iterations,
            fun=-penalized_loglike,
            jac=-adjusted_score.copy(),
        )

        return FirthBinaryLogitResult(
            params=pd.Series(coefficients, index=feature_names, name="estimate"),
            covariance=covariance,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            converged=True,
            inference_valid=True,
            loglike=loglike,
            penalized_loglike=penalized_loglike,
            nobs=design.shape[0],
            feature_names=tuple(feature_names),
            constant_features=_constant_feature_names(design, feature_names),
            score_norm=score_norm,
            n_iter=iterations,
            optimizer_result=optimizer_result,
            backend="native-firth-adjusted-score",
            covariance_type="inverse-ordinary-fisher-at-bias-reduced-estimate",
            inference_note=(
                "Approximate Wald inference from ordinary Fisher information at the "
                "bias-reduced estimate; profile penalized-likelihood inference is not computed."
            ),
            jeffreys_penalty=jeffreys_penalty,
            step_halvings=total_step_halvings,
        )


class RidgeBinaryLogit:
    """Binary Logit with an L2 penalty and optional constant-column exclusion."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        penalty: float = 1.0,
        penalize_intercept: bool = False,
        maxiter: int = 1_000,
        tolerance: float = 1e-10,
    ) -> RidgeBinaryLogitResult:
        maxiter, tolerance = _validate_optimizer_options(maxiter, tolerance)
        penalty = _validate_penalty(penalty)
        if not isinstance(penalize_intercept, (bool, np.bool_)):
            raise ValueError("penalize_intercept must be boolean.")
        design, feature_names, outcomes = _validate_binary_fit_data(X, y)
        constant_features = _constant_feature_names(design, feature_names)
        penalty_mask = np.ones(design.shape[1], dtype=float)
        if not penalize_intercept:
            for feature in constant_features:
                penalty_mask[feature_names.index(feature)] = 0.0

        def negative_penalized_loglike(coefficients: np.ndarray) -> float:
            linear_predictor = design @ coefficients
            negative_loglike = np.sum(
                np.logaddexp(0.0, linear_predictor) - outcomes * linear_predictor
            )
            return float(
                negative_loglike
                + 0.5 * penalty * np.dot(penalty_mask * coefficients, coefficients)
            )

        def penalized_gradient(coefficients: np.ndarray) -> np.ndarray:
            return (
                design.T @ (expit(design @ coefficients) - outcomes)
                + penalty * penalty_mask * coefficients
            )

        optimizer_result = minimize(
            negative_penalized_loglike,
            np.zeros(design.shape[1], dtype=float),
            jac=penalized_gradient,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": min(tolerance, 1e-12),
                "gtol": tolerance,
                "maxls": 50,
            },
        )
        coefficients = np.asarray(optimizer_result.x, dtype=float)
        score_norm = float(np.max(np.abs(penalized_gradient(coefficients))))
        converged = bool(
            optimizer_result.success or score_norm <= max(100.0 * tolerance, 1e-7)
        )
        if (
            not converged
            or not np.isfinite(optimizer_result.fun)
            or not np.isfinite(coefficients).all()
        ):
            raise RuntimeError(
                "Ridge Binary Logit optimization failed: " + str(optimizer_result.message)
            )

        linear_predictor = design @ coefficients
        probabilities = expit(linear_predictor)
        weights = probabilities * (1.0 - probabilities)
        ordinary_information = design.T @ (weights[:, None] * design)
        penalized_hessian = ordinary_information + penalty * np.diag(penalty_mask)
        inverse_penalized_hessian = _positive_definite_inverse(
            penalized_hessian, label="Ridge penalized Hessian"
        )
        covariance_values = (
            inverse_penalized_hessian
            @ ordinary_information
            @ inverse_penalized_hessian
        )
        covariance_values = 0.5 * (covariance_values + covariance_values.T)
        covariance, standard_errors, zstats, pvalues = _coefficient_inference(
            coefficients, covariance_values, feature_names
        )
        loglike = float(
            np.sum(outcomes * linear_predictor - np.logaddexp(0.0, linear_predictor))
        )
        effective_df = float(np.trace(inverse_penalized_hessian @ ordinary_information))

        return RidgeBinaryLogitResult(
            params=pd.Series(coefficients, index=feature_names, name="estimate"),
            covariance=covariance,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            converged=converged,
            inference_valid=True,
            loglike=loglike,
            penalized_loglike=-float(optimizer_result.fun),
            nobs=design.shape[0],
            feature_names=tuple(feature_names),
            constant_features=constant_features,
            score_norm=score_norm,
            n_iter=int(optimizer_result.nit),
            optimizer_result=optimizer_result,
            backend="native-ridge-logit",
            covariance_type="penalized-estimating-equation-sandwich",
            inference_note=(
                "Approximate model-based sandwich covariance H_pen^-1 I H_pen^-1; "
                "it is not an inverse-posterior-curvature covariance."
            ),
            penalty=penalty,
            penalize_intercept=bool(penalize_intercept),
            penalty_mask=pd.Series(penalty_mask, index=feature_names, name="penalty_mask"),
            effective_df=effective_df,
        )


def _selected_ordered_log_probabilities(
    design: np.ndarray,
    encoded: np.ndarray,
    coefficients: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    n_thresholds = thresholds.size
    indices = thresholds[None, :] - design @ coefficients[:, None]
    log_cumulative = log_expit(indices)
    selected_log_probabilities = np.empty(encoded.size, dtype=float)
    first = encoded == 0
    last = encoded == n_thresholds
    selected_log_probabilities[first] = log_cumulative[first, 0]
    selected_log_probabilities[last] = log_expit(-indices[last, -1])
    for category in range(1, n_thresholds):
        selected = encoded == category
        log_ratio = (
            log_cumulative[selected, category - 1] - log_cumulative[selected, category]
        )
        selected_log_probabilities[selected] = log_cumulative[
            selected, category
        ] + np.log(-np.expm1(np.minimum(log_ratio, -1e-15)))
    return selected_log_probabilities


def _numerical_gradient(function: Any, point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    steps = 1e-6 * (1.0 + np.abs(point))
    gradient = np.empty_like(point)
    for index, step in enumerate(steps):
        shift = np.zeros_like(point)
        shift[index] = step
        gradient[index] = (function(point + shift) - function(point - shift)) / (2.0 * step)
    return gradient


@dataclass(frozen=True)
class RidgeOrderedLogitResult:
    """Ridge proportional-odds result with slope-only penalization.

    The covariance is an approximate penalized-estimating-equation sandwich
    transformed from unconstrained threshold coordinates to reported ordered
    cutpoints.
    """

    params: pd.Series
    thresholds: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    categories: np.ndarray
    feature_names: tuple[str, ...]
    converged: bool
    inference_valid: bool
    loglike: float
    penalized_loglike: float
    nobs: int
    penalty: float
    effective_df: float
    score_norm: float
    n_iter: int
    optimizer_result: OptimizeResult
    covariance_type: str
    inference_note: str

    @property
    def all_params(self) -> pd.Series:
        thresholds = self.thresholds.copy()
        thresholds.index = [f"threshold: {name}" for name in thresholds.index]
        return pd.concat([self.params, thresholds]).rename("estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def backend(self) -> str:
        return "native-ridge-ordered-logit"

    @property
    def penalty_target(self) -> str:
        return "slopes-only"

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.all_params,
                "std_err": self.standard_errors,
                "z": self.zstats,
                "p_value": self.pvalues,
            }
        )

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = float(norm.ppf(0.5 + level / 2.0))
        return pd.DataFrame(
            {
                "lower": self.all_params - critical * self.standard_errors,
                "upper": self.all_params + critical * self.standard_errors,
            }
        )

    def predict_proba(self, X: Any) -> pd.DataFrame:
        design, index = _validate_prediction_data(X, self.feature_names)
        probabilities = _category_probabilities(
            design,
            self.params.to_numpy(dtype=float),
            self.thresholds.to_numpy(dtype=float),
            "logit",
        )
        return pd.DataFrame(probabilities, columns=self.categories, index=index)

    def predict(self, X: Any) -> pd.Series:
        probability_frame = self.predict_proba(X)
        probabilities = probability_frame.to_numpy()
        return pd.Series(
            self.categories[np.argmax(probabilities, axis=1)],
            index=probability_frame.index,
            name="prediction",
        )


class RidgeOrderedLogit:
    """Proportional-odds Ordered Logit with an L2 penalty on slopes only."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        penalty: float = 1.0,
        category_order: Sequence[Any] | None = None,
        maxiter: int = 1_000,
        tolerance: float = 1e-9,
    ) -> RidgeOrderedLogitResult:
        maxiter, tolerance = _validate_optimizer_options(maxiter, tolerance)
        penalty = _validate_penalty(penalty)
        design, feature_names = _as_2d_array(X)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X feature names must be unique after conversion to strings.")
        encoded, categories = _ordered_categories(y, category_order=category_order)
        if design.shape[0] != encoded.size:
            raise ValueError("X and y must contain the same number of observations.")
        constant_features = _constant_feature_names(design, feature_names)
        if constant_features:
            raise ValueError(
                "Ordered models identify location through thresholds; constant regressors "
                f"are not permitted: {list(constant_features)}."
            )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("X is rank deficient; ridge ordinal slopes are not unique.")

        n_features = design.shape[1]
        n_thresholds = categories.size - 1
        n_parameters = n_features + n_thresholds
        if design.shape[0] <= n_parameters:
            raise ValueError("Ridge Ordered Logit requires more observations than parameters.")
        cumulative_shares = np.array(
            [np.mean(encoded <= index) for index in range(n_thresholds)], dtype=float
        )
        initial_thresholds = np.log(
            np.clip(cumulative_shares, 1e-6, 1 - 1e-6)
            / np.clip(1 - cumulative_shares, 1e-6, 1 - 1e-6)
        )
        raw_thresholds = np.r_[
            initial_thresholds[0], np.log(np.diff(initial_thresholds))
        ]
        initial = np.r_[np.zeros(n_features), raw_thresholds]

        def negative_loglike(parameters: np.ndarray) -> float:
            coefficients = parameters[:n_features]
            thresholds = _unpack_thresholds(parameters[n_features:])
            selected = _selected_ordered_log_probabilities(
                design, encoded, coefficients, thresholds
            )
            return float(-np.sum(selected))

        def negative_penalized_loglike(parameters: np.ndarray) -> float:
            coefficients = parameters[:n_features]
            return float(
                negative_loglike(parameters) + 0.5 * penalty * np.dot(coefficients, coefficients)
            )

        bounds: list[tuple[float | None, float | None]] = [
            (None, None) for _ in range(n_features + 1)
        ] + [(-20.0, 20.0) for _ in range(n_thresholds - 1)]
        optimizer_result = minimize(
            negative_penalized_loglike,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "ftol": min(tolerance, 1e-12),
                "gtol": tolerance,
                "maxls": 50,
            },
        )
        fitted_parameters = np.asarray(optimizer_result.x, dtype=float)
        score = _numerical_gradient(negative_penalized_loglike, fitted_parameters)
        score_norm = float(np.max(np.abs(score)))
        converged = bool(
            optimizer_result.success or score_norm <= max(100.0 * tolerance, 1e-6)
        )
        if (
            not converged
            or not np.isfinite(optimizer_result.fun)
            or not np.isfinite(fitted_parameters).all()
        ):
            raise RuntimeError(
                "Ridge Ordered Logit optimization failed: " + str(optimizer_result.message)
            )

        penalized_hessian = _numerical_hessian(
            negative_penalized_loglike, fitted_parameters
        )
        ordinary_information = _numerical_hessian(negative_loglike, fitted_parameters)
        inverse_penalized_hessian = _positive_definite_inverse(
            penalized_hessian, label="Ridge Ordered Logit penalized Hessian"
        )
        raw_covariance = (
            inverse_penalized_hessian
            @ ordinary_information
            @ inverse_penalized_hessian
        )
        raw_covariance = 0.5 * (raw_covariance + raw_covariance.T)

        coefficients = fitted_parameters[:n_features]
        fitted_raw_thresholds = fitted_parameters[n_features:]
        thresholds = _unpack_thresholds(fitted_raw_thresholds)
        threshold_names = [
            f"{categories[index]} | {categories[index + 1]}"
            for index in range(n_thresholds)
        ]
        parameter_names = feature_names + [
            f"threshold: {name}" for name in threshold_names
        ]
        transformation = np.eye(n_parameters)
        transformation[n_features:, n_features:] = _threshold_jacobian(
            fitted_raw_thresholds
        )
        covariance_values = transformation @ raw_covariance @ transformation.T
        covariance_values = 0.5 * (covariance_values + covariance_values.T)
        reported_parameters = np.r_[coefficients, thresholds]
        covariance, standard_errors, zstats, pvalues = _coefficient_inference(
            reported_parameters, covariance_values, parameter_names
        )
        loglike = -negative_loglike(fitted_parameters)
        effective_df = float(
            np.trace(inverse_penalized_hessian @ ordinary_information)
        )

        return RidgeOrderedLogitResult(
            params=pd.Series(coefficients, index=feature_names, name="coefficient"),
            thresholds=pd.Series(thresholds, index=threshold_names, name="threshold"),
            covariance=covariance,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            categories=categories,
            feature_names=tuple(feature_names),
            converged=converged,
            inference_valid=True,
            loglike=loglike,
            penalized_loglike=-float(optimizer_result.fun),
            nobs=design.shape[0],
            penalty=penalty,
            effective_df=effective_df,
            score_norm=score_norm,
            n_iter=int(optimizer_result.nit),
            optimizer_result=optimizer_result,
            covariance_type="penalized-observed-information-sandwich",
            inference_note=(
                "Approximate slope-penalized sandwich covariance transformed to ordered "
                "cutpoints; thresholds are not directly penalized."
            ),
        )


__all__ = [
    "FirthBinaryLogit",
    "FirthBinaryLogitResult",
    "RidgeBinaryLogit",
    "RidgeBinaryLogitResult",
    "RidgeOrderedLogit",
    "RidgeOrderedLogitResult",
]
