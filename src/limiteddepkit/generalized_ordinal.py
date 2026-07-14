"""Generalized ordinal-response estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr
from scipy.stats import norm

from .ordinal import (
    OrderedLogit,
    _as_2d_array,
    _numerical_hessian,
    _numerical_jacobian,
    _ordered_categories,
)


def _interior_inference(
    objective: Any,
    parameters: np.ndarray,
    names: list[str],
    *,
    constraint_slack: float,
    boundary_tolerance: float = 1e-5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, bool]:
    valid = constraint_slack > boundary_tolerance
    if valid:
        information = _numerical_hessian(objective, parameters)
        covariance_values = np.linalg.pinv(information)
        covariance_values = (covariance_values + covariance_values.T) / 2.0
        standard_error_values = np.sqrt(np.clip(np.diag(covariance_values), 0.0, None))
        zstat_values = np.divide(
            parameters,
            standard_error_values,
            out=np.full_like(parameters, np.nan),
            where=standard_error_values > 0,
        )
        pvalue_values = 2.0 * ndtr(-np.abs(zstat_values))
    else:
        covariance_values = np.full((len(names), len(names)), np.nan)
        standard_error_values = np.full(len(names), np.nan)
        zstat_values = np.full(len(names), np.nan)
        pvalue_values = np.full(len(names), np.nan)
    return (
        pd.DataFrame(covariance_values, index=names, columns=names),
        pd.Series(standard_error_values, index=names, name="standard_error"),
        pd.Series(zstat_values, index=names, name="z_stat"),
        pd.Series(pvalue_values, index=names, name="p_value"),
        valid,
    )


def _confidence_intervals(
    parameters: pd.Series, standard_errors: pd.Series, level: float
) -> pd.DataFrame:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one.")
    critical = norm.ppf(0.5 + level / 2.0)
    return pd.DataFrame(
        {
            "lower": parameters - critical * standard_errors,
            "upper": parameters + critical * standard_errors,
        }
    )


def _generalized_probabilities(
    X: np.ndarray,
    thresholds: np.ndarray,
    slopes: np.ndarray,
    *,
    crossing_tolerance: float = 1e-9,
) -> np.ndarray:
    cumulative_indices = thresholds[None, :] - X @ slopes.T
    gaps = np.diff(cumulative_indices, axis=1)
    if gaps.size and np.min(gaps) < -crossing_tolerance:
        raise ValueError(
            "Generalized Ordered Logit cumulative probabilities cross at the supplied "
            "covariate values. Prediction outside the supported non-crossing region is unsafe."
        )
    cumulative = expit(cumulative_indices)
    bounds = np.column_stack([np.zeros(X.shape[0]), cumulative, np.ones(X.shape[0])])
    probabilities = np.diff(bounds, axis=1)
    return np.clip(probabilities, 0.0, 1.0)


def _generalized_marginal_effects(
    X: np.ndarray,
    thresholds: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    cumulative_indices = thresholds[None, :] - X @ slopes.T
    cumulative = expit(cumulative_indices)
    densities = cumulative * (1.0 - cumulative)
    weighted_slopes = densities[:, :, None] * slopes[None, :, :]
    zero_boundary = np.zeros((X.shape[0], 1, X.shape[1]), dtype=float)
    derivative_bounds = np.concatenate(
        [zero_boundary, weighted_slopes, zero_boundary], axis=1
    )
    return derivative_bounds[:, :-1, :] - derivative_bounds[:, 1:, :]


def _effects_frame(
    effects: np.ndarray, categories: np.ndarray, feature_names: tuple[str, ...]
) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [categories, feature_names], names=["category", "feature"]
    )
    return pd.DataFrame(effects.reshape(effects.shape[0], -1), columns=columns)


def _average_effects_inference(
    ame_function: Any,
    parameters: np.ndarray,
    covariance: np.ndarray,
    categories: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    inference_valid: bool,
    level: float,
) -> pd.DataFrame:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one.")
    estimates = np.asarray(ame_function(parameters), dtype=float)
    if inference_valid:
        jacobian = _numerical_jacobian(ame_function, parameters)
        ame_covariance = jacobian @ covariance @ jacobian.T
        ame_covariance = (ame_covariance + ame_covariance.T) / 2.0
        standard_errors = np.sqrt(np.clip(np.diag(ame_covariance), 0.0, None))
        zstats = np.divide(
            estimates,
            standard_errors,
            out=np.full_like(estimates, np.nan),
            where=standard_errors > 0,
        )
        pvalues = 2.0 * ndtr(-np.abs(zstats))
        critical = norm.ppf(0.5 + level / 2.0)
        lower = estimates - critical * standard_errors
        upper = estimates + critical * standard_errors
    else:
        standard_errors = np.full_like(estimates, np.nan)
        zstats = np.full_like(estimates, np.nan)
        pvalues = np.full_like(estimates, np.nan)
        lower = np.full_like(estimates, np.nan)
        upper = np.full_like(estimates, np.nan)
    index = pd.MultiIndex.from_product(
        [categories, feature_names], names=["category", "feature"]
    )
    output = pd.DataFrame(
        {
            "estimate": estimates,
            "standard_error": standard_errors,
            "z_stat": zstats,
            "p_value": pvalues,
            "lower": lower,
            "upper": upper,
        },
        index=index,
    )
    output.attrs["inference_valid"] = inference_valid
    return output


def _validate_result_X(
    X: Any, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    values, names = _as_2d_array(X)
    if values.shape[1] != len(feature_names):
        raise ValueError(f"X has {values.shape[1]} columns; expected {len(feature_names)}.")
    if isinstance(X, pd.DataFrame) and tuple(names) != feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    return values, names


def _flexible_margins(
    result: Any,
    X: Any,
    *,
    at: str | Mapping[str, float],
    kind: str,
) -> pd.Series | pd.DataFrame:
    values, _ = _validate_result_X(X, result.feature_names)
    if kind not in {"probability", "marginal_effect"}:
        raise ValueError("kind must be 'probability' or 'marginal_effect'.")

    if at == "overall":
        evaluation = X if isinstance(X, pd.DataFrame) else values
    elif at == "mean":
        evaluation = pd.DataFrame([values.mean(axis=0)], columns=result.feature_names)
    elif isinstance(at, Mapping):
        unknown = set(at) - set(result.feature_names)
        if unknown:
            raise ValueError(f"Unknown covariates in at: {sorted(unknown)}.")
        means = values.mean(axis=0)
        representative = {
            feature: means[index] for index, feature in enumerate(result.feature_names)
        }
        for feature, value in at.items():
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError("User-specified margin values must be finite.")
            representative[feature] = numeric_value
        evaluation = pd.DataFrame([representative], columns=result.feature_names)
    else:
        raise ValueError("at must be 'overall', 'mean', or a covariate-value mapping.")

    if kind == "probability":
        probabilities = result.predict_proba(evaluation).mean(axis=0)
        probabilities.index.name = "category"
        return probabilities.rename("estimate")
    effects = result.marginal_effects(evaluation).mean(axis=0).unstack("feature")
    effects.index.name = "category"
    effects.columns.name = "feature"
    return effects


@dataclass(frozen=True)
class GeneralizedOrderedLogitResult:
    """Fitted Generalized Ordered Logit result."""

    thresholds: pd.Series
    threshold_slopes: pd.DataFrame
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    categories: np.ndarray
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    minimum_index_gap: float
    constraint_slack: float
    optimizer_result: Any

    @property
    def n_params(self) -> int:
        return len(self.thresholds) + self.threshold_slopes.size

    @property
    def all_params(self) -> pd.Series:
        cuts = self.thresholds.copy()
        cuts.index = [f"threshold: {name}" for name in cuts.index]
        slope_names = [
            f"slope {split}: {feature}"
            for split in self.threshold_slopes.index
            for feature in self.threshold_slopes.columns
        ]
        slopes = pd.Series(self.threshold_slopes.to_numpy().ravel(), index=slope_names)
        return pd.concat([cuts, slopes]).rename("estimate")

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        return _confidence_intervals(self.all_params, self.standard_errors, level)

    @property
    def params(self) -> pd.Series:
        """Ecosystem-compatible alias for all fitted parameters."""
        return self.all_params

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def predict_proba(self, X: Any) -> pd.DataFrame:
        """Return category probabilities where cumulative logits do not cross."""
        values, _ = _validate_result_X(X, self.feature_names)
        probabilities = _generalized_probabilities(
            values,
            self.thresholds.to_numpy(dtype=float),
            self.threshold_slopes.to_numpy(dtype=float),
        )
        return pd.DataFrame(probabilities, columns=self.categories)

    def predict(self, X: Any) -> pd.Series:
        """Return the category with the highest predicted probability."""
        probabilities = self.predict_proba(X).to_numpy()
        return pd.Series(self.categories[np.argmax(probabilities, axis=1)], name="prediction")

    def marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return observation-level category-specific probability derivatives."""
        values, _ = _validate_result_X(X, self.feature_names)
        effects = _generalized_marginal_effects(
            values,
            self.thresholds.to_numpy(dtype=float),
            self.threshold_slopes.to_numpy(dtype=float),
        )
        return _effects_frame(effects, self.categories, self.feature_names)

    def average_marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return sample-average category-specific marginal effects."""
        average = self.marginal_effects(X).mean(axis=0).unstack("feature")
        average.index.name = "category"
        average.columns.name = "feature"
        return average

    def average_marginal_effects_inference(
        self, X: Any, *, level: float = 0.95
    ) -> pd.DataFrame:
        """Return delta-method AME inference for an interior solution."""
        values, _ = _validate_result_X(X, self.feature_names)
        n_thresholds = len(self.thresholds)
        n_features = len(self.feature_names)

        def ame(parameters: np.ndarray) -> np.ndarray:
            thresholds = parameters[:n_thresholds]
            slopes = parameters[n_thresholds:].reshape(n_thresholds, n_features)
            return _generalized_marginal_effects(values, thresholds, slopes).mean(
                axis=0
            ).reshape(-1)

        return _average_effects_inference(
            ame,
            self.all_params.to_numpy(dtype=float),
            self.covariance.to_numpy(dtype=float),
            self.categories,
            self.feature_names,
            inference_valid=self.inference_valid,
            level=level,
        )

    def margins(
        self,
        X: Any,
        *,
        at: str | Mapping[str, float] = "overall",
        kind: str = "probability",
    ) -> pd.Series | pd.DataFrame:
        """Evaluate flexible ordinal margins over observed or representative X."""
        return _flexible_margins(self, X, at=at, kind=kind)


class GeneralizedOrderedLogit:
    """Generalized Ordered Logit with threshold-specific slopes.

    Non-crossing constraints are imposed at every estimation observation. The
    fitted result checks the same condition at prediction time because arbitrary
    threshold-specific slopes cannot guarantee global ordering on unbounded
    covariate support.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        category_order: Sequence[Any] | None = None,
        maxiter: int = 2_000,
        tolerance: float = 1e-8,
        minimum_gap: float = 1e-6,
    ) -> GeneralizedOrderedLogitResult:
        values, feature_names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=category_order)
        if values.shape[0] != encoded.size:
            raise ValueError("X and y must contain the same number of observations.")
        if minimum_gap <= 0:
            raise ValueError("minimum_gap must be positive.")

        ordered_start = OrderedLogit().fit(X, y, category_order=categories)
        n_thresholds = categories.size - 1
        n_features = values.shape[1]
        initial = np.r_[
            ordered_start.thresholds.to_numpy(dtype=float),
            np.tile(ordered_start.params.to_numpy(dtype=float), n_thresholds),
        ]

        def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            thresholds = parameters[:n_thresholds]
            slopes = parameters[n_thresholds:].reshape(n_thresholds, n_features)
            return thresholds, slopes

        def cumulative_indices(parameters: np.ndarray) -> np.ndarray:
            thresholds, slopes = unpack(parameters)
            return thresholds[None, :] - values @ slopes.T

        def objective(parameters: np.ndarray) -> float:
            thresholds, slopes = unpack(parameters)
            cumulative = expit(thresholds[None, :] - values @ slopes.T)
            bounds = np.column_stack(
                [np.zeros(values.shape[0]), cumulative, np.ones(values.shape[0])]
            )
            probabilities = np.diff(bounds, axis=1)
            selected = probabilities[np.arange(encoded.size), encoded]
            return float(-np.log(np.clip(selected, 1e-15, 1.0)).sum())

        constraints = {
            "type": "ineq",
            "fun": lambda parameters: np.diff(cumulative_indices(parameters), axis=1).ravel()
            - minimum_gap,
        }
        fitted = minimize(
            objective,
            initial,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": tolerance},
        )
        if not np.isfinite(fitted.fun):
            raise RuntimeError("Generalized Ordered Logit produced a non-finite likelihood.")

        thresholds, slopes = unpack(fitted.x)
        index_gaps = np.diff(cumulative_indices(fitted.x), axis=1)
        minimum_fitted_gap = float(np.min(index_gaps)) if index_gaps.size else np.inf
        threshold_names = [
            f"{categories[index]} | {categories[index + 1]}"
            for index in range(n_thresholds)
        ]
        parameter_names = [f"threshold: {name}" for name in threshold_names] + [
            f"slope {split}: {feature}"
            for split in threshold_names
            for feature in feature_names
        ]
        covariance, standard_errors, zstats, pvalues, inference_valid = _interior_inference(
            objective,
            fitted.x,
            parameter_names,
            constraint_slack=minimum_fitted_gap - minimum_gap,
        )
        return GeneralizedOrderedLogitResult(
            thresholds=pd.Series(thresholds, index=threshold_names, name="threshold"),
            threshold_slopes=pd.DataFrame(
                slopes, index=threshold_names, columns=feature_names
            ),
            covariance=covariance,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            inference_valid=inference_valid,
            categories=categories,
            converged=bool(fitted.success and minimum_fitted_gap >= minimum_gap - 1e-7),
            loglike=float(-fitted.fun),
            nobs=values.shape[0],
            feature_names=tuple(feature_names),
            minimum_index_gap=minimum_fitted_gap,
            constraint_slack=minimum_fitted_gap - minimum_gap,
            optimizer_result=fitted,
        )


@dataclass(frozen=True)
class PartialProportionalOddsResult:
    """Fitted Partial Proportional Odds result."""

    thresholds: pd.Series
    common_params: pd.Series
    varying_params: pd.DataFrame
    threshold_slopes: pd.DataFrame
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    categories: np.ndarray
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    varying_features: tuple[str, ...]
    minimum_index_gap: float
    constraint_slack: float
    optimizer_result: Any

    @property
    def n_params(self) -> int:
        return len(self.thresholds) + len(self.common_params) + self.varying_params.size

    @property
    def all_params(self) -> pd.Series:
        cuts = self.thresholds.copy()
        cuts.index = [f"threshold: {name}" for name in cuts.index]
        common = self.common_params.copy()
        common.index = [f"common: {feature}" for feature in common.index]
        varying_names = [
            f"varying {split}: {feature}"
            for split in self.varying_params.index
            for feature in self.varying_params.columns
        ]
        varying = pd.Series(self.varying_params.to_numpy().ravel(), index=varying_names)
        return pd.concat([cuts, common, varying]).rename("estimate")

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        return _confidence_intervals(self.all_params, self.standard_errors, level)

    @property
    def params(self) -> pd.Series:
        """Ecosystem-compatible alias for all fitted parameters."""
        return self.all_params

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def predict_proba(self, X: Any) -> pd.DataFrame:
        values, _ = _validate_result_X(X, self.feature_names)
        probabilities = _generalized_probabilities(
            values,
            self.thresholds.to_numpy(dtype=float),
            self.threshold_slopes.to_numpy(dtype=float),
        )
        return pd.DataFrame(probabilities, columns=self.categories)

    def predict(self, X: Any) -> pd.Series:
        probabilities = self.predict_proba(X).to_numpy()
        return pd.Series(self.categories[np.argmax(probabilities, axis=1)], name="prediction")

    def marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return observation-level category-specific probability derivatives."""
        values, _ = _validate_result_X(X, self.feature_names)
        effects = _generalized_marginal_effects(
            values,
            self.thresholds.to_numpy(dtype=float),
            self.threshold_slopes.to_numpy(dtype=float),
        )
        return _effects_frame(effects, self.categories, self.feature_names)

    def average_marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return sample-average category-specific marginal effects."""
        average = self.marginal_effects(X).mean(axis=0).unstack("feature")
        average.index.name = "category"
        average.columns.name = "feature"
        return average

    def average_marginal_effects_inference(
        self, X: Any, *, level: float = 0.95
    ) -> pd.DataFrame:
        """Return delta-method AME inference for an interior solution."""
        values, _ = _validate_result_X(X, self.feature_names)
        n_thresholds = len(self.thresholds)
        common_features = [
            feature for feature in self.feature_names if feature not in self.varying_features
        ]
        common_indices = [self.feature_names.index(feature) for feature in common_features]
        varying_indices = [
            self.feature_names.index(feature) for feature in self.varying_features
        ]
        n_common = len(common_indices)
        n_varying = len(varying_indices)

        def ame(parameters: np.ndarray) -> np.ndarray:
            thresholds = parameters[:n_thresholds]
            common = parameters[n_thresholds : n_thresholds + n_common]
            varying = parameters[n_thresholds + n_common :].reshape(
                n_thresholds, n_varying
            )
            slopes = np.empty((n_thresholds, len(self.feature_names)), dtype=float)
            if n_common:
                slopes[:, common_indices] = common
            slopes[:, varying_indices] = varying
            return _generalized_marginal_effects(values, thresholds, slopes).mean(
                axis=0
            ).reshape(-1)

        return _average_effects_inference(
            ame,
            self.all_params.to_numpy(dtype=float),
            self.covariance.to_numpy(dtype=float),
            self.categories,
            self.feature_names,
            inference_valid=self.inference_valid,
            level=level,
        )

    def margins(
        self,
        X: Any,
        *,
        at: str | Mapping[str, float] = "overall",
        kind: str = "probability",
    ) -> pd.Series | pd.DataFrame:
        """Evaluate flexible ordinal margins over observed or representative X."""
        return _flexible_margins(self, X, at=at, kind=kind)


class PartialProportionalOdds:
    """Ordered Logit with selected threshold-varying slopes."""

    def __init__(self, *, varying: Sequence[str]) -> None:
        self.varying = tuple(str(feature) for feature in varying)
        if not self.varying:
            raise ValueError("varying must contain at least one feature name.")
        if len(set(self.varying)) != len(self.varying):
            raise ValueError("varying feature names must be unique.")

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        category_order: Sequence[Any] | None = None,
        maxiter: int = 2_000,
        tolerance: float = 1e-8,
        minimum_gap: float = 1e-6,
    ) -> PartialProportionalOddsResult:
        values, feature_names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=category_order)
        if values.shape[0] != encoded.size:
            raise ValueError("X and y must contain the same number of observations.")
        if minimum_gap <= 0:
            raise ValueError("minimum_gap must be positive.")
        unknown = set(self.varying) - set(feature_names)
        if unknown:
            raise ValueError(f"Unknown varying features: {sorted(unknown)}.")

        varying_indices = [feature_names.index(feature) for feature in self.varying]
        common_features = [feature for feature in feature_names if feature not in self.varying]
        common_indices = [feature_names.index(feature) for feature in common_features]
        n_thresholds = categories.size - 1
        n_common = len(common_indices)
        n_varying = len(varying_indices)
        ordered_start = OrderedLogit().fit(X, y, category_order=categories)
        ordered_beta = ordered_start.params.to_numpy(dtype=float)
        initial = np.r_[
            ordered_start.thresholds.to_numpy(dtype=float),
            ordered_beta[common_indices],
            np.tile(ordered_beta[varying_indices], n_thresholds),
        ]

        def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            thresholds = parameters[:n_thresholds]
            common = parameters[n_thresholds : n_thresholds + n_common]
            varying = parameters[n_thresholds + n_common :].reshape(
                n_thresholds, n_varying
            )
            slopes = np.empty((n_thresholds, len(feature_names)), dtype=float)
            if n_common:
                slopes[:, common_indices] = common
            slopes[:, varying_indices] = varying
            return thresholds, common, slopes

        def cumulative_indices(parameters: np.ndarray) -> np.ndarray:
            thresholds, _, slopes = unpack(parameters)
            return thresholds[None, :] - values @ slopes.T

        def objective(parameters: np.ndarray) -> float:
            indices = cumulative_indices(parameters)
            cumulative = expit(indices)
            bounds = np.column_stack(
                [np.zeros(values.shape[0]), cumulative, np.ones(values.shape[0])]
            )
            selected = np.diff(bounds, axis=1)[np.arange(encoded.size), encoded]
            return float(-np.log(np.clip(selected, 1e-15, 1.0)).sum())

        constraints = {
            "type": "ineq",
            "fun": lambda parameters: np.diff(cumulative_indices(parameters), axis=1).ravel()
            - minimum_gap,
        }
        fitted = minimize(
            objective,
            initial,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": tolerance},
        )
        if not np.isfinite(fitted.fun):
            raise RuntimeError("Partial Proportional Odds produced a non-finite likelihood.")

        thresholds, common, slopes = unpack(fitted.x)
        index_gaps = np.diff(cumulative_indices(fitted.x), axis=1)
        minimum_fitted_gap = float(np.min(index_gaps)) if index_gaps.size else np.inf
        threshold_names = [
            f"{categories[index]} | {categories[index + 1]}"
            for index in range(n_thresholds)
        ]
        parameter_names = (
            [f"threshold: {name}" for name in threshold_names]
            + [f"common: {feature}" for feature in common_features]
            + [
                f"varying {split}: {feature}"
                for split in threshold_names
                for feature in self.varying
            ]
        )
        covariance, standard_errors, zstats, pvalues, inference_valid = _interior_inference(
            objective,
            fitted.x,
            parameter_names,
            constraint_slack=minimum_fitted_gap - minimum_gap,
        )
        return PartialProportionalOddsResult(
            thresholds=pd.Series(thresholds, index=threshold_names, name="threshold"),
            common_params=pd.Series(common, index=common_features, name="coefficient"),
            varying_params=pd.DataFrame(
                slopes[:, varying_indices], index=threshold_names, columns=self.varying
            ),
            threshold_slopes=pd.DataFrame(
                slopes, index=threshold_names, columns=feature_names
            ),
            covariance=covariance,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            inference_valid=inference_valid,
            categories=categories,
            converged=bool(fitted.success and minimum_fitted_gap >= minimum_gap - 1e-7),
            loglike=float(-fitted.fun),
            nobs=values.shape[0],
            feature_names=tuple(feature_names),
            varying_features=self.varying,
            minimum_index_gap=minimum_fitted_gap,
            constraint_slack=minimum_fitted_gap - minimum_gap,
            optimizer_result=fitted,
        )
