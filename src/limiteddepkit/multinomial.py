"""Experimental multinomial-logit maximum-likelihood estimator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from .ordinal import _as_2d_array


def _response_values(y: Any) -> np.ndarray:
    """Return a validated one-dimensional object array of outcome labels."""
    if isinstance(y, (pd.Series, pd.Index, pd.Categorical)):
        values = np.asarray(y, dtype=object)
    else:
        values = np.asarray(y, dtype=object)
    if values.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if pd.isna(values).any():
        raise ValueError("y contains missing values.")
    for value in values:
        try:
            hash(value)
        except TypeError as error:
            raise ValueError("Outcome categories must be hashable scalar labels.") from error
    return values


def _encode_categories(
    y: Any, category_order: Sequence[Any] | None
) -> tuple[np.ndarray, tuple[Any, ...]]:
    values = _response_values(y)
    observed = list(pd.unique(pd.Series(values, dtype="object")))

    if category_order is None:
        try:
            categories = sorted(observed)
        except TypeError:
            # Heterogeneous but hashable labels have no safe natural ordering.
            # Preserve first-observed order unless the caller supplies one.
            categories = observed
    else:
        if isinstance(category_order, (str, bytes)):
            raise ValueError("category_order must be a sequence of category labels.")
        categories = list(category_order)
        if not categories:
            raise ValueError("category_order must contain at least two categories.")
        if pd.isna(np.asarray(categories, dtype=object)).any():
            raise ValueError("category_order must not contain missing labels.")
        for category in categories:
            try:
                hash(category)
            except TypeError as error:
                raise ValueError(
                    "category_order must contain hashable scalar labels."
                ) from error
        if len(set(categories)) != len(categories):
            raise ValueError("category_order must contain unique labels.")
        if set(categories) != set(observed):
            raise ValueError(
                "category_order must contain each observed category exactly once."
            )

    if len(categories) < 2:
        raise ValueError("y must contain at least two distinct categories.")
    positions = {category: index for index, category in enumerate(categories)}
    encoded = np.fromiter((positions[value] for value in values), dtype=int)
    return encoded, tuple(categories)


def _validate_design(X: Any) -> tuple[np.ndarray, list[str]]:
    design, feature_names = _as_2d_array(X)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("X must contain unique feature names.")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("X must have full column rank for multinomial-logit identification.")
    return design, feature_names


def _parameter_index(
    categories: Sequence[Any], feature_names: Sequence[str]
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [list(categories), list(feature_names)], names=["category", "feature"]
    )


@dataclass(frozen=True)
class MultinomialLogitResult:
    """Fitted experimental multinomial-logit result.

    Coefficients are identified relative to ``base_category``. ``params`` uses
    features as rows and non-base categories as columns; ``all_params`` flattens
    the same estimates in category-major order for covariance operations.
    """

    params: pd.DataFrame
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    categories: tuple[Any, ...]
    base_category: Any
    information_rank: int
    inference_valid: bool
    optimizer_result: Any

    @property
    def categories_list(self) -> list[Any]:
        return list(self.categories)

    @property
    def nonbase_categories(self) -> tuple[Any, ...]:
        return tuple(category for category in self.categories if category != self.base_category)

    @property
    def all_params(self) -> pd.Series:
        values = self.params.loc[:, list(self.nonbase_categories)].to_numpy(dtype=float).T.ravel()
        return pd.Series(values, index=self.covariance.index, name="estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def backend(self) -> str:
        return "native-mle-experimental"

    @property
    def covariance_type(self) -> str:
        return "observed-information"

    @property
    def aic(self) -> float:
        return float(2 * self.n_params - 2 * self.loglike)

    @property
    def bic(self) -> float:
        return float(np.log(self.nobs) * self.n_params - 2 * self.loglike)

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

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

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.all_params,
                "std_err": self.standard_errors,
                "z": self.zstats,
                "p_value": self.pvalues,
            }
        )

    def _prediction_design(self, X: Any) -> tuple[np.ndarray, pd.Index]:
        design, names = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {design.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
        return design, index

    def predict_proba(self, X: Any) -> pd.DataFrame:
        design, index = self._prediction_design(X)
        nonbase = self.nonbase_categories
        nonbase_utilities = design @ self.params.loc[:, list(nonbase)].to_numpy(dtype=float)
        utilities = np.zeros((len(design), len(self.categories)), dtype=float)
        category_positions = {category: position for position, category in enumerate(self.categories)}
        for column, category in enumerate(nonbase):
            utilities[:, category_positions[category]] = nonbase_utilities[:, column]
        log_denominator = logsumexp(utilities, axis=1, keepdims=True)
        probabilities = np.exp(utilities - log_denominator)
        return pd.DataFrame(probabilities, index=index, columns=list(self.categories))

    def predict(self, X: Any) -> pd.Series:
        probabilities = self.predict_proba(X)
        categories = np.asarray(self.categories, dtype=object)
        predicted = categories[np.argmax(probabilities.to_numpy(dtype=float), axis=1)]
        return pd.Series(predicted, index=probabilities.index, name="predicted")


class MultinomialLogit:
    """Experimental baseline-category multinomial logit estimated by MLE."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        category_order: Sequence[Any] | None = None,
        base_category: Any | None = None,
        maxiter: int = 300,
    ) -> MultinomialLogitResult:
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        design, feature_names = _validate_design(X)
        encoded, categories = _encode_categories(y, category_order)
        if encoded.size != design.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")

        if base_category is None:
            selected_base = categories[0]
        else:
            if base_category not in categories:
                raise ValueError("base_category must be one of the observed categories.")
            selected_base = base_category

        nonbase_categories = tuple(
            category for category in categories if category != selected_base
        )
        category_positions = {category: position for position, category in enumerate(categories)}
        nonbase_positions = np.array(
            [category_positions[category] for category in nonbase_categories], dtype=int
        )
        nobs, n_features = design.shape
        n_nonbase = len(nonbase_categories)
        chosen_rows = np.arange(nobs)

        def objective_and_gradient(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = raw_params.reshape(n_nonbase, n_features)
            nonbase_utilities = design @ coefficients.T
            utilities = np.zeros((nobs, len(categories)), dtype=float)
            utilities[:, nonbase_positions] = nonbase_utilities
            log_denominator = logsumexp(utilities, axis=1)
            negative_loglike = float(
                np.sum(log_denominator - utilities[chosen_rows, encoded])
            )
            probabilities = np.exp(utilities - log_denominator[:, None])
            residuals = probabilities[:, nonbase_positions]
            residuals = residuals.copy()
            for column, position in enumerate(nonbase_positions):
                residuals[:, column] -= encoded == position
            gradient = residuals.T @ design
            return negative_loglike, gradient.ravel()

        initial = np.zeros(n_nonbase * n_features, dtype=float)
        optimizer_result = minimize(
            objective_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        coefficients = np.asarray(optimizer_result.x, dtype=float).reshape(
            n_nonbase, n_features
        )

        nonbase_utilities = design @ coefficients.T
        utilities = np.zeros((nobs, len(categories)), dtype=float)
        utilities[:, nonbase_positions] = nonbase_utilities
        probabilities = np.exp(utilities - logsumexp(utilities, axis=1, keepdims=True))
        information = np.empty(
            (n_nonbase * n_features, n_nonbase * n_features), dtype=float
        )
        for row in range(n_nonbase):
            row_slice = slice(row * n_features, (row + 1) * n_features)
            row_probability = probabilities[:, nonbase_positions[row]]
            for column in range(n_nonbase):
                column_slice = slice(column * n_features, (column + 1) * n_features)
                column_probability = probabilities[:, nonbase_positions[column]]
                weights = row_probability * (
                    float(row == column) - column_probability
                )
                information[row_slice, column_slice] = design.T @ (
                    weights[:, None] * design
                )
        information = (information + information.T) / 2.0
        information_rank = int(np.linalg.matrix_rank(information))
        inference_valid = bool(
            optimizer_result.success and information_rank == information.shape[0]
        )
        parameter_index = _parameter_index(nonbase_categories, feature_names)
        if inference_valid:
            covariance_values = np.linalg.pinv(information)
            covariance_values = (covariance_values + covariance_values.T) / 2.0
            standard_error_values = np.sqrt(np.clip(np.diag(covariance_values), 0.0, None))
            flat_parameters = coefficients.ravel()
            z_values = np.divide(
                flat_parameters,
                standard_error_values,
                out=np.full_like(flat_parameters, np.nan),
                where=standard_error_values > 0,
            )
            p_values = 2.0 * norm.sf(np.abs(z_values))
        else:
            covariance_values = np.full((len(parameter_index), len(parameter_index)), np.nan)
            standard_error_values = np.full(len(parameter_index), np.nan)
            z_values = np.full(len(parameter_index), np.nan)
            p_values = np.full(len(parameter_index), np.nan)

        params = pd.DataFrame(
            coefficients.T,
            index=feature_names,
            columns=pd.Index(nonbase_categories, name="category"),
        )
        covariance = pd.DataFrame(
            covariance_values, index=parameter_index, columns=parameter_index
        )
        standard_errors = pd.Series(
            standard_error_values, index=parameter_index, name="std_err"
        )
        zstats = pd.Series(z_values, index=parameter_index, name="z")
        pvalues = pd.Series(p_values, index=parameter_index, name="p_value")

        return MultinomialLogitResult(
            params=params,
            covariance=covariance,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            converged=bool(optimizer_result.success),
            loglike=-float(optimizer_result.fun),
            nobs=int(nobs),
            feature_names=tuple(feature_names),
            categories=categories,
            base_category=selected_base,
            information_rank=information_rank,
            inference_valid=inference_valid,
            optimizer_result=optimizer_result,
        )
