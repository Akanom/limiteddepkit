"""Experimental sequential (continuation-ratio) logit estimator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from .ordinal import _as_2d_array


def _encode_ordered_categories(
    y: Any, category_order: Sequence[Any] | None
) -> tuple[np.ndarray, tuple[Any, ...]]:
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

    observed = list(pd.unique(pd.Series(values, dtype="object")))
    if category_order is not None:
        if isinstance(category_order, (str, bytes)):
            raise ValueError("category_order must be a sequence of category labels.")
        categories = list(category_order)
    elif isinstance(y, pd.Series) and isinstance(y.dtype, pd.CategoricalDtype):
        if not y.dtype.ordered:
            raise ValueError(
                "Categorical y must be ordered or category_order must be supplied."
            )
        categories = list(y.cat.categories)
    elif isinstance(y, (pd.Categorical, pd.CategoricalIndex)):
        if not y.ordered:
            raise ValueError(
                "Categorical y must be ordered or category_order must be supplied."
            )
        categories = list(y.categories)
    else:
        try:
            categories = sorted(observed)
        except TypeError as error:
            raise ValueError(
                "Outcome labels cannot be ordered safely; supply category_order."
            ) from error

    if len(categories) < 2:
        raise ValueError("y must contain at least two distinct categories.")
    if pd.isna(np.asarray(categories, dtype=object)).any():
        raise ValueError("category_order must not contain missing labels.")
    for category in categories:
        try:
            hash(category)
        except TypeError as error:
            raise ValueError("category_order must contain hashable scalar labels.") from error
    if len(set(categories)) != len(categories):
        raise ValueError("category_order must contain unique labels.")
    if set(categories) != set(observed):
        raise ValueError("category_order must contain each observed category exactly once.")

    positions = {category: index for index, category in enumerate(categories)}
    encoded = np.fromiter((positions[value] for value in values), dtype=int)
    return encoded, tuple(categories)


def _parameter_index(
    stages: Sequence[Any], feature_names: Sequence[str]
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [list(stages), list(feature_names)], names=["stage", "feature"]
    )


@dataclass(frozen=True)
class SequentialLogitResult:
    """Fitted experimental sequential-logit result.

    For each category except the last, a stage models the conditional probability
    of stopping in that category given that the observation reached the stage.
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
    stage_sample_sizes: pd.Series
    information_rank: int
    inference_valid: bool
    optimizer_result: Any

    @property
    def stage_categories(self) -> tuple[Any, ...]:
        return self.categories[:-1]

    @property
    def all_params(self) -> pd.Series:
        values = self.params.loc[:, list(self.stage_categories)].to_numpy(dtype=float).T.ravel()
        return pd.Series(values, index=self.covariance.index, name="estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def params_upper(self) -> pd.Series:
        """Compatibility view for the legacy three-category first-stage parameter."""
        return (-self.params.iloc[:, 0]).rename("upper")

    @property
    def params_lower(self) -> pd.Series:
        """Compatibility view for the legacy three-category second stage."""
        if len(self.categories) != 3:
            raise ValueError("params_lower is available only for three-category fits.")
        return self.params.iloc[:, 1].rename("lower")

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
        stop_probabilities = expit(
            design @ self.params.loc[:, list(self.stage_categories)].to_numpy(dtype=float)
        )
        probabilities = np.empty((len(design), len(self.categories)), dtype=float)
        remaining = np.ones(len(design), dtype=float)
        for stage in range(len(self.stage_categories)):
            probabilities[:, stage] = remaining * stop_probabilities[:, stage]
            remaining *= 1.0 - stop_probabilities[:, stage]
        probabilities[:, -1] = remaining
        return pd.DataFrame(probabilities, index=index, columns=list(self.categories))

    def predict(self, X: Any) -> pd.Series:
        probabilities = self.predict_proba(X)
        categories = np.asarray(self.categories, dtype=object)
        predicted = categories[np.argmax(probabilities.to_numpy(dtype=float), axis=1)]
        return pd.Series(predicted, index=probabilities.index, name="predicted")


class SequentialLogit:
    """Experimental sequential logit for two or more ordered categories."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        category_order: Sequence[Any] | None = None,
        maxiter: int = 300,
    ) -> SequentialLogitResult:
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        design, feature_names = _as_2d_array(X)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X must contain unique feature names.")
        encoded, categories = _encode_ordered_categories(y, category_order)
        if encoded.size != design.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")

        nobs, n_features = design.shape
        stage_categories = categories[:-1]
        n_stages = len(stage_categories)
        risk_sets: list[np.ndarray] = []
        stop_indicators: list[np.ndarray] = []
        for stage in range(n_stages):
            risk = np.flatnonzero(encoded >= stage)
            stop = (encoded[risk] == stage).astype(float)
            if np.linalg.matrix_rank(design[risk]) < n_features:
                raise ValueError(
                    f"X must have full column rank in the risk set for stage {stage_categories[stage]!r}."
                )
            if np.all(stop == stop[0]):
                raise ValueError(
                    f"Stage {stage_categories[stage]!r} must contain both stopping and continuing outcomes."
                )
            risk_sets.append(risk)
            stop_indicators.append(stop)

        def objective_and_gradient(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = raw_params.reshape(n_stages, n_features)
            negative_loglike = 0.0
            gradient = np.empty_like(coefficients)
            for stage, risk in enumerate(risk_sets):
                linear = design[risk] @ coefficients[stage]
                stop = stop_indicators[stage]
                negative_loglike += float(np.sum(np.logaddexp(0.0, linear) - stop * linear))
                gradient[stage] = design[risk].T @ (expit(linear) - stop)
            return negative_loglike, gradient.ravel()

        initial = np.zeros(n_stages * n_features, dtype=float)
        optimizer_result = minimize(
            objective_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        coefficients = np.asarray(optimizer_result.x, dtype=float).reshape(
            n_stages, n_features
        )
        information = np.zeros(
            (n_stages * n_features, n_stages * n_features), dtype=float
        )
        for stage, risk in enumerate(risk_sets):
            linear = design[risk] @ coefficients[stage]
            probabilities = expit(linear)
            weights = probabilities * (1.0 - probabilities)
            block = design[risk].T @ (weights[:, None] * design[risk])
            stage_slice = slice(stage * n_features, (stage + 1) * n_features)
            information[stage_slice, stage_slice] = block
        information = (information + information.T) / 2.0
        information_rank = int(np.linalg.matrix_rank(information))
        inference_valid = bool(
            optimizer_result.success and information_rank == information.shape[0]
        )
        parameter_index = _parameter_index(stage_categories, feature_names)
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
            columns=pd.Index(stage_categories, name="stage"),
        )
        covariance = pd.DataFrame(
            covariance_values, index=parameter_index, columns=parameter_index
        )
        standard_errors = pd.Series(
            standard_error_values, index=parameter_index, name="std_err"
        )
        zstats = pd.Series(z_values, index=parameter_index, name="z")
        pvalues = pd.Series(p_values, index=parameter_index, name="p_value")
        stage_sample_sizes = pd.Series(
            [len(risk) for risk in risk_sets],
            index=pd.Index(stage_categories, name="stage"),
            name="nobs",
        )

        return SequentialLogitResult(
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
            stage_sample_sizes=stage_sample_sizes,
            information_rank=information_rank,
            inference_valid=inference_valid,
            optimizer_result=optimizer_result,
        )
