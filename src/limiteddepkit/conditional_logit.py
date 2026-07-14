"""Experimental conditional-logit estimator for grouped choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from .ordinal import _as_2d_array


def _one_dimensional(values: Any, name: str, n_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(array) != n_rows:
        raise ValueError(f"X and {name} must contain the same number of rows.")
    if pd.isna(array).any():
        raise ValueError(f"{name} contains missing values.")
    return array


def _factorize_groups(
    n_rows: int,
    groups: Any | None,
    n_alts: int | None,
    *,
    enforce_n_alts: bool,
) -> tuple[np.ndarray, tuple[Any, ...], list[np.ndarray]]:
    if groups is None:
        if n_alts is None:
            raise ValueError("groups is required when n_alts was not specified.")
        if n_rows % n_alts != 0:
            raise ValueError("X rows must be divisible by n_alts when groups is omitted.")
        n_groups = n_rows // n_alts
        codes = np.repeat(np.arange(n_groups), n_alts)
        labels: tuple[Any, ...] = tuple(range(n_groups))
    else:
        values = _one_dimensional(groups, "groups", n_rows)
        for value in values:
            try:
                hash(value)
            except TypeError as error:
                raise ValueError("groups must contain hashable scalar labels.") from error
        codes, unique = pd.factorize(pd.Series(values, dtype="object"), sort=False)
        labels = tuple(unique.tolist())

    indices = [np.flatnonzero(codes == code) for code in range(len(labels))]
    if not indices or any(len(index) < 2 for index in indices):
        raise ValueError("Every choice set must contain at least two alternatives.")
    if enforce_n_alts and n_alts is not None and any(len(index) != n_alts for index in indices):
        raise ValueError("Every choice set must contain exactly n_alts rows.")
    return codes, labels, indices


def _alternative_values(
    alternatives: Any | None, n_rows: int, group_indices: list[np.ndarray]
) -> np.ndarray:
    if alternatives is None:
        values = np.empty(n_rows, dtype=object)
        for index in group_indices:
            values[index] = np.arange(len(index))
        return values

    values = _one_dimensional(alternatives, "alternatives", n_rows)
    for value in values:
        try:
            hash(value)
        except TypeError as error:
            raise ValueError("alternatives must contain hashable scalar labels.") from error
    for index in group_indices:
        if len(set(values[index])) != len(index):
            raise ValueError("Alternative labels must be unique within each choice set.")
    return values


@dataclass(frozen=True)
class ConditionalLogitResult:
    """Fitted experimental conditional-logit result.

    ``nobs`` and ``n_choice_sets`` count independent choice sets; ``n_rows``
    counts the long-format alternative rows used by the conditional likelihood.
    """

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    loglike: float
    nobs: int
    n_rows: int
    n_choice_sets: int
    n_alts: int | None
    feature_names: tuple[str, ...]
    information_rank: int
    inference_valid: bool
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return len(self.params)

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
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.standard_errors,
                "z": self.zstats,
                "p_value": self.pvalues,
            }
        )

    def _prediction_inputs(
        self,
        X: Any,
        groups: Any | None,
        alternatives: Any | None,
    ) -> tuple[np.ndarray, pd.Index, tuple[Any, ...], list[np.ndarray], np.ndarray]:
        design, names = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {design.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        row_index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
        _, group_labels, group_indices = _factorize_groups(
            len(design), groups, self.n_alts, enforce_n_alts=False
        )
        alternative_labels = _alternative_values(alternatives, len(design), group_indices)
        return design, row_index, group_labels, group_indices, alternative_labels

    def predict_proba(
        self, X: Any, *, groups: Any | None = None
    ) -> pd.Series:
        design, row_index, _, group_indices, _ = self._prediction_inputs(X, groups, None)
        utilities = design @ self.params.to_numpy(dtype=float)
        probabilities = np.empty(len(design), dtype=float)
        for index in group_indices:
            probabilities[index] = np.exp(utilities[index] - logsumexp(utilities[index]))
        return pd.Series(probabilities, index=row_index, name="probability")

    def predict(
        self,
        X: Any,
        *,
        groups: Any | None = None,
        alternatives: Any | None = None,
    ) -> pd.Series:
        design, _, group_labels, group_indices, alternative_labels = self._prediction_inputs(
            X, groups, alternatives
        )
        utilities = design @ self.params.to_numpy(dtype=float)
        selected = [
            alternative_labels[index[np.argmax(utilities[index])]] for index in group_indices
        ]
        return pd.Series(selected, index=pd.Index(group_labels, name="group"), name="predicted")


class ConditionalLogit:
    """Experimental conditional logit for long-format grouped choices.

    When ``groups`` is omitted, rows are interpreted as consecutive choice sets
    of size ``n_alts``. Supplying ``groups`` permits arbitrary group labels and,
    when ``n_alts`` is ``None``, unequal choice-set sizes.
    """

    def __init__(self, n_alts: int | None = None) -> None:
        if n_alts is not None and (
            isinstance(n_alts, bool) or not isinstance(n_alts, int) or n_alts < 2
        ):
            raise ValueError("n_alts must be an integer of at least two or None.")
        self.n_alts = n_alts

    def fit(
        self,
        X: Any,
        choice: Any,
        *,
        groups: Any | None = None,
        alternatives: Any | None = None,
        maxiter: int = 300,
    ) -> ConditionalLogitResult:
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        design, feature_names = _as_2d_array(X)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X must contain unique feature names.")
        raw_choice = _one_dimensional(choice, "choice", len(design))
        if not np.isin(raw_choice, [0, 1, False, True]).all():
            raise ValueError("choice must contain binary indicators coded as 0/1.")
        choices = raw_choice.astype(int)
        _, group_labels, group_indices = _factorize_groups(
            len(design), groups, self.n_alts, enforce_n_alts=True
        )
        _alternative_values(alternatives, len(design), group_indices)
        if any(int(np.sum(choices[index])) != 1 for index in group_indices):
            raise ValueError("Every choice set must contain exactly one chosen alternative.")

        within_design = design.copy()
        for index in group_indices:
            within_design[index] -= design[index].mean(axis=0)
        within_rank = int(np.linalg.matrix_rank(within_design))
        if within_rank < design.shape[1]:
            raise ValueError(
                "X is not identified after removing choice-set effects; remove constants "
                "or other within-set invariant/collinear regressors."
            )

        def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            utilities = design @ parameters
            negative_loglike = 0.0
            gradient = np.zeros(design.shape[1], dtype=float)
            for index in group_indices:
                group_utilities = utilities[index]
                log_denominator = logsumexp(group_utilities)
                chosen = int(np.flatnonzero(choices[index])[0])
                negative_loglike += log_denominator - group_utilities[chosen]
                probabilities = np.exp(group_utilities - log_denominator)
                residuals = probabilities - choices[index]
                gradient += design[index].T @ residuals
            return float(negative_loglike), gradient

        initial = np.zeros(design.shape[1], dtype=float)
        optimizer_result = minimize(
            objective_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        parameters = np.asarray(optimizer_result.x, dtype=float)
        utilities = design @ parameters
        information = np.zeros((design.shape[1], design.shape[1]), dtype=float)
        for index in group_indices:
            probabilities = np.exp(utilities[index] - logsumexp(utilities[index]))
            probability_covariance = np.diag(probabilities) - np.outer(
                probabilities, probabilities
            )
            information += design[index].T @ probability_covariance @ design[index]
        information = (information + information.T) / 2.0
        information_rank = int(np.linalg.matrix_rank(information))
        inference_valid = bool(
            optimizer_result.success and information_rank == information.shape[0]
        )
        if inference_valid:
            covariance_values = np.linalg.pinv(information)
            covariance_values = (covariance_values + covariance_values.T) / 2.0
            standard_error_values = np.sqrt(np.clip(np.diag(covariance_values), 0.0, None))
            z_values = np.divide(
                parameters,
                standard_error_values,
                out=np.full_like(parameters, np.nan),
                where=standard_error_values > 0,
            )
            p_values = 2.0 * norm.sf(np.abs(z_values))
        else:
            covariance_values = np.full((len(parameters), len(parameters)), np.nan)
            standard_error_values = np.full(len(parameters), np.nan)
            z_values = np.full(len(parameters), np.nan)
            p_values = np.full(len(parameters), np.nan)

        params = pd.Series(parameters, index=feature_names, name="estimate")
        covariance = pd.DataFrame(
            covariance_values, index=feature_names, columns=feature_names
        )
        standard_errors = pd.Series(
            standard_error_values, index=feature_names, name="std_err"
        )
        zstats = pd.Series(z_values, index=feature_names, name="z")
        pvalues = pd.Series(p_values, index=feature_names, name="p_value")
        group_sizes = {len(index) for index in group_indices}
        fitted_n_alts = next(iter(group_sizes)) if len(group_sizes) == 1 else None

        return ConditionalLogitResult(
            params=params,
            covariance=covariance,
            standard_errors=standard_errors,
            zstats=zstats,
            pvalues=pvalues,
            converged=bool(optimizer_result.success),
            loglike=-float(optimizer_result.fun),
            nobs=len(group_labels),
            n_rows=len(design),
            n_choice_sets=len(group_labels),
            n_alts=fitted_n_alts,
            feature_names=tuple(feature_names),
            information_rank=information_rank,
            inference_valid=inference_valid,
            optimizer_result=optimizer_result,
        )
