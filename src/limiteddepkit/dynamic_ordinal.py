"""Dynamic panel ordinal-response estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .ordinal import _as_2d_array, _ordered_categories
from .panel_ordinal import RandomEffectsOrderedLogit, RandomEffectsOrderedLogitResult


def _category_parameter_names(prefix: str, categories: np.ndarray) -> list[str]:
    return [f"{prefix}[{category}]" for category in categories[1:]]


@dataclass(frozen=True)
class DynamicRandomEffectsOrderedLogitResult:
    """Dynamic RE Ordered Logit with initial-conditions controls."""

    base_result: RandomEffectsOrderedLogitResult
    original_feature_names: tuple[str, ...]
    state_parameter_names: tuple[str, ...]
    initial_parameter_names: tuple[str, ...]
    initial_covariate_parameter_names: tuple[str, ...]
    mean_parameter_names: tuple[str, ...]
    initial_outcomes: pd.Series
    initial_covariates: pd.DataFrame
    entity_means: pd.DataFrame
    estimation_design: pd.DataFrame
    estimation_outcome: pd.Series
    estimation_entity: pd.Series
    estimation_index: pd.Index
    n_original_obs: int
    dropped_initial: int
    dropped_nonconsecutive: int

    @property
    def params(self) -> pd.Series:
        return self.base_result.params

    @property
    def thresholds(self) -> pd.Series:
        return self.base_result.thresholds

    @property
    def sigma_entity(self) -> float:
        return self.base_result.sigma_entity

    @property
    def all_params(self) -> pd.Series:
        return self.base_result.all_params

    @property
    def covariance(self) -> pd.DataFrame:
        return self.base_result.covariance

    @property
    def standard_errors(self) -> pd.Series:
        return self.base_result.standard_errors

    @property
    def zstats(self) -> pd.Series:
        return self.base_result.zstats

    @property
    def pvalues(self) -> pd.Series:
        return self.base_result.pvalues

    @property
    def categories(self) -> np.ndarray:
        return self.base_result.categories

    @property
    def converged(self) -> bool:
        return self.base_result.converged

    @property
    def inference_valid(self) -> bool:
        return self.base_result.inference_valid

    @property
    def loglike(self) -> float:
        return self.base_result.loglike

    @property
    def nobs(self) -> int:
        return self.base_result.nobs

    @property
    def n_groups(self) -> int:
        return self.base_result.n_groups

    @property
    def n_entities(self) -> int:
        return self.base_result.n_entities

    @property
    def n_params(self) -> int:
        return self.base_result.n_params

    @property
    def quadrature_points(self) -> int:
        """Number of Gaussian-Hermite nodes used by the fitted likelihood."""
        return self.base_result.quadrature_points

    @property
    def n_quadrature_points(self) -> int:
        """Alias matching the static random-effects result contract."""
        return self.base_result.n_quadrature_points

    @property
    def backend(self) -> str:
        return "native-dynamic-ghq"

    @property
    def covariance_type(self) -> str:
        return self.base_result.covariance_type

    @property
    def optimizer_result(self) -> Any:
        return self.base_result.optimizer_result

    @property
    def structural_params(self) -> pd.Series:
        return self.params.reindex(self.original_feature_names)

    @property
    def state_dependence_params(self) -> pd.Series:
        return self.params.reindex(self.state_parameter_names)

    @property
    def initial_condition_params(self) -> pd.Series:
        return self.params.reindex(self.initial_parameter_names)

    @property
    def initial_covariate_params(self) -> pd.Series:
        return self.params.reindex(self.initial_covariate_parameter_names)

    @property
    def correlated_effects_params(self) -> pd.Series:
        return self.params.reindex(self.mean_parameter_names)

    @property
    def fitted_probabilities(self) -> pd.DataFrame:
        probabilities = self.base_result.predict_proba(self.estimation_design)
        probabilities.index = self.estimation_index
        return probabilities

    def vcov(self) -> pd.DataFrame:
        return self.base_result.vcov()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        return self.base_result.conf_int(level=level)

    def summary_frame(self) -> pd.DataFrame:
        return self.base_result.summary_frame()

    def posterior_random_effects(self) -> pd.DataFrame:
        """Return posterior effects for the dynamic estimation sample."""
        return self.base_result.posterior_random_effects(
            self.estimation_design,
            self.estimation_outcome,
            entity=self.estimation_entity,
        )

    def _prediction_design(
        self,
        X: Any,
        *,
        entity: Any,
        lagged_y: Any,
        initial_y: Any | None,
        initial_covariates: pd.DataFrame | None,
        entity_means: pd.DataFrame | None,
    ) -> pd.DataFrame:
        values, names = _as_2d_array(X)
        if tuple(names) != self.original_feature_names:
            raise ValueError("X columns must match the original fitted features and order.")
        entities = np.asarray(entity)
        lagged = np.asarray(lagged_y)
        if entities.shape != (values.shape[0],) or lagged.shape != (values.shape[0],):
            raise ValueError("entity and lagged_y must contain one value per prediction row.")

        if initial_y is None:
            missing = [label for label in pd.unique(entities) if label not in self.initial_outcomes.index]
            if missing:
                raise ValueError(f"Initial outcomes are unavailable for entities: {missing}.")
            initial = np.array([self.initial_outcomes.loc[label] for label in entities])
        elif isinstance(initial_y, Mapping):
            missing = [label for label in pd.unique(entities) if label not in initial_y]
            if missing:
                raise ValueError(f"Initial outcomes are unavailable for entities: {missing}.")
            initial = np.array([initial_y[label] for label in entities])
        else:
            initial = np.asarray(initial_y)
            if initial.shape != (values.shape[0],):
                raise ValueError("initial_y must contain one value per row or be entity-keyed.")

        means_source = self.entity_means if entity_means is None else entity_means
        if not isinstance(means_source, pd.DataFrame):
            raise ValueError("entity_means must be a DataFrame indexed by entity.")
        if list(means_source.columns) != list(self.original_feature_names):
            raise ValueError("entity_means columns must match the original fitted features.")
        missing_means = [label for label in pd.unique(entities) if label not in means_source.index]
        if missing_means:
            raise ValueError(f"Entity means are unavailable for entities: {missing_means}.")

        allowed = set(self.categories)
        if any(value not in allowed for value in lagged):
            raise ValueError("lagged_y contains categories not present in the fitted result.")
        if any(value not in allowed for value in initial):
            raise ValueError("initial_y contains categories not present in the fitted result.")

        initial_X_source = (
            self.initial_covariates if initial_covariates is None else initial_covariates
        )
        if not isinstance(initial_X_source, pd.DataFrame):
            raise ValueError("initial_covariates must be a DataFrame indexed by entity.")
        if list(initial_X_source.columns) != list(self.original_feature_names):
            raise ValueError(
                "initial_covariates columns must match the original fitted features."
            )
        missing_initial_X = [
            label for label in pd.unique(entities) if label not in initial_X_source.index
        ]
        if missing_initial_X:
            raise ValueError(
                f"Initial covariates are unavailable for entities: {missing_initial_X}."
            )

        design = pd.DataFrame(values, columns=self.original_feature_names)
        for name, category in zip(
            self.state_parameter_names, self.categories[1:], strict=True
        ):
            design[name] = (lagged == category).astype(float)
        for name, category in zip(
            self.initial_parameter_names, self.categories[1:], strict=True
        ):
            design[name] = (initial == category).astype(float)
        for name, feature in zip(
            self.initial_covariate_parameter_names,
            self.original_feature_names,
            strict=True,
        ):
            design[name] = [initial_X_source.loc[label, feature] for label in entities]
        for name, feature in zip(
            self.mean_parameter_names, self.original_feature_names, strict=True
        ):
            design[name] = [means_source.loc[label, feature] for label in entities]
        return design

    def predict_proba(
        self,
        X: Any,
        *,
        entity: Any,
        lagged_y: Any,
        initial_y: Any | None = None,
        initial_covariates: pd.DataFrame | None = None,
        entity_means: pd.DataFrame | None = None,
        random_effects: Any | None = None,
    ) -> pd.DataFrame:
        """Predict using explicit lagged outcomes and initial-condition controls."""
        design = self._prediction_design(
            X,
            entity=entity,
            lagged_y=lagged_y,
            initial_y=initial_y,
            initial_covariates=initial_covariates,
            entity_means=entity_means,
        )
        return self.base_result.predict_proba(
            design, random_effects=random_effects, entity=entity
        )

    def predict(self, X: Any, **kwargs: Any) -> pd.Series:
        probabilities = self.predict_proba(X, **kwargs).to_numpy()
        return pd.Series(self.categories[np.argmax(probabilities, axis=1)], name="prediction")

    def posterior_predict_proba(
        self,
        X: Any,
        *,
        entity: Any,
        lagged_y: Any,
        posterior: pd.DataFrame,
        initial_y: Any | None = None,
        initial_covariates: pd.DataFrame | None = None,
        entity_means: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return posterior-predictive probabilities for observed entities."""
        design = self._prediction_design(
            X,
            entity=entity,
            lagged_y=lagged_y,
            initial_y=initial_y,
            initial_covariates=initial_covariates,
            entity_means=entity_means,
        )
        return self.base_result.posterior_predict_proba(
            design, entity=entity, posterior=posterior
        )


class DynamicRandomEffectsOrderedLogit:
    """Dynamic RE Ordered Logit with conditional initial-conditions controls."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any,
        time: Any,
        category_order: Sequence[Any] | None = None,
        quadrature_points: int = 12,
        time_step: float = 1.0,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> DynamicRandomEffectsOrderedLogitResult:
        values, feature_names = _as_2d_array(X)
        _, categories = _ordered_categories(y, category_order=category_order)
        outcomes = np.asarray(y)
        entities = np.asarray(entity)
        times = np.asarray(time)
        nobs = values.shape[0]
        if entities.shape != (nobs,) or times.shape != (nobs,) or outcomes.shape != (nobs,):
            raise ValueError("X, y, entity, and time must contain the same observations.")
        if pd.isna(entities).any() or pd.isna(times).any():
            raise ValueError("entity and time must not contain missing values.")
        if not np.issubdtype(times.dtype, np.number):
            raise ValueError("time must be numeric for exact one-period lag validation.")
        numeric_times = times.astype(float)
        if not np.isfinite(numeric_times).all():
            raise ValueError("time must contain only finite values.")
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive.")

        original_index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(nobs)
        work = pd.DataFrame(values, columns=feature_names)
        work["__outcome"] = outcomes
        work["__entity"] = entities
        work["__time"] = times
        work["__position"] = np.arange(nobs)
        work["__index"] = list(original_index)
        if work.duplicated(["__entity", "__time"]).any():
            raise ValueError("Duplicate entity-time observations are not permitted.")
        work = work.sort_values(["__entity", "__time"], kind="stable")

        grouped = work.groupby("__entity", sort=False)
        work["__lagged_outcome"] = grouped["__outcome"].shift(1)
        work["__lagged_time"] = grouped["__time"].shift(1)
        work["__initial_outcome"] = grouped["__outcome"].transform("first")
        initial_outcomes = grouped["__outcome"].first()
        initial_covariates = grouped[feature_names].first()
        has_predecessor = work["__lagged_time"].notna()
        time_differences = work["__time"] - work["__lagged_time"]
        gap = has_predecessor & ~np.isclose(
            time_differences.astype(float), time_step, rtol=0.0, atol=1e-10
        )
        at_or_after_gap = gap.groupby(work["__entity"], sort=False).cummax()
        consecutive = has_predecessor & ~at_or_after_gap
        used = work.loc[consecutive].copy()
        if used.empty:
            raise ValueError("No consecutive within-entity observations are available.")
        if used["__entity"].nunique() != initial_outcomes.size:
            raise ValueError(
                "Every entity must have at least one consecutive post-initial observation."
            )
        entity_means = used.groupby("__entity", sort=False)[feature_names].mean()

        state_names = _category_parameter_names("state", categories)
        initial_names = _category_parameter_names("initial", categories)
        initial_covariate_names = [f"initial_x[{feature}]" for feature in feature_names]
        mean_names = [f"mean[{feature}]" for feature in feature_names]
        generated_names = state_names + initial_names + initial_covariate_names + mean_names
        conflicts = set(feature_names) & set(generated_names)
        if conflicts:
            raise ValueError(f"Generated dynamic feature names conflict with X: {sorted(conflicts)}.")

        design = used[feature_names].reset_index(drop=True)
        lagged = used["__lagged_outcome"].to_numpy()
        initial = used["__initial_outcome"].to_numpy()
        used_entities = used["__entity"].to_numpy()
        for name, category in zip(state_names, categories[1:], strict=True):
            design[name] = (lagged == category).astype(float)
        for name, category in zip(initial_names, categories[1:], strict=True):
            design[name] = (initial == category).astype(float)
        for name, feature in zip(initial_covariate_names, feature_names, strict=True):
            design[name] = [initial_covariates.loc[label, feature] for label in used_entities]
        for name, feature in zip(mean_names, feature_names, strict=True):
            design[name] = [entity_means.loc[label, feature] for label in used_entities]

        design_matrix = design.to_numpy(dtype=float)
        if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
            raise ValueError(
                "The augmented dynamic design is rank deficient; check panel length, "
                "time-invariant regressors, and category support."
            )

        dynamic_outcome = used["__outcome"].reset_index(drop=True)
        _, retained_categories = _ordered_categories(
            dynamic_outcome, category_order=categories
        )
        if not np.array_equal(retained_categories, categories):
            raise ValueError("All fitted outcome categories must survive dynamic trimming.")
        base_result = RandomEffectsOrderedLogit().fit(
            design,
            dynamic_outcome,
            entity=used_entities,
            category_order=categories,
            quadrature_points=quadrature_points,
            maxiter=maxiter,
            tolerance=tolerance,
        )
        return DynamicRandomEffectsOrderedLogitResult(
            base_result=base_result,
            original_feature_names=tuple(feature_names),
            state_parameter_names=tuple(state_names),
            initial_parameter_names=tuple(initial_names),
            initial_covariate_parameter_names=tuple(initial_covariate_names),
            mean_parameter_names=tuple(mean_names),
            initial_outcomes=initial_outcomes,
            initial_covariates=initial_covariates,
            entity_means=entity_means,
            estimation_design=design,
            estimation_outcome=dynamic_outcome,
            estimation_entity=pd.Series(used_entities, name="entity"),
            estimation_index=pd.Index(used["__index"].tolist()),
            n_original_obs=nobs,
            dropped_initial=int((~has_predecessor).sum()),
            dropped_nonconsecutive=int((has_predecessor & at_or_after_gap).sum()),
        )
