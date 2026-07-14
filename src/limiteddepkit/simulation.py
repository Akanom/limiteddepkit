"""Simulation helpers for ordinal estimator validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class GeneralizedOrdinalSimulation:
    """Simulated non-crossing Generalized Ordered Logit sample and truth."""

    X: pd.DataFrame
    y: pd.Series
    thresholds: pd.Series
    threshold_slopes: pd.DataFrame
    minimum_index_gap: float


@dataclass(frozen=True)
class RandomEffectsOrderedLogitSimulation:
    """Simulated random-intercept Ordered Logit panel and truth."""

    X: pd.DataFrame
    y: pd.Series
    entity: pd.Series
    time: pd.Series
    params: pd.Series
    thresholds: pd.Series
    sigma_entity: float
    random_intercepts: pd.Series
    group_sizes: pd.Series

    @property
    def nobs(self) -> int:
        return len(self.y)

    @property
    def n_entities(self) -> int:
        return len(self.group_sizes)

    @property
    def is_balanced(self) -> bool:
        return self.group_sizes.nunique() == 1


@dataclass(frozen=True)
class DynamicOrderedLogitSimulation:
    """Dynamic conditional-RE Ordered Logit panel and parameter truth."""

    X: pd.DataFrame
    y: pd.Series
    entity: pd.Series
    time: pd.Series
    structural_params: pd.Series
    state_dependence_params: pd.Series
    initial_condition_params: pd.Series
    initial_covariate_params: pd.Series
    correlated_effects_params: pd.Series
    thresholds: pd.Series
    sigma_entity: float
    initial_outcomes: pd.Series
    initial_covariates: pd.DataFrame
    post_initial_means: pd.DataFrame
    residual_random_effects: pd.Series

    @property
    def nobs(self) -> int:
        return len(self.y)

    @property
    def n_entities(self) -> int:
        return self.entity.nunique()


def simulate_generalized_ordered_logit(
    *,
    nobs: int = 1_500,
    seed: int = 9_101,
    thresholds: Any = (-0.9, 0.9),
    threshold_slopes: Any = ((0.85, -0.4), (0.3, -0.4)),
    feature_names: tuple[str, ...] = ("x1", "x2"),
    covariate_low: float = -1.0,
    covariate_high: float = 1.0,
) -> GeneralizedOrdinalSimulation:
    """Simulate a bounded-support Generalized Ordered Logit design."""
    if nobs <= 0:
        raise ValueError("nobs must be positive.")
    if not covariate_low < covariate_high:
        raise ValueError("covariate_low must be smaller than covariate_high.")
    cuts = np.asarray(thresholds, dtype=float)
    slopes = np.asarray(threshold_slopes, dtype=float)
    if cuts.ndim != 1 or cuts.size < 2:
        raise ValueError("thresholds must contain at least two ordered cuts.")
    if slopes.shape != (cuts.size, len(feature_names)):
        raise ValueError(
            "threshold_slopes must have one row per threshold and one column per feature."
        )
    if not np.isfinite(cuts).all() or not np.isfinite(slopes).all():
        raise ValueError("Simulation parameters must be finite.")

    rng = np.random.default_rng(seed)
    values = rng.uniform(
        covariate_low, covariate_high, size=(nobs, len(feature_names))
    )
    cumulative_indices = cuts[None, :] - values @ slopes.T
    gaps = np.diff(cumulative_indices, axis=1)
    minimum_gap = float(np.min(gaps))
    if minimum_gap <= 0:
        raise ValueError(
            "The supplied design produces crossing cumulative logits on the simulated support."
        )
    cumulative = expit(cumulative_indices)
    bounds = np.column_stack([np.zeros(nobs), cumulative, np.ones(nobs)])
    probabilities = np.diff(bounds, axis=1)
    outcomes = np.array(
        [rng.choice(cuts.size + 1, p=probability) for probability in probabilities]
    )
    split_names = [f"{index} | {index + 1}" for index in range(cuts.size)]
    return GeneralizedOrdinalSimulation(
        X=pd.DataFrame(values, columns=feature_names),
        y=pd.Series(outcomes, name="y"),
        thresholds=pd.Series(cuts, index=split_names, name="threshold"),
        threshold_slopes=pd.DataFrame(
            slopes, index=split_names, columns=feature_names
        ),
        minimum_index_gap=minimum_gap,
    )


def simulate_random_effects_ordered_logit(
    *,
    n_entities: int = 160,
    n_periods: int = 6,
    seed: int = 8_821,
    thresholds: Any = (-0.8, 0.9),
    coefficients: Any = (0.8, -0.5),
    feature_names: tuple[str, ...] = ("x1", "x2"),
    sigma_entity: float = 0.7,
    unbalanced: bool = False,
    minimum_periods: int = 2,
) -> RandomEffectsOrderedLogitSimulation:
    """Simulate a balanced or unbalanced random-intercept Ordered Logit panel.

    When ``unbalanced`` is true, each entity receives a uniformly sampled number
    of observations between ``minimum_periods`` and ``n_periods`` (inclusive).
    The conditional data-generating process matches
    :class:`limiteddepkit.RandomEffectsOrderedLogit` exactly.
    """
    if n_entities < 2:
        raise ValueError("n_entities must be at least two.")
    if n_periods < 2:
        raise ValueError("n_periods must be at least two.")
    if minimum_periods < 2 or minimum_periods > n_periods:
        raise ValueError("minimum_periods must be between two and n_periods.")
    if unbalanced and minimum_periods == n_periods:
        raise ValueError("minimum_periods must be smaller than n_periods for an unbalanced panel.")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must contain unique names.")

    cuts = np.asarray(thresholds, dtype=float)
    beta = np.asarray(coefficients, dtype=float)
    if cuts.ndim != 1 or cuts.size < 2 or np.any(np.diff(cuts) <= 0):
        raise ValueError("thresholds must contain at least two strictly ordered cuts.")
    if beta.shape != (len(feature_names),):
        raise ValueError("coefficients must contain one value per feature.")
    if not np.isfinite(cuts).all() or not np.isfinite(beta).all():
        raise ValueError("Simulation parameters must be finite.")
    if not np.isfinite(sigma_entity) or sigma_entity <= 0:
        raise ValueError("sigma_entity must be finite and strictly positive.")

    rng = np.random.default_rng(seed)
    if unbalanced:
        sizes = rng.integers(minimum_periods, n_periods + 1, size=n_entities)
        if np.all(sizes == sizes[0]):
            sizes[-1] = minimum_periods if sizes[0] != minimum_periods else n_periods
    else:
        sizes = np.full(n_entities, n_periods, dtype=int)

    entity_values = np.repeat(np.arange(n_entities), sizes)
    time_values = np.concatenate([np.arange(size) for size in sizes])
    values = rng.normal(size=(entity_values.size, len(feature_names)))
    random_effect_values = rng.normal(scale=sigma_entity, size=n_entities)
    linear_predictor = values @ beta + random_effect_values[entity_values]
    cumulative = expit(cuts[None, :] - linear_predictor[:, None])
    bounds = np.column_stack(
        [np.zeros(entity_values.size), cumulative, np.ones(entity_values.size)]
    )
    probabilities = np.diff(bounds, axis=1)
    outcomes = np.array([rng.choice(cuts.size + 1, p=probability) for probability in probabilities])
    split_names = [f"{index} | {index + 1}" for index in range(cuts.size)]
    entity_index = pd.Index(np.arange(n_entities), name="entity")

    return RandomEffectsOrderedLogitSimulation(
        X=pd.DataFrame(values, columns=feature_names),
        y=pd.Series(outcomes, name="y"),
        entity=pd.Series(entity_values, name="entity"),
        time=pd.Series(time_values, name="time"),
        params=pd.Series(beta, index=feature_names, name="coefficient"),
        thresholds=pd.Series(cuts, index=split_names, name="threshold"),
        sigma_entity=float(sigma_entity),
        random_intercepts=pd.Series(
            random_effect_values, index=entity_index, name="random_intercept"
        ),
        group_sizes=pd.Series(sizes, index=entity_index, name="nobs"),
    )


def simulate_dynamic_random_effects_ordered_logit(
    *,
    n_entities: int = 180,
    n_periods: int = 7,
    seed: int = 8_821,
    thresholds: Any = (-0.85, 0.9),
    coefficients: Any = (0.55,),
    state_dependence: Any = (0.35, 0.75),
    initial_conditions: Any = (0.35, 0.75),
    initial_covariate_effects: Any = (0.25,),
    correlated_effects: Any = (0.4,),
    sigma_entity: float = 0.55,
    feature_names: tuple[str, ...] = ("x1",),
    between_sd: float = 0.55,
    within_sd: float = 0.8,
    initial_probabilities: Any = (0.34, 0.36, 0.3),
) -> DynamicOrderedLogitSimulation:
    """Simulate the package's dynamic conditional-RE ordinal specification."""
    if n_entities < 2:
        raise ValueError("n_entities must be at least two.")
    if n_periods < 3:
        raise ValueError("n_periods must be at least three for dynamic identification.")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must contain unique names.")
    cuts = np.asarray(thresholds, dtype=float)
    beta = np.asarray(coefficients, dtype=float)
    state = np.asarray(state_dependence, dtype=float)
    initial_effects = np.asarray(initial_conditions, dtype=float)
    initial_X_effects = np.asarray(initial_covariate_effects, dtype=float)
    mean_effects = np.asarray(correlated_effects, dtype=float)
    initial_probabilities_array = np.asarray(initial_probabilities, dtype=float)
    n_categories = cuts.size + 1
    if cuts.ndim != 1 or cuts.size < 2 or np.any(np.diff(cuts) <= 0):
        raise ValueError("thresholds must contain at least two strictly ordered cuts.")
    if beta.shape != (len(feature_names),):
        raise ValueError("coefficients must contain one value per feature.")
    if initial_X_effects.shape != beta.shape or mean_effects.shape != beta.shape:
        raise ValueError(
            "initial_covariate_effects and correlated_effects require one value per feature."
        )
    if state.shape != (n_categories - 1,) or initial_effects.shape != (
        n_categories - 1,
    ):
        raise ValueError(
            "state_dependence and initial_conditions require one effect per nonreference category."
        )
    if initial_probabilities_array.shape != (n_categories,):
        raise ValueError("initial_probabilities must contain one value per category.")
    if np.any(initial_probabilities_array < 0) or not np.isclose(
        initial_probabilities_array.sum(), 1.0
    ):
        raise ValueError("initial_probabilities must be nonnegative and sum to one.")
    if not np.isfinite(sigma_entity) or sigma_entity <= 0:
        raise ValueError("sigma_entity must be finite and strictly positive.")

    rng = np.random.default_rng(seed)
    entity_values = np.repeat(np.arange(n_entities), n_periods)
    time_values = np.tile(np.arange(n_periods), n_entities)
    between = rng.normal(scale=between_sd, size=(n_entities, len(feature_names)))
    values = between[entity_values] + rng.normal(
        scale=within_sd, size=(entity_values.size, len(feature_names))
    )
    value_cube = values.reshape(n_entities, n_periods, len(feature_names))
    initial_X = value_cube[:, 0, :]
    post_means = value_cube[:, 1:, :].mean(axis=1)
    initial_y = rng.choice(
        n_categories, size=n_entities, p=initial_probabilities_array
    )
    residual_effects = rng.normal(scale=sigma_entity, size=n_entities)
    conditional_intercept = (
        np.where(initial_y > 0, initial_effects[np.clip(initial_y - 1, 0, None)], 0.0)
        + initial_X @ initial_X_effects
        + post_means @ mean_effects
        + residual_effects
    )
    outcomes = np.empty((n_entities, n_periods), dtype=int)
    outcomes[:, 0] = initial_y
    for period in range(1, n_periods):
        previous = outcomes[:, period - 1]
        state_shift = np.where(
            previous > 0, state[np.clip(previous - 1, 0, None)], 0.0
        )
        eta = value_cube[:, period, :] @ beta + state_shift + conditional_intercept
        cumulative = expit(cuts[None, :] - eta[:, None])
        probabilities = np.column_stack(
            [cumulative[:, 0], np.diff(cumulative, axis=1), 1 - cumulative[:, -1]]
        )
        outcomes[:, period] = np.array(
            [rng.choice(n_categories, p=row) for row in probabilities]
        )

    split_names = [f"{category - 1} | {category}" for category in range(1, n_categories)]
    nonreference = np.arange(1, n_categories)
    entity_index = pd.Index(np.arange(n_entities), name="entity")
    return DynamicOrderedLogitSimulation(
        X=pd.DataFrame(values, columns=feature_names),
        y=pd.Series(outcomes.ravel(), name="y"),
        entity=pd.Series(entity_values, name="entity"),
        time=pd.Series(time_values, name="time"),
        structural_params=pd.Series(beta, index=feature_names, name="coefficient"),
        state_dependence_params=pd.Series(
            state, index=[f"state[{category}]" for category in nonreference]
        ),
        initial_condition_params=pd.Series(
            initial_effects, index=[f"initial[{category}]" for category in nonreference]
        ),
        initial_covariate_params=pd.Series(
            initial_X_effects,
            index=[f"initial_x[{feature}]" for feature in feature_names],
        ),
        correlated_effects_params=pd.Series(
            mean_effects, index=[f"mean[{feature}]" for feature in feature_names]
        ),
        thresholds=pd.Series(cuts, index=split_names, name="threshold"),
        sigma_entity=float(sigma_entity),
        initial_outcomes=pd.Series(initial_y, index=entity_index, name="initial_outcome"),
        initial_covariates=pd.DataFrame(
            initial_X, index=entity_index, columns=feature_names
        ),
        post_initial_means=pd.DataFrame(
            post_means, index=entity_index, columns=feature_names
        ),
        residual_random_effects=pd.Series(
            residual_effects, index=entity_index, name="residual_random_effect"
        ),
    )
