"""Simulation recovery for the dynamic conditional-RE Ordered Logit."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit import (
    DynamicRandomEffectsOrderedLogit,
    simulate_dynamic_random_effects_ordered_logit,
)

pytestmark = pytest.mark.simulation

SEEDS = (8_821, 8_822, 8_823, 8_824)


def fitted_truth_vector(result):
    return np.r_[
        result.structural_params.to_numpy(dtype=float),
        result.state_dependence_params.to_numpy(dtype=float),
        result.initial_condition_params.to_numpy(dtype=float),
        result.initial_covariate_params.to_numpy(dtype=float),
        result.correlated_effects_params.to_numpy(dtype=float),
        result.thresholds.to_numpy(dtype=float),
        result.sigma_entity,
    ]


def simulation_truth_vector(simulation):
    return np.r_[
        simulation.structural_params.to_numpy(dtype=float),
        simulation.state_dependence_params.to_numpy(dtype=float),
        simulation.initial_condition_params.to_numpy(dtype=float),
        simulation.initial_covariate_params.to_numpy(dtype=float),
        simulation.correlated_effects_params.to_numpy(dtype=float),
        simulation.thresholds.to_numpy(dtype=float),
        simulation.sigma_entity,
    ]


def test_dynamic_ordered_logit_mean_parameter_recovery():
    estimates = []
    truth = None
    for seed in SEEDS:
        simulation = simulate_dynamic_random_effects_ordered_logit(
            n_entities=250, n_periods=7, seed=seed
        )
        result = DynamicRandomEffectsOrderedLogit().fit(
            simulation.X,
            simulation.y,
            entity=simulation.entity,
            time=simulation.time,
            quadrature_points=10,
        )
        assert result.converged
        assert result.dropped_initial == simulation.n_entities
        assert result.dropped_nonconsecutive == 0
        estimates.append(fitted_truth_vector(result))
        truth = simulation_truth_vector(simulation)

    assert np.mean(estimates, axis=0) == pytest.approx(truth, abs=0.12)


def test_dynamic_simulation_and_augmented_design_construction():
    simulation = simulate_dynamic_random_effects_ordered_logit(
        n_entities=50, n_periods=6, seed=7_511
    )
    result = DynamicRandomEffectsOrderedLogit().fit(
        simulation.X,
        simulation.y,
        entity=simulation.entity,
        time=simulation.time,
        quadrature_points=8,
    )

    pd.testing.assert_series_equal(
        result.initial_outcomes.sort_index(),
        simulation.initial_outcomes.sort_index(),
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        result.initial_covariates.sort_index(),
        simulation.initial_covariates.sort_index(),
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        result.entity_means.sort_index(),
        simulation.post_initial_means.sort_index(),
        check_names=False,
    )
    first_used = result.estimation_design.iloc[0]
    first_lag = simulation.y.iloc[0]
    if first_lag > 0:
        assert first_used[f"state[{first_lag}]"] == 1.0
    else:
        assert first_used[list(result.state_parameter_names)].sum() == 0.0
    assert result.nobs == simulation.n_entities * (6 - 1)


def test_dynamic_simulator_is_deterministic():
    first = simulate_dynamic_random_effects_ordered_logit(seed=9_911, n_entities=30)
    second = simulate_dynamic_random_effects_ordered_logit(seed=9_911, n_entities=30)

    pd.testing.assert_frame_equal(first.X, second.X)
    pd.testing.assert_series_equal(first.y, second.y)
    pd.testing.assert_series_equal(first.initial_outcomes, second.initial_outcomes)
    pd.testing.assert_series_equal(
        first.residual_random_effects, second.residual_random_effects
    )
