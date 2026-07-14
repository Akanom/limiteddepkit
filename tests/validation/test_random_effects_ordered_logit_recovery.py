"""Deterministic recovery checks for random-effects Ordered Logit."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit import RandomEffectsOrderedLogit
from limiteddepkit.simulation import simulate_random_effects_ordered_logit

pytestmark = pytest.mark.simulation

SEEDS = (8_821, 8_822, 8_823, 8_824)


def parameter_vector(result):
    return np.r_[
        result.params.to_numpy(dtype=float),
        result.thresholds.to_numpy(dtype=float),
        result.sigma_entity,
    ]


@pytest.mark.parametrize("unbalanced", [False, True], ids=["balanced", "unbalanced"])
def test_random_effects_ordered_logit_mean_parameter_recovery(unbalanced):
    estimates = []
    truth = None
    for seed in SEEDS:
        simulation = simulate_random_effects_ordered_logit(
            n_entities=160,
            n_periods=6,
            minimum_periods=3,
            unbalanced=unbalanced,
            seed=seed,
        )
        result = RandomEffectsOrderedLogit().fit(
            simulation.X,
            simulation.y,
            entity=simulation.entity,
            quadrature_points=10,
        )

        assert result.converged
        assert simulation.is_balanced is not unbalanced
        estimates.append(parameter_vector(result))
        truth = np.r_[
            simulation.params.to_numpy(dtype=float),
            simulation.thresholds.to_numpy(dtype=float),
            simulation.sigma_entity,
        ]

    assert np.mean(estimates, axis=0) == pytest.approx(truth, abs=0.10)


def test_unbalanced_panel_simulation_is_deterministic_and_retains_truth():
    options = {
        "n_entities": 24,
        "n_periods": 7,
        "minimum_periods": 2,
        "unbalanced": True,
        "seed": 7_331,
    }
    first = simulate_random_effects_ordered_logit(**options)
    second = simulate_random_effects_ordered_logit(**options)

    assert not first.is_balanced
    assert first.n_entities == options["n_entities"]
    assert first.nobs == int(first.group_sizes.sum())
    assert first.group_sizes.between(options["minimum_periods"], options["n_periods"]).all()
    assert np.array_equal(
        first.entity.value_counts().sort_index().to_numpy(),
        first.group_sizes.to_numpy(),
    )
    assert first.time.groupby(first.entity).min().eq(0).all()
    assert np.array_equal(
        first.time.groupby(first.entity).count().to_numpy(),
        first.group_sizes.to_numpy(),
    )
    pd.testing.assert_frame_equal(first.X, second.X)
    pd.testing.assert_series_equal(first.y, second.y)
    pd.testing.assert_series_equal(first.entity, second.entity)
    pd.testing.assert_series_equal(first.random_intercepts, second.random_intercepts)
