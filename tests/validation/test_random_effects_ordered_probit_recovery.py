"""Deterministic recovery checks for random-effects Ordered Probit."""

import numpy as np
import pytest

from limiteddepkit import (
    RandomEffectsOrderedProbit,
    simulate_random_effects_ordered_probit,
)

pytestmark = pytest.mark.simulation

SEEDS = (4_119, 4_120, 4_121, 4_122)


def _parameter_vector(result):
    return np.r_[
        result.params.to_numpy(dtype=float),
        result.thresholds.to_numpy(dtype=float),
        result.sigma_entity,
    ]


@pytest.mark.parametrize("unbalanced", [False, True], ids=["balanced", "unbalanced"])
def test_random_effects_ordered_probit_mean_parameter_recovery(unbalanced):
    estimates = []
    truth = None
    for seed in SEEDS:
        simulation = simulate_random_effects_ordered_probit(
            n_entities=160,
            n_periods=6,
            minimum_periods=3,
            unbalanced=unbalanced,
            seed=seed,
        )
        result = RandomEffectsOrderedProbit().fit(
            simulation.X,
            simulation.y,
            entity=simulation.entity,
            quadrature_points=10,
        )

        assert result.converged
        assert simulation.is_balanced is not unbalanced
        estimates.append(_parameter_vector(result))
        truth = np.r_[
            simulation.params.to_numpy(dtype=float),
            simulation.thresholds.to_numpy(dtype=float),
            simulation.sigma_entity,
        ]

    assert np.mean(estimates, axis=0) == pytest.approx(truth, abs=0.10)


def test_unbalanced_probit_simulation_is_deterministic():
    options = {
        "n_entities": 24,
        "n_periods": 7,
        "minimum_periods": 2,
        "unbalanced": True,
        "seed": 7_332,
    }
    first = simulate_random_effects_ordered_probit(**options)
    second = simulate_random_effects_ordered_probit(**options)

    assert not first.is_balanced
    assert first.nobs == int(first.group_sizes.sum())
    assert first.group_sizes.between(2, 7).all()
    assert np.array_equal(
        first.entity.value_counts().sort_index().to_numpy(),
        first.group_sizes.to_numpy(),
    )
    assert first.X.equals(second.X)
    assert first.y.equals(second.y)
    assert first.random_intercepts.equals(second.random_intercepts)
