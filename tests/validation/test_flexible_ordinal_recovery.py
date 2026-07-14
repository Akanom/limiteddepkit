"""Deterministic Monte Carlo recovery checks for flexible ordinal estimators."""

import numpy as np
import pytest

from limiteddepkit import (
    GeneralizedOrderedLogit,
    PartialProportionalOdds,
    simulate_generalized_ordered_logit,
)

pytestmark = pytest.mark.simulation

SEEDS = (9_101, 9_102, 9_103, 9_104, 9_105)


def fitted_parameter_vector(result):
    return np.r_[
        result.thresholds.to_numpy(dtype=float),
        result.threshold_slopes.to_numpy(dtype=float).ravel(),
    ]


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedOrderedLogit(), PartialProportionalOdds(varying=["x1"])],
)
def test_flexible_ordinal_mean_parameter_recovery(estimator):
    estimates = []
    truth = None
    for seed in SEEDS:
        simulation = simulate_generalized_ordered_logit(seed=seed)
        result = estimator.fit(simulation.X, simulation.y)
        assert result.converged
        assert result.minimum_index_gap > 0
        estimates.append(fitted_parameter_vector(result))
        truth = np.r_[
            simulation.thresholds.to_numpy(dtype=float),
            simulation.threshold_slopes.to_numpy(dtype=float).ravel(),
        ]

    mean_estimate = np.mean(estimates, axis=0)
    assert mean_estimate == pytest.approx(truth, abs=0.08)


def test_simulator_rejects_crossing_design():
    with pytest.raises(ValueError, match="crossing"):
        simulate_generalized_ordered_logit(
            nobs=500,
            thresholds=(-0.1, 0.1),
            threshold_slopes=((2.0, 0.0), (-2.0, 0.0)),
        )
