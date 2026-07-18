from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.dynamic_fixed_effects_ordinal import (
    DynamicFixedEffectsOrderedLogit,
)


@pytest.mark.simulation
def test_dynamic_fixed_effects_ordered_logit_recovers_structural_parameters() -> None:
    rng = np.random.default_rng(773)
    n_entities = 15_000
    beta = 0.55
    state_dependence = 0.8
    thresholds = np.array([-1.0, 0.0, 1.15])
    covariate = np.empty((n_entities, 4), dtype=float)
    covariate[:, :3] = rng.integers(0, 3, size=(n_entities, 3))
    covariate[:, 3] = covariate[:, 2]
    # The nuisance effect is deliberately correlated with the included
    # regressor history; this is the fixed-effects case, not a random intercept.
    effects = rng.normal(scale=1.1, size=n_entities) + 0.35 * covariate.mean(axis=1)

    outcome = np.empty((n_entities, 4), dtype=int)
    outcome[:, 0] = np.digitize(
        effects + 0.2 * covariate[:, 0] + rng.logistic(size=n_entities),
        thresholds,
    )
    for period in range(1, 4):
        outcome[:, period] = np.digitize(
            effects
            + beta * covariate[:, period]
            + state_dependence * (outcome[:, period - 1] >= 2)
            + rng.logistic(size=n_entities),
            thresholds,
        )

    result = DynamicFixedEffectsOrderedLogit().fit(
        pd.DataFrame({"x": covariate.ravel()}),
        outcome.ravel(),
        entity=np.repeat(np.arange(n_entities), 4),
        time=np.tile(np.arange(4), n_entities),
        state_cutoff=2,
        category_order=[0, 1, 2, 3],
    )

    assert result.converged
    assert result.inference_valid
    assert result.params["x"] == pytest.approx(beta, abs=0.08)
    assert result.state_dependence == pytest.approx(state_dependence, abs=0.13)
    assert result.thresholds["0 | 1"] == pytest.approx(-1.0, abs=0.12)
    assert result.thresholds["1 | 2"] == 0.0
    assert result.thresholds["2 | 3"] == pytest.approx(1.15, abs=0.15)
