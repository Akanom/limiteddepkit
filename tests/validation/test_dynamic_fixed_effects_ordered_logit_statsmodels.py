from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.dynamic_fixed_effects_ordinal import (
    DynamicFixedEffectsOrderedLogit,
)

statsmodels = pytest.importorskip("statsmodels.api")


def _simulated_three_category_panel(
    n_entities: int = 2_500,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(88)
    beta = 0.45
    state_dependence = 0.65
    thresholds = np.array([0.0, 1.1])
    covariate = np.empty((n_entities, 4), dtype=float)
    covariate[:, :3] = rng.integers(0, 3, size=(n_entities, 3))
    covariate[:, 3] = covariate[:, 2]
    effects = rng.normal(size=n_entities) + 0.3 * covariate.mean(axis=1)
    outcome = np.empty((n_entities, 4), dtype=int)
    outcome[:, 0] = np.digitize(
        effects + 0.3 * covariate[:, 0] + rng.logistic(size=n_entities),
        thresholds,
    )
    for period in range(1, 4):
        outcome[:, period] = np.digitize(
            effects
            + beta * covariate[:, period]
            + state_dependence * (outcome[:, period - 1] >= 1)
            + rng.logistic(size=n_entities),
            thresholds,
        )
    return (
        pd.DataFrame({"x": covariate.ravel()}),
        outcome.ravel(),
        np.repeat(np.arange(n_entities), 4),
        np.tile(np.arange(4), n_entities),
    )


@pytest.mark.validation
def test_mrv_conditional_likelihood_matches_statsmodels_clustered_glm() -> None:
    X, y, entity, time = _simulated_three_category_panel()
    result = DynamicFixedEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        time=time,
        state_cutoff=1,
        category_order=[0, 1, 2],
    )
    conditional = result.conditional_sample_frame()
    parameter_names = result.all_params.index.tolist()
    reference = statsmodels.GLM(
        conditional["_response"],
        conditional[parameter_names],
        family=statsmodels.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": conditional["_entity"], "use_correction": True},
        use_t=False,
    )

    # The fitted threshold is interior, so the linear inequality constraint is
    # inactive and the estimates reduce exactly to this industrial binary-GLM
    # representation of the MRV conditional sample.
    assert result.inference_valid
    np.testing.assert_allclose(result.all_params, reference.params, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(
        result.covariance, reference.cov_params(), atol=5e-8, rtol=0.0
    )
    np.testing.assert_allclose(
        result.standard_errors, reference.bse, atol=2e-7, rtol=0.0
    )
