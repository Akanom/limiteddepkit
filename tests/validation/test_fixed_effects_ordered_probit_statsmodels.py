"""Industrial parity for the uncorrected FE Ordered-Probit likelihood."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.fixed_effects_ordinal import FixedEffectsOrderedProbit

statsmodels_ordinal = pytest.importorskip("statsmodels.miscmodels.ordinal_model")
OrderedModel = statsmodels_ordinal.OrderedModel


def test_uncorrected_fe_probit_matches_statsmodels_entity_dummy_mle():
    rng = np.random.default_rng(773)
    n_entities = 10
    periods = 8
    entity = np.repeat(np.arange(n_entities), periods)
    time = np.tile(np.arange(periods), n_entities)
    effects = np.linspace(-0.6, 0.6, n_entities) + rng.normal(
        scale=0.1, size=n_entities
    )
    x = rng.normal(size=len(entity))
    latent = 0.65 * x + effects[entity] + rng.normal(size=len(entity))
    y = np.digitize(latent, [-0.75, 0.10, 0.85])
    X = pd.DataFrame({"x": x})

    ours = FixedEffectsOrderedProbit().fit(
        X,
        y,
        entity=entity,
        time=time,
        category_order=[0, 1, 2, 3],
    )
    dummies = pd.get_dummies(
        pd.Series(entity, dtype="category"), drop_first=True, dtype=float
    )
    reference_design = pd.concat([X, dummies], axis=1)
    reference = OrderedModel(y, reference_design, distr="probit").fit(
        method="bfgs", maxiter=2_000, disp=False
    )

    assert reference.mle_retvals["converged"]
    assert ours.uncorrected_params["x"] == pytest.approx(
        reference.params.iloc[0], abs=8e-6
    )
    assert ours.full_loglike == pytest.approx(reference.llf, abs=2e-6)
    reference_thresholds = reference.model.transform_threshold_params(
        reference.params.iloc[-3:]
    )[1:-1]
    np.testing.assert_allclose(
        np.diff(ours.uncorrected_thresholds),
        np.diff(reference_thresholds),
        atol=2e-5,
    )
