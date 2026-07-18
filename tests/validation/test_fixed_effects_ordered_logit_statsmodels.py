"""Statsmodels conditional-likelihood parity for BUC Ordered Logit."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.discrete.conditional_models import ConditionalLogit

from limiteddepkit.fixed_effects_ordinal import FixedEffectsOrderedLogit


@pytest.mark.validation
def test_buc_coefficients_and_composite_loglike_match_blowup_conditional_logit():
    rng = np.random.default_rng(22017)
    n_entities = 55
    periods = 5
    entity = np.repeat(np.arange(n_entities), periods)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=len(entity)),
            "x2": rng.normal(size=len(entity)),
        }
    )
    fixed = rng.normal(scale=0.9, size=n_entities)
    latent = X.to_numpy() @ np.array([0.55, -0.35]) + fixed[entity] + rng.logistic(
        size=len(entity)
    )
    y = np.digitize(latent, [-0.7, 0.2, 1.0])

    native = FixedEffectsOrderedLogit().fit(X, y, entity=entity)

    expanded_X = []
    expanded_y = []
    clone_groups = []
    clone = 0
    for original_entity in range(n_entities):
        rows = np.flatnonzero(entity == original_entity)
        for cutoff in range(1, 4):
            binary = (y[rows] >= cutoff).astype(int)
            if binary.min() == binary.max():
                continue
            expanded_X.append(X.iloc[rows].to_numpy())
            expanded_y.append(binary)
            clone_groups.append(np.full(len(rows), clone))
            clone += 1

    reference = ConditionalLogit(
        np.concatenate(expanded_y),
        np.vstack(expanded_X),
        groups=np.concatenate(clone_groups),
    ).fit(method="bfgs", maxiter=1_000, disp=False)

    np.testing.assert_allclose(native.params, reference.params, rtol=0.0, atol=6e-5)
    assert native.composite_loglike == pytest.approx(reference.llf, abs=5e-7)
