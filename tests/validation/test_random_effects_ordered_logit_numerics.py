"""Numerical-stability checks for random-effects Ordered Logit."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import RandomEffectsOrderedLogit

pytestmark = pytest.mark.simulation


@pytest.fixture(scope="module")
def deterministic_panel():
    rng = np.random.default_rng(8_129)
    n_entities = 50
    n_periods = 5
    entity = np.repeat(np.arange(n_entities), n_periods)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-1.0, 1.0, size=entity.size),
            "x2": rng.normal(scale=0.7, size=entity.size),
        }
    )
    random_intercepts = rng.normal(scale=1.0, size=n_entities)
    linear_index = X.to_numpy() @ np.array([0.8, -0.55])
    linear_index += random_intercepts[entity]
    cumulative = expit(np.array([-0.8, 0.9])[None, :] - linear_index[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1.0 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, y, entity


@pytest.fixture(scope="module")
def reference_fit(deterministic_panel):
    X, y, entity = deterministic_panel
    return RandomEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        quadrature_points=20,
        maxiter=600,
    )


def test_gauss_hermite_quadrature_converges(deterministic_panel, reference_fit):
    X, y, entity = deterministic_panel
    high_order = RandomEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        quadrature_points=28,
        maxiter=600,
    )

    assert reference_fit.converged
    assert high_order.converged
    assert high_order.loglike == pytest.approx(reference_fit.loglike, abs=0.005)
    assert high_order.all_params.to_numpy() == pytest.approx(
        reference_fit.all_params.to_numpy(), abs=0.002
    )
    assert high_order.predict_proba(X.iloc[:30]).to_numpy() == pytest.approx(
        reference_fit.predict_proba(X.iloc[:30]).to_numpy(), abs=0.0002
    )


def test_fit_is_invariant_to_row_order(deterministic_panel, reference_fit):
    X, y, entity = deterministic_panel
    permutation = np.random.default_rng(2_026).permutation(len(y))
    reordered = RandomEffectsOrderedLogit().fit(
        X.iloc[permutation].reset_index(drop=True),
        y[permutation],
        entity=entity[permutation],
        quadrature_points=20,
        maxiter=600,
    )

    assert reordered.converged
    assert reordered.loglike == pytest.approx(reference_fit.loglike, abs=1e-7)
    assert reordered.all_params.to_numpy() == pytest.approx(
        reference_fit.all_params.to_numpy(), abs=2e-6
    )


def test_fit_is_invariant_to_group_labels(deterministic_panel, reference_fit):
    X, y, entity = deterministic_panel
    shuffled_codes = np.random.default_rng(99).permutation(reference_fit.n_groups)
    labels = np.asarray([f"panel-{shuffled_codes[group]:03d}" for group in entity])
    relabelled = RandomEffectsOrderedLogit().fit(
        X,
        y,
        entity=labels,
        quadrature_points=20,
        maxiter=600,
    )

    assert relabelled.converged
    assert relabelled.loglike == pytest.approx(reference_fit.loglike, abs=1e-7)
    assert relabelled.all_params.to_numpy() == pytest.approx(
        reference_fit.all_params.to_numpy(), abs=2e-6
    )
