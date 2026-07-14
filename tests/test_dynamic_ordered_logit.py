import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import DynamicRandomEffectsOrderedLogit


def make_dynamic_panel(seed=2207, n_entities=80, n_periods=6):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    entity_component = rng.normal(scale=0.5, size=(n_entities, 2))
    X = pd.DataFrame(
        entity_component[entity] + rng.normal(scale=0.8, size=(entity.size, 2)),
        columns=["x1", "x2"],
    )
    random_effect = rng.normal(scale=0.6, size=n_entities)
    y = np.empty(entity.size, dtype=int)
    thresholds = np.array([-0.8, 0.9])
    for group in range(n_entities):
        rows = np.flatnonzero(entity == group)
        previous = rng.integers(0, 3)
        for row in rows:
            state = [0.0, 0.45, 0.9][previous]
            eta = 0.65 * X.iloc[row, 0] - 0.4 * X.iloc[row, 1] + state + random_effect[group]
            cumulative = expit(thresholds - eta)
            probabilities = np.r_[cumulative[0], np.diff(cumulative), 1 - cumulative[-1]]
            y[row] = rng.choice(3, p=probabilities)
            previous = y[row]
    return X, y, entity, time


@pytest.fixture(scope="module")
def fitted_dynamic_model():
    X, y, entity, time = make_dynamic_panel()
    result = DynamicRandomEffectsOrderedLogit().fit(
        X, y, entity=entity, time=time, quadrature_points=8
    )
    return X, y, entity, time, result


def test_dynamic_ordered_logit_contract(fitted_dynamic_model):
    X, _, entity, _, result = fitted_dynamic_model

    assert result.converged
    assert result.n_original_obs == len(X)
    assert result.nobs == len(X) - len(np.unique(entity))
    assert result.n_entities == len(np.unique(entity))
    assert len(result.state_dependence_params) == 2
    assert len(result.initial_condition_params) == 2
    assert len(result.initial_covariate_params) == X.shape[1]
    assert len(result.correlated_effects_params) == X.shape[1]
    assert result.backend == "native-dynamic-ghq"
    assert result.fitted_probabilities.shape == (result.nobs, 3)


def test_dynamic_prediction_uses_known_initial_conditions(fitted_dynamic_model):
    X, y, entity, time, result = fitted_dynamic_model
    sample_rows = np.flatnonzero(time > 0)[:10]
    probabilities = result.predict_proba(
        X.iloc[sample_rows],
        entity=entity[sample_rows],
        lagged_y=y[sample_rows - 1],
    )

    assert probabilities.shape == (len(sample_rows), 3)
    assert np.all(probabilities.to_numpy() >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    posterior = result.posterior_random_effects()
    posterior_prediction = result.posterior_predict_proba(
        X.iloc[sample_rows],
        entity=entity[sample_rows],
        lagged_y=y[sample_rows - 1],
        posterior=posterior,
    )
    assert np.allclose(posterior_prediction.sum(axis=1), 1.0)


def test_dynamic_fit_does_not_bridge_time_gaps():
    X, y, entity, time = make_dynamic_panel(n_entities=30)
    keep = ~((entity == 0) & (time == 3))
    result = DynamicRandomEffectsOrderedLogit().fit(
        X.loc[keep].reset_index(drop=True),
        y[keep],
        entity=entity[keep],
        time=time[keep],
        quadrature_points=6,
    )

    balanced_dynamic_nobs = len(y) - 30
    assert result.nobs == balanced_dynamic_nobs - 3


def test_dynamic_fit_rejects_duplicate_entity_time():
    X, y, entity, time = make_dynamic_panel(n_entities=20)
    time = time.copy()
    time[1] = time[0]
    with pytest.raises(ValueError, match="Duplicate entity-time"):
        DynamicRandomEffectsOrderedLogit().fit(X, y, entity=entity, time=time)


def test_dynamic_fit_supports_shifted_time_origin_and_custom_step():
    X, y, entity, time = make_dynamic_panel(n_entities=30)
    result = DynamicRandomEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        time=2_000 + 2 * time,
        time_step=2,
        quadrature_points=6,
    )

    assert result.nobs == len(y) - 30
