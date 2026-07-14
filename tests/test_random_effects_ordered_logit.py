import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

from limiteddepkit import (
    RandomEffectsOrderedLogit,
    posterior_predict_proba,
    posterior_random_effects,
    predict_proba,
)


def make_panel_ordinal(seed=8821, n_entities=80, n_periods=5):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=entity.size),
            "x2": rng.normal(size=entity.size),
        }
    )
    random_intercepts = rng.normal(scale=0.7, size=n_entities)
    eta = X.to_numpy() @ np.array([0.8, -0.5]) + random_intercepts[entity]
    cumulative = expit(np.array([-0.8, 0.9])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, y, entity


@pytest.fixture(scope="module")
def fitted_panel_model():
    X, y, entity = make_panel_ordinal()
    result = RandomEffectsOrderedLogit().fit(X, y, entity=entity, quadrature_points=10)
    return X, y, entity, result


def test_random_effects_ordered_logit_contract(fitted_panel_model):
    X, _, entity, result = fitted_panel_model

    assert result.converged
    assert result.nobs == len(X)
    assert result.n_groups == len(np.unique(entity))
    assert result.n_entities == result.n_groups
    assert result.random_effect_sd == result.sigma_entity
    assert result.n_quadrature_points == result.quadrature_points
    assert result.backend == "native-ghq"
    assert result.covariance_type == "observed-information"
    assert result.sigma_entity > 0
    assert result.params["x1"] > 0
    assert result.params["x2"] < 0
    assert list(result.covariance.index) == list(result.all_params.index)
    assert result.inference_valid
    assert np.all(np.isfinite(result.standard_errors))
    assert np.isnan(result.zstats["sigma_entity"])
    assert np.isnan(result.pvalues["sigma_entity"])
    sigma_interval = result.sigma_conf_int()
    assert 0 < sigma_interval["lower"] < result.sigma_entity < sigma_interval["upper"]


def test_population_averaged_probabilities_are_valid(fitted_panel_model):
    X, _, _, result = fitted_panel_model
    probabilities = result.predict_proba(X.iloc[:20])

    assert probabilities.shape == (20, 3)
    assert np.all(probabilities.to_numpy() >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(result.predict(X.iloc[:20])).issubset(set(result.categories))


def test_random_effects_result_uses_ecosystem_contract(fitted_panel_model):
    _, _, _, result = fitted_panel_model
    table = result.summary_frame()

    assert list(table.columns) == ["coef", "std_err", "z", "p_value"]
    assert result.vcov().equals(result.covariance)
    assert result.conf_int().shape == (result.n_params, 2)


def test_random_effects_ordered_logit_validates_panel_inputs():
    X, y, entity = make_panel_ordinal(n_entities=10, n_periods=3)
    with pytest.raises(ValueError, match="same number"):
        RandomEffectsOrderedLogit().fit(X, y, entity=entity[:-1])
    with pytest.raises(ValueError, match="at least three"):
        RandomEffectsOrderedLogit().fit(X, y, entity=entity, quadrature_points=2)
    constant_X = X.assign(constant=1.0)
    with pytest.raises(ValueError, match="constant regressors"):
        RandomEffectsOrderedLogit().fit(constant_X, y, entity=entity)


def test_gauss_hermite_normal_scaling():
    nodes, weights = hermgauss(12)
    sigma = 0.7
    random_effects = np.sqrt(2.0) * sigma * nodes

    assert weights.sum() / np.sqrt(np.pi) == pytest.approx(1.0)
    assert np.sum(weights * random_effects**2) / np.sqrt(np.pi) == pytest.approx(
        sigma**2
    )


def test_posterior_random_effects_and_predictive_probabilities(fitted_panel_model):
    X, y, entity, result = fitted_panel_model
    posterior = result.posterior_random_effects(X, y, entity=entity)
    predictive = result.posterior_predict_proba(X.iloc[:20], entity=entity[:20], posterior=posterior)

    assert len(posterior) == result.n_entities
    assert posterior.index.name == "entity"
    assert np.all(posterior["posterior_sd"] >= 0)
    assert posterior["log_marginal_likelihood"].sum() == pytest.approx(
        result.loglike, abs=1e-7
    )
    for weights in posterior["posterior_weights"]:
        assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(predictive.to_numpy() >= 0)
    assert np.allclose(predictive.sum(axis=1), 1.0)
    wrapped_posterior = posterior_random_effects(result, X, y, entity=entity)
    wrapped_predictive = posterior_predict_proba(
        result,
        X.iloc[:20],
        entity=entity[:20],
        posterior=wrapped_posterior,
    )
    assert wrapped_predictive.to_numpy() == pytest.approx(
        predictive.to_numpy(), abs=1e-12
    )


def test_conditional_prediction_accepts_entity_keyed_effects(fitted_panel_model):
    X, y, entity, result = fitted_panel_model
    posterior = result.posterior_random_effects(X, y, entity=entity)
    sample = X.iloc[:10]
    sample_entities = entity[:10]
    conditional = result.predict_proba(
        sample,
        random_effects=posterior["posterior_mean"],
        entity=sample_entities,
    )

    assert np.allclose(conditional.sum(axis=1), 1.0)
    assert predict_proba(
        result,
        sample,
        random_effects=posterior["posterior_mean"],
        entity=sample_entities,
    ).equals(conditional)
    with pytest.raises(ValueError, match="entity is required"):
        result.predict_proba(sample, random_effects=posterior["posterior_mean"])


def test_positive_random_effect_shifts_probability_upward(fitted_panel_model):
    X, _, _, result = fitted_panel_model
    sample = X.iloc[:12]
    lower = result.predict_proba(sample, random_effects=-1.0)
    higher = result.predict_proba(sample, random_effects=1.0)

    assert np.all(higher[result.categories[-1]] > lower[result.categories[-1]])
    assert np.all(higher[result.categories[0]] < lower[result.categories[0]])


def test_posterior_effect_orders_low_and_high_outcome_histories(fitted_panel_model):
    _, _, _, result = fitted_panel_model
    X = pd.DataFrame({"x1": np.zeros(10), "x2": np.zeros(10)})
    y = np.r_[np.repeat(result.categories[0], 5), np.repeat(result.categories[-1], 5)]
    entity = np.repeat(["low", "high"], 5)
    posterior = result.posterior_random_effects(X, y, entity=entity)

    assert posterior.loc["high", "posterior_mean"] > posterior.loc["low", "posterior_mean"]
