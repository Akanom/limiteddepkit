import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import (
    DynamicRandomEffectsOrderedLogit,
    GeneralizedOrderedLogit,
    OrderedLogit,
    PartialProportionalOdds,
    RandomEffectsOrderedLogit,
    add_to_outputhub,
    to_outputhub_model,
)

outputhub = pytest.importorskip("universal_output_hub")


def fitted_example(seed=991, nobs=500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)})
    eta = X.to_numpy() @ np.array([0.8, -0.5])
    cumulative = expit(np.array([-0.7, 0.8])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, y, OrderedLogit().fit(X, y)


@pytest.fixture(scope="module")
def flexible_examples():
    rng = np.random.default_rng(6104)
    nobs = 500
    X = pd.DataFrame(
        {"x1": rng.uniform(-1, 1, nobs), "x2": rng.uniform(-1, 1, nobs)}
    )
    eta = X.to_numpy() @ np.array([0.7, -0.4])
    cumulative = expit(np.array([-0.8, 0.9])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, {
        "generalized": GeneralizedOrderedLogit().fit(X, y),
        "partial": PartialProportionalOdds(varying=["x1"]).fit(X, y),
    }


def test_outputhub_canonical_model_contains_thresholds_and_metadata():
    _, _, result = fitted_example()
    model = to_outputhub_model(result, name="Ordinal outcome", depvar="priority")

    assert model.name == "Ordinal outcome"
    assert model.depvar == "priority"
    assert list(model.params.index) == list(result.all_params.index)
    assert model.statistics["N"] == result.nobs
    assert model.statistics["Log Likelihood"] == result.loglike
    assert model.metadata["link"] == "logit"
    assert model.source == "limiteddepkit"


def test_add_to_outputhub_attaches_model_and_marginal_effects_table():
    X, _, result = fitted_example()
    hub = outputhub.OutputHub("Ordinal report")
    model = add_to_outputhub(hub, result, name="Ordered choice", X=X)

    assert hub.models == [model]
    assert len(hub.tables) == 1
    assert hub.tables[0].name == "Ordered choice average marginal effects"
    assert set(hub.tables[0].data.columns) >= {
        "category",
        "feature",
        "estimate",
        "standard_error",
        "p_value",
    }


def test_add_to_outputhub_rejects_incompatible_hub():
    _, _, result = fitted_example(nobs=300)
    with pytest.raises(TypeError, match="OutputHub-compatible"):
        add_to_outputhub(object(), result)


@pytest.mark.parametrize(
    ("result_key", "expected_name", "expected_estimator"),
    [
        ("generalized", "Generalized Ordered Logit", "generalized_ordered_logit"),
        ("partial", "Partial Proportional Odds", "partial_proportional_odds"),
    ],
)
def test_flexible_outputhub_models_include_constraint_metadata(
    flexible_examples, result_key, expected_name, expected_estimator
):
    _, results = flexible_examples
    result = results[result_key]
    model = to_outputhub_model(result)

    assert model.name == expected_name
    assert list(model.params.index) == list(result.all_params.index)
    assert model.metadata["estimator"] == expected_estimator
    assert model.metadata["link"] == "logit"
    assert model.metadata["inference_valid"] is result.inference_valid
    assert model.diagnostics["Minimum index gap"] == result.minimum_index_gap
    assert model.diagnostics["Constraint slack"] == result.constraint_slack
    if result_key == "partial":
        assert model.metadata["varying_features"] == ["x1"]


@pytest.mark.parametrize("result_key", ["generalized", "partial"])
def test_add_flexible_model_to_outputhub_attaches_ame_table(
    flexible_examples, result_key
):
    X, results = flexible_examples
    result = results[result_key]
    hub = outputhub.OutputHub("Flexible ordinal report")
    model = add_to_outputhub(hub, result, X=X)

    assert hub.models == [model]
    assert len(hub.tables) == 1
    table = hub.tables[0]
    assert table.name == f"{model.name} average marginal effects"
    assert table.metadata["estimator"] == model.metadata["estimator"]
    assert table.metadata["link"] == "logit"
    assert table.metadata["inference_valid"] is result.inference_valid
    assert set(table.data.columns) >= {
        "category",
        "feature",
        "estimate",
        "standard_error",
        "p_value",
    }


def test_random_effects_ordinal_outputhub_metadata():
    X, y, _ = fitted_example(nobs=300)
    entity = np.repeat(np.arange(60), 5)
    result = RandomEffectsOrderedLogit().fit(
        X, y, entity=entity, quadrature_points=8
    )
    model = to_outputhub_model(result)

    assert model.name == "Random-effects Ordered Logit"
    assert model.metadata["estimator"] == "random_effects_ordered_logit"
    assert model.metadata["n_entities"] == 60
    assert model.metadata["quadrature_points"] == 8
    assert model.diagnostics["Random-effect SD"] == result.sigma_entity

    hub = outputhub.OutputHub("Panel ordinal report")
    add_to_outputhub(hub, result)
    assert hub.models[0].metadata["backend"] == "native-ghq"
    with pytest.raises(NotImplementedError, match="not yet available"):
        add_to_outputhub(outputhub.OutputHub(), result, X=X)


def test_dynamic_ordinal_outputhub_metadata():
    X, y, _ = fitted_example(nobs=300)
    entity = np.repeat(np.arange(50), 6)
    time = np.tile(np.arange(6), 50)
    result = DynamicRandomEffectsOrderedLogit().fit(
        X, y, entity=entity, time=time, quadrature_points=6
    )
    model = to_outputhub_model(result)

    assert model.name == "Dynamic random-effects Ordered Logit"
    assert model.metadata["estimator"] == "dynamic_random_effects_ordered_logit"
    assert model.metadata["conditioned_initial_observations"] == 50
    assert model.metadata["truncated_gap_observations"] == 0
