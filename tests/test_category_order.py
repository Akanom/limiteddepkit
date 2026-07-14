"""Regression tests for explicit and pandas-provided ordinal category order."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import (
    DynamicRandomEffectsOrderedLogit,
    GeneralizedOrderedLogit,
    OrderedLogit,
    OrderedProbit,
    PartialProportionalOdds,
    RandomEffectsOrderedLogit,
    simulate_dynamic_random_effects_ordered_logit,
)

CATEGORY_ORDER = ("low", "medium", "high")
THRESHOLD_NAMES = ["low | medium", "medium | high"]


def _labels(codes):
    return np.asarray(CATEGORY_ORDER, dtype=object)[np.asarray(codes, dtype=int)]


def _ordered_categorical(labels):
    return pd.Series(
        pd.Categorical(labels, categories=CATEGORY_ORDER, ordered=True),
        name="outcome",
    )


def _make_cross_section(seed=9061, nobs=320):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-1.0, 1.0, size=nobs),
            "x2": rng.uniform(-1.0, 1.0, size=nobs),
        }
    )
    eta = X.to_numpy() @ np.array([0.7, -0.45])
    cumulative = expit(np.array([-0.75, 0.85])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1.0 - cumulative[:, 1]]
    )
    codes = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, _labels(codes)


def _assert_category_contract(result, X):
    assert list(result.categories) == list(CATEGORY_ORDER)
    assert list(result.thresholds.index) == THRESHOLD_NAMES
    probabilities = result.predict_proba(X.iloc[:12])
    assert list(probabilities.columns) == list(CATEGORY_ORDER)
    assert set(result.predict(X.iloc[:12])).issubset(CATEGORY_ORDER)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "estimator",
    [
        OrderedLogit(),
        OrderedProbit(),
        GeneralizedOrderedLogit(),
        PartialProportionalOdds(varying=["x1"]),
    ],
    ids=["ordered-logit", "ordered-probit", "generalized", "partial"],
)
def test_cross_section_estimators_honor_explicit_and_categorical_order(estimator):
    X, labels = _make_cross_section()
    assert list(CATEGORY_ORDER) != sorted(CATEGORY_ORDER)

    explicit = estimator.fit(X, labels, category_order=CATEGORY_ORDER)
    categorical = estimator.fit(X, _ordered_categorical(labels))

    _assert_category_contract(explicit, X)
    _assert_category_contract(categorical, X)
    np.testing.assert_allclose(
        explicit.predict_proba(X.iloc[:12]),
        categorical.predict_proba(X.iloc[:12]),
        rtol=1e-10,
        atol=1e-10,
    )


def _make_panel(seed=5108, n_entities=30, n_periods=4):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=entity.size),
            "x2": rng.normal(size=entity.size),
        }
    )
    random_effects = rng.normal(scale=0.55, size=n_entities)
    eta = X.to_numpy() @ np.array([0.65, -0.35]) + random_effects[entity]
    cumulative = expit(np.array([-0.8, 0.9])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1.0 - cumulative[:, 1]]
    )
    codes = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, _labels(codes), entity


def test_random_effects_model_honors_explicit_and_categorical_order():
    X, labels, entity = _make_panel()
    fit_options = {"entity": entity, "quadrature_points": 5, "maxiter": 500}

    explicit = RandomEffectsOrderedLogit().fit(
        X, labels, category_order=CATEGORY_ORDER, **fit_options
    )
    categorical = RandomEffectsOrderedLogit().fit(
        X, _ordered_categorical(labels), **fit_options
    )

    _assert_category_contract(explicit, X)
    _assert_category_contract(categorical, X)
    np.testing.assert_allclose(
        explicit.predict_proba(X.iloc[:12]),
        categorical.predict_proba(X.iloc[:12]),
        rtol=1e-9,
        atol=1e-9,
    )


def test_dynamic_model_honors_explicit_and_categorical_order():
    simulation = simulate_dynamic_random_effects_ordered_logit(
        n_entities=35, n_periods=5, seed=7214
    )
    labels = _labels(simulation.y)
    fit_options = {
        "entity": simulation.entity,
        "time": simulation.time,
        "quadrature_points": 4,
        "maxiter": 500,
    }

    explicit = DynamicRandomEffectsOrderedLogit().fit(
        simulation.X,
        labels,
        category_order=CATEGORY_ORDER,
        **fit_options,
    )
    categorical = DynamicRandomEffectsOrderedLogit().fit(
        simulation.X,
        _ordered_categorical(labels),
        **fit_options,
    )

    assert list(explicit.categories) == list(CATEGORY_ORDER)
    assert list(categorical.categories) == list(CATEGORY_ORDER)
    assert list(explicit.thresholds.index) == THRESHOLD_NAMES
    assert list(categorical.thresholds.index) == THRESHOLD_NAMES
    assert list(explicit.state_dependence_params.index) == ["state[medium]", "state[high]"]
    assert list(categorical.initial_condition_params.index) == [
        "initial[medium]",
        "initial[high]",
    ]
    np.testing.assert_allclose(
        explicit.fitted_probabilities,
        categorical.fitted_probabilities,
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("category_order", "message"),
    [
        (("low", "medium", "medium"), "unique labels"),
        (("low", "medium"), "each observed category exactly once"),
        (("low", None, "high"), "missing labels"),
    ],
)
def test_invalid_explicit_category_orders_are_rejected(category_order, message):
    X, labels = _make_cross_section(nobs=80)

    with pytest.raises(ValueError, match=message):
        OrderedLogit().fit(X, labels, category_order=category_order)


def test_unordered_categorical_requires_an_explicit_order():
    X, labels = _make_cross_section(nobs=80)
    unordered = pd.Series(
        pd.Categorical(labels, categories=CATEGORY_ORDER, ordered=False),
        name="outcome",
    )

    with pytest.raises(ValueError, match="Categorical y must be ordered"):
        OrderedLogit().fit(X, unordered)

    result = OrderedLogit().fit(X, unordered, category_order=CATEGORY_ORDER)
    assert list(result.categories) == list(CATEGORY_ORDER)


def test_diagnostics_reuse_the_fitted_category_order():
    X, labels = _make_cross_section(nobs=300)
    result = OrderedLogit().fit(X, labels, category_order=CATEGORY_ORDER)
    diagnostic = result.proportional_odds_test(X, labels)

    assert diagnostic.df == X.shape[1]
    assert 0 <= diagnostic.p_value <= 1


def test_explicit_order_preserves_heterogeneous_scalar_labels():
    X, labels = _make_cross_section(nobs=120)
    heterogeneous = np.array(
        [{"low": "low", "medium": 1, "high": "high"}[value] for value in labels],
        dtype=object,
    )
    result = OrderedLogit().fit(
        X, heterogeneous, category_order=["low", 1, "high"]
    )

    assert list(result.categories) == ["low", 1, "high"]
    assert list(result.predict_proba(X.iloc[:3]).columns) == ["low", 1, "high"]
