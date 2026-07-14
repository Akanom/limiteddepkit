import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

matplotlib.use("Agg")

from limiteddepkit import (
    GeneralizedOrderedLogit,
    OrderedLogit,
    PartialProportionalOdds,
    plot_marginal_effects,
    plot_probabilities,
)


def fitted_example(seed=881, nobs=500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)})
    eta = X.to_numpy() @ np.array([0.8, -0.5])
    cumulative = expit(np.array([-0.7, 0.8])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, OrderedLogit().fit(X, y)


@pytest.fixture(scope="module", params=["generalized", "partial"])
def flexible_fitted_example(request):
    rng = np.random.default_rng(428)
    X = pd.DataFrame(
        {"x1": rng.uniform(-1.0, 1.0, 400), "x2": rng.uniform(-1.0, 1.0, 400)}
    )
    eta = X.to_numpy() @ np.array([0.7, -0.4])
    cumulative = expit(np.array([-0.8, 0.9])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    if request.param == "generalized":
        estimator = GeneralizedOrderedLogit()
    else:
        estimator = PartialProportionalOdds(varying=["x1"])
    return X, estimator.fit(X, y)


def test_probability_plot_contract():
    X, result = fitted_example()
    grid = np.linspace(-1.0, 1.0, 15)
    ax = plot_probabilities(result, X, feature="x1", values=grid)

    assert len(ax.lines) == len(result.categories)
    assert ax.get_xlabel() == "x1"
    assert ax.get_ylabel() == "Predicted probability"
    assert np.array_equal(ax.lines[0].get_xdata(), grid)


def test_marginal_effect_plot_contract():
    X, result = fitted_example()
    ax = plot_marginal_effects(result, X, feature="x2")

    assert len(ax.lines) == len(result.categories) + 1
    assert ax.get_ylabel() == "Marginal effect"


def test_flexible_probability_plot_contract(flexible_fitted_example):
    X, result = flexible_fitted_example
    values = np.linspace(-0.5, 0.5, 13)
    ax = plot_probabilities(result, X, feature="x1", values=values)
    evaluation = pd.DataFrame(
        {"x1": values, "x2": np.repeat(X["x2"].mean(), values.size)}
    )
    expected = result.predict_proba(evaluation)

    assert len(ax.lines) == len(result.categories)
    assert [line.get_label() for line in ax.lines] == [
        str(category) for category in result.categories
    ]
    for line, category in zip(ax.lines, result.categories, strict=True):
        assert line.get_ydata() == pytest.approx(expected[category].to_numpy())


def test_flexible_marginal_effect_plot_contract(flexible_fitted_example):
    X, result = flexible_fitted_example
    values = np.linspace(-0.5, 0.5, 13)
    ax = plot_marginal_effects(result, X, feature="x1", values=values)
    evaluation = pd.DataFrame(
        {"x1": values, "x2": np.repeat(X["x2"].mean(), values.size)}
    )
    expected = result.marginal_effects(evaluation).xs("x1", axis=1, level="feature")

    assert len(ax.lines) == len(result.categories) + 1
    for line, category in zip(ax.lines[:-1], result.categories, strict=True):
        assert line.get_ydata() == pytest.approx(expected[category].to_numpy())
    assert np.all(np.asarray(ax.lines[-1].get_ydata()) == 0.0)


def test_plot_validation():
    X, result = fitted_example()

    with pytest.raises(ValueError, match="Unknown feature"):
        plot_probabilities(result, X, feature="unknown")
    with pytest.raises(ValueError, match="at least two"):
        plot_probabilities(result, X, feature="x1", values=[0.0])
