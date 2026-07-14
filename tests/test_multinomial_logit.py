import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from limiteddepkit.experimental import MultinomialLogit


def make_multinomial_data(seed=11, nobs=4_000):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"row-{index}" for index in range(nobs)])
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=nobs),
            "x2": rng.normal(size=nobs),
        },
        index=index,
    )
    categories = ["outside", "bus", "car", "rail"]
    coefficients = np.array(
        [
            [0.25, 0.65, -0.30],
            [-0.35, -0.45, 0.55],
            [0.10, 0.20, 0.25],
        ]
    )
    utilities = np.column_stack(
        [np.zeros(nobs), X.to_numpy(dtype=float) @ coefficients.T]
    )
    probabilities = softmax(utilities, axis=1)
    y = np.array([rng.choice(categories, p=row) for row in probabilities], dtype=object)
    return X, y, categories, coefficients


def test_multinomial_logit_recovers_four_category_model_with_inference():
    X, y, categories, coefficients = make_multinomial_data()
    result = MultinomialLogit().fit(
        X,
        y,
        category_order=categories,
        base_category="outside",
    )

    assert result.converged
    assert result.inference_valid
    assert result.categories_list == categories
    assert result.base_category == "outside"
    assert result.params.shape == (3, 3)
    np.testing.assert_allclose(
        result.params.to_numpy(dtype=float).T, coefficients, atol=0.12, rtol=0.0
    )
    assert result.n_params == 9
    assert result.information_rank == result.n_params
    assert list(result.covariance.index) == list(result.all_params.index)
    assert result.covariance.to_numpy() == pytest.approx(
        result.covariance.to_numpy().T, abs=1e-12
    )
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(result.standard_errors > 0)
    assert result.summary_frame().shape == (result.n_params, 4)
    assert result.conf_int().shape == (result.n_params, 2)


def test_multinomial_predictions_preserve_labels_index_and_remain_stable():
    X, y, categories, _ = make_multinomial_data(nobs=1_200)
    result = MultinomialLogit().fit(X, y, category_order=categories)
    sample = X.iloc[:40]

    probabilities = result.predict_proba(sample)
    predictions = result.predict(sample)

    assert list(probabilities.columns) == categories
    assert probabilities.index.equals(sample.index)
    assert predictions.index.equals(sample.index)
    assert set(predictions) <= set(categories)
    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-12)

    extreme = sample.iloc[:2].copy()
    extreme.loc[:, ["x1", "x2"]] = [[1_000.0, -1_000.0], [-1_000.0, 1_000.0]]
    extreme_probabilities = result.predict_proba(extreme)
    assert np.all(np.isfinite(extreme_probabilities.to_numpy()))
    assert extreme_probabilities.sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-12)


def test_multinomial_supports_numeric_default_order_and_nonfirst_base():
    X, y, _, _ = make_multinomial_data(nobs=1_000)
    numeric = pd.Series(y).map({"outside": 0, "bus": 1, "car": 2, "rail": 3})
    result = MultinomialLogit().fit(X.reset_index(drop=True), numeric, base_category=2)

    assert result.categories_list == [0, 1, 2, 3]
    assert result.base_category == 2
    assert list(result.params.columns) == [0, 1, 3]
    assert list(result.predict_proba(X.iloc[:5].reset_index(drop=True)).columns) == [0, 1, 2, 3]


def test_multinomial_rejects_unidentified_or_ambiguous_inputs():
    X, y, categories, _ = make_multinomial_data(nobs=300)

    with pytest.raises(ValueError, match="same number"):
        MultinomialLogit().fit(X, y[:-1], category_order=categories)
    with pytest.raises(ValueError, match="each observed category"):
        MultinomialLogit().fit(X, y, category_order=categories[:-1])
    with pytest.raises(ValueError, match="base_category"):
        MultinomialLogit().fit(X, y, category_order=categories, base_category="missing")
    with pytest.raises(ValueError, match="full column rank"):
        MultinomialLogit().fit(X.assign(copy=X["x1"]), y, category_order=categories)
    with pytest.raises(ValueError, match="columns must match"):
        MultinomialLogit().fit(X, y, category_order=categories).predict_proba(
            X[["const", "x2", "x1"]]
        )
    with pytest.raises(ValueError, match="at least two"):
        MultinomialLogit().fit(X.iloc[:20], np.repeat("only", 20))
