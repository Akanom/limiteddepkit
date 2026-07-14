import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit.experimental import SequentialLogit


def make_sequential_data(seed=41, nobs=5_000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"const": 1.0, "x": rng.normal(size=nobs)},
        index=pd.Index([f"case-{index}" for index in range(nobs)]),
    )
    categories = ["exit", "screen", "interview", "offer"]
    coefficients = np.array(
        [
            [-0.25, 0.55],
            [0.30, -0.40],
            [-0.10, 0.25],
        ]
    )
    stop_probabilities = expit(X.to_numpy() @ coefficients.T)
    probabilities = np.empty((nobs, len(categories)), dtype=float)
    remaining = np.ones(nobs, dtype=float)
    for stage in range(len(categories) - 1):
        probabilities[:, stage] = remaining * stop_probabilities[:, stage]
        remaining *= 1.0 - stop_probabilities[:, stage]
    probabilities[:, -1] = remaining
    y = np.array([rng.choice(categories, p=row) for row in probabilities], dtype=object)
    return X, y, categories, coefficients


def test_sequential_logit_recovers_four_category_continuation_model():
    X, y, categories, coefficients = make_sequential_data()
    result = SequentialLogit().fit(X, y, category_order=categories)

    assert result.converged
    assert result.inference_valid
    assert result.categories == tuple(categories)
    assert result.params.shape == (2, 3)
    np.testing.assert_allclose(
        result.params.to_numpy(dtype=float).T, coefficients, atol=0.10, rtol=0.0
    )
    assert result.stage_sample_sizes.index.tolist() == categories[:-1]
    assert result.stage_sample_sizes.is_monotonic_decreasing
    assert result.information_rank == result.n_params == 6
    assert result.covariance.to_numpy() == pytest.approx(
        result.covariance.to_numpy().T, abs=1e-12
    )
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(result.standard_errors > 0)
    assert result.summary_frame().shape == (result.n_params, 4)
    assert result.conf_int().shape == (result.n_params, 2)


def test_sequential_probabilities_preserve_category_labels_and_index():
    X, y, categories, _ = make_sequential_data(nobs=1_200)
    result = SequentialLogit().fit(X, y, category_order=categories)
    sample = X.iloc[:50]

    probabilities = result.predict_proba(sample)
    predictions = result.predict(sample)

    assert probabilities.index.equals(sample.index)
    assert predictions.index.equals(sample.index)
    assert list(probabilities.columns) == categories
    assert set(predictions) <= set(categories)
    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-12)


def test_sequential_logit_recognizes_ordered_categorical_and_binary_outcomes():
    X, y, categories, _ = make_sequential_data(nobs=1_000)
    categorical = pd.Series(
        pd.Categorical(y, categories=categories, ordered=True), index=X.index
    )
    categorical_result = SequentialLogit().fit(X, categorical)
    assert categorical_result.categories == tuple(categories)

    binary = np.where(categorical == categories[0], "stop", "continue")
    binary_result = SequentialLogit().fit(
        X,
        binary,
        category_order=["stop", "continue"],
    )
    assert binary_result.params.shape == (2, 1)
    assert binary_result.predict_proba(X.iloc[:10]).shape == (10, 2)


def test_sequential_logit_validates_order_design_and_prediction_schema():
    X, y, categories, _ = make_sequential_data(nobs=400)

    with pytest.raises(ValueError, match="same number"):
        SequentialLogit().fit(X, y[:-1], category_order=categories)
    with pytest.raises(ValueError, match="each observed category"):
        SequentialLogit().fit(X, y, category_order=categories[:-1])
    with pytest.raises(ValueError, match="ordered"):
        SequentialLogit().fit(X, pd.Series(pd.Categorical(y, categories=categories)))

    duplicate_columns = X.copy()
    duplicate_columns.columns = ["x", "x"]
    with pytest.raises(ValueError, match="unique feature names"):
        SequentialLogit().fit(duplicate_columns, y, category_order=categories)

    result = SequentialLogit().fit(X, y, category_order=categories)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict_proba(X[["x", "const"]])
    with pytest.raises(ValueError, match="strictly between"):
        result.conf_int(level=1.0)
