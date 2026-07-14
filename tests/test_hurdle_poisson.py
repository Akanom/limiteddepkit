import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit.hurdle_poisson import (
    HurdlePoisson,
    _log_positive_poisson_probability,
    _zero_truncated_mean,
)


def _draw_zero_truncated_poisson(rng, means):
    draws = rng.poisson(means)
    zero = draws == 0
    while np.any(zero):
        draws[zero] = rng.poisson(means[zero])
        zero = draws == 0
    return draws


@pytest.fixture
def fitted_hurdle():
    rng = np.random.default_rng(23)
    nobs = 850
    index = pd.Index(np.arange(4_000, 4_000 + nobs), name="row")
    X = pd.DataFrame(
        {"const": 1.0, "x": rng.normal(size=nobs)},
        index=index,
    )
    Z = pd.DataFrame(
        {"const": 1.0, "z": rng.normal(size=nobs)},
        index=index,
    )
    positive_probability = expit(Z.to_numpy() @ np.array([0.2, -0.5]))
    count_mean = np.exp(X.to_numpy() @ np.array([0.1, 0.35]))
    positive_counts = _draw_zero_truncated_poisson(rng, count_mean)
    y = np.where(rng.uniform(size=nobs) < positive_probability, positive_counts, 0)
    return X, Z, y, HurdlePoisson().fit(X, y, X_hurdle=Z)


def test_hurdle_poisson_fits_with_factorized_inference(fitted_hurdle):
    X, Z, y, result = fitted_hurdle

    assert result.converged
    assert result.inference_valid
    assert result.backend == "experimental-native-mle"
    assert result.covariance_type == "observed-information"
    assert result.nobs == len(y)
    assert result.n_positive == np.count_nonzero(y)
    assert result.n_params == X.shape[1] + Z.shape[1]
    assert result.score_norm <= 1e-7
    assert list(result.all_params.index) == [
        "hurdle: const",
        "hurdle: z",
        "count: const",
        "count: x",
    ]
    assert np.isfinite(result.standard_errors).all()
    assert np.linalg.eigvalsh(result.covariance).min() > 0.0
    assert result.covariance.iloc[:2, 2:].to_numpy() == pytest.approx(np.zeros((2, 2)))
    assert result.conf_int().shape == (4, 2)
    assert list(result.summary_frame().columns) == ["coef", "std_err", "z", "p_value"]


def test_hurdle_predictions_obey_probability_and_mean_identities(fitted_hurdle):
    X, Z, _, result = fitted_hurdle
    X_eval = X.iloc[:25]
    Z_eval = Z.iloc[:25]
    positive_probability = result.predict_positive_probability(Z_eval)
    positive_mean = result.predict_positive_mean(X_eval)
    predicted = result.predict(X_eval, X_hurdle=Z_eval)
    zero_probability = result.predict_zero_probability(X_eval, X_hurdle=Z_eval)
    pmf = result.predict_pmf(X_eval, X_hurdle=Z_eval, max_count=30)

    assert predicted.index.equals(X_eval.index)
    assert predicted.to_numpy() == pytest.approx(
        (positive_probability * positive_mean).to_numpy()
    )
    assert zero_probability.to_numpy() == pytest.approx(
        (1.0 - positive_probability).to_numpy()
    )
    assert pmf[0].to_numpy() == pytest.approx(zero_probability.to_numpy())
    assert pmf.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(pmf)), abs=1e-10)
    assert pmf.to_numpy() @ pmf.columns.to_numpy(dtype=float) == pytest.approx(
        predicted.to_numpy(), abs=1e-9
    )


def test_zero_truncation_functions_remain_stable_for_tiny_poisson_means():
    linear_index = np.array([-1_000.0, -100.0, -30.0, -10.0, 0.0])
    mean = np.exp(linear_index)
    log_probability = _log_positive_poisson_probability(linear_index, mean)
    truncated_mean = _zero_truncated_mean(mean)

    assert np.isfinite(log_probability).all()
    assert np.isfinite(truncated_mean).all()
    assert log_probability[:3] == pytest.approx(linear_index[:3], abs=1e-10)
    assert truncated_mean[:3] == pytest.approx(np.ones(3), abs=1e-10)
    assert np.all(truncated_mean >= 1.0)


def test_hurdle_rejects_nonidentified_positive_count_equation():
    X = pd.DataFrame({"const": 1.0, "x": np.arange(10.0)})
    Z = pd.DataFrame({"const": 1.0}, index=X.index)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="every positive count equals one"):
        HurdlePoisson().fit(X, y, X_hurdle=Z)


def test_hurdle_rejects_separated_hurdle_equation():
    X = pd.DataFrame(
        {"const": 1.0, "x": [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]}
    )
    y = np.array([0, 0, 0, 0, 1, 2, 1, 3])
    with pytest.raises(ValueError, match="separation"):
        HurdlePoisson().fit(X, y)


def test_hurdle_validates_separate_design_schema(fitted_hurdle):
    X, Z, y, result = fitted_hurdle
    with pytest.raises(ValueError, match="rank deficient"):
        HurdlePoisson().fit(X.assign(copy=X["x"]), y, X_hurdle=Z)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["x", "const"]], X_hurdle=Z)
    with pytest.raises(ValueError, match="same prediction rows"):
        result.predict(X.iloc[:5], X_hurdle=Z.iloc[:4])
    with pytest.raises(ValueError, match="max_count"):
        result.predict_pmf(X.iloc[:5], X_hurdle=Z.iloc[:5], max_count=1.5)


def test_hurdle_default_uses_same_design_for_both_components():
    rng = np.random.default_rng(24)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=500)})
    positive = rng.binomial(1, expit(0.2 - 0.2 * X["x"].to_numpy()))
    counts = _draw_zero_truncated_poisson(
        rng, np.exp(0.1 + 0.25 * X["x"].to_numpy())
    )
    y = positive * counts
    result = HurdlePoisson().fit(X, y)
    assert result.feature_names == result.hurdle_feature_names
    assert np.isfinite(result.predict(X)).all()
