import numpy as np
import pandas as pd
import pytest

from limiteddepkit.experimental import CensoredQuantileRegression


def test_left_censored_quantile_recovers_parameters_and_exact_objective():
    rng = np.random.default_rng(42)
    n_obs = 1_000
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})
    beta = np.array([0.5, 1.0])
    latent = X.to_numpy() @ beta + rng.normal(scale=0.7, size=n_obs)
    y = np.maximum(0.0, latent)

    result = CensoredQuantileRegression(quantile=0.5, lower=0.0).fit(
        X, y, n_starts=5
    )

    assert result.converged
    assert not result.inference_valid
    assert result.covariance_type == "not-estimated"
    np.testing.assert_allclose(result.params, beta, atol=0.1)
    residual = y - np.maximum(0.0, X.to_numpy() @ result.params.to_numpy())
    expected_objective = 0.5 * np.sum(np.abs(residual))
    assert result.objective_value == pytest.approx(expected_objective, abs=1e-9)
    assert result.n_censored_left == np.sum(y == 0.0)
    assert result.conf_int().isna().all().all()

    prediction = result.predict(X.iloc[:8])
    latent_prediction = result.predict_latent(X.iloc[:8])
    np.testing.assert_allclose(prediction, np.maximum(0.0, latent_prediction))
    assert prediction.index.equals(X.index[:8])


def test_right_censored_quantile_recovers_parameters():
    rng = np.random.default_rng(123)
    n_obs = 900
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})
    beta = np.array([0.3, -0.8])
    latent = X.to_numpy() @ beta + rng.normal(scale=0.6, size=n_obs)
    y = np.minimum(1.0, latent)

    result = CensoredQuantileRegression(
        quantile=0.5, lower=None, upper=1.0
    ).fit(X, y, n_starts=5)

    np.testing.assert_allclose(result.params, beta, atol=0.1)
    assert result.n_censored_right == np.sum(y == 1.0)
    assert np.all(result.predict(X) <= 1.0)


def test_censored_quantile_pairs_bootstrap_enables_inference():
    rng = np.random.default_rng(81)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=350)})
    latent = 0.4 + 0.8 * X["x"].to_numpy() + rng.normal(scale=0.75, size=len(X))
    y = np.maximum(0.0, latent)

    result = CensoredQuantileRegression(lower=0.0).fit(
        X,
        y,
        n_starts=4,
        n_bootstrap=20,
        random_state=19,
    )

    assert result.inference_valid
    assert result.covariance_type == "iid-pairs-bootstrap"
    assert result.bootstrap_estimates is not None
    assert len(result.bootstrap_estimates) >= 16
    assert np.isfinite(result.standard_errors).all()
    intervals = result.conf_int()
    assert np.all(intervals["lower"] < intervals["upper"])


def test_censored_quantile_validates_bounds_identification_and_schema():
    with pytest.raises(ValueError, match="At least one censoring boundary"):
        CensoredQuantileRegression(lower=None, upper=None)
    with pytest.raises(ValueError, match="strictly less"):
        CensoredQuantileRegression(lower=1.0, upper=1.0)

    X = pd.DataFrame({"const": np.ones(20), "x": np.linspace(-1.0, 1.0, 20)})
    with pytest.raises(ValueError, match="below the lower"):
        CensoredQuantileRegression(lower=0.0).fit(X, np.linspace(-1.0, 1.0, 20))
    with pytest.raises(ValueError, match="Too few uncensored"):
        CensoredQuantileRegression(lower=0.0).fit(X, np.zeros(20))

    y = np.maximum(0.0, 0.5 + X["x"].to_numpy())
    result = CensoredQuantileRegression(lower=0.0).fit(X, y, n_starts=3)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["x", "const"]])
