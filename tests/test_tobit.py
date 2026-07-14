import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

from limiteddepkit.experimental import Tobit


def _censored_sample(seed=13, nobs=700):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    latent = 0.5 + 0.8 * X["x"].to_numpy() + rng.normal(size=nobs)
    return X, np.maximum(latent, 0.0)


def test_tobit_recovers_parameters_and_matches_manual_likelihood():
    X, y = _censored_sample()
    result = Tobit().fit(X, y)

    np.testing.assert_allclose(result.params, [0.5, 0.8], atol=0.1)
    assert result.sigma == pytest.approx(1.0, abs=0.1)
    assert result.n_censored == int(np.sum(y == 0.0))

    mean = X.to_numpy() @ result.params.to_numpy()
    uncensored = y > 0.0
    manual_loglike = np.sum(
        norm.logpdf(y[uncensored], loc=mean[uncensored], scale=result.sigma)
    )
    manual_loglike += np.sum(log_ndtr(-mean[~uncensored] / result.sigma))
    assert result.loglike == pytest.approx(manual_loglike, abs=1e-8)


def test_tobit_result_contract_includes_dispersion_and_observed_mean_prediction():
    X, y = _censored_sample(nobs=450)
    result = Tobit().fit(X, y)

    labels = ["const", "x", "sigma"]
    assert result.all_params.index.tolist() == labels
    assert result.covariance.index.tolist() == labels
    assert result.standard_errors.index.tolist() == labels
    assert np.isfinite(result.standard_errors).all()
    assert result.conf_int().loc["sigma", "lower"] > 0.0
    assert result.summary_frame().index.tolist() == labels
    assert result.n_params == 3
    assert result.df_resid == result.nobs - 3
    assert np.isfinite([result.aic, result.bic]).all()

    subset = X.iloc[:12].copy()
    latent = result.predict(subset, which="latent")
    probability = result.predict(subset, which="censoring_probability")
    observed = result.predict(subset)
    standardized = -latent.to_numpy() / result.sigma
    expected = (
        latent.to_numpy() * (1.0 - ndtr(standardized))
        + result.sigma * norm.pdf(standardized)
    )
    np.testing.assert_allclose(probability, ndtr(standardized), rtol=1e-12)
    np.testing.assert_allclose(observed, expected, rtol=1e-12)
    assert observed.index.equals(subset.index)


def test_tobit_without_censoring_reduces_to_gaussian_mle():
    rng = np.random.default_rng(1301)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=260)})
    y = 0.3 - 0.6 * X["x"].to_numpy() + rng.normal(scale=1.2, size=len(X))
    result = Tobit(censoring_point=-100.0).fit(X, y)

    ols = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    mle_sigma = np.sqrt(np.mean((y - X.to_numpy() @ ols) ** 2))
    np.testing.assert_allclose(result.params, ols, atol=1e-7)
    assert result.sigma == pytest.approx(mle_sigma, abs=1e-7)


def test_tobit_rejects_invalid_outcomes_designs_and_prediction_schema():
    X, y = _censored_sample(nobs=80)
    with pytest.raises(ValueError, match="below censoring_point"):
        Tobit().fit(X, np.where(np.arange(len(y)) == 0, -0.1, y))
    with pytest.raises(ValueError, match="uncensored"):
        Tobit().fit(X, np.zeros(len(y)))
    with pytest.raises(ValueError, match="one-dimensional"):
        Tobit().fit(X, y[:, None])
    with pytest.raises(ValueError, match="rank deficient"):
        Tobit().fit(pd.DataFrame({"x1": X["x"], "x2": X["x"]}), y)
    with pytest.raises(ValueError, match="finite"):
        Tobit(censoring_point=np.inf)

    result = Tobit().fit(X, y)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["x", "const"]])
    with pytest.raises(ValueError, match="which"):
        result.predict(X, which="clipped")
