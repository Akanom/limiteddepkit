import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr
from scipy.stats import norm

from limiteddepkit.experimental import IntervalRegression
from limiteddepkit.interval_regression import _log_interval_probability


def _latent_sample(seed=44, nobs=1_000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    latent = X.to_numpy() @ np.array([1.5, 0.8]) + rng.normal(0.0, 0.9, nobs)
    return X, latent


def test_interval_regression_recovers_parameters_from_grouped_outcomes():
    X, latent = _latent_sample()
    lower = np.floor(latent * 2.0) / 2.0
    upper = lower + 0.5
    result = IntervalRegression().fit(X, lower, upper)

    np.testing.assert_allclose(result.params, [1.5, 0.8], atol=0.06)
    assert result.sigma == pytest.approx(0.9, abs=0.06)
    assert result.n_interval == len(X)
    assert result.n_exact == result.n_left_censored == result.n_right_censored == 0

    mean = X.to_numpy() @ result.params.to_numpy()
    standardized_lower = (lower - mean) / result.sigma
    standardized_upper = (upper - mean) / result.sigma
    manual = np.sum(
        np.log(norm.cdf(standardized_upper) - norm.cdf(standardized_lower))
    )
    assert result.loglike == pytest.approx(manual, abs=1e-8)


def test_exact_observations_use_density_and_reduce_to_gaussian_mle():
    X, exact = _latent_sample(seed=4401, nobs=320)
    result = IntervalRegression().fit(X, exact, exact)

    ols = np.linalg.lstsq(X.to_numpy(), exact, rcond=None)[0]
    mle_sigma = np.sqrt(np.mean((exact - X.to_numpy() @ ols) ** 2))
    np.testing.assert_allclose(result.params, ols, atol=1e-7)
    assert result.sigma == pytest.approx(mle_sigma, abs=1e-7)
    assert result.n_exact == len(X)

    manual = np.sum(norm.logpdf(exact, loc=X.to_numpy() @ ols, scale=mle_sigma))
    assert result.loglike == pytest.approx(manual, abs=1e-8)


def test_mixed_exact_interval_and_one_sided_contributions_match_manual_likelihood():
    X, latent = _latent_sample(seed=4402, nobs=600)
    lower = latent.copy()
    upper = latent.copy()
    groups = np.arange(len(X)) % 3

    interval = groups == 0
    lower[interval] = np.floor(latent[interval] * 2.0) / 2.0
    upper[interval] = lower[interval] + 0.5
    left = (groups == 1) & (latent <= 1.0)
    lower[left] = -np.inf
    upper[left] = 1.0
    right = (groups == 2) & (latent >= 2.0)
    lower[right] = 2.0
    upper[right] = np.inf

    result = IntervalRegression().fit(X, lower, upper)
    exact = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    finite_interval = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
    mean = X.to_numpy() @ result.params.to_numpy()
    manual = np.sum(
        norm.logpdf(lower[exact], loc=mean[exact], scale=result.sigma)
    )
    manual += np.sum(
        np.log(
            norm.cdf((upper[finite_interval] - mean[finite_interval]) / result.sigma)
            - norm.cdf((lower[finite_interval] - mean[finite_interval]) / result.sigma)
        )
    )
    manual += np.sum(log_ndtr((upper[left] - mean[left]) / result.sigma))
    manual += np.sum(log_ndtr(-(lower[right] - mean[right]) / result.sigma))

    assert result.loglike == pytest.approx(manual, abs=1e-8)
    assert result.n_exact == int(np.sum(exact))
    assert result.n_interval == int(np.sum(finite_interval))
    assert result.n_left_censored == int(np.sum(left))
    assert result.n_right_censored == int(np.sum(right))


def test_interval_tail_probabilities_remain_finite_and_symmetric():
    lower = np.array([-40.0, 39.5, -1e-10])
    upper = np.array([-39.5, 40.0, 1e-10])
    values = _log_interval_probability(lower, upper)

    assert np.isfinite(values).all()
    assert values[0] == pytest.approx(values[1], rel=1e-12)
    assert values[2] == pytest.approx(np.log(norm.pdf(0.0) * 2e-10), rel=1e-6)


def test_interval_result_contract_prediction_and_schema_validation():
    X, latent = _latent_sample(seed=4403, nobs=350)
    result = IntervalRegression().fit(X, latent, latent)
    labels = ["const", "x", "sigma"]

    assert result.all_params.index.tolist() == labels
    assert result.covariance.index.tolist() == labels
    assert np.isfinite(result.standard_errors).all()
    assert result.conf_int().loc["sigma", "lower"] > 0.0
    assert result.summary_frame().index.tolist() == labels
    prediction = result.predict(X.iloc[:10])
    predictive_interval = result.predict_interval(X.iloc[:10], level=0.9)
    assert prediction.name == "predicted_latent"
    assert prediction.index.equals(X.index[:10])
    assert (predictive_interval["lower"] < prediction).all()
    assert (prediction < predictive_interval["upper"]).all()

    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["x", "const"]])
    with pytest.raises(ValueError, match="level"):
        result.predict_interval(X, level=1.0)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0], "less than or equal"),
        ([-np.inf] * 4, [np.inf] * 4, "no outcome information"),
        ([np.inf] * 4, [np.inf] * 4, "lower cannot"),
        ([0.0, np.nan, 0.0, 0.0], [1.0] * 4, "missing"),
    ],
)
def test_interval_regression_rejects_invalid_bounds(lower, upper, message):
    X = pd.DataFrame({"const": 1.0, "x": [-1.0, -0.2, 0.4, 1.1]})
    with pytest.raises(ValueError, match=message):
        IntervalRegression().fit(X, lower, upper)


def test_interval_regression_rejects_nonvector_bounds_and_unidentified_designs():
    X, latent = _latent_sample(seed=4404, nobs=80)
    with pytest.raises(ValueError, match="one-dimensional"):
        IntervalRegression().fit(X, latent[:, None], latent[:, None])
    with pytest.raises(ValueError, match="rank deficient"):
        IntervalRegression().fit(
            pd.DataFrame({"x1": X["x"], "x2": X["x"]}), latent, latent
        )
