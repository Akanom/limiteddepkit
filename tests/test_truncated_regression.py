import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

from limiteddepkit import TruncatedRegression


def _truncated_sample(seed=3701, nobs=2_500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    latent = X.to_numpy() @ np.array([0.4, 0.6]) + rng.normal(0.0, 1.2, nobs)
    retained = latent > 0.0
    return X.loc[retained].reset_index(drop=True), latent[retained]


def test_truncated_regression_recovers_parameters_and_matches_manual_likelihood():
    X, y = _truncated_sample()
    result = TruncatedRegression().fit(X, y)

    np.testing.assert_allclose(result.params, [0.4, 0.6], atol=0.08)
    assert result.sigma == pytest.approx(1.2, abs=0.08)

    mean = X.to_numpy() @ result.params.to_numpy()
    standardized = (y - mean) / result.sigma
    truncation_index = -mean / result.sigma
    manual_loglike = np.sum(
        norm.logpdf(standardized)
        - np.log(result.sigma)
        - log_ndtr(-truncation_index)
    )
    assert result.loglike == pytest.approx(manual_loglike, abs=1e-8)


def test_truncated_regression_loose_tolerance_still_requires_stationarity():
    X, y = _truncated_sample(nobs=1_200)
    result = TruncatedRegression().fit(X, y, tolerance=1e6)

    assert result.converged
    assert result.inference_valid
    assert result.scaled_score_norm <= 1e-4
    assert result.optimizer_result.nit > 0


def test_truncated_result_contract_and_prediction_semantics():
    X, y = _truncated_sample(nobs=900)
    result = TruncatedRegression().fit(X, y)

    labels = ["const", "x", "sigma"]
    assert result.all_params.index.tolist() == labels
    assert result.covariance.index.tolist() == labels
    assert np.isfinite(result.standard_errors).all()
    assert result.conf_int().loc["sigma", "lower"] > 0.0
    assert result.summary_frame().index.tolist() == labels
    assert result.covariance_type == "observed-information"

    subset = X.iloc[:15]
    latent = result.predict(subset, which="latent")
    probability = result.predict(subset, which="selection_probability")
    conditional = result.predict(subset)
    truncation_index = -latent.to_numpy() / result.sigma
    mills = np.exp(norm.logpdf(truncation_index) - log_ndtr(-truncation_index))
    np.testing.assert_allclose(probability, ndtr(-truncation_index), rtol=1e-12)
    np.testing.assert_allclose(
        conditional,
        latent.to_numpy() + result.sigma * mills,
        rtol=1e-12,
    )
    assert np.all(conditional > result.truncation_point)


def test_distant_truncation_reduces_to_gaussian_mle():
    rng = np.random.default_rng(3711)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=260)})
    y = 0.3 - 0.6 * X["x"].to_numpy() + rng.normal(scale=1.2, size=len(X))
    result = TruncatedRegression(truncation_point=-100.0).fit(X, y)

    ols = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    mle_sigma = np.sqrt(np.mean((y - X.to_numpy() @ ols) ** 2))
    np.testing.assert_allclose(result.params, ols, atol=1e-7)
    assert result.sigma == pytest.approx(mle_sigma, abs=1e-7)


def test_truncated_regression_validates_support_design_and_prediction_schema():
    X, y = _truncated_sample(nobs=180)
    invalid_y = y.copy()
    invalid_y[0] = 0.0
    with pytest.raises(ValueError, match="strictly greater"):
        TruncatedRegression().fit(X, invalid_y)
    with pytest.raises(ValueError, match="one-dimensional"):
        TruncatedRegression().fit(X, y[:, None])
    with pytest.raises(ValueError, match="rank deficient"):
        TruncatedRegression().fit(
            pd.DataFrame({"x1": X["x"], "x2": X["x"]}), y
        )
    with pytest.raises(ValueError, match="finite"):
        TruncatedRegression(truncation_point=-np.inf)

    result = TruncatedRegression().fit(X, y)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["x", "const"]])
    with pytest.raises(ValueError, match="which"):
        result.predict(X, which="unconditional")


def test_right_truncation_matches_reflected_left_truncation_problem():
    rng = np.random.default_rng(3712)
    full_X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=2_000)})
    latent = 0.4 + 0.6 * full_X["x"].to_numpy() + rng.normal(
        scale=1.2,
        size=len(full_X),
    )
    retained = latent < 0.0
    X = full_X.loc[retained].reset_index(drop=True)
    y = latent[retained]

    right = TruncatedRegression(side="right").fit(X, y)
    reflected = TruncatedRegression(side="left").fit(X, -y)

    np.testing.assert_allclose(right.params, -reflected.params, rtol=1e-10, atol=1e-10)
    assert right.sigma == pytest.approx(reflected.sigma, rel=1e-12)
    assert right.loglike == pytest.approx(reflected.loglike, abs=1e-10)
    np.testing.assert_allclose(
        right.predict(X.iloc[:20]),
        -reflected.predict(X.iloc[:20]),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.all(right.predict(X.iloc[:20]) < right.truncation_point)
    with pytest.raises(ValueError, match="side"):
        TruncatedRegression(side="both")


def test_truncated_regression_supports_robust_and_cluster_covariance():
    X, y = _truncated_sample(seed=3713, nobs=1_200)
    robust = TruncatedRegression().fit(X, y, covariance_type="robust")
    clusters = np.arange(len(X)) // 9
    clustered = TruncatedRegression().fit(
        X,
        y,
        covariance_type="cluster",
        clusters=clusters,
    )

    assert robust.covariance_type == "robust"
    assert clustered.covariance_type == "cluster"
    assert clustered.n_clusters == len(np.unique(clusters))
    assert np.isfinite(robust.standard_errors).all()
    assert np.isfinite(clustered.standard_errors).all()
