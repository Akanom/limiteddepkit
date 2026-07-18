import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from limiteddepkit.count import NegativeBinomial, PoissonRegressor


def _data(seed=831, nobs=600):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=nobs),
            "x2": rng.normal(size=nobs),
        }
    )
    offset = rng.normal(scale=0.15, size=nobs)
    exposure = rng.uniform(0.4, 2.2, size=nobs)
    mean = np.exp(X.to_numpy() @ np.array([0.05, 0.4, -0.25]) + offset) * exposure
    return rng, X, offset, exposure, mean


@pytest.mark.validation
def test_poisson_offset_exposure_and_frequency_weights_match_statsmodels_glm():
    rng, X, offset, exposure, mean = _data()
    y = rng.poisson(mean)
    weights = rng.integers(1, 4, size=len(X))

    result = PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
    )
    reference = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
    ).fit()

    np.testing.assert_allclose(result.params, reference.params, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(result.covariance, reference.cov_params(), rtol=2e-7, atol=2e-8)
    assert result.loglike == pytest.approx(reference.llf, rel=1e-11, abs=1e-8)
    assert result.aic == pytest.approx(reference.aic, rel=1e-11, abs=1e-8)
    assert result.bic == pytest.approx(reference.bic_llf, rel=1e-11, abs=1e-8)


@pytest.mark.validation
def test_poisson_analytic_weights_match_statsmodels_var_weights():
    rng, X, offset, exposure, mean = _data(seed=832)
    y = rng.poisson(mean)
    weights = rng.uniform(0.4, 2.0, size=len(X))

    result = PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        analytic_weights=weights,
    )
    reference = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=offset,
        exposure=exposure,
        var_weights=weights,
    ).fit()

    np.testing.assert_allclose(result.params, reference.params, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result.covariance, reference.cov_params(), rtol=2e-6, atol=2e-7)
    assert result.loglike == pytest.approx(reference.llf, rel=1e-10, abs=1e-7)

    robust_result = PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        analytic_weights=weights,
        cov_type="HC0",
    )
    robust_reference = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=offset,
        exposure=exposure,
        var_weights=weights,
    ).fit(cov_type="HC0")
    np.testing.assert_allclose(
        robust_result.covariance,
        robust_reference.cov_params(),
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.validation
@pytest.mark.parametrize("cov_type", ["HC0", "cluster"])
def test_poisson_robust_covariance_matches_statsmodels(cov_type):
    rng, X, offset, exposure, mean = _data(seed=833)
    y = rng.poisson(mean)
    groups = np.repeat(np.arange(len(X) // 6), 6)
    fit_kwargs = {}
    reference_kwargs = {}
    if cov_type == "cluster":
        fit_kwargs = {"clusters": groups}
        reference_kwargs = {
            "cov_kwds": {"groups": groups, "use_correction": True}
        }

    result = PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        cov_type=cov_type,
        **fit_kwargs,
    )
    reference = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=offset,
        exposure=exposure,
    ).fit(cov_type=cov_type, **reference_kwargs)

    np.testing.assert_allclose(result.covariance, reference.cov_params(), rtol=2e-6, atol=2e-7)


def _transform_nb2_covariance(reference):
    alpha = float(reference.params.iloc[-1])
    transform = np.eye(len(reference.params))
    transform[-1, -1] = 1.0 / alpha
    return transform @ reference.cov_params() @ transform.T


@pytest.mark.validation
def test_nb2_offset_exposure_likelihood_and_covariance_match_statsmodels():
    rng, X, offset, exposure, mean = _data(seed=901, nobs=850)
    alpha = 0.7
    y = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))

    result = NegativeBinomial().fit(X, y, offset=offset, exposure=exposure)
    reference = sm.NegativeBinomial(
        y,
        X,
        offset=offset,
        exposure=exposure,
        loglike_method="nb2",
    ).fit(disp=False, method="newton", maxiter=500)

    np.testing.assert_allclose(result.params, reference.params.iloc[:-1], rtol=2e-5, atol=2e-5)
    assert result.alpha == pytest.approx(float(reference.params.iloc[-1]), rel=2e-5, abs=2e-5)
    assert result.loglike == pytest.approx(reference.llf, rel=1e-9, abs=2e-6)
    np.testing.assert_allclose(
        result.covariance,
        _transform_nb2_covariance(reference),
        rtol=2e-3,
        atol=2e-4,
    )


@pytest.mark.validation
@pytest.mark.parametrize("cov_type", ["HC0", "cluster"])
def test_nb2_robust_covariance_matches_statsmodels(cov_type):
    rng, X, offset, exposure, mean = _data(seed=902, nobs=720)
    alpha = 0.5
    y = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))
    groups = np.repeat(np.arange(len(X) // 6), 6)
    fit_kwargs = {}
    reference_fit_kwargs = {"cov_type": cov_type}
    if cov_type == "cluster":
        fit_kwargs = {"clusters": groups}
        reference_fit_kwargs["cov_kwds"] = {
            "groups": groups,
            "use_correction": True,
        }

    result = NegativeBinomial().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        cov_type=cov_type,
        **fit_kwargs,
    )
    reference = sm.NegativeBinomial(
        y,
        X,
        offset=offset,
        exposure=exposure,
        loglike_method="nb2",
    ).fit(
        disp=False,
        method="newton",
        maxiter=500,
        **reference_fit_kwargs,
    )

    np.testing.assert_allclose(
        result.covariance,
        _transform_nb2_covariance(reference),
        rtol=3e-3,
        atol=3e-4,
    )


@pytest.mark.validation
def test_nb2_frequency_weights_match_statsmodels_expanded_rows():
    rng, X, offset, exposure, mean = _data(seed=903, nobs=320)
    alpha = 0.8
    y = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))
    weights = rng.integers(1, 4, size=len(X))
    repeated = np.repeat(np.arange(len(X)), weights)

    result = NegativeBinomial().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
    )
    reference = sm.NegativeBinomial(
        y[repeated],
        X.iloc[repeated].reset_index(drop=True),
        offset=offset[repeated],
        exposure=exposure[repeated],
        loglike_method="nb2",
    ).fit(disp=False, method="newton", maxiter=500)

    np.testing.assert_allclose(result.params, reference.params.iloc[:-1], rtol=3e-5, atol=3e-5)
    assert result.alpha == pytest.approx(float(reference.params.iloc[-1]), rel=3e-5, abs=3e-5)
    assert result.loglike == pytest.approx(reference.llf, rel=1e-9, abs=3e-6)
