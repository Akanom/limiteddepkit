import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from limiteddepkit.experimental import GaussianMixtureRegression


@pytest.mark.validation
def test_gaussian_mixture_regression_recovers_components_and_exact_likelihood():
    rng = np.random.default_rng(9773)
    n_obs = 1_200
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})
    probabilities = np.array([0.4, 0.6])
    betas = np.array([[-1.2, 0.25], [1.7, -0.45]])
    sigmas = np.array([0.7, 1.0])
    component = rng.choice(2, size=n_obs, p=probabilities)
    means = np.sum(X.to_numpy() * betas[component], axis=1)
    y = means + rng.normal(scale=sigmas[component])

    result = GaussianMixtureRegression(
        n_regimes=2, n_starts=6, random_state=123
    ).fit(X, y)

    fitted_betas = np.vstack([params.to_numpy() for params in result.params_regimes])
    assert result.converged
    assert result.inference_valid
    np.testing.assert_allclose(fitted_betas, betas, atol=0.14)
    np.testing.assert_allclose(result.sigma_regimes, sigmas, atol=0.12)
    np.testing.assert_allclose(result.mixture_probs, probabilities, atol=0.07)

    fitted_means = result.predict_component_means(X).to_numpy()
    log_joint = np.log(result.mixture_probs)[None, :] + norm.logpdf(
        y[:, None], loc=fitted_means, scale=np.asarray(result.sigma_regimes)[None, :]
    )
    expected_loglike = np.sum(logsumexp(log_joint, axis=1))
    assert result.loglike == pytest.approx(expected_loglike, rel=1e-11, abs=1e-8)
    membership = result.predict_membership(X, y)
    np.testing.assert_allclose(membership.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        result.predict(X), fitted_means @ result.mixture_probs, atol=1e-12
    )


def test_gaussian_mixture_regression_validates_identification_and_schema():
    with pytest.raises(ValueError, match="at least two"):
        GaussianMixtureRegression(n_regimes=1)

    X = pd.DataFrame({"const": np.ones(30), "x": np.linspace(-1.0, 1.0, 30)})
    y = np.r_[np.linspace(-2.0, -1.0, 15), np.linspace(1.0, 2.0, 15)]
    result = GaussianMixtureRegression(n_starts=2).fit(X, y)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["x", "const"]])
