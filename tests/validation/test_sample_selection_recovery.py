import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr
from scipy.stats import norm

from limiteddepkit.experimental import SampleSelection


@pytest.mark.validation
def test_sample_selection_recovers_correlated_equations_and_exact_likelihood():
    rng = np.random.default_rng(4140)
    n_obs = 2_500
    x = rng.normal(size=n_obs)
    excluded = rng.normal(size=n_obs)
    X = pd.DataFrame({"const": 1.0, "x": x})
    Z = pd.DataFrame({"const": 1.0, "x": x, "excluded": excluded})
    beta = np.array([0.8, 0.55])
    gamma = np.array([-0.15, 0.3, 0.75])
    sigma = 1.1
    rho = 0.45

    selection_error = rng.normal(size=n_obs)
    outcome_innovation = rng.normal(size=n_obs)
    outcome_error = sigma * (
        rho * selection_error + np.sqrt(1.0 - rho**2) * outcome_innovation
    )
    selected = Z.to_numpy() @ gamma + selection_error > 0.0
    latent_y = X.to_numpy() @ beta + outcome_error
    y = np.where(selected, latent_y, np.nan)

    result = SampleSelection().fit(X, y, Z)

    assert result.converged
    assert result.inference_valid
    np.testing.assert_allclose(result.params_outcome, beta, atol=0.12)
    np.testing.assert_allclose(result.params_selection, gamma, atol=0.12)
    assert result.sigma == pytest.approx(sigma, abs=0.12)
    assert result.rho == pytest.approx(rho, abs=0.15)

    residual = (
        y[selected] - X.to_numpy()[selected] @ result.params_outcome.to_numpy()
    ) / result.sigma
    selection_index = Z.to_numpy() @ result.params_selection.to_numpy()
    conditional_index = (
        selection_index[selected] + result.rho * residual
    ) / np.sqrt(1.0 - result.rho**2)
    expected_loglike = np.sum(
        norm.logpdf(residual) - np.log(result.sigma) + log_ndtr(conditional_index)
    ) + np.sum(log_ndtr(-selection_index[~selected]))
    assert result.loglike == pytest.approx(expected_loglike, rel=1e-10, abs=1e-7)

    sigma_lower, sigma_upper = result.sigma_conf_int()
    rho_lower, rho_upper = result.rho_conf_int()
    assert 0.0 < sigma_lower < result.sigma < sigma_upper
    assert -1.0 < rho_lower < result.rho < rho_upper < 1.0


def test_sample_selection_requires_the_full_selection_sample():
    X = pd.DataFrame({"const": np.ones(8), "x": np.arange(8.0)})
    Z = pd.DataFrame(
        {"const": np.ones(8), "x": np.arange(8.0), "z": np.tile([0.0, 1.0], 4)}
    )
    with pytest.raises(ValueError, match="Both selected and unselected"):
        SampleSelection().fit(X, np.arange(8.0), Z)
