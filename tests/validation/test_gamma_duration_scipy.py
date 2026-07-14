import numpy as np
import pandas as pd
import pytest
from scipy.stats import gamma

from limiteddepkit.experimental import GammaDuration


@pytest.mark.validation
def test_gamma_duration_likelihood_matches_scipy_and_recovers_parameters():
    rng = np.random.default_rng(8521)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=1_800)})
    beta = np.array([0.2, -0.35])
    shape = 2.1
    scale = np.exp(X.to_numpy() @ beta)
    latent = rng.gamma(shape, scale)
    censoring = rng.exponential(scale=4.0, size=len(X))
    duration = np.minimum(latent, censoring)
    event = (latent <= censoring).astype(int)

    result = GammaDuration().fit(X, duration, event)

    fitted_scale = np.exp(X.to_numpy() @ result.params.to_numpy())
    expected_loglike = np.sum(
        event * gamma.logpdf(duration, result.shape_param, scale=fitted_scale)
        + (1 - event) * gamma.logsf(duration, result.shape_param, scale=fitted_scale)
    )
    assert result.converged
    assert result.inference_valid
    assert result.loglike == pytest.approx(expected_loglike, rel=1e-10, abs=1e-7)
    np.testing.assert_allclose(result.params, beta, atol=0.1)
    assert result.shape_param == pytest.approx(shape, abs=0.15)
    lower, upper = result.shape_conf_int()
    assert 0.0 < lower < result.shape_param < upper


def test_gamma_duration_rejects_fractional_event_and_bad_prediction_schema():
    X = pd.DataFrame(
        {"const": [1.0, 1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="binary"):
        GammaDuration().fit(X, [0.5, 1.0, 1.5, 2.0], [1, 0.5, 0, 1])

    fitted = GammaDuration().fit(X, [0.5, 1.0, 1.5, 2.0], [1, 1, 0, 1])
    with pytest.raises(ValueError, match="must contain 2 regressors"):
        fitted.predict(np.ones((3, 1)))
