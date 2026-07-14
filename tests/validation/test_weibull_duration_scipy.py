import numpy as np
import pandas as pd
import pytest
from scipy.stats import weibull_min

from limiteddepkit.experimental import WeibullDuration


@pytest.mark.validation
def test_weibull_duration_likelihood_matches_scipy_and_recovers_parameters():
    rng = np.random.default_rng(7719)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=2_000)})
    beta = np.array([0.3, -0.45])
    shape = 1.6
    scale = np.exp(X.to_numpy() @ beta)
    latent = weibull_min.rvs(shape, scale=scale, random_state=rng)
    censoring = rng.exponential(scale=3.0, size=len(X))
    duration = np.minimum(latent, censoring)
    event = (latent <= censoring).astype(int)

    result = WeibullDuration().fit(X, duration, event)

    fitted_scale = np.exp(X.to_numpy() @ result.params.to_numpy())
    expected_loglike = np.sum(
        event * weibull_min.logpdf(duration, result.shape_param, scale=fitted_scale)
        + (1 - event)
        * weibull_min.logsf(duration, result.shape_param, scale=fitted_scale)
    )
    assert result.converged
    assert result.inference_valid
    assert result.loglike == pytest.approx(expected_loglike, rel=1e-10, abs=1e-7)
    np.testing.assert_allclose(result.params, beta, atol=0.08)
    assert result.shape_param == pytest.approx(shape, abs=0.1)
    lower, upper = result.shape_conf_int()
    assert 0.0 < lower < result.shape_param < upper


def test_weibull_duration_rejects_no_events_and_bad_prediction_schema():
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="At least one observed event"):
        WeibullDuration().fit(X, [1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0])

    fitted = WeibullDuration().fit(X, [0.5, 1.0, 1.5, 2.0], [1, 1, 0, 1])
    with pytest.raises(ValueError, match="must contain 2 regressors"):
        fitted.predict(np.ones((3, 1)))
