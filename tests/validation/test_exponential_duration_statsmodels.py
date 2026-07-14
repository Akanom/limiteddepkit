import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from limiteddepkit.experimental import ExponentialDuration


@pytest.mark.validation
def test_exponential_duration_matches_poisson_exposure_representation():
    rng = np.random.default_rng(1203)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=800)})
    beta = np.array([0.35, -0.5])
    latent = rng.exponential(scale=np.exp(X.to_numpy() @ beta))
    censoring = rng.exponential(scale=2.5, size=len(X))
    duration = np.minimum(latent, censoring)
    event = (latent <= censoring).astype(int)

    result = ExponentialDuration().fit(X, duration, event)
    reference = sm.GLM(
        event,
        -X,
        family=sm.families.Poisson(),
        offset=np.log(duration),
    ).fit()

    assert result.converged
    assert result.inference_valid
    np.testing.assert_allclose(result.params, reference.params, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        result.covariance, reference.cov_params(), rtol=2e-6, atol=2e-7
    )
    expected_loglike = reference.llf - np.sum(event * np.log(duration))
    assert result.loglike == pytest.approx(expected_loglike, rel=1e-11, abs=1e-8)


def test_exponential_duration_rejects_no_events_and_nonbinary_event():
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0]})
    with pytest.raises(ValueError, match="At least one observed event"):
        ExponentialDuration().fit(X, [1.0, 2.0, 3.0], [0, 0, 0])
    with pytest.raises(ValueError, match="binary"):
        ExponentialDuration().fit(X, [1.0, 2.0, 3.0], [0, 2, 1])
