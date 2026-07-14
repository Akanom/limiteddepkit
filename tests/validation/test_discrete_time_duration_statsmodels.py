import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from limiteddepkit.experimental import DiscreteTimeDuration


@pytest.mark.validation
def test_grouped_discrete_duration_matches_person_period_logit():
    rng = np.random.default_rng(6024)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=500)})
    hazard = 1.0 / (1.0 + np.exp(-(X.to_numpy() @ np.array([-1.1, 0.45]))))
    latent_duration = np.array([rng.geometric(probability) for probability in hazard])
    censoring = rng.integers(2, 9, size=len(X))
    duration = np.minimum(latent_duration, censoring)
    event = (latent_duration <= censoring).astype(int)

    result = DiscreteTimeDuration().fit(X, duration, event)

    person_period_X = np.repeat(X.to_numpy(), duration, axis=0)
    person_period_event = np.zeros(int(np.sum(duration)))
    final_rows = np.cumsum(duration) - 1
    person_period_event[final_rows] = event
    reference = sm.Logit(person_period_event, person_period_X).fit(disp=False)

    assert result.converged
    assert result.inference_valid
    np.testing.assert_allclose(result.params, reference.params, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        result.covariance, reference.cov_params(), rtol=2e-6, atol=2e-7
    )
    assert result.loglike == pytest.approx(reference.llf, rel=1e-11, abs=1e-8)

    predicted_hazard = result.predict_hazard(X.iloc[:5])
    np.testing.assert_allclose(
        result.predict_survival(X.iloc[:5], 3), (1.0 - predicted_hazard) ** 3
    )
    np.testing.assert_allclose(result.predict(X.iloc[:5]), 1.0 / predicted_hazard)


def test_discrete_duration_rejects_fractional_durations_and_no_events():
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0]})
    with pytest.raises(ValueError, match="positive integers"):
        DiscreteTimeDuration().fit(X, [1.0, 2.5, 3.0], [1, 0, 1])
    with pytest.raises(ValueError, match="Both event"):
        DiscreteTimeDuration().fit(X, [1, 2, 3], [0, 0, 0])
