import numpy as np
import pandas as pd

from limiteddepkit.experimental import ExponentialDuration


def test_exponential_duration_fits_and_predicts():
    rng = np.random.default_rng(46)
    n_obs = 280

    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})

    # True parameters
    beta = np.array([0.5, -0.4])

    # Generate durations: T ~ Exponential(lambda = exp(-X*beta))
    linear_pred = X.to_numpy() @ beta
    lambda_rate = np.exp(-linear_pred)
    durations = rng.exponential(scale=1.0/lambda_rate)

    # Censor some observations at random (right censoring)
    censoring_time = rng.exponential(scale=2.0, size=n_obs)
    observed_duration = np.minimum(durations, censoring_time)
    is_censored = durations > censoring_time
    event_indicator = ~is_censored

    result = ExponentialDuration().fit(X, observed_duration, event_indicator)

    assert result.converged
    assert result.nobs == n_obs
    assert result.n_events == event_indicator.sum()
    predictions = result.predict(X)
    assert predictions.shape == (n_obs,)
    assert np.all(np.isfinite(predictions.to_numpy()))
    assert np.all(predictions.to_numpy() > 0.0)
