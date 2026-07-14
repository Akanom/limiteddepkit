import numpy as np
import pandas as pd

from limiteddepkit.experimental import WeibullDuration


def test_weibull_duration_fits_and_predicts():
    rng = np.random.default_rng(47)
    n_obs = 300

    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})

    # True parameters
    beta = np.array([0.3, -0.5])
    shape_param = 1.5  # alpha > 1 = increasing hazard

    # Generate durations: T ~ Weibull(shape=alpha, scale=exp(X*beta))
    linear_pred = X.to_numpy() @ beta
    scale = np.exp(linear_pred)
    durations = scale * (rng.exponential(size=n_obs) ** (1.0 / shape_param))

    # Right censoring
    censoring_time = rng.exponential(scale=2.5, size=n_obs)
    observed_duration = np.minimum(durations, censoring_time)
    event_indicator = durations <= censoring_time

    result = WeibullDuration().fit(X, observed_duration, event_indicator)

    assert result.converged
    assert result.nobs == n_obs
    assert result.n_events == event_indicator.sum()
    assert result.shape_param > 0  # Shape parameter should be positive
    predictions = result.predict(X)
    assert predictions.shape == (n_obs,)
    assert np.all(np.isfinite(predictions.to_numpy()))
    assert np.all(predictions.to_numpy() > 0.0)
