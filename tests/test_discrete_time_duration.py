import numpy as np
import pandas as pd

from limiteddepkit.experimental import DiscreteTimeDuration


def test_discrete_time_duration_fits_and_predicts():
    rng = np.random.default_rng(49)
    n_obs = 300

    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})

    # True parameters
    beta = np.array([0.2, -0.35])

    # Generate duration in discrete periods (1, 2, 3, ...)
    # P(event at period t) = logit(X*beta + alpha*t) where alpha captures duration dependence
    durations = np.zeros(n_obs, dtype=int)
    for i in range(n_obs):
        for t in range(1, 20):  # Max duration 20 periods
            eta = X.iloc[i].to_numpy() @ beta + 0.1 * t  # Positive alpha = increasing hazard
            prob_event = 1.0 / (1.0 + np.exp(-eta))
            if rng.uniform() < prob_event:
                durations[i] = t
                break
        if durations[i] == 0:
            durations[i] = 20  # Censored at period 20

    event_indicator = (durations < 20).astype(int)

    result = DiscreteTimeDuration().fit(X, durations, event_indicator)

    assert result.converged
    assert result.nobs == n_obs
    assert result.n_events == event_indicator.sum()
    predictions = result.predict(X)
    assert predictions.shape == (n_obs,)
    assert np.all(np.isfinite(predictions.to_numpy()))
