import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaincc, gammaln

from limiteddepkit.experimental import GammaDuration


def test_gamma_duration_fits_and_predicts():
    rng = np.random.default_rng(50)
    n_obs = 280

    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})

    # True parameters
    beta = np.array([0.4, -0.3])
    shape_param = 2.0  # Gamma shape parameter

    # Generate durations: T ~ Gamma(shape, scale) where scale = exp(X*beta)
    linear_pred = X.to_numpy() @ beta
    scale = np.exp(linear_pred)
    durations = rng.gamma(shape=shape_param, scale=scale)

    # Right censoring
    censoring_time = rng.exponential(scale=2.0, size=n_obs)
    observed_duration = np.minimum(durations, censoring_time)
    event_indicator = durations <= censoring_time

    result = GammaDuration().fit(X, observed_duration, event_indicator)

    assert result.converged
    assert result.nobs == n_obs
    assert result.n_events == event_indicator.sum()
    assert result.shape_param > 0  # Shape parameter should be positive
    predictions = result.predict(X)
    assert predictions.shape == (n_obs,)
    assert np.all(np.isfinite(predictions.to_numpy()))
    assert np.all(predictions.to_numpy() > 0.0)


def test_gamma_duration_loglike_matches_gamma_density_and_survival():
    X = pd.DataFrame(
        {"const": np.ones(6), "x": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]}
    )
    duration = np.array([1.2, 0.8, 1.7, 0.6, 2.1, 1.1])
    event = np.array([1, 0, 1, 1, 0, 1])

    result = GammaDuration().fit(X, duration, event)

    beta = result.params.to_numpy()
    shape = result.shape_param
    eta = X.to_numpy() @ beta
    scaled = duration / np.exp(eta)
    event_mask = event == 1
    expected = np.sum(
        (shape - 1.0) * np.log(duration[event_mask])
        - scaled[event_mask]
        - shape * eta[event_mask]
        - gammaln(shape)
    ) + np.sum(
        np.log(gammaincc(shape, scaled[~event_mask]))
    )
    assert result.loglike == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_gamma_duration_rejects_an_unidentified_all_censored_sample():
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0]})

    with pytest.raises(ValueError, match="At least one observed event"):
        GammaDuration().fit(X, [1.0, 2.0, 3.0], [0, 0, 0])
