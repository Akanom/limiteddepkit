import numpy as np
import pandas as pd

from limiteddepkit.experimental import SampleSelection


def test_sample_selection_fits_and_predicts():
    rng = np.random.default_rng(43)
    n_obs = 700

    X = pd.DataFrame({"const": 1.0, "x1": rng.normal(size=n_obs)})
    Z = pd.DataFrame(
        {"const": 1.0, "x1": X["x1"], "z1": rng.normal(size=n_obs)}
    )

    selection_beta = np.array([0.2, 0.25, 0.5])
    selection_error = rng.normal(size=n_obs)
    outcome_innovation = rng.normal(size=n_obs)
    rho = 0.4
    outcome_sigma = 1.2
    outcome_error = outcome_sigma * (
        rho * selection_error + np.sqrt(1.0 - rho**2) * outcome_innovation
    )
    selection_latent = Z.to_numpy() @ selection_beta + selection_error
    observed = selection_latent > 0

    outcome_beta = np.array([1.0, 0.6])
    outcome_latent = X.to_numpy() @ outcome_beta + outcome_error

    y = np.full(n_obs, np.nan)
    y[observed] = outcome_latent[observed]

    result = SampleSelection().fit(X, y, Z)

    assert result.converged
    assert result.nobs_observed == observed.sum()
    assert result.nobs_total == n_obs
    predictions = result.predict(X)
    assert predictions.shape == (n_obs,)
    assert np.all(np.isfinite(predictions.to_numpy()))
    selection_probability = result.predict_selection(Z)
    assert np.all((selection_probability > 0.0) & (selection_probability < 1.0))
    observed_prediction = result.predict_observed(X, Z)
    assert observed_prediction.shape == (n_obs,)
