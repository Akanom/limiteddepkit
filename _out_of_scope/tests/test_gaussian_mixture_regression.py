import numpy as np
import pandas as pd
import pytest

from limiteddepkit.experimental import GaussianMixtureRegression, SwitchingRegression


def test_switching_regression_fits_and_predicts():
    rng = np.random.default_rng(48)
    n_obs = 350

    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=n_obs)})

    # Regime 1 (50% of obs): high mean, lower variance
    regime_1 = rng.choice(n_obs, size=n_obs // 2, replace=False)
    regime_1_mask = np.zeros(n_obs, dtype=bool)
    regime_1_mask[regime_1] = True

    # Regime 2: low mean, higher variance
    regime_2_mask = ~regime_1_mask

    y = np.zeros(n_obs)
    beta_1 = np.array([2.0, 0.5])
    beta_2 = np.array([-1.0, 0.3])
    sigma_1 = 0.5
    sigma_2 = 1.2

    y[regime_1_mask] = X[regime_1_mask].to_numpy() @ beta_1 + rng.normal(0, sigma_1, regime_1_mask.sum())
    y[regime_2_mask] = X[regime_2_mask].to_numpy() @ beta_2 + rng.normal(0, sigma_2, regime_2_mask.sum())

    y = pd.Series(y)

    result = GaussianMixtureRegression(n_regimes=2).fit(X, y)

    assert result.converged
    assert result.nobs == n_obs
    assert result.n_regimes == 2
    assert np.sum(result.mixture_probs) == pytest.approx(1.0)
    predictions = result.predict(X)
    assert predictions.shape == (n_obs,)
    assert np.all(np.isfinite(predictions.to_numpy()))


def test_historical_switching_name_warns_that_the_model_is_an_iid_mixture():
    with pytest.warns(FutureWarning, match="iid Gaussian mixture"):
        model = SwitchingRegression(n_regimes=2)
    assert isinstance(model, GaussianMixtureRegression)
