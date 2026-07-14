import numpy as np
import pandas as pd

from limiteddepkit.experimental import NegativeBinomial


def test_negative_binomial_fits_and_predicts():
    rng = np.random.default_rng(29)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=280)})

    beta = np.array([0.1, 0.35])
    alpha = 0.8

    mean = np.exp(X.to_numpy() @ beta)
    p = alpha * mean / (1.0 + alpha * mean)
    r = 1.0 / alpha

    y = rng.negative_binomial(r, 1.0 - p)

    result = NegativeBinomial().fit(X, y)

    assert result.converged
    assert result.nobs == 280
    predictions = result.predict(X)
    assert predictions.shape == (280,)
    assert np.all(predictions >= 0.0)
