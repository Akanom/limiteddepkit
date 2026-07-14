import numpy as np
import pandas as pd

from limiteddepkit.experimental import PoissonRegressor


def test_poisson_regressor_fits_and_predicts_counts():
    rng = np.random.default_rng(17)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=400)})
    linear = -0.2 + 0.6 * X["x"].to_numpy()
    counts = rng.poisson(np.exp(linear))

    result = PoissonRegressor().fit(X, counts)

    assert result.converged
    assert result.params.index.tolist() == ["const", "x"]
    assert result.predict(X).shape == (400,)
    assert result.nobs == 400
