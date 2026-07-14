import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from limiteddepkit.experimental import PoissonRegressor


@pytest.mark.validation
def test_poisson_matches_statsmodels_coefficients_likelihood_and_covariance():
    rng = np.random.default_rng(90210)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=700)})
    y = rng.poisson(np.exp(X.to_numpy() @ np.array([-0.15, 0.55])))

    result = PoissonRegressor().fit(X, y)
    reference = sm.Poisson(y, X).fit(disp=False)

    assert result.converged
    assert result.inference_valid
    np.testing.assert_allclose(result.params, reference.params, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        result.covariance, reference.cov_params(), rtol=2e-5, atol=2e-6
    )
    assert result.loglike == pytest.approx(reference.llf, rel=1e-10, abs=1e-8)


@pytest.mark.parametrize("invalid_y", [[0.0, 1.5, 2.0], [0.0, 0.0, 0.0]])
def test_poisson_rejects_invalid_or_unidentified_counts(invalid_y):
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0]})

    with pytest.raises(ValueError):
        PoissonRegressor().fit(X, invalid_y)


def test_poisson_rejects_rank_deficiency_and_wrong_prediction_width():
    X = pd.DataFrame(
        {"const": [1.0, 1.0, 1.0, 1.0], "duplicate": [1.0, 1.0, 1.0, 1.0]}
    )
    with pytest.raises(ValueError, match="full column rank"):
        PoissonRegressor().fit(X, [0, 1, 2, 1])

    fitted = PoissonRegressor().fit(
        pd.DataFrame({"const": [1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0]}),
        [0, 1, 2],
    )
    with pytest.raises(ValueError, match="must contain 2 regressors"):
        fitted.predict(np.ones((2, 1)))
