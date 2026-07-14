import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from limiteddepkit.experimental import NegativeBinomial


@pytest.mark.validation
def test_negative_binomial_nb2_matches_statsmodels():
    rng = np.random.default_rng(1117)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=1_000)})
    beta = np.array([0.25, -0.4])
    alpha = 0.65
    mean = np.exp(X.to_numpy() @ beta)
    y = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))

    result = NegativeBinomial().fit(X, y)
    reference = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(disp=False)

    assert result.converged
    assert result.inference_valid
    np.testing.assert_allclose(
        result.params, reference.params.iloc[:-1], rtol=2e-5, atol=2e-5
    )
    reference_alpha = float(reference.params.iloc[-1])
    assert result.alpha == pytest.approx(reference_alpha, rel=3e-5, abs=3e-5)
    assert result.loglike == pytest.approx(reference.llf, rel=1e-9, abs=2e-6)

    transform = np.eye(3)
    transform[-1, -1] = 1.0 / reference_alpha
    reference_covariance = transform @ reference.cov_params() @ transform.T
    np.testing.assert_allclose(
        result.covariance, reference_covariance, rtol=2e-3, atol=2e-4
    )
    lower, upper = result.alpha_conf_int()
    assert 0.0 < lower < result.alpha < upper


@pytest.mark.parametrize("invalid_y", [[0.0, 2.5, 1.0], [0.0, 0.0, 0.0]])
def test_negative_binomial_rejects_invalid_or_unidentified_counts(invalid_y):
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0]})

    with pytest.raises(ValueError):
        NegativeBinomial().fit(X, invalid_y)
