import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from limiteddepkit.experimental import CensoredQuantileRegression


@pytest.mark.validation
def test_inactive_censoring_reduces_to_statsmodels_quantile_regression():
    rng = np.random.default_rng(91)
    quantile = 0.35
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=600)})
    y = 0.3 - 0.6 * X["x"].to_numpy() + rng.standard_t(4, size=len(X))

    result = CensoredQuantileRegression(
        quantile=quantile, lower=-1_000_000.0
    ).fit(X, y, n_starts=3)
    reference = sm.QuantReg(y, X).fit(q=quantile, max_iter=5_000, p_tol=1e-10)

    np.testing.assert_allclose(result.params, reference.params, rtol=2e-5, atol=2e-6)
    residual = y - X.to_numpy() @ reference.params.to_numpy()
    reference_objective = np.sum(
        np.where(
            residual >= 0.0,
            quantile * residual,
            (quantile - 1.0) * residual,
        )
    )
    assert result.objective_value == pytest.approx(reference_objective, abs=2e-5)
