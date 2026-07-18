"""Reference-package gate for the Gaussian exact-observation submodel."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit import IntervalRegression

sm = pytest.importorskip("statsmodels.api")


@pytest.mark.validation
def test_exact_interval_regression_matches_statsmodels_gaussian_hc0():
    rng = np.random.default_rng(20260715)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=600)})
    scale = 0.7 + 0.3 * np.abs(X["x"].to_numpy())
    y = 0.4 - 0.65 * X["x"].to_numpy() + rng.normal(scale=scale)

    result = IntervalRegression().fit(X, y, y, covariance_type="robust")
    reference = sm.OLS(y, X).fit(cov_type="HC0")

    np.testing.assert_allclose(result.params, reference.params, atol=2e-7)
    assert result.loglike == pytest.approx(reference.llf, abs=2e-7)
    np.testing.assert_allclose(
        result.covariance.loc[X.columns, X.columns],
        reference.cov_params(),
        atol=3e-9,
    )
