"""Independent numerical validation of stable binary-response MLEs."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from limiteddepkit import BinaryLogit, BinaryProbit

statsmodels_api = pytest.importorskip("statsmodels.api")

pytestmark = pytest.mark.validation


@pytest.mark.parametrize(
    ("native_estimator", "reference_model", "probability"),
    [
        (BinaryLogit, statsmodels_api.Logit, expit),
        (BinaryProbit, statsmodels_api.Probit, norm.cdf),
    ],
)
def test_binary_mle_matches_statsmodels(native_estimator, reference_model, probability):
    rng = np.random.default_rng(8421)
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=1_500),
            "x2": rng.normal(size=1_500),
        }
    )
    truth = np.array([-0.25, 0.65, -0.4])
    y = rng.binomial(1, probability(X.to_numpy() @ truth))

    native = native_estimator().fit(X, y)
    reference = reference_model(y, X).fit(
        method="newton", maxiter=1_000, tol=1e-12, disp=False
    )

    assert native.converged
    assert reference.mle_retvals["converged"]
    assert native.params.to_numpy() == pytest.approx(reference.params, abs=2e-7)
    assert native.loglike == pytest.approx(reference.llf, abs=1e-9)
    assert native.covariance.to_numpy() == pytest.approx(reference.cov_params(), abs=2e-7)
    assert native.standard_errors.to_numpy() == pytest.approx(reference.bse, abs=2e-7)
    assert native.predict_proba(X.iloc[:50])[1].to_numpy() == pytest.approx(
        reference.predict(X.iloc[:50]), abs=1e-8
    )
    assert native.aic == pytest.approx(reference.aic, abs=1e-9)
    assert native.bic == pytest.approx(reference.bic, abs=1e-9)
    assert native.average_marginal_effects(X).to_numpy() == pytest.approx(
        reference.get_margeff(at="overall").margeff, abs=2e-9
    )


def test_logit_likelihood_remains_finite_for_extreme_misclassification():
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x": [-20.0, -10.0, -2.0, -1.0, 1.0, 2.0, 10.0, 20.0],
        }
    )
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    result = BinaryLogit().fit(X, y)
    direct = np.sum(
        y * np.log(result.predict_proba(X)[1])
        + (1 - y) * np.log(result.predict_proba(X)[0])
    )
    assert np.isfinite(result.loglike)
    assert result.loglike == pytest.approx(direct, abs=1e-10)
