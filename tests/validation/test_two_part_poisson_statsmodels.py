"""Independent validation of experimental ZIP and hurdle-Poisson models."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit.hurdle_poisson import HurdlePoisson
from limiteddepkit.zero_inflated_poisson import ZeroInflatedPoisson

statsmodels_api = pytest.importorskip("statsmodels.api")
count_models = pytest.importorskip("statsmodels.discrete.count_model")
truncated_models = pytest.importorskip("statsmodels.discrete.truncated_model")

pytestmark = pytest.mark.validation


def _draw_zero_truncated_poisson(rng, means):
    draws = rng.poisson(means)
    zero = draws == 0
    while np.any(zero):
        draws[zero] = rng.poisson(means[zero])
        zero = draws == 0
    return draws


def test_zero_inflated_poisson_matches_statsmodels_likelihood_and_predictions():
    rng = np.random.default_rng(91)
    nobs = 1_200
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    Z = pd.DataFrame({"const": 1.0, "z": rng.normal(size=nobs)})
    mean = np.exp(X.to_numpy() @ np.array([0.2, 0.3]))
    inflation = expit(Z.to_numpy() @ np.array([-1.0, 0.5]))
    y = rng.poisson(mean)
    y[rng.uniform(size=nobs) < inflation] = 0

    native = ZeroInflatedPoisson().fit(X, y, X_inflation=Z)
    reference_model = count_models.ZeroInflatedPoisson(
        y, X, exog_infl=Z, inflation="logit"
    )
    reference = reference_model.fit(
        method="bfgs", maxiter=1_000, gtol=1e-8, disp=False
    )

    assert native.converged
    assert reference.mle_retvals["converged"]
    assert native.all_params.to_numpy() == pytest.approx(reference.params, abs=2e-7)
    assert native.loglike == pytest.approx(reference.llf, abs=1e-9)
    assert native.predict(X.iloc[:40], X_inflation=Z.iloc[:40]).to_numpy() == pytest.approx(
        reference_model.predict(
            reference.params,
            exog=X.iloc[:40],
            exog_infl=Z.iloc[:40],
            which="mean",
        ),
        abs=2e-8,
    )
    assert native.predict_zero_probability(
        X.iloc[:40], X_inflation=Z.iloc[:40]
    ).to_numpy() == pytest.approx(
        reference_model.predict(
            reference.params,
            exog=X.iloc[:40],
            exog_infl=Z.iloc[:40],
            which="prob-zero",
        ),
        abs=2e-8,
    )
    assert native.predict_pmf(
        X.iloc[:20], X_inflation=Z.iloc[:20], max_count=6
    ).to_numpy() == pytest.approx(
        reference_model.predict(
            reference.params,
            exog=X.iloc[:20],
            exog_infl=Z.iloc[:20],
            which="prob",
            y_values=np.arange(7),
        ),
        abs=2e-8,
    )


def test_logit_hurdle_matches_statsmodels_factor_models():
    rng = np.random.default_rng(92)
    nobs = 1_100
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    Z = pd.DataFrame({"const": 1.0, "z": rng.normal(size=nobs)})
    count_mean = np.exp(X.to_numpy() @ np.array([0.15, 0.35]))
    positive_probability = expit(Z.to_numpy() @ np.array([0.25, -0.45]))
    positive_counts = _draw_zero_truncated_poisson(rng, count_mean)
    y = np.where(rng.uniform(size=nobs) < positive_probability, positive_counts, 0)

    native = HurdlePoisson().fit(X, y, X_hurdle=Z)
    reference_hurdle = statsmodels_api.Logit((y > 0).astype(int), Z).fit(
        method="newton", maxiter=1_000, tol=1e-12, disp=False
    )
    reference_count = truncated_models.TruncatedLFPoisson(y, X).fit(
        method="bfgs", maxiter=1_000, gtol=1e-8, disp=False
    )
    reference_full = truncated_models.HurdleCountModel(y, X).fit(
        method="bfgs", maxiter=1_000, gtol=1e-8, disp=False
    )

    assert native.params_hurdle.to_numpy() == pytest.approx(
        reference_hurdle.params, abs=2e-8
    )
    assert native.params_poisson.to_numpy() == pytest.approx(
        reference_count.params, abs=2e-7
    )
    assert native.params_poisson.to_numpy() == pytest.approx(
        reference_full.results_count.params, abs=2e-7
    )
    assert native.loglike == pytest.approx(
        reference_hurdle.llf + reference_count.llf, abs=1e-9
    )
    assert native.covariance.iloc[: Z.shape[1], : Z.shape[1]].to_numpy() == pytest.approx(
        reference_hurdle.cov_params(), abs=2e-9
    )
    assert native.covariance.iloc[Z.shape[1] :, Z.shape[1] :].to_numpy() == pytest.approx(
        reference_count.cov_params(), abs=2e-8
    )
    assert native.predict_positive_probability(Z.iloc[:40]).to_numpy() == pytest.approx(
        reference_hurdle.predict(Z.iloc[:40]), abs=2e-9
    )
    assert native.predict_positive_mean(X.iloc[:40]).to_numpy() == pytest.approx(
        reference_count.model.predict(
            reference_count.params, exog=X.iloc[:40], which="mean"
        ),
        abs=2e-8,
    )
