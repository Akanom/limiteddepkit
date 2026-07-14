"""Numerical certification for experimental choice estimators."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag
from scipy.special import expit, softmax

from limiteddepkit.experimental import ConditionalLogit, MultinomialLogit, SequentialLogit

statsmodels_discrete = pytest.importorskip("statsmodels.discrete.discrete_model")
statsmodels_conditional = pytest.importorskip("statsmodels.discrete.conditional_models")
Logit = statsmodels_discrete.Logit
MNLogit = statsmodels_discrete.MNLogit
ReferenceConditionalLogit = statsmodels_conditional.ConditionalLogit

pytestmark = pytest.mark.validation


def test_multinomial_logit_matches_statsmodels_parameters_likelihood_and_inference():
    rng = np.random.default_rng(8102)
    nobs = 1_400
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=nobs),
            "x2": rng.normal(size=nobs),
        }
    )
    coefficients = np.array(
        [[0.20, 0.65, -0.30], [-0.40, -0.45, 0.55], [0.10, 0.20, 0.20]]
    )
    probabilities = softmax(
        np.column_stack([np.zeros(nobs), X.to_numpy() @ coefficients.T]), axis=1
    )
    y = np.array([rng.choice(4, p=row) for row in probabilities])

    native = MultinomialLogit().fit(X, y, category_order=[0, 1, 2, 3])
    reference = MNLogit(y, X).fit(method="newton", disp=False, maxiter=100)

    assert native.converged
    assert reference.mle_retvals["converged"]
    assert native.params.to_numpy() == pytest.approx(reference.params.to_numpy(), abs=2e-6)
    assert native.loglike == pytest.approx(reference.llf, abs=1e-7)
    assert native.predict_proba(X.iloc[:75]).to_numpy() == pytest.approx(
        reference.predict(X.iloc[:75]).to_numpy(), abs=1e-6
    )
    assert native.covariance.to_numpy() == pytest.approx(
        reference.cov_params().to_numpy(), abs=2e-6
    )


def test_conditional_logit_matches_statsmodels_group_likelihood_and_inference():
    rng = np.random.default_rng(8203)
    n_choice_sets = 600
    n_alts = 4
    groups = np.repeat(np.arange(n_choice_sets), n_alts)
    X = pd.DataFrame(
        {
            "price": rng.normal(size=n_choice_sets * n_alts),
            "quality": rng.normal(size=n_choice_sets * n_alts),
        }
    )
    coefficients = np.array([-0.65, 0.40])
    utilities = (X.to_numpy() @ coefficients).reshape(n_choice_sets, n_alts)
    probabilities = softmax(utilities, axis=1)
    selected = np.array([rng.choice(n_alts, p=row) for row in probabilities])
    choice = np.zeros(n_choice_sets * n_alts, dtype=int)
    choice[np.arange(n_choice_sets) * n_alts + selected] = 1

    native = ConditionalLogit().fit(X, choice, groups=groups)
    reference = ReferenceConditionalLogit(choice, X, groups=groups).fit(
        method="newton", disp=False
    )

    assert native.converged
    assert native.params.to_numpy() == pytest.approx(
        np.asarray(reference.params), abs=2e-5
    )
    assert native.loglike == pytest.approx(reference.llf, abs=2e-7)
    assert native.covariance.to_numpy() == pytest.approx(
        np.asarray(reference.cov_params()), abs=2e-6
    )


def test_sequential_logit_equals_its_stagewise_statsmodels_decomposition():
    rng = np.random.default_rng(8304)
    nobs = 2_000
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    coefficients = np.array([[-0.20, 0.50], [0.30, -0.40], [-0.10, 0.25]])
    stop_probabilities = expit(X.to_numpy() @ coefficients.T)
    probabilities = np.empty((nobs, 4), dtype=float)
    remaining = np.ones(nobs, dtype=float)
    for stage in range(3):
        probabilities[:, stage] = remaining * stop_probabilities[:, stage]
        remaining *= 1.0 - stop_probabilities[:, stage]
    probabilities[:, -1] = remaining
    y = np.array([rng.choice(4, p=row) for row in probabilities])

    native = SequentialLogit().fit(X, y, category_order=[0, 1, 2, 3])
    reference_results = []
    for stage in range(3):
        risk = y >= stage
        stop = (y[risk] == stage).astype(int)
        reference_results.append(Logit(stop, X.loc[risk]).fit(method="newton", disp=False))

    reference_params = np.vstack(
        [result.params.to_numpy() for result in reference_results]
    )
    reference_covariance = block_diag(
        *[result.cov_params().to_numpy() for result in reference_results]
    )

    assert native.converged
    np.testing.assert_allclose(
        native.params.to_numpy().T, reference_params, atol=2e-6, rtol=0.0
    )
    assert native.loglike == pytest.approx(
        sum(result.llf for result in reference_results), abs=2e-7
    )
    assert native.covariance.to_numpy() == pytest.approx(reference_covariance, abs=2e-6)
