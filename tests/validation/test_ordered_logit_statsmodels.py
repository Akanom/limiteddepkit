"""Numerical validation against statsmodels OrderedModel."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, ndtr

from limiteddepkit import OrderedLogit, OrderedProbit

statsmodels_ordinal = pytest.importorskip("statsmodels.miscmodels.ordinal_model")
OrderedModel = statsmodels_ordinal.OrderedModel

pytestmark = pytest.mark.validation


def make_reference_data(link, seed=4102, nobs=1_500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)})
    beta = np.array([0.9, -0.6])
    thresholds = np.array([-0.8, 0.7])
    latent_bounds = thresholds[None, :] - X.to_numpy() @ beta[:, None]
    cumulative = expit(latent_bounds) if link == "logit" else ndtr(latent_bounds)
    probabilities = np.column_stack(
        [cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, y


def transformed_statsmodels_covariance(reference_result, n_features, n_thresholds):
    """Map statsmodels' raw threshold covariance to ordered-cut covariance."""
    raw = reference_result.params.to_numpy(dtype=float)
    transformation = np.eye(n_features + n_thresholds)
    threshold_start = n_features
    transformation[threshold_start:, threshold_start] = 1.0
    for offset in range(1, n_thresholds):
        column = threshold_start + offset
        transformation[column:, column] = np.exp(raw[column])
    raw_covariance = reference_result.cov_params().to_numpy(dtype=float)
    return transformation @ raw_covariance @ transformation.T


@pytest.mark.parametrize(
    ("link", "estimator"),
    [("logit", OrderedLogit), ("probit", OrderedProbit)],
)
def test_ordered_model_matches_statsmodels(link, estimator):
    X, y = make_reference_data(link)
    native = estimator().fit(X, y)

    reference_model = OrderedModel(y, X, distr=link)
    reference = reference_model.fit(method="bfgs", disp=False)
    reference_thresholds = reference_model.transform_threshold_params(
        reference.params.iloc[X.shape[1] :]
    )[1:-1]
    reference_probabilities = reference_model.predict(reference.params, exog=X.iloc[:50])
    transformed_covariance = transformed_statsmodels_covariance(
        reference, X.shape[1], len(native.thresholds)
    )

    assert native.converged
    assert reference.mle_retvals["converged"]
    assert native.params.to_numpy() == pytest.approx(
        reference.params.iloc[: X.shape[1]].to_numpy(), abs=1e-5
    )
    assert native.thresholds.to_numpy() == pytest.approx(reference_thresholds, abs=3e-5)
    assert native.loglike == pytest.approx(reference.llf, abs=1e-6)
    assert native.predict_proba(X.iloc[:50]).to_numpy() == pytest.approx(
        reference_probabilities, abs=1e-5
    )
    assert native.standard_errors.to_numpy() == pytest.approx(
        np.sqrt(np.diag(transformed_covariance)), abs=1e-6
    )
