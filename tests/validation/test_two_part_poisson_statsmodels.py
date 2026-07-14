"""Independent validation of experimental ZIP and hurdle-Poisson models."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, gammaln

from limiteddepkit.hurdle_poisson import HurdlePoisson
from limiteddepkit.zero_inflated_poisson import ZeroInflatedPoisson

statsmodels_api = pytest.importorskip("statsmodels.api")
count_models = pytest.importorskip("statsmodels.discrete.count_model")
truncated_models = pytest.importorskip("statsmodels.discrete.truncated_model")

pytestmark = pytest.mark.validation


def _draw_zero_truncated_poisson(
    rng: np.random.Generator,
    means: np.ndarray,
) -> np.ndarray:
    """Draw Poisson observations conditional on strictly positive counts."""

    draws = rng.poisson(means)
    zero_mask = draws == 0

    while np.any(zero_mask):
        draws[zero_mask] = rng.poisson(means[zero_mask])
        zero_mask = draws == 0

    return draws


def _zip_pmf_from_reference_params(
    reference_model: Any,
    params: Any,
    X: pd.DataFrame | np.ndarray,
    X_inflation: pd.DataFrame | np.ndarray,
    *,
    max_count: int,
) -> np.ndarray:
    """Evaluate a logit-inflated Poisson PMF from reference parameters.

    This calculates the ZIP probability mass function directly from the
    parameters estimated by Statsmodels. It avoids Statsmodels' probability
    prediction path, which can depend on private SciPy interfaces that differ
    across SciPy releases.

    Parameters
    ----------
    reference_model
        Fitted-model specification used to determine the number of inflation
        parameters.
    params
        Combined Statsmodels parameter vector. Inflation parameters precede
        count parameters.
    X
        Count-equation design matrix.
    X_inflation
        Inflation-equation design matrix.
    max_count
        Largest count for which probabilities are evaluated.

    Returns
    -------
    numpy.ndarray
        Matrix with one row per observation and columns corresponding to
        counts ``0, ..., max_count``.
    """

    if max_count < 0:
        raise ValueError("max_count must be non-negative.")

    X_array = np.asarray(X, dtype=float)
    Z_array = np.asarray(X_inflation, dtype=float)
    params_array = np.asarray(params, dtype=float)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    if Z_array.ndim != 2:
        raise ValueError("X_inflation must be a two-dimensional design matrix.")

    if X_array.shape[0] != Z_array.shape[0]:
        raise ValueError(
            "X and X_inflation must contain the same number of observations."
        )

    k_inflate = int(reference_model.k_inflate)
    inflation_params = params_array[:k_inflate]
    count_params = params_array[k_inflate:]

    if Z_array.shape[1] != inflation_params.size:
        raise ValueError(
            "X_inflation column count does not match the reference "
            "inflation parameter count."
        )

    if X_array.shape[1] != count_params.size:
        raise ValueError(
            "X column count does not match the reference count parameter count."
        )

    inflation_probability = expit(Z_array @ inflation_params)
    poisson_mean = np.exp(X_array @ count_params)

    counts = np.arange(max_count + 1, dtype=float)

    log_poisson_pmf = (
        -poisson_mean[:, None]
        + counts[None, :] * np.log(poisson_mean)[:, None]
        - gammaln(counts + 1.0)[None, :]
    )

    pmf = (
        1.0 - inflation_probability[:, None]
    ) * np.exp(log_poisson_pmf)

    # A zero can arise either from the structural-zero process or from the
    # ordinary Poisson process.
    pmf[:, 0] += inflation_probability

    return pmf


def test_zero_inflated_poisson_matches_statsmodels_likelihood_and_predictions():
    """Match ZIP estimates and predictions against Statsmodels."""

    rng = np.random.default_rng(91)
    nobs = 1_200

    X = pd.DataFrame(
        {
            "const": 1.0,
            "x": rng.normal(size=nobs),
        }
    )
    Z = pd.DataFrame(
        {
            "const": 1.0,
            "z": rng.normal(size=nobs),
        }
    )

    mean = np.exp(X.to_numpy() @ np.array([0.2, 0.3]))
    inflation = expit(Z.to_numpy() @ np.array([-1.0, 0.5]))

    y = rng.poisson(mean)
    y[rng.uniform(size=nobs) < inflation] = 0

    native = ZeroInflatedPoisson().fit(
        X,
        y,
        X_inflation=Z,
    )

    reference_model = count_models.ZeroInflatedPoisson(
        y,
        X,
        exog_infl=Z,
        inflation="logit",
    )
    reference = reference_model.fit(
        method="bfgs",
        maxiter=1_000,
        gtol=1e-8,
        disp=False,
    )

    assert native.converged
    assert reference.mle_retvals["converged"]

    assert native.all_params.to_numpy() == pytest.approx(
        reference.params,
        abs=2e-7,
    )
    assert native.loglike == pytest.approx(
        reference.llf,
        abs=1e-9,
    )

    expected_mean = reference_model.predict(
        reference.params,
        exog=X.iloc[:40],
        exog_infl=Z.iloc[:40],
        which="mean",
    )

    assert native.predict(
        X.iloc[:40],
        X_inflation=Z.iloc[:40],
    ).to_numpy() == pytest.approx(
        expected_mean,
        abs=2e-8,
    )

    expected_zero_probability = reference_model.predict(
        reference.params,
        exog=X.iloc[:40],
        exog_infl=Z.iloc[:40],
        which="prob-zero",
    )

    assert native.predict_zero_probability(
        X.iloc[:40],
        X_inflation=Z.iloc[:40],
    ).to_numpy() == pytest.approx(
        expected_zero_probability,
        abs=2e-8,
    )

    expected_pmf = _zip_pmf_from_reference_params(
        reference_model,
        reference.params,
        X.iloc[:20],
        Z.iloc[:20],
        max_count=6,
    )

    assert native.predict_pmf(
        X.iloc[:20],
        X_inflation=Z.iloc[:20],
        max_count=6,
    ).to_numpy() == pytest.approx(
        expected_pmf,
        abs=2e-8,
    )


def test_logit_hurdle_matches_statsmodels_factor_models():
    """Match hurdle-Logit and zero-truncated Poisson components."""

    rng = np.random.default_rng(92)
    nobs = 1_100

    X = pd.DataFrame(
        {
            "const": 1.0,
            "x": rng.normal(size=nobs),
        }
    )
    Z = pd.DataFrame(
        {
            "const": 1.0,
            "z": rng.normal(size=nobs),
        }
    )

    count_mean = np.exp(X.to_numpy() @ np.array([0.15, 0.35]))
    positive_probability = expit(
        Z.to_numpy() @ np.array([0.25, -0.45])
    )

    positive_counts = _draw_zero_truncated_poisson(
        rng,
        count_mean,
    )
    y = np.where(
        rng.uniform(size=nobs) < positive_probability,
        positive_counts,
        0,
    )

    native = HurdlePoisson().fit(
        X,
        y,
        X_hurdle=Z,
    )

    reference_hurdle = statsmodels_api.Logit(
        (y > 0).astype(int),
        Z,
    ).fit(
        method="newton",
        maxiter=1_000,
        tol=1e-12,
        disp=False,
    )

    reference_count = truncated_models.TruncatedLFPoisson(
        y,
        X,
    ).fit(
        method="bfgs",
        maxiter=1_000,
        gtol=1e-8,
        disp=False,
    )

    reference_full = truncated_models.HurdleCountModel(
        y,
        X,
    ).fit(
        method="bfgs",
        maxiter=1_000,
        gtol=1e-8,
        disp=False,
    )

    assert native.params_hurdle.to_numpy() == pytest.approx(
        reference_hurdle.params,
        abs=2e-8,
    )

    assert native.params_poisson.to_numpy() == pytest.approx(
        reference_count.params,
        abs=2e-7,
    )

    assert native.params_poisson.to_numpy() == pytest.approx(
        reference_full.results_count.params,
        abs=2e-7,
    )

    assert native.loglike == pytest.approx(
        reference_hurdle.llf + reference_count.llf,
        abs=1e-9,
    )

    hurdle_covariance = native.covariance.iloc[
        : Z.shape[1],
        : Z.shape[1],
    ].to_numpy()

    assert hurdle_covariance == pytest.approx(
        reference_hurdle.cov_params(),
        abs=2e-9,
    )

    count_covariance = native.covariance.iloc[
        Z.shape[1] :,
        Z.shape[1] :,
    ].to_numpy()

    assert count_covariance == pytest.approx(
        reference_count.cov_params(),
        abs=2e-8,
    )

    assert native.predict_positive_probability(
        Z.iloc[:40]
    ).to_numpy() == pytest.approx(
        reference_hurdle.predict(Z.iloc[:40]),
        abs=2e-9,
    )

    expected_positive_mean = reference_count.model.predict(
        reference_count.params,
        exog=X.iloc[:40],
        which="mean",
    )

    assert native.predict_positive_mean(
        X.iloc[:40]
    ).to_numpy() == pytest.approx(
        expected_positive_mean,
        abs=2e-8,
    )