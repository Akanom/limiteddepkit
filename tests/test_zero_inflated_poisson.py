import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, gammaln, log_expit

from limiteddepkit.ordinal import _numerical_hessian
from limiteddepkit.zero_inflated_poisson import ZeroInflatedPoisson


@pytest.fixture
def fitted_zip():
    rng = np.random.default_rng(19)
    nobs = 900
    index = pd.Index(np.arange(3_000, 3_000 + nobs), name="row")
    X = pd.DataFrame(
        {"const": 1.0, "x": rng.normal(size=nobs)},
        index=index,
    )
    Z = pd.DataFrame(
        {"const": 1.0, "z": rng.normal(size=nobs)},
        index=index,
    )
    count_mean = np.exp(X.to_numpy() @ np.array([0.15, 0.4]))
    inflation_probability = expit(Z.to_numpy() @ np.array([-0.9, -0.45]))
    y = rng.poisson(count_mean)
    y[rng.uniform(size=nobs) < inflation_probability] = 0
    return X, Z, y, ZeroInflatedPoisson().fit(X, y, X_inflation=Z)


def test_zip_fits_with_complete_inference_contract(fitted_zip):
    X, Z, _, result = fitted_zip

    assert result.converged
    assert result.inference_valid
    assert result.backend == "experimental-native-mle"
    assert result.covariance_type == "observed-information"
    assert result.nobs == len(X)
    assert result.n_params == X.shape[1] + Z.shape[1]
    assert result.df_resid == result.nobs - result.n_params
    assert result.score_norm <= 1e-7
    assert list(result.all_params.index) == [
        "inflation: const",
        "inflation: z",
        "count: const",
        "count: x",
    ]
    assert np.linalg.eigvalsh(result.covariance).min() > 0.0
    assert np.isfinite(result.standard_errors).all()
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert result.conf_int().shape == (result.n_params, 2)
    assert result.vcov().equals(result.covariance)
    assert result.vcov() is not result.covariance
    assert list(result.summary_frame().columns) == ["coef", "std_err", "z", "p_value"]


def test_zip_predictions_obey_mixture_identities_and_preserve_index(fitted_zip):
    X, Z, _, result = fitted_zip
    X_eval = X.iloc[:30]
    Z_eval = Z.iloc[:30]
    inflation = result.predict_inflation_probability(Z_eval)
    count_mean = result.predict_count_mean(X_eval)
    expected = result.predict(X_eval, X_inflation=Z_eval)
    zero_probability = result.predict_zero_probability(X_eval, X_inflation=Z_eval)
    pmf = result.predict_pmf(X_eval, X_inflation=Z_eval, max_count=30)

    assert expected.index.equals(X_eval.index)
    assert expected.to_numpy() == pytest.approx(
        ((1.0 - inflation) * count_mean).to_numpy()
    )
    assert zero_probability.to_numpy() == pytest.approx(
        (inflation + (1.0 - inflation) * np.exp(-count_mean)).to_numpy()
    )
    assert pmf[0].to_numpy() == pytest.approx(zero_probability.to_numpy())
    assert (pmf.to_numpy() >= 0.0).all()
    assert pmf.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(pmf)), abs=1e-10)
    pmf_mean = pmf.to_numpy() @ pmf.columns.to_numpy(dtype=float)
    assert pmf_mean == pytest.approx(expected.to_numpy(), abs=1e-9)


def test_zip_analytical_information_matches_numerical_likelihood_hessian(fitted_zip):
    X, Z, y, result = fitted_zip
    count_design = X.to_numpy()
    inflation_design = Z.to_numpy()
    zero = y == 0

    def negative_loglike(parameters):
        inflation_index = inflation_design @ parameters[: Z.shape[1]]
        count_index = count_design @ parameters[Z.shape[1] :]
        mean = np.exp(count_index)
        contributions = np.empty(len(y))
        contributions[zero] = np.logaddexp(
            log_expit(inflation_index[zero]),
            log_expit(-inflation_index[zero]) - mean[zero],
        )
        positive = ~zero
        contributions[positive] = (
            log_expit(-inflation_index[positive])
            + y[positive] * count_index[positive]
            - mean[positive]
            - gammaln(y[positive] + 1.0)
        )
        return -contributions.sum()

    numerical_information = _numerical_hessian(
        negative_loglike, result.all_params.to_numpy()
    )
    analytical_information = np.linalg.inv(result.covariance.to_numpy())
    assert analytical_information == pytest.approx(
        numerical_information, rel=2e-6, abs=3e-5
    )
    assert np.max(np.abs(result.covariance.iloc[:2, 2:].to_numpy())) > 1e-6


@pytest.mark.parametrize(
    ("y", "match"),
    [
        (np.array([0.0, 1.5, 2.0, 0.0, 1.0, 2.0]), "integer counts"),
        (np.zeros(6), "both zero and positive"),
        (np.arange(1.0, 7.0), "both zero and positive"),
        (np.array([[0, 1], [0, 2], [1, 0]]), "one-dimensional"),
    ],
)
def test_zip_rejects_invalid_count_samples(y, match):
    X = pd.DataFrame({"const": 1.0, "x": np.arange(np.asarray(y).size)})
    with pytest.raises(ValueError, match=match):
        ZeroInflatedPoisson().fit(X, y)


def test_zip_rejects_rank_deficiency_and_prediction_schema_mismatch(fitted_zip):
    X, Z, y, result = fitted_zip
    bad_Z = Z.assign(copy=Z["z"])
    with pytest.raises(ValueError, match="rank deficient"):
        ZeroInflatedPoisson().fit(X, y, X_inflation=bad_Z)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["x", "const"]], X_inflation=Z)
    with pytest.raises(ValueError, match="indices must match"):
        result.predict(X.iloc[:10], X_inflation=Z.iloc[:10].reset_index(drop=True))
    with pytest.raises(ValueError, match="max_count"):
        result.predict_pmf(X.iloc[:3], X_inflation=Z.iloc[:3], max_count=-1)


def test_zip_same_design_default_is_backward_compatible():
    rng = np.random.default_rng(22)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=500)})
    mean = np.exp(0.1 + 0.2 * X["x"].to_numpy())
    inflation = expit(-1.0 - 0.3 * X["x"].to_numpy())
    y = rng.poisson(mean)
    y[rng.uniform(size=len(y)) < inflation] = 0
    result = ZeroInflatedPoisson().fit(X, y)
    assert result.feature_names == result.inflation_feature_names
    assert np.isfinite(result.predict(X)).all()
