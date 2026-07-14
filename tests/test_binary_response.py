import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import limiteddepkit.binary as binary_module
from limiteddepkit import BinaryLogit


@pytest.fixture
def fitted_logit():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=600),
            "x2": rng.normal(size=600),
        },
        index=pd.Index(np.arange(1_000, 1_600), name="row"),
    )
    beta = np.array([-0.4, 0.8, -0.3])
    y = rng.binomial(1, expit(X.to_numpy() @ beta))
    return X, y, BinaryLogit().fit(X, y)


def test_binary_logit_fits_with_observed_information(fitted_logit):
    X, _, result = fitted_logit

    assert result.converged
    assert result.inference_valid
    assert result.backend == "native-mle"
    assert result.covariance_type == "observed-information"
    assert result.params.index.tolist() == list(X.columns)
    assert result.nobs == len(X)
    assert result.n_params == X.shape[1]
    assert result.df_resid == len(X) - X.shape[1]
    assert result.constant_features == ("const",)
    assert result.score_norm <= 1e-7
    assert np.isfinite(result.loglike)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert np.all(np.isfinite(result.standard_errors))

    probabilities = result.predict_proba(X.iloc[:25])
    assert probabilities.index.equals(X.index[:25])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all().all()
    assert set(result.predict(X.iloc[:25]).unique()) <= {0, 1}


def test_binary_logit_covariance_is_inverse_analytical_information(fitted_logit):
    X, _, result = fitted_logit
    probabilities = expit(X.to_numpy() @ result.params.to_numpy())
    weights = probabilities * (1.0 - probabilities)
    information = X.to_numpy().T @ (weights[:, None] * X.to_numpy())

    assert result.covariance.to_numpy() == pytest.approx(
        np.linalg.inv(information), rel=1e-10, abs=1e-12
    )
    assert result.vcov().equals(result.covariance)
    assert result.vcov() is not result.covariance
    assert list(result.summary_frame().columns) == ["coef", "std_err", "z", "p_value"]
    assert result.conf_int().shape == (X.shape[1], 2)


def test_binary_logit_common_finite_mle_uses_fast_separation_certificate(
    fitted_logit, monkeypatch
):
    X, y, _ = fitted_logit

    def unexpected_exact_check(*_args, **_kwargs):
        raise AssertionError("ordinary finite fits should not need the exact LP check")

    monkeypatch.setattr(binary_module, "linprog", unexpected_exact_check)
    result = BinaryLogit().fit(X, y)

    assert result.converged
    assert result.score_norm <= 1e-7


def test_binary_logit_marginal_effects_match_probability_finite_differences(fitted_logit):
    X, _, result = fitted_logit
    evaluation = X.iloc[:12].copy()
    analytical = result.marginal_effects(evaluation)
    assert list(analytical.columns) == ["x1", "x2"]
    step = 1e-6

    for feature in ["x1", "x2"]:
        higher = evaluation.copy()
        lower = evaluation.copy()
        higher[feature] += step
        lower[feature] -= step
        numerical = (
            result.predict_proba(higher)[1].to_numpy()
            - result.predict_proba(lower)[1].to_numpy()
        ) / (2.0 * step)
        assert analytical[feature].to_numpy() == pytest.approx(numerical, abs=2e-8)

    pd.testing.assert_series_equal(
        result.average_marginal_effects(evaluation),
        analytical.mean(axis=0).rename("estimate"),
    )


@pytest.mark.parametrize(
    ("X", "y", "match"),
    [
        (
            pd.DataFrame({"const": 1.0, "x": np.arange(8.0)}),
            np.zeros(8),
            "both binary outcome classes",
        ),
        (
            pd.DataFrame({"const": 1.0, "x": np.arange(8.0), "duplicate": np.arange(8.0)}),
            np.array([0, 1] * 4),
            "rank deficient",
        ),
        (
            pd.DataFrame({"const": 1.0, "x": np.arange(8.0)}),
            np.array([[0, 1], [1, 0], [0, 1], [1, 0]]),
            "one-dimensional",
        ),
    ],
)
def test_binary_logit_rejects_unidentified_or_invalid_samples(X, y, match):
    with pytest.raises(ValueError, match=match):
        BinaryLogit().fit(X, y)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 0]),
    ],
)
def test_binary_logit_rejects_complete_and_quasi_complete_separation(y):
    if len(y) == 6:
        x = np.array([-2.0, -1.0, 0.0, 0.0, 1.0, 2.0])
    else:
        x = np.array([-4.0, -3.0, -2.0, 1.0, 2.0, 3.0, 4.0, -1.0])
    X = pd.DataFrame({"const": 1.0, "x": x})
    with pytest.raises(ValueError, match="separation"):
        BinaryLogit().fit(X, y)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("tolerance", [1e-8, 0.1, 1.0])
def test_binary_logit_separation_check_is_scale_and_tolerance_invariant(
    scale, tolerance
):
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x": scale * np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="separation"):
        BinaryLogit().fit(X, y, tolerance=tolerance)


def test_binary_logit_validates_prediction_contract_and_threshold(fitted_logit):
    X, _, result = fitted_logit
    with pytest.raises(ValueError, match="columns must match"):
        result.predict_proba(X[["x1", "const", "x2"]])
    with pytest.raises(ValueError, match="expected 3"):
        result.predict_proba(X[["const", "x1"]].to_numpy())
    with pytest.raises(ValueError, match="threshold"):
        result.predict(X, threshold=1.0)


def test_binary_logit_extreme_prediction_indices_remain_finite(fitted_logit):
    _, _, result = fitted_logit
    X = pd.DataFrame(
        {
            "const": [1.0, 1.0],
            "x1": [-1e6, 1e6],
            "x2": [1e6, -1e6],
        }
    )
    probabilities = result.predict_proba(X)
    assert np.isfinite(probabilities.to_numpy()).all()
    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(np.ones(2))
