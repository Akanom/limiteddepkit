import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from limiteddepkit import BinaryProbit


def _probit_data():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=650),
            "x2": rng.normal(size=650),
        },
        index=pd.Index(np.arange(2_000, 2_650), name="row"),
    )
    beta = np.array([0.3, 0.7, -0.25])
    y = rng.binomial(1, norm.cdf(X.to_numpy() @ beta))
    return X, y


@pytest.fixture
def fitted_probit():
    X, y = _probit_data()
    return X, y, BinaryProbit().fit(X, y)


def test_binary_probit_fits_with_complete_result_contract(fitted_probit):
    X, _, result = fitted_probit

    assert result.converged
    assert result.inference_valid
    assert result.backend == "native-mle"
    assert result.covariance_type == "observed-information"
    assert result.params.index.tolist() == list(X.columns)
    assert result.nobs == len(X)
    assert result.df_resid == len(X) - X.shape[1]
    assert result.constant_features == ("const",)
    assert result.score_norm <= 1e-7
    assert np.all(np.isfinite(result.standard_errors))

    probabilities = result.predict_proba(X.iloc[:30])
    assert probabilities.index.equals(X.index[:30])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all().all()
    assert set(result.predict(X.iloc[:30]).unique()) <= {0, 1}
    assert list(result.summary_frame().index) == list(X.columns)


def test_binary_probit_fixture_converges_without_bfgs(monkeypatch):
    X, y = _probit_data()

    def reject_bfgs(*args, **kwargs):
        raise AssertionError("BFGS should not run for this regular Probit fit")

    monkeypatch.setattr("limiteddepkit.binary_probit.minimize", reject_bfgs)
    result = BinaryProbit().fit(X, y)

    assert result.converged
    assert result.optimizer_result.method == "damped-newton-irls"
    assert result.score_norm <= 1e-7


def test_binary_probit_marginal_effects_match_probability_finite_differences(fitted_probit):
    X, _, result = fitted_probit
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
            result.predict_proba(higher)[1].to_numpy() - result.predict_proba(lower)[1].to_numpy()
        ) / (2.0 * step)
        assert analytical[feature].to_numpy() == pytest.approx(numerical, abs=2e-8)


def test_binary_probit_rejects_separation_and_invalid_optimizer_options():
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="separation"):
        BinaryProbit().fit(X, y)
    with pytest.raises(ValueError, match="maxiter"):
        BinaryProbit().fit(X, y, maxiter=0)
    with pytest.raises(ValueError, match="tolerance"):
        BinaryProbit().fit(X, y, tolerance=0.0)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_binary_probit_separation_check_is_scale_invariant(scale):
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x": scale * np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="separation"):
        BinaryProbit().fit(X, y, tolerance=0.1)


def test_binary_probit_prediction_contract_and_extreme_indices(fitted_probit):
    X, _, result = fitted_probit
    with pytest.raises(ValueError, match="columns must match"):
        result.predict_proba(X[["x2", "x1", "const"]])
    with pytest.raises(ValueError, match="threshold"):
        result.predict(X, threshold=np.nan)

    extreme = pd.DataFrame(
        {
            "const": [1.0, 1.0],
            "x1": [-1e6, 1e6],
            "x2": [1e6, -1e6],
        }
    )
    probabilities = result.predict_proba(extreme)
    assert np.isfinite(probabilities.to_numpy()).all()
    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(np.ones(2))
