import numpy as np
import pandas as pd
import pytest

import limiteddepkit.poisson as poisson_module
from limiteddepkit.count import PoissonRegressor


def test_poisson_regressor_fits_and_predicts_counts():
    rng = np.random.default_rng(17)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=400)})
    linear = -0.2 + 0.6 * X["x"].to_numpy()
    counts = rng.poisson(np.exp(linear))

    result = PoissonRegressor().fit(X, counts)

    assert result.converged
    assert result.params.index.tolist() == ["const", "x"]
    assert result.predict(X).shape == (400,)
    assert result.nobs == 400


def test_poisson_prepares_active_likelihood_constants_once(monkeypatch):
    rng = np.random.default_rng(726)
    source = rng.normal(size=(600, 4))
    noncontiguous = source[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    X = pd.DataFrame(
        np.column_stack([np.ones(len(source)), noncontiguous]),
        columns=["const", "x1", "x2"],
    )
    exposure = rng.uniform(0.5, 2.0, size=len(X))
    linear = X.to_numpy() @ np.array([0.15, 0.25, -0.2]) + np.log(exposure)
    counts = rng.poisson(np.exp(linear))
    weights = np.ones(len(X), dtype=int)
    weights[::11] = 0
    original_gammaln = poisson_module.gammaln
    calls = 0

    def tracked_gammaln(values):
        nonlocal calls
        calls += 1
        return original_gammaln(values)

    monkeypatch.setattr(poisson_module, "gammaln", tracked_gammaln)
    result = PoissonRegressor().fit(
        X,
        counts,
        exposure=exposure,
        freq_weights=weights,
        engine="accelerated",
    )

    assert result.converged
    assert result.inference_valid
    assert calls == 1
    assert np.isfinite(result.params.to_numpy()).all()


def test_poisson_accelerated_engine_is_bitwise_equal_and_reference_is_default():
    rng = np.random.default_rng(902)
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=800),
            "x2": rng.normal(size=800),
        }
    )
    exposure = rng.uniform(0.75, 1.5, size=len(X))
    offset = rng.normal(scale=0.05, size=len(X))
    mean = np.exp(X.to_numpy() @ np.array([0.1, 0.35, -0.2]) + offset + np.log(exposure))
    counts = rng.poisson(mean)
    weights = np.ones(len(X), dtype=int)
    weights[::13] = 0

    default = PoissonRegressor().fit(
        X,
        counts,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
    )
    reference = PoissonRegressor().fit(
        X,
        counts,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
        engine="reference",
    )
    accelerated = PoissonRegressor().fit(
        X,
        counts,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
        engine="accelerated",
    )

    for left, right in ((default, reference), (reference, accelerated)):
        np.testing.assert_array_equal(left.params.to_numpy(), right.params.to_numpy())
        np.testing.assert_array_equal(left.covariance.to_numpy(), right.covariance.to_numpy())
        np.testing.assert_array_equal(
            left.standard_errors.to_numpy(), right.standard_errors.to_numpy()
        )
        np.testing.assert_array_equal(left.fitted_values.to_numpy(), right.fitted_values.to_numpy())
        assert left.loglike == right.loglike
        assert left.score_norm == right.score_norm
        assert left.scaled_score_norm == right.scaled_score_norm
        assert left.pearson_chi2 == right.pearson_chi2
        assert left.deviance == right.deviance
        assert left.optimizer_result.nit == right.optimizer_result.nit
        assert left.optimizer_result.nfev == right.optimizer_result.nfev
        assert left.optimizer_result.njev == right.optimizer_result.njev


def test_poisson_rejects_unknown_engine():
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0], "x": [-1.0, 0.0, 1.0]})

    with pytest.raises(ValueError, match="engine"):
        PoissonRegressor().fit(X, [0, 1, 2], engine="auto")
    with pytest.raises(TypeError, match="engine"):
        PoissonRegressor().fit(X, [0, 1, 2], engine=None)
