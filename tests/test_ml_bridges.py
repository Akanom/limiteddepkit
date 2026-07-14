"""Tests for lazy generic, scikit-learn and statsmodels bridges."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit.ml.bridges import (
    PredictProbabilityProtocol,
    PredictProtocol,
    ProbabilityBridgedResult,
    generic_bridge,
    sklearn_bridge,
    statsmodels_bridge,
)
from limiteddepkit.ml.split import EntityHoldoutSplit
from limiteddepkit.ml.validation import cross_validate


@dataclass
class ToyResult:
    slope: float


def test_generic_probability_bridge_normalizes_callbacks_and_diagnostics():
    def fit_function(X, y, *, scale=1.0):
        del y
        return ToyResult(slope=scale * float(np.mean(np.asarray(X))))

    def predict_function(result, X, *, shift=0.0):
        probability = expit(result.slope * np.asarray(X, dtype=float) + shift)
        return np.column_stack((1.0 - probability, probability))

    bridge = generic_bridge(
        fit_function,
        predict_function,
        probability_output=True,
        fit_options={"scale": 2.0},
        diagnostics=lambda result: {"converged": result.slope > 0},
    )
    result = bridge.fit(np.array([1.0, 2.0]), np.array([0, 1]))

    assert isinstance(result, ProbabilityBridgedResult)
    assert isinstance(result, PredictProtocol)
    assert isinstance(result, PredictProbabilityProtocol)
    assert result.converged
    assert result.raw_result.slope == pytest.approx(3.0)
    assert result.predict_proba([0.0, 1.0], shift=0.1).shape == (2, 2)


def test_generic_optional_dependency_is_checked_only_when_fit_is_called():
    bridge = generic_bridge(
        lambda X, y: ToyResult(1.0),
        lambda result, X: np.zeros(len(X)),
        dependency="limiteddepkit_dependency_that_does_not_exist",
    )

    with pytest.raises(ImportError, match="required only when this bridge is used"):
        bridge.fit([1.0], [0])


def test_generic_bridge_rejects_invented_or_nonboolean_diagnostics():
    invalid_name = generic_bridge(
        lambda X, y: ToyResult(1.0),
        lambda result, X: np.zeros(len(X)),
        diagnostics=lambda result: {"validated_by_magic": True},
    )
    with pytest.raises(ValueError, match="Unsupported bridge diagnostics"):
        invalid_name.fit([1.0], [0])

    invalid_value = generic_bridge(
        lambda X, y: ToyResult(1.0),
        lambda result, X: np.zeros(len(X)),
        diagnostics=lambda result: {"converged": 1},
    )
    with pytest.raises(TypeError, match="must be Boolean"):
        invalid_value.fit([1.0], [0])


def test_sklearn_bridge_clones_estimator_and_returns_probabilities():
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    estimator = sklearn_linear.LogisticRegression(C=1e6, solver="lbfgs")
    X = np.arange(-4.0, 5.0).reshape(-1, 1)
    y = (X[:, 0] > 0.0).astype(int)

    bridge = sklearn_bridge(estimator)
    result = bridge.fit(X, y)

    assert not hasattr(estimator, "coef_")
    assert isinstance(result, ProbabilityBridgedResult)
    probability = result.predict_proba(X)
    assert probability.shape == (len(X), 2)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert not hasattr(result, "inference_valid")


def test_sklearn_bridge_rejects_unavailable_prediction_method():
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    estimator = sklearn_linear.LinearRegression()
    bridge = sklearn_bridge(estimator, prediction_method="predict_proba")
    with pytest.raises(TypeError, match=r"does not expose predict_proba\(\)"):
        bridge.fit([[0.0], [1.0]], [0.0, 1.0])


def test_statsmodels_bridge_supports_lazy_factory_constant_and_diagnostics():
    statsmodels_api = pytest.importorskip("statsmodels.api")
    X = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    probability = expit(-0.4 + 1.2 * X[:, 0])
    rng = np.random.default_rng(42)
    y = rng.binomial(1, probability)

    bridge = statsmodels_bridge(
        "statsmodels.api:Logit",
        prediction_kind="probability",
        add_constant=True,
        fit_options={"disp": False},
        diagnostics=lambda result: {
            "converged": bool(result.mle_retvals["converged"]),
            "inference_valid": bool(np.all(np.isfinite(result.cov_params()))),
        },
    )
    result = bridge.fit(X, y)

    expected = result.raw_result.predict(statsmodels_api.add_constant(X))
    assert result.converged
    assert result.inference_valid
    assert result.predict_proba(X) == pytest.approx(expected)
    assert result.bridge_name == "statsmodels"


def test_statsmodels_lazy_factory_must_remain_inside_dependency_namespace():
    bridge = statsmodels_bridge(
        "scipy.special:expit",
        prediction_kind="probability",
    )
    with pytest.raises(ValueError, match="outside dependency"):
        bridge.fit([[0.0], [1.0]], [0, 1])


def test_statsmodels_bridge_requires_explicit_prediction_semantics():
    with pytest.raises(TypeError, match="prediction_kind"):
        statsmodels_bridge("statsmodels.api:Logit")  # type: ignore[call-arg]

    bridge = statsmodels_bridge(
        "statsmodels.api:Logit",
        prediction_kind="auto",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="explicitly"):
        bridge.fit([[0.0], [1.0]], [0, 1])


def test_sklearn_and_statsmodels_bridges_strip_workflow_entity_metadata():
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(901)
    nobs = 120
    X = pd.DataFrame({"x": rng.normal(size=nobs)})
    entity = np.repeat(np.arange(30), 4)
    y = rng.binomial(1, expit(-0.2 + 0.8 * X["x"].to_numpy()))
    splitter = EntityHoldoutSplit(3)

    sklearn_result = cross_validate(
        lambda: sklearn_bridge(
            sklearn_linear.LogisticRegression(max_iter=2_000),
        ),
        X,
        y,
        splitter=splitter,
        entity=entity,
        outcome="binary",
        prediction_target="new_entity",
        require_converged=False,
        require_inference_valid=False,
    )
    statsmodels_result = cross_validate(
        lambda: statsmodels_bridge(
            "statsmodels.api:Logit",
            prediction_kind="probability",
            add_constant=True,
            fit_options={"disp": False},
        ),
        X,
        y,
        splitter=splitter,
        entity=entity,
        outcome="binary",
        prediction_target="new_entity",
        require_converged=False,
        require_inference_valid=False,
    )

    assert sklearn_result.successful_folds == 3
    assert statsmodels_result.successful_folds == 3


def test_sklearn_probability_bridge_preserves_arbitrary_class_labels():
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(88)
    labels = np.array(["bronze", "silver", "gold"])
    y = np.tile(labels, 30)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=len(y)),
            "x2": rng.normal(size=len(y)),
        },
        index=pd.Index([f"row-{index}" for index in range(len(y))]),
    )

    result = cross_validate(
        lambda: sklearn_bridge(
            sklearn_linear.LogisticRegression(max_iter=2_000),
        ),
        X,
        y,
        splitter=EntityHoldoutSplit(3),
        entity=np.repeat(np.arange(30), 3),
        outcome="multiclass",
        prediction_target="new_entity",
        require_converged=False,
        require_inference_valid=False,
    )

    assert result.successful_folds == 3
    probability_columns = {
        "prediction_bronze",
        "prediction_silver",
        "prediction_gold",
    }
    assert probability_columns.issubset(result.out_of_fold_predictions().columns)
