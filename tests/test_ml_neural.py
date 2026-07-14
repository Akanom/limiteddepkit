"""Tests for the optional residual neural-network prediction challenger."""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import limiteddepkit.ml.neural as neural
from limiteddepkit.ml.neural import ResidualBinaryMLP
from limiteddepkit.ml.split import StratifiedKFold
from limiteddepkit.ml.tuning import nested_cross_validate
from limiteddepkit.ml.validation import cross_validate

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")


def _binary_data(nobs: int = 72) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7301)
    X = pd.DataFrame(
        {
            "income": rng.normal(size=nobs),
            "age": rng.normal(size=nobs),
            "exposure": rng.normal(size=nobs),
        },
        index=pd.Index([f"person-{row}" for row in range(nobs)], name="person"),
    )
    probability = expit(0.8 * X["income"] - 0.55 * X["age"] + 0.25 * X["exposure"])
    y = pd.Series(rng.binomial(1, probability), index=X.index, name="participates")
    assert y.value_counts().min() >= 12
    return X, y


def _quick_model(*, random_state: int = 91, dropout: float = 0.2) -> ResidualBinaryMLP:
    return ResidualBinaryMLP(
        hidden_width=8,
        n_blocks=1,
        dropout=dropout,
        learning_rate=0.01,
        weight_decay=1e-3,
        batch_size=64,
        max_epochs=20,
        validation_fraction=0.2,
        patience=4,
        min_delta=1e-4,
        gradient_clip_norm=2.0,
        temperature_scaling=True,
        random_state=random_state,
        device="cpu",
    )


def _tiny_cv_model() -> ResidualBinaryMLP:
    return ResidualBinaryMLP(
        hidden_width=4,
        n_blocks=1,
        dropout=0.1,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=4,
        validation_fraction=0.2,
        patience=1,
        min_delta=10.0,
        gradient_clip_norm=2.0,
        temperature_scaling=False,
        random_state=19,
    )


def test_torch_import_is_deferred_until_fit(monkeypatch):
    calls = []

    def missing_import(name):
        calls.append(name)
        raise ImportError("deliberately unavailable")

    monkeypatch.setattr(neural.importlib, "import_module", missing_import)
    model = ResidualBinaryMLP(hidden_width=4, n_blocks=1, max_epochs=1)
    assert calls == []

    X, y = _binary_data(24)
    with pytest.raises(ImportError, match="optional and requires PyTorch"):
        model.fit(X, y)
    assert calls == ["torch"]


def test_probability_challenger_defaults_to_unweighted_binary_loss():
    assert ResidualBinaryMLP().positive_class_weight is None


def test_fit_signature_exposes_and_rejects_panel_or_time_metadata_before_torch(monkeypatch):
    calls = []

    def tracked_import(name):
        calls.append(name)
        raise AssertionError("metadata rejection must precede the optional import")

    monkeypatch.setattr(neural.importlib, "import_module", tracked_import)
    parameters = inspect.signature(ResidualBinaryMLP.fit).parameters
    assert parameters["entity"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["time"].kind is inspect.Parameter.KEYWORD_ONLY

    X, y = _binary_data(24)
    with pytest.raises(ValueError, match="independent rows only.*entity metadata"):
        ResidualBinaryMLP().fit(X, y, entity=np.repeat(np.arange(6), 4))
    with pytest.raises(ValueError, match="independent rows only.*time metadata"):
        ResidualBinaryMLP().fit(X, y, time=np.arange(len(X)))
    with pytest.raises(ValueError, match="entity and time metadata"):
        ResidualBinaryMLP().fit(
            X,
            y,
            entity=np.repeat(np.arange(6), 4),
            time=np.tile(np.arange(4), 6),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("hidden_width", 0),
        ("n_blocks", 0),
        ("dropout", 1.0),
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("batch_size", True),
        ("validation_fraction", 1.0),
        ("patience", 0),
        ("gradient_clip_norm", 0.0),
        ("positive_class_weight", "minority"),
        ("temperature_scaling", 1),
        ("random_state", None),
        ("device", ""),
    ],
)
def test_constructor_rejects_invalid_hyperparameters(keyword, value):
    with pytest.raises(ValueError):
        ResidualBinaryMLP(**{keyword: value})


@requires_torch
def test_fit_is_deterministic_and_standardization_excludes_validation_rows():
    X, y = _binary_data()
    first = _quick_model().fit(X, y)
    second = _quick_model().fit(X, y)

    pd.testing.assert_frame_equal(first.predict_proba(X), second.predict_proba(X))
    pd.testing.assert_frame_equal(
        first.training_history_frame(), second.training_history_frame()
    )
    np.testing.assert_array_equal(
        first.internal_training_indices, second.internal_training_indices
    )
    np.testing.assert_array_equal(
        first.internal_validation_indices, second.internal_validation_indices
    )
    expected_means = X.iloc[first.internal_training_indices].mean(axis=0)
    expected_scales = X.iloc[first.internal_training_indices].std(axis=0, ddof=0)
    pd.testing.assert_series_equal(first.feature_means, expected_means.rename("mean"))
    pd.testing.assert_series_equal(first.feature_scales, expected_scales.rename("scale"))
    assert set(first.internal_training_indices).isdisjoint(
        set(first.internal_validation_indices)
    )
    assert first.calibrated_validation_loss <= first.best_validation_loss + 1e-6
    assert not first.inference_valid
    assert first.training_completed


@requires_torch
def test_labeled_probabilities_schema_validation_and_early_stopping_metadata():
    X, y = _binary_data()
    model = ResidualBinaryMLP(
        hidden_width=8,
        n_blocks=1,
        dropout=0.2,
        learning_rate=0.01,
        batch_size=128,
        max_epochs=20,
        patience=2,
        min_delta=10.0,
        random_state=8,
    )
    result = model.fit(X, y)
    probabilities = result.predict_proba(X.iloc[:7])
    predictions = result.predict(X.iloc[:7])

    assert probabilities.index.equals(X.index[:7])
    assert probabilities.columns.tolist() == [0, 1]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    assert predictions.index.equals(X.index[:7])
    assert set(predictions.unique()).issubset({0, 1})
    assert result.converged
    assert result.training_completed
    assert result.stopped_early
    assert result.n_epochs == 3
    assert result.best_epoch == 1
    assert len(result.training_history_frame()) == result.n_epochs
    assert result.diagnostics()["internal_validation_n"] == len(
        result.internal_validation_indices
    )

    with pytest.raises(ValueError, match="columns must match"):
        result.predict_proba(X[["age", "income", "exposure"]])
    with pytest.raises(ValueError, match="non-finite"):
        result.predict_proba(X.assign(income=np.nan))
    with pytest.raises(ValueError, match="strictly between"):
        result.predict(X, threshold=1.0)


@requires_torch
def test_epoch_limit_is_training_completion_not_optimizer_convergence():
    X, y = _binary_data(32)
    result = ResidualBinaryMLP(
        hidden_width=4,
        n_blocks=1,
        max_epochs=1,
        patience=3,
        temperature_scaling=False,
        random_state=12,
    ).fit(X, y)

    assert result.training_completed
    assert not result.stopped_early
    assert not result.converged


@requires_torch
def test_mc_dropout_returns_reproducible_probability_uncertainty():
    X, y = _binary_data()
    result = _quick_model(dropout=0.4).fit(X, y)

    first = result.mc_dropout_probabilities(X.iloc[:9], n_draws=24, random_state=44)
    second = result.mc_dropout_probabilities(X.iloc[:9], n_draws=24, random_state=44)
    pd.testing.assert_frame_equal(first, second)
    assert first.index.equals(X.index[:9])
    assert first.columns.name == "draw"
    assert np.all((first.to_numpy() >= 0.0) & (first.to_numpy() <= 1.0))
    assert np.any(first.std(axis=1).to_numpy() > 1e-8)

    uncertainty = result.predict_proba_uncertainty(
        X.iloc[:9], n_draws=24, level=0.9, random_state=44
    )
    assert uncertainty.index.equals(X.index[:9])
    assert set(uncertainty) == {
        "probability_mean",
        "probability_std",
        "probability_lower",
        "probability_upper",
        "n_draws",
    }
    assert np.all(uncertainty["probability_lower"] <= uncertainty["probability_mean"])
    assert np.all(uncertainty["probability_mean"] <= uncertainty["probability_upper"])
    assert np.all(uncertainty["n_draws"] == 24)


@requires_torch
def test_neural_challenger_integrates_with_cross_validation_and_nested_selection():
    X, y = _binary_data(48)
    ordinary = cross_validate(
        _tiny_cv_model,
        X,
        y,
        splitter=StratifiedKFold(2, shuffle=True, random_state=3),
        outcome="binary",
        require_inference_valid=False,
    )
    assert ordinary.successful_folds == 2
    assert ordinary.eligible
    assert len(ordinary.out_of_fold_predictions()) == len(X)

    nested = nested_cross_validate(
        {"residual-mlp": _tiny_cv_model},
        X,
        y,
        outer_splitter=StratifiedKFold(2, shuffle=True, random_state=11),
        inner_splitter_factory=lambda: StratifiedKFold(
            2, shuffle=True, random_state=13
        ),
        outcome="binary",
        require_inference_valid=False,
        selection_rule="one_se",
    )
    assert nested.eligible
    assert nested.selected_models == ("residual-mlp", "residual-mlp")
    assert len(nested.out_of_fold_predictions()) == len(X)
