from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.ml.split import StratifiedKFold
from limiteddepkit.ml.tuning import TuningCandidate, nested_cross_validate


@dataclass(frozen=True)
class _ConstantBinaryResult:
    probability: float
    converged: bool = True
    inference_valid: bool = True

    def predict_proba(self, X):
        probability = np.full(len(X), self.probability)
        return pd.DataFrame({0: 1.0 - probability, 1: probability})


class _ConstantBinaryEstimator:
    fitted_probabilities: list[float] = []

    def fit(self, X, y, *, probability=0.5):
        del X, y
        self.fitted_probabilities.append(float(probability))
        return _ConstantBinaryResult(float(probability))


def _balanced_data():
    X = pd.DataFrame({"constant": np.ones(36), "row": np.arange(36)})
    y = np.tile([0, 1], 18)
    return X, y


def test_nested_cv_uses_one_se_simplicity_and_candidate_fit_kwargs():
    X, y = _balanced_data()
    _ConstantBinaryEstimator.fitted_probabilities.clear()
    candidates = {
        "complex-first": TuningCandidate(
            _ConstantBinaryEstimator,
            fit_kwargs={"probability": 0.5},
            complexity=2.0,
        ),
        "simple": TuningCandidate(
            _ConstantBinaryEstimator,
            fit_kwargs={"probability": 0.5},
            complexity=1.0,
        ),
    }

    result = nested_cross_validate(
        candidates,
        X,
        y,
        outer_splitter=StratifiedKFold(3, shuffle=True, random_state=1),
        inner_splitter_factory=lambda: StratifiedKFold(
            2, shuffle=True, random_state=2
        ),
        entity=np.repeat(np.arange(12), 3),
        outcome="binary",
        selection_rule="one_se",
    )

    assert result.selected_models == ("simple", "simple", "simple")
    assert result.eligible
    assert result.outer_result.successful_folds == 3
    assert result.fold_frame()["selected_model"].tolist() == ["simple"] * 3
    assert len(result.out_of_fold_predictions()) == len(X)
    assert set(_ConstantBinaryEstimator.fitted_probabilities) == {0.5}


def test_nested_cv_best_rule_keeps_outer_test_rows_out_of_inner_comparison():
    X, y = _balanced_data()
    result = nested_cross_validate(
        {
            "calibrated": TuningCandidate(
                _ConstantBinaryEstimator,
                fit_kwargs={"probability": 0.5},
                complexity=1.0,
            ),
            "miscalibrated": TuningCandidate(
                _ConstantBinaryEstimator,
                fit_kwargs={"probability": 0.8},
                complexity=1.0,
            ),
        },
        X,
        y,
        outer_splitter=StratifiedKFold(3, shuffle=True, random_state=10),
        inner_splitter_factory=lambda: StratifiedKFold(
            3, shuffle=True, random_state=20
        ),
        outcome="binary",
        primary_metric="log_loss",
        selection_rule="best",
    )

    assert result.selected_models == ("calibrated",) * 3
    for selection in result.selections:
        outer_test = set(selection.test_index)
        assert all(
            outer_test.isdisjoint(set(selection.train_index[fold.test_index]))
            for comparison in selection.inner_comparison.cv_results.values()
            for fold in comparison.folds
        )
    assert result.summary_frame().loc["log_loss", "mean"] == pytest.approx(
        np.log(2.0)
    )


def test_nested_cv_rejects_non_nested_or_ambiguous_splitter_contracts():
    X, y = _balanced_data()
    with pytest.raises(TypeError, match="outer_splitter"):
        nested_cross_validate(
            {"model": _ConstantBinaryEstimator},
            X,
            y,
            outer_splitter=StratifiedKFold(3),
            inner_splitter_factory=lambda: StratifiedKFold(2),
            splitter=StratifiedKFold(3),
            outcome="binary",
        )
    with pytest.raises(ValueError, match="selection_rule"):
        nested_cross_validate(
            {"model": _ConstantBinaryEstimator},
            X,
            y,
            outer_splitter=StratifiedKFold(3),
            inner_splitter_factory=lambda: StratifiedKFold(2),
            selection_rule="optimistic",
            outcome="binary",
        )
    with pytest.raises(TypeError, match=r"TuningCandidate\(complexity"):
        nested_cross_validate(
            {"first": _ConstantBinaryEstimator, "second": _ConstantBinaryEstimator},
            X,
            y,
            outer_splitter=StratifiedKFold(3),
            inner_splitter_factory=lambda: StratifiedKFold(2),
            outcome="binary",
            selection_rule="one_se",
        )


def test_nested_cv_refits_transformer_only_on_each_inner_and_outer_training_set():
    X, y = _balanced_data()
    fitted_rows = []

    class RowSpyTransformer:
        def fit(self, X, y=None):
            del y
            self.rows_ = tuple(np.asarray(X["row"], dtype=int))
            fitted_rows.append(self.rows_)
            return self

        def transform(self, X):
            return X.copy()

    result = nested_cross_validate(
        {"constant": _ConstantBinaryEstimator},
        X,
        y,
        outer_splitter=StratifiedKFold(3, shuffle=True, random_state=40),
        inner_splitter_factory=lambda: StratifiedKFold(
            2, shuffle=True, random_state=41
        ),
        outcome="binary",
        transformer_factory=RowSpyTransformer,
    )

    assert len(fitted_rows) == 9
    for outer_offset, selection in enumerate(result.selections):
        inner_result = selection.inner_comparison.cv_results["constant"]
        expected = [
            tuple(selection.train_index[fold.train_index])
            for fold in inner_result.folds
        ]
        expected.append(tuple(selection.train_index))
        observed = fitted_rows[3 * outer_offset : 3 * outer_offset + 3]
        assert observed == expected
        assert all(
            set(rows).isdisjoint(set(selection.test_index)) for rows in observed
        )
