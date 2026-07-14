"""Tests for limited-outcome cross-validation and comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import expit

from limiteddepkit.ml.adapter import infer_outcome, result_eligibility
from limiteddepkit.ml.compare import compare_models
from limiteddepkit.ml.split import (
    EntityHoldoutSplit,
    ForwardPanelSplit,
    KFold,
    StratifiedKFold,
)
from limiteddepkit.ml.validation import (
    CrossValidationResult,
    FoldEvaluation,
    cross_validate,
)


@dataclass
class ToyBinaryResult:
    slope: float
    converged: bool = True
    inference_valid: bool = True

    def predict_proba(self, X):
        values = np.asarray(X, dtype=float).reshape(len(X), -1)
        probability = expit(self.slope * values[:, 0])
        return pd.DataFrame({0: 1.0 - probability, 1: probability})


class ToyBinaryModel:
    def __init__(self, slope=1.0, *, inference_valid=True):
        self.slope = slope
        self.inference_valid = inference_valid

    def fit(self, X, y):
        return ToyBinaryResult(self.slope, inference_valid=self.inference_valid)


def test_binary_cross_validation_returns_fold_summary_and_oof_probabilities():
    X = pd.DataFrame({"x": np.linspace(-2.0, 2.0, 60)})
    y = pd.Series((X["x"] > 0).astype(int), index=X.index)

    result = cross_validate(
        lambda: ToyBinaryModel(slope=3.0),
        X,
        y,
        splitter=StratifiedKFold(3, shuffle=True, random_state=44),
        outcome="binary",
    )

    assert result.successful_folds == 3
    assert result.eligible_folds == 3
    assert result.eligible
    assert set(["log_loss", "brier_score", "accuracy"]).issubset(
        result.fold_frame().columns
    )
    assert result.summary_frame().loc["log_loss", "folds"] == 3
    oof = result.out_of_fold_predictions()
    assert len(oof) == len(X)
    assert {"prediction_0", "prediction_1"}.issubset(oof.columns)


def test_pooled_oof_uses_available_values_for_mixed_prediction_schemas():
    result = CrossValidationResult(
        folds=(
            FoldEvaluation(
                fold=1,
                train_index=np.array([1]),
                test_index=np.array([0]),
                outcome="count",
                prediction_target="pooled",
                prediction={"mean": np.array([1.0])},
            ),
            FoldEvaluation(
                fold=2,
                train_index=np.array([1]),
                test_index=np.array([0]),
                outcome="count",
                prediction_target="pooled",
                prediction={
                    "mean": np.array([3.0]),
                    "zero_probability": np.array([0.2]),
                },
            ),
        ),
        outcome="count",
        model_name="mixed-count",
        row_labels=np.array(["held-out", "training"]),
    )

    pooled = result.pooled_out_of_fold_predictions()

    assert pooled.loc[0, "prediction_mean"] == pytest.approx(2.0)
    assert pooled.loc[0, "prediction_mean__count"] == 2
    assert pooled.loc[0, "prediction_zero_probability"] == pytest.approx(0.2)
    assert pooled.loc[0, "prediction_zero_probability__count"] == 1
    assert pooled.loc[0, "prediction_zero_probability__weight_sum"] == pytest.approx(
        1.0
    )


def test_pooled_oof_keeps_duplicate_labels_distinct_and_weights_string_modes():
    result = CrossValidationResult(
        folds=(
            FoldEvaluation(
                fold=1,
                train_index=np.array([2]),
                test_index=np.array([0, 1]),
                outcome="continuous",
                prediction_target="pooled",
                prediction=pd.Series(["left-a", "left-b"]),
            ),
            FoldEvaluation(
                fold=2,
                train_index=np.array([2]),
                test_index=np.array([0, 1]),
                outcome="continuous",
                prediction_target="pooled",
                prediction=pd.Series(["right-a", "right-b"]),
            ),
        ),
        outcome="continuous",
        model_name="string-mode",
        row_labels=np.array(["duplicate", "duplicate", "training"]),
    )

    pooled = result.pooled_out_of_fold_predictions(
        fold_weights={1: 1.0, 2: 2.0}
    )

    assert pooled["row_position"].tolist() == [0, 1]
    assert pooled["row_index"].tolist() == ["duplicate", "duplicate"]
    assert pooled["prediction"].tolist() == ["right-a", "right-b"]
    assert pooled["prediction__count"].tolist() == [2, 2]
    assert pooled["prediction__weight_sum"].tolist() == [3.0, 3.0]


def test_fold_transformer_is_fitted_only_on_each_training_partition():
    fitted_transformers = []

    class MeanCenterer:
        def fit(self, X, y=None):
            del y
            self.mean_ = float(np.asarray(X)[:, 0].mean())
            self.fit_rows_ = tuple(np.asarray(X)[:, 0])
            fitted_transformers.append(self)
            return self

        def transform(self, X):
            values = np.asarray(X, dtype=float).copy()
            values[:, 0] -= self.mean_
            return values

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = np.tile([0, 1], 6)
    result = cross_validate(
        ToyBinaryModel,
        X,
        y,
        splitter=KFold(3),
        outcome="binary",
        transformer_factory=MeanCenterer,
    )

    assert len(fitted_transformers) == 3
    for fold, transformer in zip(result.folds, fitted_transformers, strict=True):
        expected_rows = X.iloc[fold.train_index, 0].to_numpy()
        np.testing.assert_array_equal(transformer.fit_rows_, expected_rows)
        assert transformer.mean_ == pytest.approx(float(expected_rows.mean()))
        assert fold.transformer is transformer


def test_fold_transformer_accepts_standard_sparse_matrix_outputs():
    class SparseTransformer:
        def fit(self, X, y=None):
            del X, y
            return self

        def transform(self, X):
            return sparse.csr_matrix(np.asarray(X, dtype=float))

    @dataclass
    class SparseResult:
        converged: bool = True
        inference_valid: bool = True

        def predict_proba(self, X):
            assert sparse.issparse(X)
            probability = np.full(X.shape[0], 0.5)
            return np.column_stack((1.0 - probability, probability))

    class SparseModel:
        def fit(self, X, y):
            assert sparse.issparse(X)
            assert X.shape[0] == len(y)
            return SparseResult()

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = np.tile([0, 1], 6)
    result = cross_validate(
        SparseModel,
        X,
        y,
        splitter=KFold(3),
        transformer_factory=SparseTransformer,
        outcome="binary",
    )

    assert result.successful_folds == 3
    assert all(sparse.issparse(fold.transformer.transform(X)) for fold in result.folds)


def test_model_comparison_ranks_only_valid_results():
    X = pd.DataFrame({"x": np.linspace(-3.0, 3.0, 90)})
    y = pd.Series((X["x"] > 0).astype(int))

    comparison = compare_models(
        {
            "strong": lambda: ToyBinaryModel(slope=5.0),
            "weak": lambda: ToyBinaryModel(slope=0.1),
            "invalid": lambda: ToyBinaryModel(slope=5.0, inference_valid=False),
        },
        X,
        y,
        splitter=StratifiedKFold(3),
        outcome="binary",
    )

    assert comparison.primary_metric == "log_loss"
    assert comparison.best_model == "strong"
    invalid = comparison.table.set_index("model").loc["invalid"]
    assert not bool(invalid["eligible"])
    assert pd.isna(invalid["rank"])


def test_model_comparison_rejects_names_that_collide_after_string_conversion():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = np.tile([0, 1], 10)

    with pytest.raises(ValueError, match="unique after conversion"):
        compare_models(
            {1: ToyBinaryModel, "1": ToyBinaryModel},
            X,
            y,
            splitter=StratifiedKFold(2),
            outcome="binary",
        )


@dataclass
class ToyPanelResult:
    converged: bool = True
    inference_valid: bool = True

    def predict_proba(self, X, *, entity=None):
        probability = np.full(len(X), 0.5)
        return pd.DataFrame({0: 1.0 - probability, 1: probability})

    def posterior_random_effects(self, X, y, *, entity):
        labels = pd.unique(np.asarray(entity))
        return pd.DataFrame({"posterior_mean": 0.0}, index=labels)

    def posterior_predict_proba(self, X, *, entity, posterior):
        assert set(pd.unique(np.asarray(entity))).issubset(set(posterior.index))
        probability = np.full(len(X), 0.75)
        return pd.DataFrame({0: 1.0 - probability, 1: probability})


class ToyPanelModel:
    def fit(self, X, y, *, entity):
        assert len(entity) == len(X)
        return ToyPanelResult()


def _panel_data(n_entities=6, n_periods=6):
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    X = pd.DataFrame({"x": np.tile(np.linspace(-1.0, 1.0, n_periods), n_entities)})
    y = pd.Series((time % 2).astype(int))
    return X, y, entity, time


def test_entity_holdout_uses_population_average_new_entity_target():
    X, y, entity, time = _panel_data()
    result = cross_validate(
        ToyPanelModel,
        X,
        y,
        splitter=EntityHoldoutSplit(3),
        entity=entity,
        time=time,
        outcome="binary",
        prediction_target="auto",
    )

    assert set(result.fold_frame()["prediction_target"]) == {"new_entity"}


def test_forward_panel_uses_training_history_for_posterior_prediction():
    X, y, entity, time = _panel_data(n_entities=4, n_periods=7)
    result = cross_validate(
        ToyPanelModel,
        X,
        y,
        splitter=ForwardPanelSplit(n_splits=2, min_train_periods=3),
        entity=entity,
        time=time,
        outcome="binary",
        prediction_target="known_entity_future",
    )

    assert set(result.fold_frame()["prediction_target"]) == {"known_entity_future"}
    assert np.allclose(result.out_of_fold_predictions()["prediction_1"], 0.75)


def test_forward_split_rejects_new_entity_estimand():
    X, y, entity, time = _panel_data(n_entities=4, n_periods=7)
    with pytest.raises(ValueError, match="complete held-out entities"):
        cross_validate(
            ToyPanelModel,
            X,
            y,
            splitter=ForwardPanelSplit(n_splits=1, min_train_periods=3),
            entity=entity,
            time=time,
            outcome="binary",
            prediction_target="new_entity",
        )


class FixedSplitter:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def split(self, X, y=None):
        del X, y
        yield self.train, self.test


def test_known_entity_future_rejects_training_observations_after_test_rows():
    X, y, entity, time = _panel_data(n_entities=2, n_periods=4)
    train = np.array([0, 2, 3, 4, 6, 7])
    test = np.array([1, 5])

    with pytest.raises(ValueError, match="all training observations to precede"):
        cross_validate(
            ToyPanelModel,
            X,
            y,
            splitter=FixedSplitter(train, test),
            entity=entity,
            time=time,
            outcome="binary",
            prediction_target="known_entity_future",
        )


def test_auto_known_entity_target_requires_time_metadata():
    X, y, entity, _ = _panel_data(n_entities=2, n_periods=4)
    train = np.array([0, 1, 2, 4, 5, 6])
    test = np.array([3, 7])

    with pytest.raises(ValueError, match="without time values"):
        cross_validate(
            ToyPanelModel,
            X,
            y,
            splitter=FixedSplitter(train, test),
            entity=entity,
            outcome="binary",
        )


@dataclass
class ToyDynamicPanelResult(ToyPanelResult):
    def posterior_random_effects(self):
        return pd.DataFrame({"posterior_mean": [0.0]})

    def posterior_predict_proba(self, X, *, entity, lagged_y, posterior):
        del entity, lagged_y, posterior
        probability = np.full(len(X), 0.6)
        return pd.DataFrame({0: 1.0 - probability, 1: probability})


class ToyDynamicPanelModel:
    def fit(self, X, y, *, entity, time):
        del X, y, entity, time
        return ToyDynamicPanelResult()


def test_dynamic_posterior_cv_accepts_only_last_training_outcome_as_test_lag():
    X, y, entity, time = _panel_data(n_entities=3, n_periods=6)
    lagged_y = np.empty(len(y), dtype=int)
    for label in np.unique(entity):
        rows = np.flatnonzero(entity == label)
        lagged_y[rows[0]] = 0
        lagged_y[rows[1:]] = np.asarray(y)[rows[:-1]]

    result = cross_validate(
        ToyDynamicPanelModel,
        X,
        y,
        splitter=ForwardPanelSplit(n_splits=1, min_train_periods=3),
        entity=entity,
        time=time,
        outcome="binary",
        prediction_target="known_entity_future",
        predict_context={"lagged_y": lagged_y},
    )
    assert result.successful_folds == 1

    bad_lags = 1 - lagged_y
    with pytest.raises(ValueError, match="last observed training outcome"):
        cross_validate(
            ToyDynamicPanelModel,
            X,
            y,
            splitter=ForwardPanelSplit(n_splits=1, min_train_periods=3),
            entity=entity,
            time=time,
            outcome="binary",
            prediction_target="known_entity_future",
            predict_context={"lagged_y": bad_lags},
        )


def test_dynamic_posterior_cv_rejects_embargoed_observed_lag():
    X, y, entity, time = _panel_data(n_entities=3, n_periods=7)
    lagged_y = np.roll(np.asarray(y), 1)
    with pytest.raises(ValueError, match="embargoed observed lags"):
        cross_validate(
            ToyDynamicPanelModel,
            X,
            y,
            splitter=ForwardPanelSplit(
                n_splits=1,
                min_train_periods=3,
                gap_periods=1,
            ),
            entity=entity,
            time=time,
            outcome="binary",
            prediction_target="known_entity_future",
            predict_context={"lagged_y": lagged_y},
        )


def test_conditional_prediction_target_requires_fold_aware_custom_predictor():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = np.tile([0, 1], 10)
    with pytest.raises(ValueError, match="custom predict callback"):
        cross_validate(
            ToyBinaryModel,
            X,
            y,
            splitter=StratifiedKFold(2),
            outcome="binary",
            prediction_target="conditional",
            predict_context={"random_effects": np.zeros(len(y))},
        )

    with pytest.raises(ValueError, match="cannot use conditional random_effects"):
        cross_validate(
            ToyBinaryModel,
            X,
            y,
            splitter=StratifiedKFold(2),
            outcome="binary",
            prediction_target="pooled",
            predict_context={"random_effects": np.zeros(len(y))},
        )


def test_user_context_typo_is_not_silently_discarded():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = np.tile([0, 1], 10)
    with pytest.raises(TypeError, match="misspelled"):
        cross_validate(
            ToyBinaryModel,
            X,
            y,
            splitter=StratifiedKFold(2),
            outcome="binary",
            fit_context={"misspelled": np.ones(len(y))},
        )


def test_boolean_split_masks_work_and_duplicate_indices_are_rejected():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = np.tile([0, 1], 10)
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[::2] = True
    result = cross_validate(
        ToyBinaryModel,
        X,
        y,
        splitter=FixedSplitter(~test_mask, test_mask),
        outcome="binary",
    )
    assert result.folds[0].test_index.tolist() == np.flatnonzero(test_mask).tolist()

    with pytest.raises(ValueError, match="must be unique"):
        cross_validate(
            ToyBinaryModel,
            X,
            y,
            splitter=FixedSplitter([0, 0, 1, 2], [3, 4]),
            outcome="binary",
        )


def test_quantile_is_inherited_from_result_for_automatic_scoring():
    @dataclass
    class CensoredQuantileRegressionResult:
        quantile: float = 0.25
        converged: bool = True
        inference_valid: bool = True

        def predict(self, X):
            return np.zeros(len(X))

    class QuantileModel:
        def __init__(self, quantile=0.25):
            self.quantile = quantile

        def fit(self, X, y):
            del X, y
            return CensoredQuantileRegressionResult(quantile=self.quantile)

    X = np.arange(24, dtype=float)[:, None]
    y = np.linspace(-1.0, 1.0, 24)
    result = cross_validate(
        QuantileModel,
        X,
        y,
        splitter=KFold(3),
        outcome="auto",
    )
    assert result.outcome == "quantile"
    assert "check_loss" in result.fold_frame()

    with pytest.raises(ValueError, match="must match the fitted"):
        cross_validate(
            QuantileModel,
            X,
            y,
            splitter=KFold(3),
            outcome="auto",
            score_context={"quantile": 0.75},
        )

    with pytest.raises(ValueError, match="different quantiles"):
        compare_models(
            {
                "q25": lambda: QuantileModel(0.25),
                "q75": lambda: QuantileModel(0.75),
            },
            X,
            y,
            splitter=KFold(3),
            outcome="auto",
        )


def test_interval_regression_default_cv_is_rejected_as_unidentified():
    @dataclass
    class IntervalRegressionResult:
        converged: bool = True
        inference_valid: bool = True

        def predict(self, X):
            return np.zeros(len(X))

    class IntervalModel:
        def fit(self, X, lower, *, upper):
            del X, lower, upper
            return IntervalRegressionResult()

    X = np.arange(20, dtype=float)[:, None]
    lower = np.linspace(0.0, 1.0, 20)
    upper = lower + 1.0
    with pytest.raises(ValueError, match="no observed point target"):
        cross_validate(
            IntervalModel,
            X,
            lower,
            splitter=KFold(2),
            outcome="continuous",
            fit_context={"upper": upper},
        )


def test_partial_proportional_odds_result_infers_ordinal_scoring():
    class PartialProportionalOddsResult:
        pass

    assert infer_outcome(PartialProportionalOddsResult()) == "ordinal"


def test_model_comparison_materializes_one_shared_split_design():
    class CountingSplitter:
        def __init__(self):
            self.calls = 0

        def split(self, X, y):
            del y
            self.calls += 1
            indices = np.arange(len(X))
            yield indices[10:], indices[:10]
            yield indices[:10], indices[10:]

    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = np.tile([0, 1], 10)
    splitter = CountingSplitter()
    comparison = compare_models(
        {"a": ToyBinaryModel, "b": ToyBinaryModel},
        X,
        y,
        splitter=splitter,
        outcome="binary",
    )

    assert splitter.calls == 1
    assert np.array_equal(
        comparison.cv_results["a"].folds[0].test_index,
        comparison.cv_results["b"].folds[0].test_index,
    )
    assert comparison.to_markdown().startswith("| model | rank |")


def test_shared_comparison_split_preserves_forward_time_step_metadata():
    X, y, entity, time = _panel_data(n_entities=3, n_periods=6)
    time = time * 2
    lagged_y = np.empty(len(y), dtype=int)
    for label in np.unique(entity):
        rows = np.flatnonzero(entity == label)
        lagged_y[rows[0]] = 0
        lagged_y[rows[1:]] = np.asarray(y)[rows[:-1]]

    comparison = compare_models(
        {"a": ToyDynamicPanelModel, "b": ToyDynamicPanelModel},
        X,
        y,
        splitter=ForwardPanelSplit(
            n_splits=1,
            min_train_periods=3,
            time_step=2,
        ),
        entity=entity,
        time=time,
        outcome="binary",
        prediction_target="known_entity_future",
        predict_context={"lagged_y": lagged_y},
    )

    assert comparison.best_model in {"a", "b"}
    assert comparison.table["eligible"].all()


def test_comparison_does_not_rank_an_incomplete_primary_metric():
    class PartlyOneClassSplitter:
        def split(self, X, y):
            del y
            rows = np.arange(len(X))
            for test in (np.arange(5), np.array([5, 15])):
                yield np.setdiff1d(rows, test), test

    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    comparison = compare_models(
        {"toy": ToyBinaryModel},
        X,
        y,
        splitter=PartlyOneClassSplitter(),
        outcome="binary",
        primary_metric="roc_auc",
    )

    assert comparison.best_model is None
    assert comparison.table.loc[0, "eligible"]
    assert not comparison.table.loc[0, "primary_metric_complete"]
    assert pd.isna(comparison.table.loc[0, "rank"])


def test_eligibility_requires_explicit_true_econometric_diagnostics():
    class MissingDiagnostics:
        pass

    missing = result_eligibility(MissingDiagnostics())
    assert not missing.eligible
    assert "converged diagnostic is unavailable" in missing.reasons
    assert "inference_valid diagnostic is unavailable" in missing.reasons

    invalid = result_eligibility(
        type(
            "Result",
            (),
            {"converged": np.bool_(False), "inference_valid": np.bool_(True)},
        )()
    )
    assert not invalid.eligible
    assert "optimizer did not converge" in invalid.reasons
