"""Tests for uncertainty-aware small-sample validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from limiteddepkit import BinaryLogit
from limiteddepkit.ml.metrics import binary_log_loss
from limiteddepkit.ml.uncertainty import (
    RepeatedGroupKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    one_standard_error_select,
    paired_bootstrap_interval,
    paired_fold_score_differences,
    weighted_fold_summary,
    weighted_score_summary,
)
from limiteddepkit.ml.validation import cross_validate


def test_paired_fold_differences_have_candidate_oriented_sign():
    candidate = [0.20, 0.25, 0.22]
    reference = [0.24, 0.27, 0.21]

    minimized = paired_fold_score_differences(
        candidate,
        reference,
        fold_labels=["r1f1", "r1f2", "r1f3"],
    )
    np.testing.assert_allclose(
        minimized["candidate_advantage"],
        [0.04, 0.02, -0.01],
    )
    assert minimized["fold"].tolist() == ["r1f1", "r1f2", "r1f3"]

    maximized = paired_fold_score_differences(
        candidate,
        reference,
        higher_is_better=True,
    )
    np.testing.assert_allclose(
        maximized["candidate_advantage"],
        [-0.04, -0.02, 0.01],
    )


def test_paired_fold_differences_reject_misalignment_and_nonfinite_values():
    with pytest.raises(ValueError, match="same length"):
        paired_fold_score_differences([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="finite"):
        paired_fold_score_differences([0.1, np.nan], [0.1, 0.2])
    with pytest.raises(ValueError, match="one label"):
        paired_fold_score_differences([0.1, 0.2], [0.2, 0.3], fold_labels=[1])


def test_paired_bootstrap_on_losses_is_deterministic_and_cluster_aware():
    candidate_loss = np.array([0.0, 0.0, 2.0, 2.0, 2.0])
    reference_loss = np.ones(5)
    entity = np.array(["a", "a", "b", "b", "b"])

    first = paired_bootstrap_interval(
        candidate_loss,
        reference_loss,
        clusters=entity,
        n_resamples=500,
        random_state=9201,
    )
    second = paired_bootstrap_interval(
        candidate_loss,
        reference_loss,
        clusters=entity,
        n_resamples=500,
        random_state=9201,
    )

    assert first == second
    assert first.estimate == pytest.approx(-0.2)
    assert first.clustered
    assert first.n_observations == 5
    assert first.n_sampling_units == 2
    assert first.lower <= first.estimate <= first.upper
    assert first.as_series()["n_resamples"] == 500


def test_paired_bootstrap_scorer_path_uses_paired_predictions():
    y = np.tile([0, 1], 40)
    candidate_probability = np.tile([0.05, 0.95], 40)
    reference_probability = np.tile([0.40, 0.60], 40)

    result = paired_bootstrap_interval(
        candidate_probability,
        reference_probability,
        y_true=y,
        scorer=binary_log_loss,
        n_resamples=300,
        random_state=711,
    )

    expected = binary_log_loss(y, reference_probability) - binary_log_loss(
        y, candidate_probability
    )
    assert result.estimate == pytest.approx(expected)
    assert result.lower == pytest.approx(expected)
    assert result.upper == pytest.approx(expected)
    assert result.excludes_zero
    assert not result.clustered


def test_paired_bootstrap_identical_values_has_zero_interval():
    losses = np.array([0.1, 0.4, 0.2, 0.3])
    result = paired_bootstrap_interval(
        losses,
        losses,
        n_resamples=100,
        random_state=90,
    )

    assert result.estimate == 0.0
    assert result.standard_error == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0
    assert not result.excludes_zero


def test_paired_bootstrap_validates_sampling_contract():
    with pytest.raises(ValueError, match="same observations"):
        paired_bootstrap_interval([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="at least two distinct clusters"):
        paired_bootstrap_interval([0.1, 0.2], [0.2, 0.3], clusters=["a", "a"])
    with pytest.raises(ValueError, match="required when scorer"):
        paired_bootstrap_interval([0.1, 0.2], [0.2, 0.3], scorer=np.mean)


def _assert_repeat_partition(items, nobs, n_splits, n_repeats):
    assert len(items) == n_splits * n_repeats
    for item in items:
        assert np.intersect1d(item.train_index, item.test_index).size == 0
        assert sorted(np.concatenate([item.train_index, item.test_index])) == list(
            range(nobs)
        )
    for repeat in range(1, n_repeats + 1):
        tests = [item.test_index for item in items if item.repeat == repeat]
        np.testing.assert_array_equal(np.sort(np.concatenate(tests)), np.arange(nobs))


def test_repeated_kfold_is_reproducible_and_exposes_repeat_metadata():
    X = np.zeros((18, 2))
    splitter = RepeatedKFold(3, n_repeats=3, random_state=188)

    first = list(splitter.split_with_repeats(X))
    second = list(splitter.split_with_repeats(X))

    _assert_repeat_partition(first, 18, 3, 3)
    assert splitter.get_n_splits() == 9
    assert [(item.repeat, item.fold) for item in first] == [
        (repeat, fold) for repeat in range(1, 4) for fold in range(1, 4)
    ]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.train_index, right.train_index)
        np.testing.assert_array_equal(left.test_index, right.test_index)
    assert len({tuple(first[offset].test_index) for offset in (0, 3, 6)}) > 1


def test_repeated_stratified_kfold_retains_every_category():
    y = np.repeat(["low", "middle", "high"], 12)
    X = np.zeros((len(y), 2))
    splitter = RepeatedStratifiedKFold(3, n_repeats=2, random_state=991)
    items = list(splitter.split_with_repeats(X, y))

    _assert_repeat_partition(items, len(y), 3, 2)
    for item in items:
        assert set(y[item.test_index]) == {"low", "middle", "high"}
    standard = list(splitter.split(X, y))
    assert len(standard) == 6


def test_repeated_stratified_splitter_integrates_with_cross_validate():
    X = pd.DataFrame({"const": np.ones(40)})
    y = np.tile([0, 1], 20)
    result = cross_validate(
        BinaryLogit,
        X,
        y,
        splitter=RepeatedStratifiedKFold(2, n_repeats=2, random_state=77),
        outcome="binary",
    )

    assert result.successful_folds == 4
    assert result.eligible_folds == 4
    predictions = result.out_of_fold_predictions()
    assert len(predictions) == 2 * len(y)
    assert (predictions["row_index"].value_counts() == 2).all()
    pooled = result.pooled_out_of_fold_predictions()
    assert len(pooled) == len(y)
    assert (pooled["prediction_count"] == 2).all()
    assert np.allclose(pooled["prediction_1"], 0.5)
    weighted = result.weighted_summary_frame()
    assert weighted.loc["log_loss", "weight_sum"] == 2 * len(y)

    fold_weights = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    unequally_pooled = result.pooled_out_of_fold_predictions(
        fold_weights=fold_weights
    ).set_index("row_position")
    expected_weight = predictions.assign(
        weight=predictions["fold"].map(fold_weights)
    ).groupby("row_position")["weight"].sum()
    pd.testing.assert_series_equal(
        unequally_pooled["weight_sum"],
        expected_weight,
        check_names=False,
    )
    assert np.allclose(unequally_pooled["prediction_1"], 0.5)

    invalid_weights = (
        {1: 1.0, 2: 1.0, 3: 1.0},
        {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
        {1: 1.0, 2: 1.0, 3: 1.0, 4.5: 1.0},
        {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
        {1: 1.0, 2: 1.0, 3: 1.0, 4: np.nan},
    )
    for weights in invalid_weights:
        with pytest.raises(ValueError, match="fold_weights"):
            result.pooled_out_of_fold_predictions(fold_weights=weights)


def test_repeated_group_kfold_never_splits_an_entity():
    entity = np.repeat(np.arange(12), 2)
    X = np.zeros((len(entity), 1))
    splitter = RepeatedGroupKFold(3, n_repeats=2, random_state=881)
    items = list(splitter.split_with_repeats(X, groups=entity))

    _assert_repeat_partition(items, len(entity), 3, 2)
    for item in items:
        assert set(entity[item.train_index]).isdisjoint(set(entity[item.test_index]))
    assert len(list(splitter.split(X, groups=entity))) == 6


def test_weighted_score_summary_reports_effective_fold_count():
    ordinary = weighted_score_summary([1.0, 2.0, 3.0])
    assert ordinary["mean"] == pytest.approx(2.0)
    assert ordinary["std"] == pytest.approx(1.0)
    assert ordinary["standard_error"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert ordinary["effective_n"] == pytest.approx(3.0)

    unequal = weighted_score_summary([1.0, 2.0, 3.0], [1.0, 1.0, 2.0])
    assert unequal["mean"] == pytest.approx(2.25)
    assert unequal["weight_sum"] == pytest.approx(4.0)
    assert unequal["effective_n"] == pytest.approx(16.0 / 6.0)


def test_weighted_fold_summary_uses_test_size_and_ignores_metadata():
    folds = pd.DataFrame(
        {
            "fold": [1, 2],
            "train_n": [40, 20],
            "test_n": [10, 30],
            "eligible": [True, True],
            "log_loss": [0.2, 0.4],
            "brier_score": [0.1, np.nan],
        }
    )

    summary = weighted_fold_summary(folds)

    assert list(summary.index) == ["log_loss", "brier_score"]
    assert summary.loc["log_loss", "mean"] == pytest.approx(0.35)
    assert summary.loc["log_loss", "weight_sum"] == pytest.approx(40.0)
    assert summary.loc["brier_score", "mean"] == pytest.approx(0.1)
    assert summary.loc["brier_score", "n"] == 1


def test_one_standard_error_selects_simplest_eligible_candidate():
    table = pd.DataFrame(
        {
            "model": ["flexible", "ordered", "invalid_shortcut", "intermediate"],
            "score": [0.20, 0.22, 0.18, 0.21],
            "se": [0.03, 0.02, 0.01, 0.02],
            "complexity": [4, 1, 0, 2],
            "eligible": [True, True, False, True],
        }
    )

    selection = one_standard_error_select(
        table,
        score_column="score",
        standard_error_column="se",
        complexity_column="complexity",
    )

    assert selection.best_model == "flexible"
    assert selection.cutoff == pytest.approx(0.23)
    assert selection.selected_model == "ordered"
    assert selection.selected_complexity == 1
    assert set(selection.candidate_models) == {"flexible", "ordered", "intermediate"}


def test_one_standard_error_supports_maximized_scores_and_validates_inputs():
    table = pd.DataFrame(
        {
            "model": ["large", "small", "outside"],
            "auc": [0.80, 0.76, 0.74],
            "se": [0.05, 0.02, 0.02],
            "complexity": [5, 1, 0],
            "eligible": [True, True, True],
        }
    )
    selection = one_standard_error_select(
        table,
        score_column="auc",
        standard_error_column="se",
        complexity_column="complexity",
        higher_is_better=True,
    )
    assert selection.best_model == "large"
    assert selection.cutoff == pytest.approx(0.75)
    assert selection.selected_model == "small"

    table.loc[1, "se"] = np.nan
    with pytest.raises(ValueError, match="finite scores"):
        one_standard_error_select(
            table,
            score_column="auc",
            standard_error_column="se",
            complexity_column="complexity",
            higher_is_better=True,
        )
