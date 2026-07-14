"""Tests for dependency-light pooled and panel splitters."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.ml.split import (
    EntityHoldoutSplit,
    ForwardPanelSplit,
    GroupKFold,
    GroupPanelSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


def test_kfold_covers_each_independent_observation_once():
    splitter = KFold(4, shuffle=True, random_state=12)
    splits = list(splitter.split(np.zeros((23, 2)), np.arange(23)))

    assert len(splits) == 4
    assert sorted(np.concatenate([test for _, test in splits]).tolist()) == list(range(23))
    assert all(not np.intersect1d(train, test).size for train, test in splits)
    assert [test.tolist() for _, test in splits] == [
        test.tolist()
        for _, test in KFold(4, shuffle=True, random_state=12).split(np.zeros(23))
    ]


def test_kfold_validates_lengths_and_number_of_rows():
    with pytest.raises(ValueError, match="at least n_splits"):
        list(KFold(4).split(np.zeros(3)))
    with pytest.raises(ValueError, match="same length"):
        list(KFold(2).split(np.zeros(4), np.zeros(3)))


def _assert_partition(train, test, nobs):
    assert train.dtype == np.int64
    assert test.dtype == np.int64
    assert np.intersect1d(train, test).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(nobs))


def test_stratified_kfold_is_deterministic_and_balanced():
    y = np.array(["low"] * 7 + ["middle"] * 8 + ["high"] * 9)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=905)

    first = list(splitter.split(y))
    second = list(splitter.split(X=np.zeros((y.size, 2)), y=y))

    assert splitter.get_n_splits() == 4
    for (train, test), (train_again, test_again) in zip(first, second, strict=True):
        _assert_partition(train, test, y.size)
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        assert set(y[test]) == {"low", "middle", "high"}
    fold_sizes = [test.size for _, test in first]
    assert max(fold_sizes) - min(fold_sizes) <= 1


def test_stratified_kfold_balances_class_remainders_across_total_fold_sizes():
    y = np.repeat(["a", "b", "c"], [7, 5, 4])

    splits = list(StratifiedKFold(n_splits=4).split(y))

    assert [len(test) for _, test in splits] == [4, 4, 4, 4]
    for label in ("a", "b", "c"):
        per_fold = [int(np.count_nonzero(y[test] == label)) for _, test in splits]
        assert max(per_fold) - min(per_fold) <= 1


def test_stratified_kfold_rejects_sparse_categories_and_bad_metadata():
    with pytest.raises(ValueError, match="at least n_splits"):
        list(StratifiedKFold(n_splits=3).split(["a", "a", "b", "b", "b", "b"]))
    with pytest.raises(ValueError, match="missing"):
        list(StratifiedKFold(n_splits=2).split(["a", "a", None, "b"]))
    with pytest.raises(ValueError, match="same length"):
        list(StratifiedKFold(n_splits=2).split(np.zeros((5, 1)), [0, 0, 1, 1]))


def test_stratified_group_kfold_preserves_groups_and_all_categories():
    groups = np.repeat(np.arange(12), 4)
    y = np.tile(["low", "low", "middle", "high"], 12)
    X = np.zeros((len(y), 2))

    splits = list(
        StratifiedGroupKFold(4, shuffle=True, random_state=17).split(
            X, y, groups=groups
        )
    )

    assert len(splits) == 4
    seen_groups = []
    for train, test in splits:
        _assert_partition(train, test, len(y))
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        assert train_groups.isdisjoint(test_groups)
        assert set(y[test]) == {"low", "middle", "high"}
        seen_groups.extend(test_groups)
    assert sorted(seen_groups) == list(range(12))


def test_stratified_group_kfold_exact_fallback_recovers_feasible_coverage():
    group_class_counts = np.array(
        [
            [0, 0, 3],
            [1, 0, 2],
            [2, 3, 2],
            [0, 2, 3],
            [3, 1, 0],
        ]
    )
    groups = []
    outcomes = []
    for group, counts in enumerate(group_class_counts):
        for category, count in enumerate(counts):
            groups.extend([group] * int(count))
            outcomes.extend([category] * int(count))
    groups = np.asarray(groups)
    outcomes = np.asarray(outcomes)

    splits = list(StratifiedGroupKFold(3).split(outcomes, groups=groups))

    assert len(splits) == 3
    for train, test in splits:
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert set(outcomes[test]) == {0, 1, 2}


def test_stratified_group_kfold_rejects_structurally_impossible_design():
    groups = np.repeat(np.arange(6), 3)
    y = np.tile([0, 0, 1], 6)
    y[groups >= 2] = 0

    with pytest.raises(ValueError, match="distinct groups"):
        list(StratifiedGroupKFold(3).split(y, groups=groups))


@pytest.mark.parametrize("splitter_type", [EntityHoldoutSplit, GroupKFold, GroupPanelSplit])
def test_entity_holdout_never_splits_an_entity(splitter_type):
    entity = np.repeat(["a", "b", "c", "d", "e", "f"], [5, 1, 4, 2, 3, 2])
    splitter = splitter_type(n_splits=3)
    seen_test_entities = []

    for train, test in splitter.split(groups=entity):
        _assert_partition(train, test, entity.size)
        train_entities = set(entity[train])
        test_entities = set(entity[test])
        assert train_entities.isdisjoint(test_entities)
        seen_test_entities.extend(test_entities)

    assert sorted(seen_test_entities) == sorted(set(entity))


def test_entity_holdout_supports_feature_matrix_and_validates_groups():
    X = np.zeros((8, 2))
    entity = np.repeat(np.arange(4), 2)
    first = list(EntityHoldoutSplit(2, shuffle=True, random_state=44).split(X, entity=entity))
    second = list(EntityHoldoutSplit(2, shuffle=True, random_state=44).split(entity))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])

    with pytest.raises(ValueError, match="only one"):
        list(EntityHoldoutSplit(2).split(entity=entity, groups=entity))
    with pytest.raises(ValueError, match="unique entities"):
        list(EntityHoldoutSplit(5).split(entity))


def _balanced_panel(n_entities=3, n_periods=7):
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    return entity, time


def test_forward_panel_split_has_common_cutoffs_and_no_leakage():
    entity, time = _balanced_panel()
    splitter = ForwardPanelSplit(n_splits=3, min_train_periods=3, test_periods=1)
    splits = list(splitter.split(entity, time))

    assert splitter.get_n_splits() == 3
    assert [np.unique(time[test]).item() for _, test in splits] == [4, 5, 6]
    for train, test in splits:
        assert np.max(time[train]) < np.min(time[test])
        assert set(entity[train]) == set(entity[test]) == {0, 1, 2}
        for code in np.unique(entity):
            assert np.count_nonzero(entity[train] == code) >= 3
        assert np.all(np.diff(train) > 0)
        assert np.all(np.diff(test) > 0)


def test_forward_panel_split_drops_the_configured_gap():
    entity, time = _balanced_panel(n_entities=2, n_periods=8)
    splits = list(
        ForwardPanelSplit(
            n_splits=2,
            min_train_periods=3,
            test_periods=1,
            gap_periods=1,
        ).split(X=np.zeros((entity.size, 1)), entity=entity, time=time)
    )

    for train, test in splits:
        train_last = np.max(time[train])
        test_first = np.min(time[test])
        assert test_first - train_last == 2
        gap_time = train_last + 1
        assert not np.any(time[train] == gap_time)
        assert not np.any(time[test] == gap_time)


def test_forward_panel_split_supports_exact_datetime_steps():
    entity = np.repeat(["left", "right"], 5)
    time = np.tile(pd.date_range("2026-01-01", periods=5, freq="D"), 2)
    splitter = ForwardPanelSplit(
        n_splits=2,
        min_train_periods=2,
        time_step=pd.Timedelta(days=1),
    )

    splits = list(splitter.split(entity, time))

    assert len(splits) == 2
    assert pd.Timestamp(time[splits[-1][1][0]]) == pd.Timestamp("2026-01-05")


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([0, 1, 3, 4], "exact time_step"),
        ([0, 2, 1, 3], "strictly increasing"),
        ([0, 1, 1, 2], "strictly increasing"),
    ],
)
def test_forward_panel_split_rejects_noncontiguous_dynamic_history(time, message):
    entity = np.zeros(4, dtype=int)
    with pytest.raises(ValueError, match=message):
        list(
            ForwardPanelSplit(n_splits=1, min_train_periods=2).split(
                entity, np.asarray(time)
            )
        )


def test_forward_panel_split_validates_lengths_and_period_budget():
    with pytest.raises(ValueError, match="same length"):
        list(ForwardPanelSplit(n_splits=1).split([0, 0, 0], [0, 1]))

    entity, time = _balanced_panel(n_entities=2, n_periods=4)
    with pytest.raises(ValueError, match="too few calendar periods"):
        list(
            ForwardPanelSplit(
                n_splits=2,
                min_train_periods=3,
                gap_periods=1,
            ).split(entity, time)
        )
