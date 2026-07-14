"""Dependency-light data splitters for limited-dependent-variable validation.

The splitters in this module intentionally return integer row positions rather
than sliced data.  This keeps them compatible with NumPy and pandas inputs
without requiring scikit-learn.
"""

from __future__ import annotations

from collections.abc import Iterator
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

IndexSplit = tuple[np.ndarray, np.ndarray]

__all__ = [
    "EntityHoldoutSplit",
    "ForwardPanelSplit",
    "GroupKFold",
    "GroupPanelSplit",
    "KFold",
    "StratifiedGroupKFold",
    "StratifiedKFold",
]


def _validate_integer(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return int(value)


def _as_1d(values: Any, *, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required.")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    try:
        missing = np.asarray(pd.isna(array), dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain scalar values.") from exc
    if missing.any():
        raise ValueError(f"{name} must not contain missing values.")
    return array


def _validate_x_length(X: Any, n_samples: int) -> None:
    if X is None:
        return
    try:
        x_length = len(X)
    except TypeError as exc:
        raise ValueError("X must be sized when supplied.") from exc
    if x_length != n_samples:
        raise ValueError("X and the splitting metadata must have the same length.")


def _factorize(values: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        codes, levels = pd.factorize(values, sort=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain hashable scalar values.") from exc
    if np.any(codes < 0):
        raise ValueError(f"{name} must not contain missing values.")
    return np.asarray(codes, dtype=np.int64), np.asarray(levels)


class KFold:
    """Dependency-light K-fold splitting for independent observations.

    This splitter is appropriate only when rows are exchangeable.  Panel,
    repeated-measures, duration-spell, and grouped-choice data should instead
    use complete-group or forward-time splitting.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = _validate_integer("n_splits", n_splits, minimum=2)
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None.")
        self.shuffle = shuffle
        self.random_state = None if random_state is None else int(random_state)

    def get_n_splits(self, *_args: Any, **_kwargs: Any) -> int:
        """Return the configured number of folds."""
        return self.n_splits

    def split(self, X: Any, y: Any = None) -> Iterator[IndexSplit]:
        """Yield positional train/test indices, optionally ignoring ``y``."""
        try:
            n_samples = len(X)
        except TypeError as exc:
            raise ValueError("X must be sized.") from exc
        if n_samples < self.n_splits:
            raise ValueError("The number of observations must be at least n_splits.")
        if y is not None:
            try:
                y_length = len(y)
            except TypeError as exc:
                raise ValueError("y must be sized when supplied.") from exc
            if y_length != n_samples:
                raise ValueError("X and y must have the same length.")

        indices = np.arange(n_samples, dtype=np.int64)
        if self.shuffle:
            indices = np.random.default_rng(self.random_state).permutation(indices)
        for test_indices in np.array_split(indices, self.n_splits):
            test_indices = np.sort(np.asarray(test_indices, dtype=np.int64))
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False
            yield np.flatnonzero(train_mask).astype(np.int64), test_indices


class StratifiedKFold:
    """Deterministic K-fold splits that preserve categorical outcome shares.

    Fold allocation follows the standard round-robin class-count design: each
    class differs by at most one observation across folds and total test-fold
    sizes also differ by at most one.  Categories with fewer observations than
    ``n_splits`` are rejected because the package's probability metrics require
    every requested category to be represented in every test fold.

    Parameters
    ----------
    n_splits:
        Number of folds.  Every observed category must occur at least this many
        times so that each test fold contains every category.
    shuffle:
        Shuffle observations independently within each category before assigning
        folds.  Shuffling is repeatable for every call to :meth:`split`.
    random_state:
        Seed used when ``shuffle=True``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = _validate_integer("n_splits", n_splits, minimum=2)
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None.")
        self.shuffle = shuffle
        self.random_state = None if random_state is None else int(random_state)

    def get_n_splits(self, *_args: Any, **_kwargs: Any) -> int:
        """Return the configured number of folds."""
        return self.n_splits

    def split(self, X: Any = None, y: Any = None) -> Iterator[IndexSplit]:
        """Yield ``(train_indices, test_indices)`` pairs.

        ``split(y)`` is accepted as a compact form.  ``split(X, y)`` and
        ``split(X=X, y=y)`` provide a scikit-learn-like calling convention.
        """
        direct_target = y is None
        target = X if direct_target else y
        target_array = _as_1d(target, name="y")
        if not direct_target:
            _validate_x_length(X, target_array.size)

        codes, categories = _factorize(target_array, name="y")
        counts = np.bincount(codes, minlength=categories.size)
        too_small = np.flatnonzero(counts < self.n_splits)
        if too_small.size:
            labels = categories[too_small].tolist()
            raise ValueError(
                "Every category must contain at least n_splits observations; "
                f"insufficient categories: {labels!r}."
            )

        # Distribute the sorted class-code sequence round-robin, then use those
        # per-fold class counts to assign observations within each category.
        # Independently array-splitting every category would put every class
        # remainder in the first folds and can leave total fold sizes badly
        # imbalanced even though each class is individually balanced.
        sorted_codes = np.sort(codes)
        allocation = np.asarray(
            [
                np.bincount(sorted_codes[fold :: self.n_splits], minlength=categories.size)
                for fold in range(self.n_splits)
            ],
            dtype=np.int64,
        )
        rng = np.random.default_rng(self.random_state)
        fold_assignment = np.empty(target_array.size, dtype=np.int64)
        for category_code in range(categories.size):
            category_folds = np.repeat(
                np.arange(self.n_splits, dtype=np.int64),
                allocation[:, category_code],
            )
            if self.shuffle:
                category_folds = rng.permutation(category_folds)
            fold_assignment[codes == category_code] = category_folds

        all_indices = np.arange(target_array.size, dtype=np.int64)
        for fold in range(self.n_splits):
            test_indices = np.flatnonzero(fold_assignment == fold).astype(
                np.int64, copy=False
            )
            train_mask = np.ones(target_array.size, dtype=bool)
            train_mask[test_indices] = False
            yield all_indices[train_mask], test_indices


class StratifiedGroupKFold:
    """Stratified K-fold splitting while keeping complete groups together.

    A deterministic greedy assignment targets class and observation balance.
    When its local repair cannot cover every class in every fold, an exact
    mixed-integer feasibility fallback searches for a minimum-move assignment.
    Every category must occur in at least ``n_splits`` distinct groups.  A
    structurally infeasible or solver-unresolved design is rejected rather than
    silently producing a one-class probability-scoring fold.

    ``shuffle=True`` randomizes equal-priority group ordering reproducibly.  It
    does not promise index identity with scikit-learn, whose tie-breaking and
    random-number implementation may differ.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = _validate_integer("n_splits", n_splits, minimum=2)
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None.")
        self.shuffle = shuffle
        self.random_state = None if random_state is None else int(random_state)

    def get_n_splits(self, *_args: Any, **_kwargs: Any) -> int:
        """Return the configured number of folds."""
        return self.n_splits

    def _exact_coverage_repair(
        self,
        group_class_counts: np.ndarray,
        initial_assignment: np.ndarray,
    ) -> np.ndarray | None:
        """Find a minimum-move feasible assignment when greedy repair stalls."""
        n_groups, n_classes = group_class_counts.shape
        n_variables = n_groups * self.n_splits
        n_constraints = n_groups + self.n_splits * n_classes
        matrix = lil_matrix((n_constraints, n_variables), dtype=float)
        lower = np.empty(n_constraints, dtype=float)
        upper = np.empty(n_constraints, dtype=float)

        for group in range(n_groups):
            start = group * self.n_splits
            matrix[group, start : start + self.n_splits] = 1.0
            lower[group] = 1.0
            upper[group] = 1.0

        present = group_class_counts > 0
        row = n_groups
        for fold in range(self.n_splits):
            for category in range(n_classes):
                groups = np.flatnonzero(present[:, category])
                matrix[row, groups * self.n_splits + fold] = 1.0
                lower[row] = 1.0
                upper[row] = np.inf
                row += 1

        # Keep as much of the balanced greedy assignment as possible while
        # enforcing the exact every-class/every-fold coverage constraints.
        objective = np.ones(n_variables, dtype=float)
        objective[
            np.arange(n_groups, dtype=np.int64) * self.n_splits
            + initial_assignment
        ] = 0.0
        solution = milp(
            c=objective,
            integrality=np.ones(n_variables, dtype=np.int8),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"time_limit": 10.0},
        )
        if not solution.success or solution.x is None:
            return None
        assignment_matrix = np.asarray(solution.x).reshape(n_groups, self.n_splits)
        assignment = np.argmax(assignment_matrix, axis=1).astype(np.int64)
        coverage = np.zeros((self.n_splits, n_classes), dtype=np.int64)
        for group, fold in enumerate(assignment):
            coverage[fold] += group_class_counts[group]
        return assignment if np.all(coverage > 0) else None

    @staticmethod
    def _imbalance_score(
        fold_class_counts: np.ndarray,
        fold_sizes: np.ndarray,
        class_totals: np.ndarray,
    ) -> float:
        proportions = fold_class_counts / class_totals[None, :]
        class_imbalance = float(np.mean(np.std(proportions, axis=0)))
        size_scale = max(float(np.sum(fold_sizes)), 1.0)
        size_imbalance = float(np.std(fold_sizes / size_scale))
        return class_imbalance + 0.05 * size_imbalance

    def split(
        self,
        X: Any = None,
        y: Any = None,
        *,
        groups: Any = None,
    ) -> Iterator[IndexSplit]:
        """Yield complete-group train/test indices with approximate stratification."""
        direct_target = y is None
        target = X if direct_target else y
        target_array = _as_1d(target, name="y")
        if not direct_target:
            _validate_x_length(X, target_array.size)
        group_array = _as_1d(groups, name="groups")
        if group_array.size != target_array.size:
            raise ValueError("y and groups must have the same length.")

        class_codes, classes = _factorize(target_array, name="y")
        group_codes, group_labels = _factorize(group_array, name="groups")
        n_groups = group_labels.size
        if n_groups < self.n_splits:
            raise ValueError(
                "The number of unique groups must be greater than or equal to n_splits."
            )

        n_classes = classes.size
        group_class_counts = np.zeros((n_groups, n_classes), dtype=np.int64)
        np.add.at(group_class_counts, (group_codes, class_codes), 1)
        groups_per_class = np.sum(group_class_counts > 0, axis=0)
        insufficient = np.flatnonzero(groups_per_class < self.n_splits)
        if insufficient.size:
            labels = classes[insufficient].tolist()
            raise ValueError(
                "Every category must occur in at least n_splits distinct groups; "
                f"insufficient categories: {labels!r}."
            )

        group_sizes = group_class_counts.sum(axis=1)
        priority = np.std(group_class_counts, axis=1)
        order = np.arange(n_groups, dtype=np.int64)
        if self.shuffle:
            order = np.random.default_rng(self.random_state).permutation(order)
        order = order[np.argsort(-group_sizes[order], kind="stable")]
        order = order[np.argsort(-priority[order], kind="stable")]

        class_totals = group_class_counts.sum(axis=0).astype(float)
        fold_class_counts = np.zeros((self.n_splits, n_classes), dtype=np.int64)
        fold_sizes = np.zeros(self.n_splits, dtype=np.int64)
        group_fold = np.full(n_groups, -1, dtype=np.int64)

        for group_code in order:
            counts = group_class_counts[group_code]
            size = group_sizes[group_code]
            candidates = []
            for fold in range(self.n_splits):
                trial_counts = fold_class_counts.copy()
                trial_sizes = fold_sizes.copy()
                trial_counts[fold] += counts
                trial_sizes[fold] += size
                candidates.append(
                    (
                        self._imbalance_score(trial_counts, trial_sizes, class_totals),
                        int(fold_sizes[fold]),
                        fold,
                    )
                )
            selected = min(candidates)[2]
            group_fold[group_code] = selected
            fold_class_counts[selected] += counts
            fold_sizes[selected] += size

        # Repair missing class/fold cells by moving a donor group only when the
        # donor fold retains every class represented by that group.
        for _ in range(n_classes * self.n_splits):
            missing = np.argwhere(fold_class_counts == 0)
            if not missing.size:
                break
            repaired = False
            for target_fold, class_code in missing:
                moves = []
                for group_code in np.flatnonzero(
                    (group_fold != target_fold)
                    & (group_class_counts[:, class_code] > 0)
                ):
                    donor = int(group_fold[group_code])
                    counts = group_class_counts[group_code]
                    if np.any((counts > 0) & (fold_class_counts[donor] - counts <= 0)):
                        continue
                    trial_counts = fold_class_counts.copy()
                    trial_sizes = fold_sizes.copy()
                    trial_counts[donor] -= counts
                    trial_counts[target_fold] += counts
                    trial_sizes[donor] -= group_sizes[group_code]
                    trial_sizes[target_fold] += group_sizes[group_code]
                    moves.append(
                        (
                            self._imbalance_score(
                                trial_counts, trial_sizes, class_totals
                            ),
                            int(group_sizes[group_code]),
                            int(group_code),
                            donor,
                        )
                    )
                if not moves:
                    continue
                _, _, group_code, donor = min(moves)
                counts = group_class_counts[group_code]
                fold_class_counts[donor] -= counts
                fold_class_counts[target_fold] += counts
                fold_sizes[donor] -= group_sizes[group_code]
                fold_sizes[target_fold] += group_sizes[group_code]
                group_fold[group_code] = target_fold
                repaired = True
            if not repaired:
                break

        if np.any(fold_class_counts == 0):
            exact_assignment = self._exact_coverage_repair(
                group_class_counts,
                group_fold,
            )
            if exact_assignment is None:
                missing = [
                    (int(fold), classes[int(category)])
                    for fold, category in np.argwhere(fold_class_counts == 0)
                ]
                raise ValueError(
                    "Unable to construct stratified complete-group folds with every "
                    "category represented after greedy and exact feasibility searches; "
                    f"unresolved fold/category cells: {missing!r}."
                )
            group_fold = exact_assignment

        all_indices = np.arange(target_array.size, dtype=np.int64)
        for fold in range(self.n_splits):
            test_mask = group_fold[group_codes] == fold
            yield all_indices[~test_mask], all_indices[test_mask]


class EntityHoldoutSplit:
    """K-fold holdout of complete entities.

    Entities are assigned greedily to folds by observation count.  The algorithm
    is deterministic and keeps every entity entirely in either training or test
    data for a given fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = _validate_integer("n_splits", n_splits, minimum=2)
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None.")
        self.shuffle = shuffle
        self.random_state = None if random_state is None else int(random_state)

    def get_n_splits(self, *_args: Any, **_kwargs: Any) -> int:
        """Return the configured number of folds."""
        return self.n_splits

    def split(
        self,
        X: Any = None,
        *,
        entity: Any = None,
        groups: Any = None,
    ) -> Iterator[IndexSplit]:
        """Yield complete-entity train/test splits.

        Supply entity labels as ``entity=`` or ``groups=``.  ``split(entity)``
        is also accepted when no feature matrix is needed.
        """
        if entity is not None and groups is not None:
            raise ValueError("Supply only one of entity or groups.")
        metadata_is_explicit = entity is not None or groups is not None
        entity_values = entity if entity is not None else groups
        if entity_values is None:
            entity_values = X
        entity_array = _as_1d(entity_values, name="entity")
        if metadata_is_explicit:
            _validate_x_length(X, entity_array.size)

        codes, entities = _factorize(entity_array, name="entity")
        n_entities = entities.size
        if n_entities < self.n_splits:
            raise ValueError(
                "The number of unique entities must be greater than or equal to n_splits."
            )

        counts = np.bincount(codes, minlength=n_entities)
        order = np.arange(n_entities, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            order = rng.permutation(order)
        order = order[np.argsort(-counts[order], kind="stable")]

        fold_loads = np.zeros(self.n_splits, dtype=np.int64)
        entity_fold = np.empty(n_entities, dtype=np.int64)
        for entity_code in order:
            fold = int(np.argmin(fold_loads))
            entity_fold[entity_code] = fold
            fold_loads[fold] += counts[entity_code]

        all_indices = np.arange(entity_array.size, dtype=np.int64)
        for fold in range(self.n_splits):
            test_mask = entity_fold[codes] == fold
            yield all_indices[~test_mask], all_indices[test_mask]


class GroupKFold(EntityHoldoutSplit):
    """Alias spelling for :class:`EntityHoldoutSplit`."""


class GroupPanelSplit(EntityHoldoutSplit):
    """Panel-oriented alias for :class:`EntityHoldoutSplit`."""


class ForwardPanelSplit:
    """Expanding, time-ordered panel splits with common calendar cutoffs.

    Each fold trains on earlier periods and tests on a later common time window.
    ``gap_periods`` calendar periods between the training and test windows are
    deliberately omitted.  The final fold always ends at the latest observed
    period, while earlier folds move backward by ``test_periods``.

    Dynamic limited-dependent-variable models need an uninterrupted outcome
    history to construct state variables and posterior predictions.  This
    splitter therefore rejects duplicate, out-of-order, or noncontiguous rows
    within *any* entity.  Consecutive times must differ by exactly ``time_step``;
    it never silently stitches observations across a gap.  Entities without at
    least ``min_train_periods`` observations before a fold or without its complete
    test window are omitted from both sides of that fold.

    Parameters
    ----------
    n_splits:
        Number of expanding-window folds.
    min_train_periods:
        Minimum number of observations required for each test entity in every
        fold's training window.
    test_periods:
        Number of consecutive calendar periods in each test window.
    gap_periods:
        Number of consecutive calendar periods deliberately dropped between the
        training and test windows.
    time_step:
        Exact expected difference between adjacent time values.  Use, for
        example, ``1`` for annual integer indices or ``pd.Timedelta(days=1)`` for
        daily timestamps.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        min_train_periods: int = 2,
        test_periods: int = 1,
        gap_periods: int = 0,
        time_step: Any = 1,
    ) -> None:
        self.n_splits = _validate_integer("n_splits", n_splits, minimum=1)
        self.min_train_periods = _validate_integer(
            "min_train_periods", min_train_periods, minimum=1
        )
        self.test_periods = _validate_integer("test_periods", test_periods, minimum=1)
        self.gap_periods = _validate_integer("gap_periods", gap_periods, minimum=0)
        if time_step is None:
            raise ValueError("time_step must be supplied explicitly.")
        try:
            zero_step = time_step - time_step
            is_positive = bool(time_step > zero_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_step must be a positive scalar difference.") from exc
        if not is_positive:
            raise ValueError("time_step must be positive.")
        self.time_step = time_step

    def get_n_splits(self, *_args: Any, **_kwargs: Any) -> int:
        """Return the configured number of folds."""
        return self.n_splits

    def _is_exact_step(self, earlier: Any, later: Any) -> bool:
        try:
            return bool(later - earlier == self.time_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "time values must support subtraction compatible with time_step."
            ) from exc

    def split(
        self,
        X: Any = None,
        time: Any = None,
        *,
        entity: Any = None,
    ) -> Iterator[IndexSplit]:
        """Yield expanding-window ``(train_indices, test_indices)`` pairs.

        The preferred form is ``split(X, entity=entity, time=time)``.  For direct
        metadata use, ``split(entity, time)`` is also accepted.
        """
        metadata_is_explicit = entity is not None
        entity_values = X if entity is None else entity
        entity_array = _as_1d(entity_values, name="entity")
        time_array = _as_1d(time, name="time")
        if entity_array.size != time_array.size:
            raise ValueError("entity and time must have the same length.")
        if metadata_is_explicit:
            _validate_x_length(X, entity_array.size)

        entity_codes, entities = _factorize(entity_array, name="entity")
        for entity_code, entity_label in enumerate(entities):
            entity_indices = np.flatnonzero(entity_codes == entity_code)
            entity_times = time_array[entity_indices]
            for earlier, later in zip(entity_times[:-1], entity_times[1:], strict=True):
                if not self._is_exact_step(earlier, later):
                    raise ValueError(
                        "Each entity must be in strictly increasing, contiguous input order "
                        f"with exact time_step; entity {entity_label!r} violates this contract."
                    )

        try:
            unique_times = np.asarray(pd.Index(time_array).unique().sort_values())
        except (TypeError, ValueError) as exc:
            raise ValueError("time values must be mutually orderable scalars.") from exc
        for earlier, later in zip(unique_times[:-1], unique_times[1:], strict=True):
            if not self._is_exact_step(earlier, later):
                raise ValueError(
                    "The panel calendar must be contiguous with the exact time_step."
                )

        n_periods = unique_times.size
        initial_train_periods = (
            n_periods - self.gap_periods - self.n_splits * self.test_periods
        )
        if initial_train_periods < self.min_train_periods:
            required = (
                self.min_train_periods
                + self.gap_periods
                + self.n_splits * self.test_periods
            )
            raise ValueError(
                "The panel has too few calendar periods for the requested splits; "
                f"at least {required} are required."
            )

        time_positions = pd.Index(unique_times).get_indexer(time_array)
        if np.any(time_positions < 0):  # pragma: no cover - defensive pandas guard
            raise RuntimeError("Failed to map panel time values to calendar positions.")

        for fold in range(self.n_splits):
            train_stop = initial_train_periods + fold * self.test_periods
            test_start = train_stop + self.gap_periods
            test_stop = test_start + self.test_periods

            before_cutoff = time_positions < train_stop
            in_test_window = (time_positions >= test_start) & (time_positions < test_stop)
            eligible_entities: list[int] = []
            expected_test_positions = np.arange(test_start, test_stop)
            for entity_code in range(entities.size):
                is_entity = entity_codes == entity_code
                train_count = int(np.count_nonzero(is_entity & before_cutoff))
                observed_test_positions = np.sort(time_positions[is_entity & in_test_window])
                if train_count >= self.min_train_periods and np.array_equal(
                    observed_test_positions, expected_test_positions
                ):
                    eligible_entities.append(entity_code)

            if not eligible_entities:
                raise ValueError(
                    f"Fold {fold} has no entity with the required training history "
                    "and complete test window."
                )
            eligible_mask = np.isin(entity_codes, np.asarray(eligible_entities))
            train_indices = np.flatnonzero(before_cutoff & eligible_mask).astype(
                np.int64, copy=False
            )
            test_indices = np.flatnonzero(in_test_window & eligible_mask).astype(
                np.int64, copy=False
            )
            yield train_indices, test_indices
