"""Uncertainty-aware helpers for small-sample model validation.

The routines in this module are deliberately estimator-agnostic.  Positive
paired differences always mean that the candidate model performs better than
the reference model, regardless of whether the underlying score is minimized
or maximized.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .split import EntityHoldoutSplit, KFold, StratifiedKFold

__all__ = [
    "BootstrapDifference",
    "OneStandardErrorSelection",
    "RepeatedEntityHoldoutSplit",
    "RepeatedGroupKFold",
    "RepeatedKFold",
    "RepeatedSplit",
    "RepeatedStratifiedKFold",
    "one_standard_error_select",
    "paired_bootstrap_interval",
    "paired_fold_score_differences",
    "weighted_fold_summary",
    "weighted_score_summary",
]

ScoreFunction = Callable[[Any, Any], float]


def _validate_integer(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return int(value)


def _validate_random_state(random_state: int | None) -> int | None:
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, Integral)
    ):
        raise ValueError("random_state must be an integer or None.")
    return None if random_state is None else int(random_state)


def _numeric_vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values.")
    return numeric


def _advantage(candidate_score: float, reference_score: float, *, higher: bool) -> float:
    if higher:
        return float(candidate_score - reference_score)
    return float(reference_score - candidate_score)


def paired_fold_score_differences(
    candidate_scores: Sequence[float] | np.ndarray | pd.Series,
    reference_scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    higher_is_better: bool = False,
    fold_labels: Sequence[Any] | np.ndarray | pd.Index | None = None,
) -> pd.DataFrame:
    """Return paired fold scores and candidate-oriented differences.

    The two score vectors must already use the same materialized folds in the
    same order.  ``candidate_advantage`` is positive when the candidate is
    better: ``candidate - reference`` for maximized scores and ``reference -
    candidate`` for minimized scores.
    """

    candidate = _numeric_vector(candidate_scores, name="candidate_scores")
    reference = _numeric_vector(reference_scores, name="reference_scores")
    if candidate.shape != reference.shape:
        raise ValueError("candidate_scores and reference_scores must have the same length.")
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be a boolean.")

    if fold_labels is None:
        labels = np.arange(1, candidate.size + 1)
    else:
        labels = np.asarray(fold_labels)
        if labels.ndim != 1 or labels.size != candidate.size:
            raise ValueError("fold_labels must contain one label per paired fold.")
        if pd.isna(labels).any():
            raise ValueError("fold_labels must not contain missing values.")

    difference = candidate - reference if higher_is_better else reference - candidate
    return pd.DataFrame(
        {
            "fold": labels,
            "candidate_score": candidate,
            "reference_score": reference,
            "candidate_advantage": difference,
        }
    )


def _prediction_nobs(values: Any, *, name: str) -> int:
    if isinstance(values, Mapping):
        lengths = []
        for value in values.values():
            if value is None or isinstance(value, (str, bytes)) or np.isscalar(value):
                continue
            try:
                lengths.append(len(value))
            except TypeError:
                continue
        if not lengths:
            raise ValueError(f"{name} must contain at least one row-aligned value.")
        if len(set(lengths)) != 1:
            raise ValueError(f"Row-aligned values in {name} must have the same length.")
        return int(lengths[0])
    try:
        nobs = len(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain row-aligned predictions.") from exc
    if nobs == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    return int(nobs)


def _take_rows(values: Any, indices: np.ndarray, *, nobs: int) -> Any:
    if isinstance(values, Mapping):
        return {
            name: _take_rows(value, indices, nobs=nobs)
            if not isinstance(value, (str, bytes))
            and value is not None
            and not np.isscalar(value)
            and hasattr(value, "__len__")
            and len(value) == nobs
            else value
            for name, value in values.items()
        }
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.iloc[indices]
    if isinstance(values, pd.Index):
        return values.take(indices)
    return np.asarray(values)[indices]


def _cluster_codes(clusters: Any, *, nobs: int) -> tuple[np.ndarray, int]:
    cluster_array = np.asarray(clusters)
    if cluster_array.ndim != 1 or cluster_array.size != nobs:
        raise ValueError("clusters must contain one label per observation.")
    if pd.isna(cluster_array).any():
        raise ValueError("clusters must not contain missing values.")
    try:
        codes, levels = pd.factorize(cluster_array, sort=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("clusters must contain hashable scalar labels.") from exc
    if len(levels) < 2:
        raise ValueError("Cluster bootstrap requires at least two distinct clusters.")
    return np.asarray(codes, dtype=np.int64), int(len(levels))


@dataclass(frozen=True)
class BootstrapDifference:
    """Percentile-bootstrap interval for a candidate's paired advantage."""

    estimate: float
    standard_error: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    n_observations: int
    n_sampling_units: int
    clustered: bool
    higher_is_better: bool

    @property
    def excludes_zero(self) -> bool:
        """Whether the percentile interval excludes no performance difference."""
        return bool(self.lower > 0.0 or self.upper < 0.0)

    def as_series(self) -> pd.Series:
        """Return a report-friendly representation."""
        return pd.Series(
            {
                "estimate": self.estimate,
                "standard_error": self.standard_error,
                "lower": self.lower,
                "upper": self.upper,
                "confidence_level": self.confidence_level,
                "n_resamples": self.n_resamples,
                "n_observations": self.n_observations,
                "n_sampling_units": self.n_sampling_units,
                "clustered": self.clustered,
                "higher_is_better": self.higher_is_better,
            },
            name="paired_bootstrap_difference",
        )


def paired_bootstrap_interval(
    candidate: Any,
    reference: Any,
    *,
    y_true: Any | None = None,
    scorer: ScoreFunction | None = None,
    clusters: Any | None = None,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    higher_is_better: bool = False,
) -> BootstrapDifference:
    """Bootstrap a paired candidate-versus-reference score difference.

    With no ``scorer``, ``candidate`` and ``reference`` are observation-level
    losses or rewards.  With ``scorer``, they are row-aligned predictions and
    ``scorer(y_subset, prediction_subset)`` must return one finite score.
    Supplying ``clusters`` resamples complete entities rather than individual
    rows.  Positive estimates always favor the candidate.
    """

    n_resamples = _validate_integer("n_resamples", n_resamples, minimum=2)
    random_state = _validate_random_state(random_state)
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be a boolean.")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and strictly between zero and one.")

    if scorer is None:
        if y_true is not None:
            raise ValueError("y_true is used only when scorer is supplied.")
        candidate_values = _numeric_vector(candidate, name="candidate")
        reference_values = _numeric_vector(reference, name="reference")
        if candidate_values.shape != reference_values.shape:
            raise ValueError("candidate and reference must contain the same observations.")
        nobs = int(candidate_values.size)

        def evaluate(values: np.ndarray, indices: np.ndarray) -> float:
            return float(np.mean(values[indices]))

        def difference(indices: np.ndarray) -> float:
            return _advantage(
                evaluate(candidate_values, indices),
                evaluate(reference_values, indices),
                higher=higher_is_better,
            )

    else:
        if not callable(scorer):
            raise TypeError("scorer must be callable.")
        if y_true is None:
            raise ValueError("y_true is required when scorer is supplied.")
        nobs = _prediction_nobs(y_true, name="y_true")
        if _prediction_nobs(candidate, name="candidate") != nobs:
            raise ValueError("candidate and y_true must contain the same observations.")
        if _prediction_nobs(reference, name="reference") != nobs:
            raise ValueError("reference and y_true must contain the same observations.")

        def evaluate(values: Any, indices: np.ndarray) -> float:
            target_subset = _take_rows(y_true, indices, nobs=nobs)
            prediction_subset = _take_rows(values, indices, nobs=nobs)
            try:
                score = float(scorer(target_subset, prediction_subset))
            except Exception as exc:
                raise ValueError("scorer failed on a bootstrap sample.") from exc
            if not np.isfinite(score):
                raise ValueError("scorer must return a finite scalar for every sample.")
            return score

        def difference(indices: np.ndarray) -> float:
            return _advantage(
                evaluate(candidate, indices),
                evaluate(reference, indices),
                higher=higher_is_better,
            )

    full_index = np.arange(nobs, dtype=np.int64)
    estimate = difference(full_index)
    if not np.isfinite(estimate):  # pragma: no cover - guarded by the paths above
        raise ValueError("The paired score difference must be finite.")

    cluster_codes: np.ndarray | None = None
    if clusters is None:
        n_sampling_units = nobs
        if n_sampling_units < 2:
            raise ValueError("Bootstrap requires at least two observations.")
    else:
        cluster_codes, n_sampling_units = _cluster_codes(clusters, nobs=nobs)

    rng = np.random.default_rng(random_state)
    bootstrap = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        sampled_units = rng.integers(0, n_sampling_units, size=n_sampling_units)
        if cluster_codes is None:
            indices = sampled_units
        else:
            indices = np.concatenate(
                [np.flatnonzero(cluster_codes == unit) for unit in sampled_units]
            )
        bootstrap[draw] = difference(np.asarray(indices, dtype=np.int64))

    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    return BootstrapDifference(
        estimate=float(estimate),
        standard_error=float(np.std(bootstrap, ddof=1)),
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        n_resamples=n_resamples,
        n_observations=nobs,
        n_sampling_units=n_sampling_units,
        clustered=clusters is not None,
        higher_is_better=higher_is_better,
    )


@dataclass(frozen=True)
class RepeatedSplit:
    """One fold with explicit repeat and within-repeat identifiers."""

    repeat: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray


class _RepeatedBase:
    def __init__(
        self,
        n_splits: int,
        *,
        n_repeats: int,
        random_state: int | None,
    ) -> None:
        self.n_splits = _validate_integer("n_splits", n_splits, minimum=2)
        self.n_repeats = _validate_integer("n_repeats", n_repeats, minimum=1)
        self.random_state = _validate_random_state(random_state)

    def get_n_splits(self, *_args: Any, **_kwargs: Any) -> int:
        """Return the total number of folds across every repeat."""
        return self.n_splits * self.n_repeats

    def _repeat_seeds(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.integers(0, np.iinfo(np.int32).max, size=self.n_repeats)


class RepeatedKFold(_RepeatedBase):
    """Repeated shuffled K-fold splitting for independent rows."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        n_repeats: int = 10,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    def split_with_repeats(self, X: Any, y: Any = None) -> Iterator[RepeatedSplit]:
        """Yield folds with explicit repeat metadata."""
        for repeat, seed in enumerate(self._repeat_seeds(), start=1):
            splitter = KFold(self.n_splits, shuffle=True, random_state=int(seed))
            for fold, (train, test) in enumerate(splitter.split(X, y), start=1):
                yield RepeatedSplit(repeat, fold, train, test)

    def split(self, X: Any, y: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield standard train/test pairs for cross-validation consumers."""
        for repeated in self.split_with_repeats(X, y):
            yield repeated.train_index, repeated.test_index


class RepeatedStratifiedKFold(_RepeatedBase):
    """Repeated shuffled stratified folds for categorical outcomes."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        n_repeats: int = 10,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    def split_with_repeats(self, X: Any = None, y: Any = None) -> Iterator[RepeatedSplit]:
        """Yield stratified folds with explicit repeat metadata."""
        for repeat, seed in enumerate(self._repeat_seeds(), start=1):
            splitter = StratifiedKFold(
                self.n_splits,
                shuffle=True,
                random_state=int(seed),
            )
            for fold, (train, test) in enumerate(splitter.split(X, y), start=1):
                yield RepeatedSplit(repeat, fold, train, test)

    def split(
        self, X: Any = None, y: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield standard train/test pairs for cross-validation consumers."""
        for repeated in self.split_with_repeats(X, y):
            yield repeated.train_index, repeated.test_index


class RepeatedEntityHoldoutSplit(_RepeatedBase):
    """Repeated complete-entity holdout with shuffled tie-breaking."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        n_repeats: int = 10,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    def split_with_repeats(
        self,
        X: Any = None,
        *,
        entity: Any = None,
        groups: Any = None,
    ) -> Iterator[RepeatedSplit]:
        """Yield complete-entity folds with explicit repeat metadata."""
        for repeat, seed in enumerate(self._repeat_seeds(), start=1):
            splitter = EntityHoldoutSplit(
                self.n_splits,
                shuffle=True,
                random_state=int(seed),
            )
            splits = splitter.split(X, entity=entity, groups=groups)
            for fold, (train, test) in enumerate(splits, start=1):
                yield RepeatedSplit(repeat, fold, train, test)

    def split(
        self,
        X: Any = None,
        *,
        entity: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield standard train/test pairs for cross-validation consumers."""
        repeated = self.split_with_repeats(X, entity=entity, groups=groups)
        for item in repeated:
            yield item.train_index, item.test_index


class RepeatedGroupKFold(RepeatedEntityHoldoutSplit):
    """Group-oriented spelling of :class:`RepeatedEntityHoldoutSplit`."""


def weighted_score_summary(
    scores: Sequence[float] | np.ndarray | pd.Series,
    weights: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> pd.Series:
    """Summarize scores with non-negative reliability weights.

    Uniform weights reproduce the ordinary sample standard deviation and its
    standard error.  ``effective_n`` makes the loss of precision from unequal
    fold weights explicit.  The reported standard error treats the supplied
    scores as independent; overlapping repeated-CV folds violate that
    approximation, so paired observation/entity bootstrap intervals should be
    used for inferential comparisons.
    """

    values = _numeric_vector(scores, name="scores")
    if weights is None:
        weight = np.ones(values.size, dtype=float)
    else:
        weight = _numeric_vector(weights, name="weights")
        if weight.shape != values.shape:
            raise ValueError("weights must contain one value per score.")
        if np.any(weight < 0.0):
            raise ValueError("weights must be non-negative.")
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0.0:
        raise ValueError("At least one weight must be positive.")

    mean = float(np.average(values, weights=weight))
    squared_weight_sum = float(np.sum(weight**2))
    effective_n = weight_sum**2 / squared_weight_sum
    denominator = weight_sum - squared_weight_sum / weight_sum
    if denominator > 0.0:
        variance = float(np.sum(weight * (values - mean) ** 2) / denominator)
        standard_deviation = float(np.sqrt(max(variance, 0.0)))
        standard_error = float(standard_deviation / np.sqrt(effective_n))
    else:
        standard_deviation = 0.0
        standard_error = 0.0

    return pd.Series(
        {
            "mean": mean,
            "std": standard_deviation,
            "standard_error": standard_error,
            "min": float(np.min(values[weight > 0.0])),
            "max": float(np.max(values[weight > 0.0])),
            "n": int(np.count_nonzero(weight > 0.0)),
            "weight_sum": weight_sum,
            "effective_n": float(effective_n),
        },
        name="weighted_score_summary",
    )


def weighted_fold_summary(
    fold_frame: pd.DataFrame,
    *,
    weight_column: str = "test_n",
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return one weighted summary row per numeric fold metric."""

    if not isinstance(fold_frame, pd.DataFrame):
        raise TypeError("fold_frame must be a pandas DataFrame.")
    if weight_column not in fold_frame:
        raise ValueError(f"weight_column={weight_column!r} is not present in fold_frame.")
    weights = pd.to_numeric(fold_frame[weight_column], errors="coerce").to_numpy()
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("Fold weights must be finite, non-negative, and not all zero.")

    reserved = {
        "fold",
        "repeat",
        "train_n",
        "test_n",
        "outcome",
        "prediction_target",
        "eligible",
        "eligibility_reasons",
        "error",
        weight_column,
    }
    if metric_columns is None:
        metrics = [
            str(column)
            for column in fold_frame.columns
            if column not in reserved
            and pd.api.types.is_numeric_dtype(fold_frame[column])
        ]
    else:
        metrics = [str(column) for column in metric_columns]
        missing = [column for column in metrics if column not in fold_frame]
        if missing:
            raise ValueError(f"metric_columns are absent from fold_frame: {missing!r}.")

    rows = []
    for metric in metrics:
        values = pd.to_numeric(fold_frame[metric], errors="coerce").to_numpy()
        usable = np.isfinite(values) & (weights > 0.0)
        if not np.any(usable):
            continue
        summary = weighted_score_summary(values[usable], weights[usable])
        rows.append({"metric": metric, **summary.to_dict()})

    columns = [
        "mean",
        "std",
        "standard_error",
        "min",
        "max",
        "n",
        "weight_sum",
        "effective_n",
    ]
    if not rows:
        return pd.DataFrame(columns=columns).rename_axis("metric")
    return pd.DataFrame(rows).set_index("metric")[columns]


@dataclass(frozen=True)
class OneStandardErrorSelection:
    """Result of choosing the simplest model inside the best model's one-SE band."""

    selected_model: Any
    best_model: Any
    best_score: float
    best_standard_error: float
    cutoff: float
    candidate_models: tuple[Any, ...]
    selected_complexity: float
    higher_is_better: bool

    def as_series(self) -> pd.Series:
        """Return a report-friendly representation."""
        return pd.Series(
            {
                "selected_model": self.selected_model,
                "best_model": self.best_model,
                "best_score": self.best_score,
                "best_standard_error": self.best_standard_error,
                "cutoff": self.cutoff,
                "candidate_models": self.candidate_models,
                "selected_complexity": self.selected_complexity,
                "higher_is_better": self.higher_is_better,
            },
            name="one_standard_error_selection",
        )


def one_standard_error_select(
    table: pd.DataFrame,
    *,
    score_column: str,
    standard_error_column: str,
    complexity_column: str,
    model_column: str = "model",
    eligible_column: str | None = "eligible",
    higher_is_better: bool = False,
) -> OneStandardErrorSelection:
    """Select the simplest eligible model within one SE of the best score.

    Complexity must increase with model flexibility.  Ties in complexity are
    resolved by score, then by the table's original row order.
    """

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("table must be a non-empty pandas DataFrame.")
    required = {model_column, score_column, standard_error_column, complexity_column}
    if eligible_column is not None:
        required.add(eligible_column)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"table is missing required columns: {missing!r}.")
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be a boolean.")
    if table[model_column].duplicated().any():
        raise ValueError("model_column must contain unique model labels.")

    work = table.reset_index(drop=True).copy()
    work["__row_order"] = np.arange(len(work))
    if eligible_column is None:
        eligible = np.ones(len(work), dtype=bool)
    else:
        eligible = work[eligible_column].eq(True).to_numpy(dtype=bool)
    if not np.any(eligible):
        raise ValueError("No eligible model is available for one-standard-error selection.")

    for column in (score_column, standard_error_column, complexity_column):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    eligible_rows = work.loc[eligible]
    invalid = (
        ~np.isfinite(eligible_rows[score_column])
        | ~np.isfinite(eligible_rows[standard_error_column])
        | (eligible_rows[standard_error_column] < 0.0)
        | ~np.isfinite(eligible_rows[complexity_column])
        | (eligible_rows[complexity_column] < 0.0)
    )
    if invalid.any():
        labels = eligible_rows.loc[invalid, model_column].tolist()
        raise ValueError(f"Eligible models require finite scores, SEs, and complexity: {labels!r}.")

    best_position = (
        eligible_rows[score_column].idxmax()
        if higher_is_better
        else eligible_rows[score_column].idxmin()
    )
    best = work.loc[best_position]
    best_score = float(best[score_column])
    best_se = float(best[standard_error_column])
    cutoff = best_score - best_se if higher_is_better else best_score + best_se
    if higher_is_better:
        inside = eligible_rows[score_column] >= cutoff
    else:
        inside = eligible_rows[score_column] <= cutoff
    candidates = eligible_rows.loc[inside].copy()

    minimum_complexity = float(candidates[complexity_column].min())
    simplest = candidates.loc[candidates[complexity_column] == minimum_complexity]
    simplest = simplest.sort_values(
        [score_column, "__row_order"],
        ascending=[not higher_is_better, True],
        kind="stable",
    )
    selected = simplest.iloc[0]
    return OneStandardErrorSelection(
        selected_model=selected[model_column],
        best_model=best[model_column],
        best_score=best_score,
        best_standard_error=best_se,
        cutoff=float(cutoff),
        candidate_models=tuple(candidates[model_column]),
        selected_complexity=minimum_complexity,
        higher_is_better=higher_is_better,
    )
