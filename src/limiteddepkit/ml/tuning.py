"""Leakage-safe nested cross-validation for estimator and penalty selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .adapter import accepted_kwargs, subset_context, take_rows
from .compare import ModelComparisonResult, _FixedSplitter, compare_models
from .uncertainty import OneStandardErrorSelection, one_standard_error_select
from .validation import (
    CrossValidationResult,
    _nobs,
    _row_labels,
    _split_iterator,
    _validated_indices,
    cross_validate,
)

__all__ = [
    "NestedCrossValidationResult",
    "NestedFoldSelection",
    "TuningCandidate",
    "nested_cross_validate",
]


@dataclass(frozen=True)
class TuningCandidate:
    """One estimator configuration considered only inside inner CV folds.

    ``complexity`` must increase with model flexibility.  For ridge paths this
    normally means assigning *smaller* complexity to stronger regularization;
    the value is deliberately explicit because penalty scales are estimator-
    specific and cannot be inferred safely.
    """

    model_factory: Callable[[], Any]
    fit_kwargs: Mapping[str, Any] = field(default_factory=dict)
    complexity: float = 0.0

    def __post_init__(self) -> None:
        if not callable(self.model_factory):
            raise TypeError("model_factory must be callable.")
        if not isinstance(self.fit_kwargs, Mapping):
            raise TypeError("fit_kwargs must be a mapping.")
        try:
            complexity = float(self.complexity)
        except (TypeError, ValueError) as exc:
            raise ValueError("complexity must be a finite non-negative number.") from exc
        if not np.isfinite(complexity) or complexity < 0.0:
            raise ValueError("complexity must be a finite non-negative number.")
        object.__setattr__(self, "fit_kwargs", dict(self.fit_kwargs))
        object.__setattr__(self, "complexity", complexity)


class _ConfiguredEstimator:
    """Apply candidate-specific fit keywords without changing result types."""

    def __init__(self, estimator: Any, fit_kwargs: Mapping[str, Any]) -> None:
        self._estimator = estimator
        self._fit_kwargs = dict(fit_kwargs)

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> Any:
        configured = {**fit_kwargs, **self._fit_kwargs}
        # cross_validate supplies entity/time as optional estimator metadata.
        # Its normal adapter filters metadata against the real estimator's
        # signature; preserve that behavior despite this wrapper's **kwargs.
        metadata = {
            name: configured.pop(name)
            for name in ("entity", "time")
            if name in configured
        }
        configured.update(accepted_kwargs(self._estimator.fit, metadata))
        return self._estimator.fit(X, y, **configured)


@dataclass(frozen=True)
class NestedFoldSelection:
    """Inner-CV evidence and chosen configuration for one outer fold."""

    outer_fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    selected_model: str
    inner_comparison: ModelComparisonResult
    one_standard_error: OneStandardErrorSelection | None = None


@dataclass(frozen=True)
class NestedCrossValidationResult:
    """Outer-fold generalization scores plus every inner selection decision."""

    outer_result: CrossValidationResult
    selections: tuple[NestedFoldSelection, ...]
    primary_metric: str
    higher_is_better: bool
    selection_rule: str

    @property
    def selected_models(self) -> tuple[str, ...]:
        """Return the configuration selected independently in each outer fold."""
        return tuple(selection.selected_model for selection in self.selections)

    @property
    def eligible(self) -> bool:
        return self.outer_result.eligible

    def fold_frame(self) -> pd.DataFrame:
        """Return outer-fold scores with their fold-specific selected models."""
        table = self.outer_result.fold_frame().copy()
        selected = {item.outer_fold: item.selected_model for item in self.selections}
        table.insert(1, "selected_model", table["fold"].map(selected))
        return table

    def summary_frame(self) -> pd.DataFrame:
        """Return the ordinary outer-CV score summary."""
        return self.outer_result.summary_frame()

    def out_of_fold_predictions(self) -> pd.DataFrame:
        """Return predictions made only by outer-fold selected configurations."""
        return self.outer_result.out_of_fold_predictions()


def _normalise_candidates(
    candidates: Mapping[str, TuningCandidate | Callable[[], Any]],
) -> dict[str, TuningCandidate]:
    if not isinstance(candidates, Mapping) or not candidates:
        raise TypeError("candidates must be a non-empty mapping.")
    normalised: dict[str, TuningCandidate] = {}
    for position, (raw_name, value) in enumerate(candidates.items()):
        name = str(raw_name)
        if not name:
            raise ValueError("Candidate names must not be empty.")
        if name in normalised:
            raise ValueError(f"Candidate names must be unique after conversion: {name!r}.")
        if isinstance(value, TuningCandidate):
            normalised[name] = value
        elif callable(value):
            normalised[name] = TuningCandidate(value, complexity=float(position))
        else:
            raise TypeError(
                f"Candidate {name!r} must be callable or a TuningCandidate."
            )
    return normalised


def _configured_factory(candidate: TuningCandidate) -> Callable[[], Any]:
    def factory() -> _ConfiguredEstimator:
        return _ConfiguredEstimator(candidate.model_factory(), candidate.fit_kwargs)

    return factory


def _subset_cv_kwargs(
    values: Mapping[str, Any],
    indices: np.ndarray,
    *,
    nobs: int,
) -> dict[str, Any]:
    output = dict(values)
    for name in ("entity", "time", "split_y"):
        if output.get(name) is not None:
            output[name] = take_rows(output[name], indices)
    for name in ("fit_context", "predict_context", "score_context"):
        if output.get(name) is not None:
            output[name] = subset_context(output[name], indices, nobs=nobs)
    return output


def _selection_table(
    comparison: ModelComparisonResult,
    candidates: Mapping[str, TuningCandidate],
) -> pd.DataFrame:
    table = comparison.table.copy()
    score_column = f"{comparison.primary_metric}_mean"
    std_column = f"{comparison.primary_metric}_std"
    if score_column not in table or std_column not in table:
        raise ValueError("Inner CV did not produce a complete primary-metric summary.")
    fold_count = pd.to_numeric(table["successful_folds"], errors="coerce")
    standard_deviation = pd.to_numeric(table[std_column], errors="coerce")
    table["primary_standard_error"] = standard_deviation / np.sqrt(fold_count)
    table["complexity"] = table["model"].map(
        {name: candidate.complexity for name, candidate in candidates.items()}
    )
    table["selection_eligible"] = (
        table["eligible"].eq(True)
        & table["error"].eq("")
        & table["primary_metric_complete"].eq(True)
        & np.isfinite(pd.to_numeric(table[score_column], errors="coerce"))
        & np.isfinite(table["primary_standard_error"])
    )
    return table


def nested_cross_validate(
    candidates: Mapping[str, TuningCandidate | Callable[[], Any]],
    X: Any,
    y: Any,
    *,
    outer_splitter: Any,
    inner_splitter_factory: Callable[[], Any],
    primary_metric: str | None = None,
    higher_is_better: bool | None = None,
    selection_rule: str = "one_se",
    **cross_validation_kwargs: Any,
) -> NestedCrossValidationResult:
    """Select configurations in inner folds and score them in untouched outer folds.

    This is the supported path for tuning Firth/ridge choices or penalty
    strengths.  Every outer training partition receives a fresh inner splitter;
    preprocessing factories are consequently fitted again inside every inner
    fold.  The outer test partition is used exactly once, after selection.
    """
    if "splitter" in cross_validation_kwargs:
        raise TypeError("Use outer_splitter=; do not also supply splitter=.")
    if "model_name" in cross_validation_kwargs:
        raise TypeError("nested_cross_validate assigns fold-specific model names.")
    if not callable(inner_splitter_factory):
        raise TypeError("inner_splitter_factory must be callable and return a fresh splitter.")
    rule = str(selection_rule).strip().lower()
    if rule not in {"best", "one_se"}:
        raise ValueError("selection_rule must be 'best' or 'one_se'.")

    specifications = _normalise_candidates(candidates)
    if rule == "one_se" and len(specifications) > 1:
        implicit = [
            str(name)
            for name, candidate in candidates.items()
            if not isinstance(candidate, TuningCandidate)
        ]
        if implicit:
            raise TypeError(
                "one_se selection with multiple candidates requires an explicit "
                "TuningCandidate(complexity=...) for every model; implicit candidates: "
                f"{implicit!r}."
            )
    nobs = _nobs(X, name="X")
    if _nobs(y, name="y") != nobs:
        raise ValueError("X and y must contain the same number of rows.")
    entity = cross_validation_kwargs.get("entity")
    time = cross_validation_kwargs.get("time")
    splitting_target = cross_validation_kwargs.get("split_y", y)
    raw_outer_splits = tuple(
        _split_iterator(
            outer_splitter,
            X,
            splitting_target,
            entity=entity,
            time=time,
        )
    )
    if not raw_outer_splits:
        raise ValueError("outer_splitter did not produce any folds.")

    outer_folds = tuple(
        _validated_indices(train, test, nobs=nobs) for train, test in raw_outer_splits
    )
    selections: list[NestedFoldSelection] = []
    outer_evaluations = []
    resolved_metric: str | None = None
    resolved_direction: bool | None = None

    configured_candidates = {
        name: _configured_factory(candidate)
        for name, candidate in specifications.items()
    }
    for outer_fold, (train_index, test_index) in enumerate(outer_folds, start=1):
        inner_kwargs = _subset_cv_kwargs(
            cross_validation_kwargs,
            train_index,
            nobs=nobs,
        )
        inner_kwargs["splitter"] = inner_splitter_factory()
        comparison = compare_models(
            configured_candidates,
            take_rows(X, train_index),
            take_rows(y, train_index),
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
            raise_on_error=False,
            **inner_kwargs,
        )
        if resolved_metric is None:
            resolved_metric = comparison.primary_metric
            resolved_direction = comparison.higher_is_better
        elif (
            comparison.primary_metric != resolved_metric
            or comparison.higher_is_better != resolved_direction
        ):
            raise RuntimeError("Inner folds resolved inconsistent scoring contracts.")

        selection_table = _selection_table(comparison, specifications)
        one_se_result = None
        if rule == "best":
            usable = selection_table.loc[selection_table["selection_eligible"]]
            if usable.empty:
                raise ValueError(f"Outer fold {outer_fold} has no eligible inner candidate.")
            selected_name = str(
                usable.sort_values(
                    f"{comparison.primary_metric}_mean",
                    ascending=not comparison.higher_is_better,
                    kind="stable",
                ).iloc[0]["model"]
            )
        else:
            one_se_result = one_standard_error_select(
                selection_table,
                score_column=f"{comparison.primary_metric}_mean",
                standard_error_column="primary_standard_error",
                complexity_column="complexity",
                eligible_column="selection_eligible",
                higher_is_better=comparison.higher_is_better,
            )
            selected_name = str(one_se_result.selected_model)

        outer_kwargs = dict(cross_validation_kwargs)
        outer_kwargs["splitter"] = _FixedSplitter(
            ((train_index, test_index),),
            outer_splitter,
        )
        outer_run = cross_validate(
            configured_candidates[selected_name],
            X,
            y,
            model_name=selected_name,
            **outer_kwargs,
        )
        outer_evaluations.append(replace(outer_run.folds[0], fold=outer_fold))
        selections.append(
            NestedFoldSelection(
                outer_fold=outer_fold,
                train_index=train_index.copy(),
                test_index=test_index.copy(),
                selected_model=selected_name,
                inner_comparison=comparison,
                one_standard_error=one_se_result,
            )
        )

    resolved_outcome = next(
        (fold.outcome for fold in outer_evaluations if fold.outcome != "auto"),
        str(cross_validation_kwargs.get("outcome", "auto")),
    )
    outer_result = CrossValidationResult(
        folds=tuple(outer_evaluations),
        outcome=resolved_outcome,
        model_name="nested-selected",
        row_labels=_row_labels(X, nobs),
    )
    if resolved_metric is None or resolved_direction is None:  # pragma: no cover
        raise RuntimeError("Nested CV did not resolve a scoring contract.")
    return NestedCrossValidationResult(
        outer_result=outer_result,
        selections=tuple(selections),
        primary_metric=resolved_metric,
        higher_is_better=resolved_direction,
        selection_rule=rule,
    )
