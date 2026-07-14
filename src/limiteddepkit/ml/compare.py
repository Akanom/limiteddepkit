"""Cross-validated model comparison for limited outcomes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .validation import CrossValidationResult, cross_validate

_DEFAULT_PRIMARY = {
    "binary": "log_loss",
    "choice": "log_loss",
    "multiclass": "log_loss",
    "ordinal": "ranked_probability_score",
    "count": "poisson_deviance",
    "continuous": "rmse",
    "quantile": "check_loss",
    "duration": "concordance_index",
    "selection": "selection_log_loss",
}
_HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "concordance_index",
    "roc_auc",
}


@dataclass(frozen=True)
class ModelComparisonResult:
    """Ranked comparison table plus the underlying fold evidence."""

    table: pd.DataFrame
    cv_results: Mapping[str, CrossValidationResult]
    primary_metric: str
    higher_is_better: bool

    @property
    def best_model(self) -> str | None:
        if "rank" not in self.table:
            return None
        usable = self.table.loc[
            self.table["eligible"]
            & self.table["error"].eq("")
            & self.table["rank"].notna()
        ]
        return None if usable.empty else str(usable.iloc[0]["model"])

    def to_markdown(self) -> str:
        """Return a dependency-free report-ready comparison table."""
        columns = [str(column) for column in self.table.columns]

        def render(value: Any) -> str:
            if pd.isna(value):
                return ""
            return str(value).replace("|", "\\|").replace("\n", " ")

        rows = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        rows.extend(
            "| " + " | ".join(render(value) for value in row) + " |"
            for row in self.table.itertuples(index=False, name=None)
        )
        return "\n".join(rows)


class _FixedSplitter:
    """Replay one materialized split design for every compared model."""

    def __init__(self, splits: tuple[tuple[Any, Any], ...], source: Any) -> None:
        self._splits = splits
        self._source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def split(self, X: Any = None, y: Any = None):
        del X, y
        for train, test in self._splits:
            yield np.asarray(train).copy(), np.asarray(test).copy()


def _normalise_models(
    models: Mapping[str, Callable[[], Any]],
) -> list[tuple[str, Callable[[], Any]]]:
    if not isinstance(models, Mapping) or not models:
        raise TypeError("models must be a non-empty mapping of names to model factories.")
    output = []
    normalized_names: set[str] = set()
    for name, factory in models.items():
        if not callable(factory):
            raise TypeError(f"Model factory for {name!r} is not callable.")
        normalized = str(name)
        if normalized in normalized_names:
            raise ValueError(
                "Model names must be unique after conversion to strings; "
                f"duplicate {normalized!r}."
            )
        normalized_names.add(normalized)
        output.append((normalized, factory))
    return output


def _summary_record(name: str, result: CrossValidationResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model": name,
        "eligible": result.eligible,
        "eligible_folds": result.eligible_folds,
        "successful_folds": result.successful_folds,
        "total_folds": len(result.folds),
        "error": "; ".join(fold.error for fold in result.folds if fold.error),
    }
    summary = result.summary_frame()
    for metric, row in summary.iterrows():
        record[f"{metric}_mean"] = float(row["mean"])
        record[f"{metric}_std"] = float(row["std"])
    return record


def _resolve_primary(
    requested: str | None,
    outcome: str,
    table: pd.DataFrame,
) -> str:
    if requested is not None:
        metric = str(requested)
        if f"{metric}_mean" not in table.columns:
            raise ValueError(f"primary_metric={metric!r} is not available in the CV results.")
        return metric

    preferred = _DEFAULT_PRIMARY.get(outcome)
    if preferred is not None and f"{preferred}_mean" in table.columns:
        return preferred
    candidates = [column.removesuffix("_mean") for column in table if column.endswith("_mean")]
    if not candidates:
        raise ValueError("No common numeric validation metric is available for ranking.")
    return candidates[0]


def compare_models(
    models: Mapping[str, Callable[[], Any]],
    X: Any,
    y: Any,
    *,
    primary_metric: str | None = None,
    higher_is_better: bool | None = None,
    raise_on_error: bool = False,
    **cross_validation_kwargs: Any,
) -> ModelComparisonResult:
    """Cross-validate model factories and rank only econometrically eligible fits."""
    results: dict[str, CrossValidationResult] = {}
    rows: list[dict[str, Any]] = []
    requested_outcome = str(cross_validation_kwargs.get("outcome", "auto"))
    if "splitter" not in cross_validation_kwargs:
        raise TypeError("compare_models requires splitter= in cross-validation arguments.")
    if "model_name" in cross_validation_kwargs:
        raise TypeError("compare_models assigns model_name from the models mapping.")

    from .validation import _split_iterator

    original_splitter = cross_validation_kwargs["splitter"]
    splitting_target = cross_validation_kwargs.get("split_y")
    if splitting_target is None:
        splitting_target = y
    shared_splits = tuple(
        _split_iterator(
            original_splitter,
            X,
            splitting_target,
            entity=cross_validation_kwargs.get("entity"),
            time=cross_validation_kwargs.get("time"),
        )
    )
    if not shared_splits:
        raise ValueError("splitter did not produce any folds.")
    validation_kwargs = dict(cross_validation_kwargs)
    validation_kwargs["splitter"] = _FixedSplitter(shared_splits, original_splitter)

    for name, factory in _normalise_models(models):
        try:
            result = cross_validate(
                factory,
                X,
                y,
                model_name=name,
                **validation_kwargs,
            )
            results[name] = result
            rows.append(_summary_record(name, result))
        except Exception as exc:
            if raise_on_error:
                raise
            rows.append(
                {
                    "model": name,
                    "eligible": False,
                    "eligible_folds": 0,
                    "successful_folds": 0,
                    "total_folds": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    table = pd.DataFrame(rows)
    if not results:
        metric = primary_metric or _DEFAULT_PRIMARY.get(requested_outcome, "score")
        return ModelComparisonResult(
            table=table,
            cv_results=results,
            primary_metric=metric,
            higher_is_better=bool(higher_is_better),
        )

    outcomes = {result.outcome for result in results.values()}
    if len(outcomes) != 1:
        raise ValueError(f"Compared models resolved to different outcome families: {outcomes}.")
    resolved_outcome = outcomes.pop()
    if resolved_outcome == "quantile":
        fitted_quantiles = {
            float(fold.result.quantile)
            for result in results.values()
            for fold in result.folds
            if not fold.error and hasattr(fold.result, "quantile")
        }
        if len(fitted_quantiles) > 1:
            raise ValueError(
                "Compared quantile models target different quantiles and cannot be ranked "
                "on one check-loss scale."
            )
    if primary_metric is not None and f"{primary_metric}_mean" not in table:
        metric_was_scored = any(
            primary_metric in fold.metrics
            for result in results.values()
            for fold in result.folds
        )
        if metric_was_scored:
            table[f"{primary_metric}_mean"] = np.nan
            table[f"{primary_metric}_std"] = np.nan
    metric = _resolve_primary(primary_metric, resolved_outcome, table)
    maximize = metric in _HIGHER_IS_BETTER if higher_is_better is None else higher_is_better
    score_column = f"{metric}_mean"

    primary_complete = {
        name: bool(result.folds)
        and all(
            not fold.error
            and metric in fold.metrics
            and np.isfinite(float(fold.metrics[metric]))
            for fold in result.folds
        )
        for name, result in results.items()
    }
    table["primary_metric_complete"] = (
        table["model"].map(primary_complete).fillna(False).astype(bool)
    )

    usable = (
        table["eligible"]
        & table["error"].eq("")
        & table["primary_metric_complete"]
        & np.isfinite(pd.to_numeric(table.get(score_column), errors="coerce"))
    )
    ranked = table.loc[usable].sort_values(score_column, ascending=not maximize).copy()
    ranked.insert(1, "rank", np.arange(1, len(ranked) + 1))
    excluded = table.loc[~usable].copy()
    excluded.insert(1, "rank", pd.NA)
    table = pd.concat([ranked, excluded], ignore_index=True)

    return ModelComparisonResult(
        table=table,
        cv_results=results,
        primary_metric=metric,
        higher_is_better=bool(maximize),
    )
