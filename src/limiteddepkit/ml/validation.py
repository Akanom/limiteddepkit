"""Cross-validation for probability and limited-outcome estimators."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .adapter import (
    accepted_kwargs,
    default_fit,
    default_predict,
    normalize_outcome,
    prediction_frame,
    result_eligibility,
    subset_context,
    take_rows,
    validated_kwargs,
)
from .metrics import score_predictions

FitCallback = Callable[..., Any]
PredictCallback = Callable[..., Any]
TransformerFactory = Callable[[], Any]


@dataclass(frozen=True)
class FoldEvaluation:
    """One fitted and scored validation fold."""

    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    outcome: str
    prediction_target: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    eligible: bool = False
    eligibility_reasons: tuple[str, ...] = ()
    prediction: Any = None
    result: Any = None
    transformer: Any = None
    error: str = ""

    def as_record(self) -> dict[str, Any]:
        """Return a flat row for reporting."""
        return {
            "fold": self.fold,
            "train_n": len(self.train_index),
            "test_n": len(self.test_index),
            "outcome": self.outcome,
            "prediction_target": self.prediction_target,
            "eligible": self.eligible,
            "eligibility_reasons": "; ".join(self.eligibility_reasons),
            "error": self.error,
            **dict(self.metrics),
        }


@dataclass(frozen=True)
class CrossValidationResult:
    """Structured fold results, summaries, and out-of-fold predictions."""

    folds: tuple[FoldEvaluation, ...]
    outcome: str
    model_name: str
    row_labels: np.ndarray

    @property
    def successful_folds(self) -> int:
        return sum(not fold.error for fold in self.folds)

    @property
    def eligible_folds(self) -> int:
        return sum(fold.eligible and not fold.error for fold in self.folds)

    @property
    def eligible(self) -> bool:
        return bool(self.folds) and self.eligible_folds == len(self.folds)

    def fold_frame(self) -> pd.DataFrame:
        """Return one row per fold."""
        return pd.DataFrame([fold.as_record() for fold in self.folds])

    def summary_frame(self) -> pd.DataFrame:
        """Return unweighted macro summaries of numeric scores across folds."""
        table = self.fold_frame()
        reserved = {
            "fold",
            "train_n",
            "test_n",
            "outcome",
            "prediction_target",
            "eligible",
            "eligibility_reasons",
            "error",
        }
        metric_columns = [
            column
            for column in table.columns
            if column not in reserved and pd.api.types.is_numeric_dtype(table[column])
        ]
        rows = []
        for metric in metric_columns:
            values = pd.to_numeric(table[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "folds": int(len(values)),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["mean", "std", "min", "max", "folds"]).rename_axis(
                "metric"
            )
        return pd.DataFrame(rows).set_index("metric")

    def weighted_summary_frame(self, *, weight_column: str = "test_n") -> pd.DataFrame:
        """Return fold-score summaries weighted by test size or another column.

        This is descriptive aggregation.  Because repeated and overlapping CV
        scores are dependent, its standard errors are not a substitute for a
        paired observation/entity bootstrap comparison.
        """
        from .uncertainty import weighted_fold_summary

        return weighted_fold_summary(
            self.fold_frame(),
            weight_column=weight_column,
        )

    def out_of_fold_predictions(self) -> pd.DataFrame:
        """Return row-labelled predictions from all successful folds."""
        frames = []
        for fold in self.folds:
            if fold.error or fold.prediction is None:
                continue
            labels = self.row_labels[fold.test_index]
            frame = prediction_frame(
                fold.prediction,
                fold=fold.fold,
                row_index=labels,
            )
            frame.insert(1, "row_position", fold.test_index)
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["fold", "row_position", "row_index"])
        return pd.concat(frames, ignore_index=True)

    def pooled_out_of_fold_predictions(
        self,
        *,
        fold_weights: Mapping[int, float] | None = None,
    ) -> pd.DataFrame:
        """Average repeated OOF predictions once per original observation.

        Numeric prediction columns use a weighted mean.  Non-numeric point
        predictions use a deterministic weighted mode.  ``row_position`` is
        the grouping key, so duplicate pandas index labels remain distinct.
        """
        table = self.out_of_fold_predictions()
        prediction_columns = [
            column
            for column in table.columns
            if column not in {"fold", "row_position", "row_index"}
        ]
        if table.empty:
            return pd.DataFrame(
                columns=[
                    "row_position",
                    "row_index",
                    "prediction_count",
                    "weight_sum",
                    *prediction_columns,
                ]
            )
        if fold_weights is None:
            weights_by_fold = {
                int(fold): 1.0 for fold in pd.unique(table["fold"])
            }
        else:
            if not isinstance(fold_weights, Mapping):
                raise TypeError("fold_weights must be a mapping from fold number to weight.")
            observed_folds = {int(fold) for fold in pd.unique(table["fold"])}
            weights_by_fold = {}
            for fold, raw_weight in fold_weights.items():
                if isinstance(fold, (bool, np.bool_)) or not isinstance(fold, Integral):
                    raise ValueError("fold_weights keys must be integer fold numbers.")
                fold_number = int(fold)
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError) as exc:
                    raise ValueError("fold_weights must contain finite positive values.") from exc
                if not np.isfinite(weight) or weight <= 0.0:
                    raise ValueError("fold_weights must contain finite positive values.")
                weights_by_fold[fold_number] = weight
            supplied_folds = set(weights_by_fold)
            missing = sorted(observed_folds - supplied_folds)
            extra = sorted(supplied_folds - observed_folds)
            if missing or extra:
                raise ValueError(
                    "fold_weights keys must exactly match observed folds; "
                    f"missing={missing!r}, extra={extra!r}."
                )
        table = table.copy()
        table["__weight"] = table["fold"].map(weights_by_fold)
        if table["__weight"].isna().any():  # pragma: no cover - guarded above
            raise RuntimeError("Failed to align fold weights.")

        rows: list[dict[str, Any]] = []
        for row_position, group in table.groupby("row_position", sort=True):
            weights = group["__weight"].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "row_position": int(row_position),
                "row_index": group["row_index"].iloc[0],
                "prediction_count": int(len(group)),
                "weight_sum": float(np.sum(weights)),
            }
            for column in prediction_columns:
                values = group[column]
                numeric = pd.to_numeric(values, errors="coerce")
                numeric_usable = numeric.notna() & values.notna()
                if numeric_usable.any() and numeric_usable.sum() == values.notna().sum():
                    selected_weights = weights[numeric_usable.to_numpy()]
                    row[column] = float(
                        np.average(
                            numeric.loc[numeric_usable].to_numpy(dtype=float),
                            weights=selected_weights,
                        )
                    )
                    row[f"{column}__count"] = int(numeric_usable.sum())
                    row[f"{column}__weight_sum"] = float(np.sum(selected_weights))
                    continue
                weight_by_value: dict[Any, float] = {}
                first_position: dict[Any, int] = {}
                nonmissing = values.notna().to_numpy()
                for position, (value, weight) in enumerate(
                    zip(
                        values.loc[nonmissing].tolist(),
                        weights[nonmissing],
                        strict=True,
                    )
                ):
                    try:
                        weight_by_value[value] = weight_by_value.get(value, 0.0) + float(
                            weight
                        )
                        first_position.setdefault(value, position)
                    except TypeError as exc:
                        raise ValueError(
                            f"Non-numeric prediction column {column!r} must contain "
                            "hashable scalar values."
                        ) from exc
                if not weight_by_value:
                    row[column] = np.nan
                    row[f"{column}__count"] = 0
                    row[f"{column}__weight_sum"] = 0.0
                    continue
                row[column] = min(
                    weight_by_value,
                    key=lambda value: (-weight_by_value[value], first_position[value]),
                )
                row[f"{column}__count"] = int(np.count_nonzero(nonmissing))
                row[f"{column}__weight_sum"] = float(np.sum(weights[nonmissing]))
            rows.append(row)
        return pd.DataFrame(rows)


def _nobs(values: Any, *, name: str) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) >= 1 and shape[0] is not None:
        try:
            length = int(shape[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must have a finite row dimension.") from exc
    else:
        try:
            length = len(values)
        except TypeError as exc:
            raise TypeError(f"{name} must be row-oriented and sized.") from exc
    if length == 0:
        raise ValueError(f"{name} must not be empty.")
    return int(length)


def _row_labels(X: Any, nobs: int) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.index.to_numpy(copy=True)
    return np.arange(nobs)


def _split_iterator(
    splitter: Any,
    X: Any,
    y: Any,
    *,
    entity: Any | None,
    time: Any | None,
) -> Iterator[tuple[Any, Any]]:
    if not hasattr(splitter, "split"):
        raise TypeError("splitter must expose a split() method.")
    method = splitter.split
    signature = inspect.signature(method)
    available = {
        "X": X,
        "x": X,
        "data": X,
        "y": y,
        "groups": entity,
        "group": entity,
        "entity": entity,
        "time": time,
    }
    keywords: dict[str, Any] = {}
    missing: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if name == "groups" and "entity" in signature.parameters and entity is not None:
            continue
        if name in available and available[name] is not None:
            keywords[name] = available[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(
            f"Cannot supply required splitter arguments {missing}; expected names like "
            "X, y, entity/groups, or time."
        )
    return iter(method(**keywords))


def _index_positions(values: Any, *, name: str, nobs: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("Splitter indices must be one-dimensional.")
    if np.issubdtype(raw.dtype, np.bool_):
        if len(raw) != nobs:
            raise ValueError(f"Boolean {name} mask must contain one value per observation.")
        positions = np.flatnonzero(raw).astype(int)
    else:
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError(f"Splitter {name} indices must be integers or a Boolean mask.")
        positions = raw.astype(int, copy=False)
    if np.unique(positions).size != positions.size:
        raise ValueError(f"Splitter {name} indices must be unique.")
    return positions


def _validated_indices(train: Any, test: Any, *, nobs: int) -> tuple[np.ndarray, np.ndarray]:
    train_index = _index_positions(train, name="train", nobs=nobs)
    test_index = _index_positions(test, name="test", nobs=nobs)
    if train_index.size == 0 or test_index.size == 0:
        raise ValueError("Every fold must contain non-empty train and test samples.")
    if (
        np.any(train_index < 0)
        or np.any(test_index < 0)
        or np.any(train_index >= nobs)
        or np.any(test_index >= nobs)
    ):
        raise IndexError("Splitter returned indices outside the available rows.")
    if np.intersect1d(train_index, test_index).size:
        raise ValueError("Train and test indices overlap.")
    return train_index, test_index


def _panel_target(
    requested: str,
    result: Any,
    entity_train: Any | None,
    entity_test: Any | None,
    time_train: Any | None,
    time_test: Any | None,
) -> str:
    target = str(requested).strip().lower()
    if target != "auto":
        return target
    if entity_train is None or entity_test is None:
        return "pooled"
    train_labels = set(pd.unique(np.asarray(entity_train)))
    test_labels = set(pd.unique(np.asarray(entity_test)))
    if train_labels.isdisjoint(test_labels):
        return "new_entity"
    if test_labels.issubset(train_labels) and hasattr(result, "posterior_predict_proba"):
        if time_train is None or time_test is None:
            raise ValueError(
                "prediction_target='auto' cannot infer a leakage-safe known-entity "
                "forecast without time values; supply time or choose an explicit target."
            )
        return "known_entity_future"
    return "pooled"


def _validate_panel_target(
    target: str,
    entity_train: Any | None,
    entity_test: Any | None,
    time_train: Any | None,
    time_test: Any | None,
) -> None:
    if target not in {"new_entity", "known_entity_future"}:
        return
    if entity_train is None or entity_test is None:
        raise ValueError(f"prediction_target='{target}' requires entity labels.")
    train_labels = set(pd.unique(np.asarray(entity_train)))
    test_labels = set(pd.unique(np.asarray(entity_test)))
    if target == "new_entity" and not train_labels.isdisjoint(test_labels):
        raise ValueError("new_entity evaluation requires complete held-out entities.")
    if target == "known_entity_future" and not test_labels.issubset(train_labels):
        raise ValueError("known_entity_future requires every test entity in the training fold.")
    if target != "known_entity_future":
        return
    if time_train is None or time_test is None:
        raise ValueError("known_entity_future requires time values for leakage checks.")

    train_entities = np.asarray(entity_train)
    test_entities = np.asarray(entity_test)
    train_times = np.asarray(time_train)
    test_times = np.asarray(time_test)
    if np.any(pd.isna(train_times)) or np.any(pd.isna(test_times)):
        raise ValueError("known_entity_future time values cannot be missing.")
    for label in pd.unique(test_entities):
        entity_train_times = train_times[train_entities == label]
        entity_test_times = test_times[test_entities == label]
        try:
            latest_train = max(entity_train_times.tolist())
            earliest_test = min(entity_test_times.tolist())
            ordered = bool(latest_train < earliest_test)
        except (TypeError, ValueError) as exc:
            raise ValueError("known_entity_future time values must be mutually orderable.") from exc
        if not ordered:
            raise ValueError(
                "known_entity_future requires all training observations to precede all "
                f"test observations within entity {label!r}."
            )


def _requires_lagged_outcome(result: Any) -> bool:
    method = getattr(result, "posterior_predict_proba", None)
    if method is None:
        return False
    try:
        return "lagged_y" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _validate_one_step_lagged_outcome(
    result: Any,
    *,
    y_train: Any,
    entity_train: Any,
    entity_test: Any,
    time_train: Any,
    time_test: Any,
    predict_context: Mapping[str, Any],
    time_step: Any,
) -> None:
    """Ensure dynamic posterior CV uses only the final observed training state."""
    if not _requires_lagged_outcome(result):
        return
    if "lagged_y" not in predict_context:
        raise ValueError(
            "Dynamic known-entity forecasting requires row-aligned lagged_y in "
            "predict_context."
        )
    overridden_history = {
        name
        for name in ("initial_y", "initial_covariates", "entity_means")
        if name in predict_context
    }
    if overridden_history:
        names = ", ".join(sorted(overridden_history))
        raise ValueError(
            "Known-entity dynamic validation uses initial conditions and entity means "
            f"stored from each training fit; do not override: {names}."
        )

    train_entities = np.asarray(entity_train)
    test_entities = np.asarray(entity_test)
    train_times = np.asarray(time_train)
    test_times = np.asarray(time_test)
    training_outcomes = np.asarray(y_train)
    test_lags = np.asarray(predict_context["lagged_y"])
    if test_lags.ndim != 1 or len(test_lags) != len(test_entities):
        raise ValueError("lagged_y must contain one value per test observation.")

    for label in pd.unique(test_entities):
        train_rows = np.flatnonzero(train_entities == label)
        test_rows = np.flatnonzero(test_entities == label)
        if len(test_rows) != 1:
            raise ValueError(
                "Default dynamic posterior validation supports one-step test windows only; "
                "use a custom predict callback for recursive multi-step forecasts."
            )
        entity_train_times = train_times[train_rows]
        latest_time = max(entity_train_times.tolist())
        latest_rows = train_rows[entity_train_times == latest_time]
        if len(latest_rows) != 1:
            raise ValueError("Dynamic training histories require one row per entity-time pair.")
        test_row = int(test_rows[0])
        try:
            difference = test_times[test_row] - latest_time
            if isinstance(difference, (float, np.floating)) or isinstance(
                time_step, (float, np.floating)
            ):
                adjacent = bool(
                    np.isclose(
                        float(difference),
                        float(time_step),
                        rtol=0.0,
                        atol=1e-10,
                    )
                )
            else:
                adjacent = bool(difference == time_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("Dynamic time values are incompatible with time_step.") from exc
        if not adjacent:
            raise ValueError(
                "Dynamic known-entity forecasting requires the test row to immediately "
                "follow training history; embargoed observed lags are not used automatically."
            )
        latest_outcome = training_outcomes[int(latest_rows[0])]
        if pd.isna(latest_outcome) or pd.isna(test_lags[test_row]):
            raise ValueError("Dynamic lagged outcomes cannot be missing.")
        if not bool(test_lags[test_row] == latest_outcome):
            raise ValueError(
                "Each dynamic test lag must equal that entity's last observed training "
                "outcome; future or embargoed outcomes are not accepted."
            )


def _custom_fit(
    callback: FitCallback,
    model: Any,
    X_train: Any,
    y_train: Any,
    context: Mapping[str, Any],
    metadata_context: Mapping[str, Any],
    fit_kwargs: Mapping[str, Any],
) -> Any:
    user_keywords = {**dict(context), **dict(fit_kwargs)}
    keywords = accepted_kwargs(callback, metadata_context)
    keywords.update(
        validated_kwargs(
            callback,
            user_keywords,
            label="fit_context/fit_kwargs",
        )
    )
    return callback(model, X_train, y_train, **keywords)


def _custom_predict(
    callback: PredictCallback,
    result: Any,
    X_test: Any,
    context: Mapping[str, Any],
    metadata_context: Mapping[str, Any],
    predict_kwargs: Mapping[str, Any],
) -> Any:
    user_keywords = {**dict(context), **dict(predict_kwargs)}
    keywords = accepted_kwargs(callback, metadata_context)
    keywords.update(
        validated_kwargs(
            callback,
            user_keywords,
            label="predict_context/predict_kwargs",
        )
    )
    return callback(result, X_test, **keywords)


def _fit_fold_transformer(
    transformer_factory: TransformerFactory,
    X_train: Any,
    y_train: Any,
    X_test: Any,
) -> tuple[Any, Any, Any]:
    """Fit one fresh transformer on training rows and transform both partitions."""
    if not callable(transformer_factory):
        raise TypeError("transformer_factory must be callable.")
    transformer = transformer_factory()
    fit_method = getattr(transformer, "fit", None)
    transform_method = getattr(transformer, "transform", None)
    if not callable(fit_method) or not callable(transform_method):
        raise TypeError(
            "transformer_factory must return an object exposing fit() and transform()."
        )

    try:
        signature = inspect.signature(fit_method)
    except (TypeError, ValueError):
        signature = None
    accepts_y = signature is None or any(
        parameter.kind
        == inspect.Parameter.VAR_POSITIONAL
        or name == "y"
        for name, parameter in (signature.parameters.items() if signature else ())
    )
    if accepts_y:
        fit_method(X_train, y_train)
    else:
        fit_method(X_train)
    transformed_train = transform_method(X_train)
    transformed_test = transform_method(X_test)
    if _nobs(transformed_train, name="transformed X_train") != _nobs(
        X_train, name="X_train"
    ):
        raise ValueError("A fold transformer must preserve the number of training rows.")
    if _nobs(transformed_test, name="transformed X_test") != _nobs(
        X_test, name="X_test"
    ):
        raise ValueError("A fold transformer must preserve the number of test rows.")
    return transformed_train, transformed_test, transformer


def cross_validate(
    model_factory: Callable[[], Any],
    X: Any,
    y: Any,
    *,
    splitter: Any,
    outcome: str = "auto",
    entity: Any | None = None,
    time: Any | None = None,
    prediction_target: str = "auto",
    fit: FitCallback | None = None,
    predict: PredictCallback | None = None,
    transformer_factory: TransformerFactory | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,
    fit_context: Mapping[str, Any] | None = None,
    predict_context: Mapping[str, Any] | None = None,
    score_context: Mapping[str, Any] | None = None,
    split_y: Any | None = None,
    require_converged: bool = True,
    require_inference_valid: bool = True,
    continue_on_error: bool = False,
    model_name: str | None = None,
) -> CrossValidationResult:
    """Fit and score a limited-outcome estimator across leakage-safe folds.

    Row-aligned values placed in ``fit_context``, ``predict_context``, or
    ``score_context`` are sliced automatically.  Custom callbacks remain available for
    multi-equation estimators whose fit or prediction contracts cannot be inferred.
    A ``transformer_factory`` must return a fresh ``fit``/``transform`` object for each
    fold; it is fitted only on training rows before either partition is transformed.
    """
    if not callable(model_factory):
        raise TypeError("model_factory must be callable and return an unfitted model.")
    nobs = _nobs(X, name="X")
    if _nobs(y, name="y") != nobs:
        raise ValueError("X and y must contain the same number of rows.")
    for name, values in (("entity", entity), ("time", time)):
        if values is not None and _nobs(values, name=name) != nobs:
            raise ValueError(f"{name} must contain one value per row.")
    if entity is not None and np.any(pd.isna(np.asarray(entity))):
        raise ValueError("entity cannot contain missing values.")
    if split_y is not None and _nobs(split_y, name="split_y") != nobs:
        raise ValueError("split_y must contain one value per row.")

    evaluations: list[FoldEvaluation] = []
    splitting_target = y if split_y is None else split_y
    split_iterator = _split_iterator(
        splitter, X, splitting_target, entity=entity, time=time
    )
    for fold_number, (train, test) in enumerate(split_iterator, start=1):
        train_index, test_index = _validated_indices(train, test, nobs=nobs)
        X_train, X_test = take_rows(X, train_index), take_rows(X, test_index)
        y_train, y_test = take_rows(y, train_index), take_rows(y, test_index)
        entity_train = take_rows(entity, train_index) if entity is not None else None
        entity_test = take_rows(entity, test_index) if entity is not None else None
        time_train = take_rows(time, train_index) if time is not None else None
        time_test = take_rows(time, test_index) if time is not None else None
        fold_outcome = str(outcome)
        fold_target = str(prediction_target)
        fold_transformer = None

        try:
            if transformer_factory is not None:
                X_train, X_test, fold_transformer = _fit_fold_transformer(
                    transformer_factory,
                    X_train,
                    y_train,
                    X_test,
                )
            model = model_factory()
            if fit is None and "intervalregression" in type(model).__name__.lower():
                raise ValueError(
                    "Interval regression has no observed point target for default CV. "
                    "Use a custom fit callback with an explicit evaluation target, then "
                    "set outcome='continuous', or use a purpose-built interval scorer."
                )
            train_context = subset_context(fit_context, train_index, nobs=nobs)
            train_metadata: dict[str, Any] = {}
            if entity_train is not None:
                train_metadata["entity"] = entity_train
            if time_train is not None:
                train_metadata["time"] = time_train

            if fit is None:
                result = default_fit(
                    model,
                    X_train,
                    y_train,
                    context=train_context,
                    metadata_context=train_metadata,
                    fit_kwargs=fit_kwargs,
                )
            else:
                result = _custom_fit(
                    fit,
                    model,
                    X_train,
                    y_train,
                    train_context,
                    train_metadata,
                    dict(fit_kwargs or {}),
                )

            fold_outcome = normalize_outcome(outcome, result=result, y=y_train)
            is_interval_result = "intervalregression" in type(result).__name__.lower()
            if is_interval_result and fit is None:
                raise ValueError(
                    "Interval regression has no observed point target for default CV. "
                    "Use a custom fit callback with an explicit evaluation target, then "
                    "set outcome='continuous', or use a purpose-built interval scorer."
                )
            if fold_outcome == "interval":
                raise ValueError(
                    "Interval regression requires an explicit evaluation target and "
                    "outcome rather than outcome='auto'."
                )
            fold_target = _panel_target(
                prediction_target,
                result,
                entity_train,
                entity_test,
                time_train,
                time_test,
            )
            _validate_panel_target(
                fold_target,
                entity_train,
                entity_test,
                time_train,
                time_test,
            )
            if fold_target == "conditional" and predict is None:
                raise ValueError(
                    "Conditional cross-validation requires a custom predict callback "
                    "that derives random effects from training-fold data or uses effects "
                    "known independently of held-out outcomes."
                )

            test_predict_context = subset_context(predict_context, test_index, nobs=nobs)

            if predict is None and fold_target == "known_entity_future":
                configured_step = getattr(
                    splitter,
                    "time_step",
                    dict(fit_kwargs or {}).get("time_step", 1),
                )
                _validate_one_step_lagged_outcome(
                    result,
                    y_train=y_train,
                    entity_train=entity_train,
                    entity_test=entity_test,
                    time_train=time_train,
                    time_test=time_test,
                    predict_context=test_predict_context,
                    time_step=configured_step,
                )

            if predict is None:
                prediction = default_predict(
                    result,
                    X_test,
                    outcome=fold_outcome,
                    prediction_target=fold_target,
                    X_train=X_train,
                    y_train=y_train,
                    entity_train=entity_train,
                    entity_test=entity_test,
                    time_test=time_test,
                    predict_context=test_predict_context,
                    predict_kwargs=predict_kwargs,
                )
            else:
                callback_metadata = {
                    "X_train": X_train,
                    "y_train": y_train,
                    "entity_train": entity_train,
                    "entity_test": entity_test,
                    "time_train": time_train,
                    "time_test": time_test,
                    "prediction_target": fold_target,
                }
                prediction = _custom_predict(
                    predict,
                    result,
                    X_test,
                    test_predict_context,
                    callback_metadata,
                    dict(predict_kwargs or {}),
                )

            test_score_context = subset_context(score_context, test_index, nobs=nobs)
            if fold_outcome == "quantile":
                fitted_quantile = getattr(result, "quantile", None)
                requested_quantile = test_score_context.get("quantile")
                if requested_quantile is None and fitted_quantile is not None:
                    test_score_context["quantile"] = fitted_quantile
                elif fitted_quantile is not None and not np.isclose(
                    float(requested_quantile),
                    float(fitted_quantile),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "score_context quantile must match the fitted result quantile."
                    )
            metrics = score_predictions(
                y_test,
                prediction,
                outcome=fold_outcome,
                **test_score_context,
            )
            decision = result_eligibility(
                result,
                require_converged=require_converged,
                require_inference_valid=require_inference_valid,
            )
            evaluations.append(
                FoldEvaluation(
                    fold=fold_number,
                    train_index=train_index,
                    test_index=test_index,
                    outcome=fold_outcome,
                    prediction_target=fold_target,
                    metrics=metrics,
                    eligible=decision.eligible,
                    eligibility_reasons=decision.reasons,
                    prediction=prediction,
                    result=result,
                    transformer=fold_transformer,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            evaluations.append(
                FoldEvaluation(
                    fold=fold_number,
                    train_index=train_index,
                    test_index=test_index,
                    outcome=fold_outcome,
                    prediction_target=fold_target,
                    eligibility_reasons=("fold failed",),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if not evaluations:
        raise ValueError("splitter did not produce any folds.")
    resolved_outcome = next(
        (fold.outcome for fold in evaluations if fold.outcome != "auto"), str(outcome)
    )
    return CrossValidationResult(
        folds=tuple(evaluations),
        outcome=resolved_outcome,
        model_name=model_name or getattr(model_factory, "__name__", "model"),
        row_labels=_row_labels(X, nobs),
    )
