"""Result adaptation for the experimental limited-outcome ML workflow layer."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_OUTCOME_ALIASES = {
    "categorical": "multiclass",
    "censored": "continuous",
    "regression": "continuous",
}


@dataclass(frozen=True)
class ResultEligibility:
    """Eligibility decision used when ranking cross-validated estimators."""

    eligible: bool
    reasons: tuple[str, ...]


def take_rows(values: Any, indices: np.ndarray) -> Any:
    """Select rows without discarding pandas labels or column metadata."""
    positions = np.asarray(indices, dtype=int)
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.iloc[positions]
    if isinstance(values, pd.Index):
        return values.take(positions)
    array = np.asarray(values)
    if array.ndim == 0:
        return values
    return array[positions]


def subset_context(
    context: Mapping[str, Any] | None,
    indices: np.ndarray,
    *,
    nobs: int,
) -> dict[str, Any]:
    """Slice row-aligned context values while retaining scalar configuration."""
    output: dict[str, Any] = {}
    for name, value in dict(context or {}).items():
        if isinstance(value, (str, bytes)) or value is None or np.isscalar(value):
            output[name] = value
            continue
        try:
            length = len(value)
        except TypeError:
            output[name] = value
            continue
        output[name] = take_rows(value, indices) if length == nobs else value
    return output


def _accepts_keyword(function: Any, name: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return True
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return True
    parameter = signature.parameters.get(name)
    return parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY


def accepted_kwargs(function: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    """Return keyword arguments accepted by a callable's visible signature."""
    return {name: value for name, value in values.items() if _accepts_keyword(function, name)}


def validated_kwargs(
    function: Any,
    values: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Reject misspelled user context instead of silently changing a model specification."""
    unsupported = [name for name in values if not _accepts_keyword(function, name)]
    if unsupported:
        names = ", ".join(repr(name) for name in unsupported)
        raise TypeError(f"{label} contains unsupported keyword(s): {names}.")
    return dict(values)


def infer_outcome(result: Any, y: Any | None = None) -> str:
    """Infer the scoring family conservatively from the fitted result contract."""
    class_name = type(result).__name__.lower()
    if "censoredquantile" in class_name:
        return "quantile"
    if "duration" in class_name:
        return "duration"
    if "sampleselection" in class_name:
        return "selection"
    if "conditional" in class_name:
        return "choice"
    if any(token in class_name for token in ("poisson", "negativebinomial")):
        return "count"
    if "intervalregression" in class_name:
        return "interval"
    if any(token in class_name for token in ("tobit", "truncated")):
        return "continuous"
    if "binary" in class_name:
        return "binary"
    if any(
        token in class_name
        for token in ("ordered", "ordinal", "sequential", "partialproportional")
    ):
        return "ordinal"
    if "multinomial" in class_name:
        return "multiclass"

    if hasattr(result, "predict_proba") and y is not None:
        unique = pd.unique(np.asarray(y))
        return "binary" if len(unique) == 2 else "multiclass"
    return "continuous"


def normalize_outcome(outcome: str, *, result: Any | None = None, y: Any | None = None) -> str:
    """Normalize an explicit outcome name or infer it from a result."""
    normalized = str(outcome).strip().lower().replace("-", "_")
    if normalized == "auto":
        if result is None:
            raise ValueError("outcome='auto' requires a fitted result.")
        return infer_outcome(result, y)
    return _OUTCOME_ALIASES.get(normalized, normalized)


def result_eligibility(
    result: Any,
    *,
    require_converged: bool = True,
    require_inference_valid: bool = True,
    crossing_tolerance: float = 1e-10,
) -> ResultEligibility:
    """Apply econometric validity gates before a model can be ranked."""
    reasons: list[str] = []

    def require_true(attribute: str, failure: str) -> None:
        try:
            value = getattr(result, attribute)
        except (AttributeError, TypeError, ValueError):
            reasons.append(f"{attribute} diagnostic is unavailable")
            return
        if not isinstance(value, (bool, np.bool_)):
            reasons.append(f"{attribute} diagnostic is not a scalar boolean")
            return
        valid = bool(value)
        if not valid:
            reasons.append(failure)

    if require_converged:
        require_true("converged", "optimizer did not converge")
    if require_inference_valid:
        require_true(
            "inference_valid",
            "ordinary inference is not valid for the fitted solution",
        )

    minimum_gap = getattr(result, "minimum_index_gap", None)
    if minimum_gap is not None:
        try:
            numeric_gap = float(minimum_gap)
            if not np.isfinite(numeric_gap):
                reasons.append("non-crossing diagnostic is not finite")
            elif numeric_gap < -crossing_tolerance:
                reasons.append("predicted cumulative probabilities cross on estimation support")
        except (TypeError, ValueError):
            reasons.append("non-crossing diagnostic is not numeric")

    return ResultEligibility(eligible=not reasons, reasons=tuple(reasons))


def default_fit(
    model: Any,
    X: Any,
    y: Any,
    *,
    context: Mapping[str, Any] | None = None,
    metadata_context: Mapping[str, Any] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Fit a regular estimator, filtering metadata but validating user keywords."""
    if not hasattr(model, "fit"):
        raise TypeError("model_factory must return an object with a fit() method.")
    user_keywords = {**dict(context or {}), **dict(fit_kwargs or {})}
    keywords = accepted_kwargs(model.fit, dict(metadata_context or {}))
    keywords.update(
        validated_kwargs(
            model.fit,
            user_keywords,
            label="fit_context/fit_kwargs",
        )
    )
    return model.fit(X, y, **keywords)


def _posterior_prediction(
    result: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    *,
    entity_train: Any,
    entity_test: Any,
    predict_context: Mapping[str, Any],
) -> Any:
    if not hasattr(result, "posterior_random_effects") or not hasattr(
        result, "posterior_predict_proba"
    ):
        raise ValueError(
            "prediction_target='known_entity_future' requires posterior prediction support."
        )

    posterior_method = result.posterior_random_effects
    signature = inspect.signature(posterior_method)
    required_names = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    if required_names:
        posterior = posterior_method(X_train, y_train, entity=entity_train)
    else:
        posterior = posterior_method()

    method = result.posterior_predict_proba
    keywords = accepted_kwargs(
        method,
        {"entity": entity_test, "posterior": posterior},
    )
    keywords.update(
        validated_kwargs(
            method,
            predict_context,
            label="predict_context/predict_kwargs",
        )
    )
    return method(X_test, **keywords)


def default_predict(
    result: Any,
    X_test: Any,
    *,
    outcome: str,
    prediction_target: str,
    X_train: Any | None = None,
    y_train: Any | None = None,
    entity_train: Any | None = None,
    entity_test: Any | None = None,
    time_test: Any | None = None,
    predict_context: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Generate outcome-appropriate predictions without silently changing estimands."""
    context = {**dict(predict_context or {}), **dict(predict_kwargs or {})}
    metadata = {"entity": entity_test, "time": time_test}
    metadata = {name: value for name, value in metadata.items() if value is not None}
    normalized_target = str(prediction_target).strip().lower()

    if normalized_target == "known_entity_future":
        if X_train is None or y_train is None or entity_train is None or entity_test is None:
            raise ValueError("Known-entity posterior prediction requires training and entity data.")
        return _posterior_prediction(
            result,
            X_train,
            y_train,
            X_test,
            entity_train=entity_train,
            entity_test=entity_test,
            predict_context=context,
        )

    if normalized_target not in {"auto", "pooled", "new_entity", "conditional"}:
        raise ValueError(
            "prediction_target must be 'auto', 'pooled', 'new_entity', "
            "'known_entity_future', or 'conditional'."
        )

    if normalized_target == "conditional" and "random_effects" not in context:
        raise ValueError(
            "prediction_target='conditional' requires explicit random_effects in "
            "predict_context."
        )
    if normalized_target in {"pooled", "new_entity"} and "random_effects" in context:
        raise ValueError(
            f"prediction_target='{normalized_target}' cannot use conditional random_effects; "
            "choose prediction_target='conditional'."
        )

    if outcome == "selection":
        if "Z" not in context:
            raise ValueError("Selection prediction requires row-aligned Z in predict_context.")
        Z = context.pop("Z")
        if context:
            names = ", ".join(repr(name) for name in context)
            raise TypeError(f"Selection prediction received unsupported keyword(s): {names}.")
        required = ("predict_selection", "predict", "predict_observed")
        missing = [name for name in required if not hasattr(result, name)]
        if missing:
            raise TypeError(f"Selection result is missing prediction methods: {missing}.")
        return {
            "selection_probability": result.predict_selection(Z),
            "outcome": result.predict(X_test),
            "observed_outcome": result.predict_observed(X_test, Z),
        }

    if outcome == "count":
        if not hasattr(result, "predict"):
            raise TypeError(f"{type(result).__name__} does not expose predict().")
        mean_keywords = accepted_kwargs(result.predict, metadata)
        mean_keywords.update(
            validated_kwargs(
                result.predict,
                context,
                label="predict_context/predict_kwargs",
            )
        )
        mean = result.predict(X_test, **mean_keywords)
        prediction: dict[str, Any] = {"mean": mean}
        if hasattr(result, "predict_zero_probability"):
            zero_method = result.predict_zero_probability
            zero_keywords = accepted_kwargs(zero_method, metadata)
            zero_keywords.update(
                validated_kwargs(
                    zero_method,
                    context,
                    label="predict_context/predict_kwargs",
                )
            )
            prediction["zero_probability"] = zero_method(X_test, **zero_keywords)
        return prediction

    if outcome == "duration":
        if not hasattr(result, "predict"):
            raise TypeError(f"{type(result).__name__} does not expose predict().")
        keywords = accepted_kwargs(result.predict, metadata)
        keywords.update(
            validated_kwargs(
                result.predict,
                context,
                label="predict_context/predict_kwargs",
            )
        )
        return {"expected_duration": result.predict(X_test, **keywords)}

    if outcome == "choice":
        if "groups" not in context:
            raise ValueError(
                "Grouped-choice prediction requires row-aligned groups in predict_context."
            )
        if not hasattr(result, "predict_proba"):
            raise TypeError(f"{type(result).__name__} does not expose predict_proba().")
        method = result.predict_proba
        keywords = accepted_kwargs(method, metadata)
        keywords.update(
            validated_kwargs(
                method,
                context,
                label="predict_context/predict_kwargs",
            )
        )
        return method(X_test, **keywords)

    if outcome in {"binary", "multiclass", "ordinal"}:
        if not hasattr(result, "predict_proba"):
            raise TypeError(f"{type(result).__name__} does not expose predict_proba().")
        method = result.predict_proba
    else:
        if not hasattr(result, "predict"):
            raise TypeError(f"{type(result).__name__} does not expose predict().")
        method = result.predict

    keywords = accepted_kwargs(method, metadata)
    keywords.update(
        validated_kwargs(
            method,
            context,
            label="predict_context/predict_kwargs",
        )
    )
    return method(X_test, **keywords)


def prediction_frame(prediction: Any, *, fold: int, row_index: Any) -> pd.DataFrame:
    """Convert scalar, probability, or mapping predictions to an OOF table."""
    index_values = np.asarray(row_index)
    if isinstance(prediction, pd.DataFrame):
        frame = prediction.reset_index(drop=True).copy()
        frame.columns = [f"prediction_{column}" for column in frame.columns]
    elif isinstance(prediction, pd.Series):
        frame = prediction.reset_index(drop=True).to_frame("prediction")
    elif isinstance(prediction, Mapping):
        columns: dict[str, Any] = {}
        for name, value in prediction.items():
            array = np.asarray(value)
            if array.ndim == 1 and len(array) == len(index_values):
                columns[f"prediction_{name}"] = array
        if not columns:
            return pd.DataFrame({"fold": fold, "row_index": index_values})
        frame = pd.DataFrame(columns)
    else:
        array = np.asarray(prediction)
        if array.ndim == 1:
            frame = pd.DataFrame({"prediction": array})
        elif array.ndim == 2:
            frame = pd.DataFrame(
                array,
                columns=[f"prediction_{column}" for column in range(array.shape[1])],
            )
        else:
            return pd.DataFrame({"fold": fold, "row_index": index_values})

    if len(frame) != len(index_values):
        raise ValueError("Prediction output must contain one row per test observation.")
    frame.insert(0, "row_index", index_values)
    frame.insert(0, "fold", fold)
    return frame
