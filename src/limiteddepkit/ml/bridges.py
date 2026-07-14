"""Lazy adapters for external estimators and prediction protocols.

The bridge layer does not import optional libraries at module import time.
Named integrations are intentionally narrow: scikit-learn and statsmodels
have stable generic contracts, while packages such as Biogeme, lifelines and
Bambi require model-specific user callbacks through ``GenericEstimatorBridge``.
This avoids claiming coverage for incompatible estimands or prediction shapes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "BridgeProtocol",
    "BridgedResult",
    "GenericEstimatorBridge",
    "PredictProbabilityProtocol",
    "PredictProtocol",
    "ProbabilityBridgedResult",
    "SklearnBridge",
    "StatsmodelsBridge",
    "generic_bridge",
    "sklearn_bridge",
    "statsmodels_bridge",
]

PredictionKind = Literal["probability", "value"]


@runtime_checkable
class PredictProtocol(Protocol):
    """Structural protocol for a fitted point-prediction object."""

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Predict one result per supplied row."""


@runtime_checkable
class PredictProbabilityProtocol(Protocol):
    """Structural protocol for a fitted probability-prediction object."""

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        """Predict probabilities for supplied rows."""


@runtime_checkable
class BridgeProtocol(Protocol):
    """Structural protocol for a fold-safe external estimator bridge."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> BridgedResult:
        """Fit and return a result exposing a normalized prediction method."""


@dataclass(frozen=True)
class BridgedResult:
    """External fitted object with normalized point prediction.

    Attributes not defined by the wrapper are delegated to ``raw_result``.
    Econometric diagnostics are never invented: ``converged`` and
    ``inference_valid`` are available only when the external result or an
    explicit diagnostics callback supplies them.
    """

    raw_result: Any
    _prediction_function: Callable[..., Any]
    bridge_name: str
    _diagnostics: Mapping[str, bool] = field(default_factory=dict)

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Return bridge-normalized predictions."""

        return self._prediction_function(self.raw_result, X, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in self._diagnostics:
            return self._diagnostics[name]
        return getattr(self.raw_result, name)


@dataclass(frozen=True)
class ProbabilityBridgedResult(BridgedResult):
    """External fitted object exposing normalized probability prediction."""

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        """Return bridge-normalized probabilities."""

        return self._prediction_function(self.raw_result, X, **kwargs)


def _require_dependency(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Optional dependency {module!r} is required only when this bridge is used."
        ) from exc


def _resolve_symbol(value: Any, *, dependency: str) -> Callable[..., Any]:
    if callable(value):
        return value
    if not isinstance(value, str) or ":" not in value:
        raise TypeError("A lazy factory must be callable or use 'module:attribute' syntax.")
    module_name, attribute_path = value.split(":", 1)
    if module_name != dependency and not module_name.startswith(f"{dependency}."):
        raise ValueError(f"Lazy factory {value!r} is outside dependency {dependency!r}.")
    resolved: Any = importlib.import_module(module_name)
    for name in attribute_path.split("."):
        resolved = getattr(resolved, name)
    if not callable(resolved):
        raise TypeError(f"Resolved factory {value!r} is not callable.")
    return resolved


def _call_prediction(
    result: Any,
    X: Any,
    *,
    method: str,
    entity: Any = None,
    time: Any = None,
    **kwargs: Any,
) -> Any:
    del entity, time
    prediction = getattr(result, method)(X, **kwargs)
    if method == "predict_proba" and hasattr(result, "classes_"):
        values = np.asarray(prediction)
        classes = np.asarray(result.classes_)
        if values.ndim == 2 and values.shape[1] == len(classes):
            index = X.index.copy() if isinstance(X, pd.DataFrame) else None
            return pd.DataFrame(values, columns=classes, index=index)
    return prediction


def _validated_diagnostics(
    result: Any,
    diagnostics: Callable[[Any], Mapping[str, bool]] | None,
) -> dict[str, bool]:
    if diagnostics is None:
        return {}
    values = dict(diagnostics(result))
    unsupported = set(values) - {"converged", "inference_valid"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported bridge diagnostics: {names}.")
    for name, value in values.items():
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Bridge diagnostic {name!r} must be Boolean.")
    return values


@dataclass(frozen=True)
class GenericEstimatorBridge:
    """Callback adapter for an arbitrary optional external estimator.

    ``fit_function`` receives ``(X, y, **kwargs)``. ``predict_function``
    receives ``(fitted_result, X, **kwargs)``. Supplying ``dependency`` makes
    availability validation lazy, which is the recommended route for
    Biogeme, lifelines, Bambi and other model-specific ecosystems.
    """

    fit_function: Callable[..., Any]
    predict_function: Callable[..., Any]
    probability_output: bool = False
    dependency: str | None = None
    name: str = "generic"
    fit_options: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Callable[[Any], Mapping[str, bool]] | None = None

    def fit(self, X: Any, y: Any, **kwargs: Any) -> BridgedResult:
        """Fit through the supplied callback and normalize its prediction API."""

        if self.dependency is not None:
            _require_dependency(self.dependency)
        options = {**dict(self.fit_options), **kwargs}
        raw_result = self.fit_function(X, y, **options)
        diagnostics = _validated_diagnostics(raw_result, self.diagnostics)
        result_type = ProbabilityBridgedResult if self.probability_output else BridgedResult
        return result_type(
            raw_result=raw_result,
            _prediction_function=self.predict_function,
            bridge_name=self.name,
            _diagnostics=diagnostics,
        )


def generic_bridge(
    fit_function: Callable[..., Any],
    predict_function: Callable[..., Any],
    *,
    probability_output: bool = False,
    dependency: str | None = None,
    name: str = "generic",
    fit_options: Mapping[str, Any] | None = None,
    diagnostics: Callable[[Any], Mapping[str, bool]] | None = None,
) -> GenericEstimatorBridge:
    """Construct a callback bridge without importing its optional dependency."""

    return GenericEstimatorBridge(
        fit_function=fit_function,
        predict_function=predict_function,
        probability_output=probability_output,
        dependency=dependency,
        name=name,
        fit_options=dict(fit_options or {}),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class SklearnBridge:
    """Fold-safe lazy adapter for a scikit-learn estimator instance."""

    estimator: Any
    prediction_method: str = "auto"
    clone_estimator: bool = True
    fit_options: Mapping[str, Any] = field(default_factory=dict)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any = None,
        time: Any = None,
        **kwargs: Any,
    ) -> BridgedResult:
        """Clone, fit and expose the selected scikit-learn prediction method."""

        del entity, time

        method = self.prediction_method
        if method not in {"auto", "predict", "predict_proba", "decision_function"}:
            raise ValueError(
                "prediction_method must be 'auto', 'predict', 'predict_proba', "
                "or 'decision_function'."
            )
        if method == "auto":
            method = (
                "predict_proba" if hasattr(self.estimator, "predict_proba") else "predict"
            )
        if not hasattr(self.estimator, method):
            raise TypeError(f"The estimator does not expose {method}().")
        sklearn_base = _require_dependency("sklearn.base")
        estimator = sklearn_base.clone(self.estimator) if self.clone_estimator else self.estimator
        estimator.fit(X, y, **{**dict(self.fit_options), **kwargs})
        result_type = ProbabilityBridgedResult if method == "predict_proba" else BridgedResult
        return result_type(
            raw_result=estimator,
            _prediction_function=partial(_call_prediction, method=method),
            bridge_name="sklearn",
        )


def sklearn_bridge(
    estimator: Any,
    *,
    prediction_method: str = "auto",
    clone_estimator: bool = True,
    fit_options: Mapping[str, Any] | None = None,
) -> SklearnBridge:
    """Construct a scikit-learn bridge; import scikit-learn only on ``fit``."""

    return SklearnBridge(
        estimator=estimator,
        prediction_method=prediction_method,
        clone_estimator=clone_estimator,
        fit_options=dict(fit_options or {}),
    )


def _statsmodels_prediction(
    result: Any,
    X: Any,
    *,
    method: str,
    add_constant: bool,
    has_constant: str,
    entity: Any = None,
    time: Any = None,
    **kwargs: Any,
) -> Any:
    del entity, time
    design = X
    if add_constant:
        statsmodels_api = _require_dependency("statsmodels.api")
        design = statsmodels_api.add_constant(X, has_constant=has_constant)
    return getattr(result, method)(design, **kwargs)


@dataclass(frozen=True)
class StatsmodelsBridge:
    """Lazy adapter for array-style statsmodels model constructors.

    The model factory is called as ``factory(y, X, **model_options)``. Formula
    APIs and models requiring extra structures should use
    ``GenericEstimatorBridge`` because their data contracts are model-specific.
    """

    model_factory: Callable[..., Any] | str
    prediction_kind: PredictionKind
    prediction_method: str = "predict"
    add_constant: bool = False
    has_constant: Literal["raise", "add", "skip"] = "skip"
    model_options: Mapping[str, Any] = field(default_factory=dict)
    fit_options: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Callable[[Any], Mapping[str, bool]] | None = None

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any = None,
        time: Any = None,
        **kwargs: Any,
    ) -> BridgedResult:
        """Construct and fit a statsmodels model, then normalize prediction."""

        del entity, time

        if self.prediction_kind not in {"probability", "value"}:
            raise ValueError(
                "prediction_kind must be explicitly 'probability' or 'value'."
            )
        if not isinstance(self.prediction_method, str) or not self.prediction_method:
            raise ValueError("prediction_method must be a non-empty string.")
        statsmodels_api = _require_dependency("statsmodels.api")
        factory = _resolve_symbol(self.model_factory, dependency="statsmodels")
        design = (
            statsmodels_api.add_constant(X, has_constant=self.has_constant)
            if self.add_constant
            else X
        )
        model = factory(y, design, **dict(self.model_options))
        if not hasattr(model, self.prediction_method):
            raise TypeError(
                f"The statsmodels model does not expose {self.prediction_method}()."
            )
        raw_result = model.fit(**{**dict(self.fit_options), **kwargs})
        if not hasattr(raw_result, self.prediction_method):
            raise TypeError(
                f"The fitted statsmodels result does not expose {self.prediction_method}()."
            )
        diagnostics = _validated_diagnostics(raw_result, self.diagnostics)
        result_type = (
            ProbabilityBridgedResult
            if self.prediction_kind == "probability"
            else BridgedResult
        )
        return result_type(
            raw_result=raw_result,
            _prediction_function=partial(
                _statsmodels_prediction,
                method=self.prediction_method,
                add_constant=self.add_constant,
                has_constant=self.has_constant,
            ),
            bridge_name="statsmodels",
            _diagnostics=diagnostics,
        )


def statsmodels_bridge(
    model_factory: Callable[..., Any] | str,
    *,
    prediction_kind: PredictionKind,
    prediction_method: str = "predict",
    add_constant: bool = False,
    has_constant: Literal["raise", "add", "skip"] = "skip",
    model_options: Mapping[str, Any] | None = None,
    fit_options: Mapping[str, Any] | None = None,
    diagnostics: Callable[[Any], Mapping[str, bool]] | None = None,
) -> StatsmodelsBridge:
    """Construct a statsmodels bridge; import statsmodels only on ``fit``."""

    return StatsmodelsBridge(
        model_factory=model_factory,
        prediction_kind=prediction_kind,
        prediction_method=prediction_method,
        add_constant=add_constant,
        has_constant=has_constant,
        model_options=dict(model_options or {}),
        fit_options=dict(fit_options or {}),
        diagnostics=diagnostics,
    )
