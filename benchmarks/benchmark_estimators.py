"""Deterministic fit, prediction, and post-estimation performance benchmarks."""

from __future__ import annotations

import argparse
import cProfile
import gc
import importlib.metadata
import io
import json
import os
import platform
import pstats
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, softmax

from limiteddepkit import (
    BinaryLogit,
    BinaryProbit,
    GeneralizedOrderedLogit,
    NegativeBinomial,
    OrderedLogit,
    OrderedProbit,
    PoissonRegressor,
    RandomEffectsOrderedLogit,
    Tobit,
    TruncatedRegression,
    marginal_effects,
    summary_frame,
)
from limiteddepkit.experimental import (
    ConditionalLogit,
    HurdlePoisson,
    MultinomialLogit,
    SampleSelection,
    ZeroInflatedPoisson,
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    estimator: str
    nobs: int


QUICK_CASES = (
    BenchmarkCase("binary_logit", "binary_logit", 4_000),
    BenchmarkCase("binary_probit", "binary_probit", 4_000),
    BenchmarkCase("ordered_logit", "ordered_logit", 1_500),
    BenchmarkCase("ordered_probit", "ordered_probit", 1_500),
    BenchmarkCase("generalized_ordered_logit", "generalized_ordered_logit", 800),
    BenchmarkCase("multinomial_logit", "multinomial_logit", 2_000),
    BenchmarkCase("poisson", "poisson", 10_000),
    BenchmarkCase("negative_binomial", "negative_binomial", 3_000),
    BenchmarkCase("zero_inflated_poisson", "zero_inflated_poisson", 1_000),
    BenchmarkCase("hurdle_poisson", "hurdle_poisson", 1_000),
    BenchmarkCase("tobit", "tobit", 1_500),
    BenchmarkCase("truncated_regression", "truncated_regression", 3_000),
    BenchmarkCase("sample_selection", "sample_selection", 1_000),
    BenchmarkCase("conditional_logit", "conditional_logit", 600),
    BenchmarkCase("random_effects_ordered_logit", "random_effects_ordered_logit", 300),
)
FULL_CASES = (
    BenchmarkCase("binary_logit", "binary_logit", 40_000),
    BenchmarkCase("binary_probit", "binary_probit", 40_000),
    BenchmarkCase("ordered_logit", "ordered_logit", 6_000),
    BenchmarkCase("ordered_probit", "ordered_probit", 6_000),
    BenchmarkCase("generalized_ordered_logit", "generalized_ordered_logit", 3_000),
    BenchmarkCase("multinomial_logit", "multinomial_logit", 12_000),
    BenchmarkCase("poisson", "poisson", 100_000),
    BenchmarkCase("negative_binomial", "negative_binomial", 20_000),
    BenchmarkCase("zero_inflated_poisson", "zero_inflated_poisson", 5_000),
    BenchmarkCase("hurdle_poisson", "hurdle_poisson", 5_000),
    BenchmarkCase("tobit", "tobit", 10_000),
    BenchmarkCase("truncated_regression", "truncated_regression", 20_000),
    BenchmarkCase("sample_selection", "sample_selection", 8_000),
    BenchmarkCase("conditional_logit", "conditional_logit", 4_000),
    BenchmarkCase("random_effects_ordered_logit", "random_effects_ordered_logit", 1_000),
)


def _problem(
    case: BenchmarkCase,
) -> tuple[Any, pd.DataFrame, np.ndarray, dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(20260728)
    if case.estimator.startswith("binary"):
        X = pd.DataFrame(
            {"const": 1.0, "x1": rng.normal(size=case.nobs), "x2": rng.normal(size=case.nobs)}
        )
        linear = X.to_numpy() @ np.array([-0.15, 0.55, -0.35])
        if case.estimator == "binary_logit":
            y = rng.binomial(1, expit(linear))
            return BinaryLogit(), X, y, {}, {}
        y = (linear + rng.normal(size=case.nobs) > 0.0).astype(int)
        return BinaryProbit(), X, y, {}, {}
    if case.estimator.startswith("ordered") or case.estimator == "generalized_ordered_logit":
        X = pd.DataFrame({"x1": rng.normal(size=case.nobs), "x2": rng.normal(size=case.nobs)})
        latent = X.to_numpy() @ np.array([0.65, -0.3]) + rng.logistic(size=case.nobs)
        y = np.digitize(latent, bins=np.array([-0.55, 0.65]))
        if case.estimator == "ordered_logit":
            return OrderedLogit(), X, y, {}, {}
        if case.estimator == "ordered_probit":
            return OrderedProbit(), X, y, {}, {}
        return GeneralizedOrderedLogit(), X, y, {}, {}
    if case.estimator == "multinomial_logit":
        X = pd.DataFrame(
            {"const": 1.0, "x1": rng.normal(size=case.nobs), "x2": rng.normal(size=case.nobs)}
        )
        utilities = np.column_stack(
            [
                np.zeros(case.nobs),
                X.to_numpy() @ np.array([0.25, 0.55, -0.3]),
                X.to_numpy() @ np.array([-0.2, -0.35, 0.45]),
            ]
        )
        probabilities = softmax(utilities, axis=1)
        y = np.array([rng.choice(3, p=row) for row in probabilities])
        return MultinomialLogit(), X, y, {"category_order": [0, 1, 2]}, {}
    if case.estimator in {"tobit", "truncated_regression"}:
        X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=case.nobs)})
        latent = 0.4 + 0.65 * X["x"].to_numpy() + rng.normal(scale=1.1, size=case.nobs)
        if case.estimator == "tobit":
            return Tobit(), X, np.maximum(latent, 0.0), {}, {}
        retained = latent > 0.0
        return (
            TruncatedRegression(),
            X.loc[retained].reset_index(drop=True),
            latent[retained],
            {},
            {},
        )
    if case.estimator == "sample_selection":
        X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=case.nobs)})
        Z = pd.DataFrame({"const": 1.0, "x": X["x"], "z": rng.normal(size=case.nobs)})
        selection_error = rng.normal(size=case.nobs)
        outcome_error = 1.1 * (
            0.35 * selection_error + np.sqrt(1.0 - 0.35**2) * rng.normal(size=case.nobs)
        )
        observed = Z.to_numpy() @ np.array([0.15, 0.2, 0.5]) + selection_error > 0.0
        y = np.full(case.nobs, np.nan)
        y[observed] = X.to_numpy()[observed] @ np.array([0.8, 0.55]) + outcome_error[observed]
        return SampleSelection(), X, y, {"_third": Z}, {"_selection": Z}
    if case.estimator == "conditional_logit":
        n_alternatives = 4
        groups = np.repeat(np.arange(case.nobs), n_alternatives)
        alternatives = np.tile(np.arange(n_alternatives), case.nobs)
        X = pd.DataFrame(
            {"price": rng.normal(size=len(groups)), "quality": rng.normal(size=len(groups))}
        )
        probabilities = softmax(
            (X.to_numpy() @ np.array([-0.7, 0.45])).reshape(case.nobs, n_alternatives),
            axis=1,
        )
        selected = np.array([rng.choice(n_alternatives, p=row) for row in probabilities])
        y = np.zeros(len(groups), dtype=int)
        y[np.arange(case.nobs) * n_alternatives + selected] = 1
        fit_kwargs = {"groups": groups, "alternatives": alternatives}
        prediction_kwargs = {"groups": groups}
        return ConditionalLogit(n_alts=n_alternatives), X, y, fit_kwargs, prediction_kwargs
    if case.estimator == "random_effects_ordered_logit":
        n_periods = 5
        n_entities = max(case.nobs // n_periods, 8)
        entity = np.repeat(np.arange(n_entities), n_periods)
        X = pd.DataFrame({"x1": rng.normal(size=len(entity)), "x2": rng.normal(size=len(entity))})
        random_effect = rng.normal(scale=0.65, size=n_entities)
        latent = (
            X.to_numpy() @ np.array([0.7, -0.4])
            + random_effect[entity]
            + rng.logistic(size=len(entity))
        )
        y = np.digitize(latent, bins=np.array([-0.6, 0.7]))
        return RandomEffectsOrderedLogit(), X, y, {"entity": entity, "quadrature_points": 8}, {}
    X = pd.DataFrame(
        {"const": 1.0, "x1": rng.normal(size=case.nobs), "x2": rng.normal(size=case.nobs)}
    )
    exposure = rng.uniform(0.5, 2.0, size=case.nobs)
    mean = np.exp(X.to_numpy() @ np.array([0.12, 0.3, -0.22]) + np.log(exposure))
    if case.estimator == "negative_binomial":
        alpha = 0.7
        probability = 1.0 / (1.0 + alpha * mean)
        y = rng.negative_binomial(1.0 / alpha, probability)
        return NegativeBinomial(), X, y, {"exposure": exposure}, {}
    if case.estimator in {"zero_inflated_poisson", "hurdle_poisson"}:
        Z = pd.DataFrame({"const": 1.0, "z": rng.normal(size=case.nobs)})
        gate_probability = expit(Z.to_numpy() @ np.array([-0.4, -0.45]))
        if case.estimator == "zero_inflated_poisson":
            y = rng.poisson(mean)
            y[rng.random(case.nobs) < gate_probability] = 0
            return ZeroInflatedPoisson(), X, y, {"X_inflation": Z}, {"X_inflation": Z}
        y = rng.poisson(mean)
        zero = y == 0
        while np.any(zero):
            y[zero] = rng.poisson(mean[zero])
            zero = y == 0
        y = np.where(rng.random(case.nobs) < gate_probability, y, 0)
        return HurdlePoisson(), X, y, {"X_hurdle": Z}, {"X_hurdle": Z}
    y = rng.poisson(mean)
    weights = np.ones(case.nobs, dtype=int)
    weights[::17] = 0
    return PoissonRegressor(), X, y, {"exposure": exposure, "freq_weights": weights}, {}


def _fit(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    fit_kwargs: dict[str, Any],
    engine: str,
) -> Any:
    kwargs = dict(fit_kwargs)
    third = kwargs.pop("_third", None)
    if isinstance(estimator, PoissonRegressor):
        kwargs["engine"] = engine
    if third is not None:
        return estimator.fit(X, y, third, **kwargs)
    return estimator.fit(X, y, **kwargs)


def _seconds(function: Callable[[], Any]) -> tuple[float, Any]:
    gc.collect()
    started = time.perf_counter()
    value = function()
    return time.perf_counter() - started, value


def _post_estimation_supported(result: Any) -> bool:
    if hasattr(result, "summary_frame"):
        return True
    parameters = getattr(result, "all_params", getattr(result, "params", None))
    return isinstance(parameters, pd.Series)


def _post_estimation(result: Any, X: pd.DataFrame) -> None:
    if hasattr(result, "summary_frame"):
        result.summary_frame()
    elif _post_estimation_supported(result):
        summary_frame(result)
    if hasattr(result, "marginal_effects"):
        marginal_effects(result, X)


def _prediction(result: Any, X: pd.DataFrame, prediction_kwargs: dict[str, Any]) -> None:
    kwargs = dict(prediction_kwargs)
    selection = kwargs.pop("_selection", None)
    if selection is not None:
        result.predict(X)
        result.predict_selection(selection)
        result.predict_observed(X, selection)
        return
    if hasattr(result, "predict_proba"):
        result.predict_proba(X, **kwargs)
    else:
        result.predict(X, **kwargs)


def _measure(
    case: BenchmarkCase,
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    fit_kwargs: dict[str, Any],
    prediction_kwargs: dict[str, Any],
    engine: str,
    repetitions: int,
) -> tuple[dict[str, Any], Any]:
    fit = partial(_fit, estimator, X, y, fit_kwargs, engine)
    cold_seconds, result = _seconds(fit)
    warm_seconds: list[float] = []
    for _ in range(repetitions):
        elapsed, result = _seconds(fit)
        warm_seconds.append(elapsed)

    prediction_seconds = [
        _seconds(lambda: _prediction(result, X, prediction_kwargs))[0] for _ in range(repetitions)
    ]
    post_seconds = [_seconds(lambda: _post_estimation(result, X))[0] for _ in range(repetitions)]

    gc.collect()
    tracemalloc.start()
    fit()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    optimizer = getattr(result, "optimizer_result", None)
    parameters = getattr(result, "all_params", getattr(result, "params", pd.Series(dtype=float)))
    return (
        {
            "engine": engine if case.estimator == "poisson" else "native",
            "cold_fit_seconds": cold_seconds,
            "warm_fit_seconds": warm_seconds,
            "warm_fit_median_seconds": statistics.median(warm_seconds),
            "prediction_median_seconds": statistics.median(prediction_seconds),
            "post_estimation_median_seconds": statistics.median(post_seconds),
            "post_estimation_supported": _post_estimation_supported(result),
            "python_peak_bytes": peak_bytes,
            "converged": bool(getattr(result, "converged", True)),
            "inference_valid": bool(getattr(result, "inference_valid", True)),
            "loglike": float(getattr(result, "loglike", np.nan)),
            "iterations": int(getattr(optimizer, "nit", 0)),
            "parameters": {str(key): float(value) for key, value in parameters.items()},
        },
        result,
    )


def _exact_poisson_parity(left: Any, right: Any) -> bool:
    arrays_equal = all(
        np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        for a, b in (
            (left.params, right.params),
            (left.covariance, right.covariance),
            (left.standard_errors, right.standard_errors),
            (left.fitted_values, right.fitted_values),
        )
    )
    scalars_equal = all(
        getattr(left, name) == getattr(right, name)
        for name in ("loglike", "score_norm", "scaled_score_norm", "pearson_chi2", "deviance")
    )
    return arrays_equal and scalars_equal


def _profile(function: Callable[[], Any], limit: int) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(limit)
    return output.getvalue()


def _environment() -> dict[str, Any]:
    numpy_config = io.StringIO()
    with redirect_stdout(numpy_config):
        np.show_config()
    try:
        numba_version = importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        numba_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "limiteddepkit": importlib.metadata.version("limiteddepkit"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "numba": numba_version,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "numpy_config": numpy_config.getvalue(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--case", help="Run one named case from the selected suite.")
    parser.add_argument(
        "--poisson-engine", choices=("reference", "accelerated", "both"), default="both"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile-case", help="Include a cumulative cProfile fit report.")
    parser.add_argument("--profile-limit", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least one")

    cases = list(QUICK_CASES if args.suite == "quick" else FULL_CASES)
    if args.case:
        cases = [case for case in cases if case.name == args.case]
        if not cases:
            parser.error(f"unknown case for {args.suite!r} suite: {args.case}")

    records: list[dict[str, Any]] = []
    profiles: dict[str, str] = {}
    for case in cases:
        estimator, X, y, fit_kwargs, prediction_kwargs = _problem(case)
        engines = ["reference"]
        if case.estimator == "poisson":
            engines = (
                [args.poisson_engine]
                if args.poisson_engine != "both"
                else ["reference", "accelerated"]
            )
        measurements: list[dict[str, Any]] = []
        results: dict[str, Any] = {}
        for engine in engines:
            measurement, result = _measure(
                case,
                estimator,
                X,
                y,
                fit_kwargs,
                prediction_kwargs,
                engine,
                args.repetitions,
            )
            measurements.append(measurement)
            results[engine] = result
            if args.profile_case == case.name:
                profile_fit = partial(_fit, estimator, X, y, fit_kwargs, engine)
                profiles[f"{case.name}:{engine}"] = _profile(profile_fit, args.profile_limit)
        exact_parity = None
        if case.estimator == "poisson" and args.poisson_engine == "both":
            exact_parity = _exact_poisson_parity(results["reference"], results["accelerated"])
        records.append(
            {
                "case": asdict(case),
                "measurements": measurements,
                "reference_accelerated_exact_parity": exact_parity,
            }
        )

    payload = {
        "benchmark": "limiteddepkit-estimators",
        "timing_contract": "data generation excluded; perf_counter wall time",
        "memory_contract": "tracemalloc Python allocation peak from a separate fit; not process RSS",
        "suite": args.suite,
        "repetitions": args.repetitions,
        "environment": _environment(),
        "records": records,
        "profiles": profiles,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
