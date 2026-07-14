"""Prepare deterministic data and limiteddepkit references for Stata parity checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ldk = importlib.import_module("limiteddepkit")

N_CROSS_SECTION = 1_500
QUADRATURE_POINTS = 12
PANEL_OPTIMIZER_TOLERANCE = 1e-12
ORDERED_OPTIMIZER_TOLERANCE = 1e-13
ORDERED_OPTIMIZER_MAXITER = 5_000
PREDICTION_ROWS = 25
STATA_TIMESTAMP = datetime(2000, 1, 1)
STALE_STATA_ARTIFACTS = (
    "stata_run.log",
    "estimates_raw.csv",
    "covariance_raw.csv",
    "fit.csv",
    "predictions.csv",
    "metadata.txt",
    "estimates_canonical.csv",
    "covariance_canonical.csv",
)
STALE_COMPARISON_ARTIFACTS = (
    "comparison_report.csv",
    "comparison_summary.md",
    "parity_certificate.json",
)
STALE_R_ARTIFACTS = (
    "estimates.csv",
    "covariance.csv",
    "fit.csv",
    "predictions.csv",
    "metadata.csv",
    "comparison_report.csv",
    "comparison_summary.md",
    "parity_certificate.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "work",
        help="Working directory for data, Python references, and returned Stata files.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_stale_evidence(workdir: Path) -> None:
    """Remove only maintained outputs that could be mistaken for this run."""
    candidates = (
        *(workdir / "stata" / name for name in STALE_STATA_ARTIFACTS),
        *(workdir / name for name in STALE_COMPARISON_ARTIFACTS),
        *(workdir / "r" / name for name in STALE_R_ARTIFACTS),
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def _write_dataset(frame: pd.DataFrame, stem: Path) -> dict[str, str]:
    csv_path = stem.with_suffix(".csv")
    dta_path = stem.with_suffix(".dta")
    _write_csv(frame, csv_path)
    frame.to_stata(
        dta_path,
        write_index=False,
        version=118,
        time_stamp=STATA_TIMESTAMP,
    )
    return {
        csv_path.name: _sha256(csv_path),
        dta_path.name: _sha256(dta_path),
    }


def _sample_ordinal(
    rng: np.random.Generator,
    linear_predictor: np.ndarray,
    thresholds: np.ndarray,
    *,
    link: str,
) -> np.ndarray:
    indices = thresholds[None, :] - linear_predictor[:, None]
    if link == "logit":
        cumulative = expit(indices)
    elif link == "probit":
        cumulative = norm.cdf(indices)
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"Unknown ordinal link: {link}")
    uniforms = rng.random(linear_predictor.size)
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int32)


def _cross_section_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    binary_rng = np.random.default_rng(8_421)
    x1 = binary_rng.normal(size=N_CROSS_SECTION)
    x2 = 0.35 * x1 + binary_rng.normal(scale=0.9, size=N_CROSS_SECTION)
    binary_X = pd.DataFrame({"intercept": 1.0, "x1": x1, "x2": x2})

    logit_beta = np.array([-0.35, 0.75, -0.55])
    probit_beta = np.array([0.15, -0.60, 0.45])
    y_logit = (binary_rng.random(N_CROSS_SECTION) < expit(binary_X.to_numpy() @ logit_beta)).astype(
        np.int32
    )
    y_probit = (
        binary_rng.random(N_CROSS_SECTION) < norm.cdf(binary_X.to_numpy() @ probit_beta)
    ).astype(np.int32)

    ordinal_rng = np.random.default_rng(4_102)
    ox1 = ordinal_rng.normal(size=N_CROSS_SECTION)
    ox2 = ordinal_rng.normal(size=N_CROSS_SECTION)
    ordinal_X = pd.DataFrame({"ox1": ox1, "ox2": ox2})
    ordinal_index = ordinal_X.to_numpy() @ np.array([0.65, -0.40])
    thresholds = np.array([-0.70, 0.80])
    y_ologit = _sample_ordinal(ordinal_rng, ordinal_index, thresholds, link="logit")
    y_oprobit = _sample_ordinal(ordinal_rng, ordinal_index, thresholds, link="probit")

    flexible_simulation = ldk.simulate_generalized_ordered_logit(
        nobs=N_CROSS_SECTION,
        seed=9_101,
        thresholds=(-0.9, 0.9),
        threshold_slopes=((0.85, -0.4), (0.3, -0.4)),
    )
    flexible_X = flexible_simulation.X.rename(columns={"x1": "gx1", "x2": "gx2"})

    data = pd.DataFrame({"obs_id": np.arange(1, N_CROSS_SECTION + 1)})
    data = pd.concat(
        [
            data,
            binary_X,
            pd.DataFrame({"y_logit": y_logit, "y_probit": y_probit}),
            ordinal_X,
            pd.DataFrame({"y_ologit": y_ologit, "y_oprobit": y_oprobit}),
            flexible_X,
            pd.DataFrame({"y_gologit": flexible_simulation.y.to_numpy(dtype=np.int32)}),
        ],
        axis=1,
    )
    designs = {
        "binary": binary_X,
        "ordinal": ordinal_X,
        "flexible": flexible_X,
    }
    outcomes = {
        "y_logit": pd.Series(y_logit, name="y_logit"),
        "y_probit": pd.Series(y_probit, name="y_probit"),
        "y_ologit": pd.Series(y_ologit, name="y_ologit"),
        "y_oprobit": pd.Series(y_oprobit, name="y_oprobit"),
        "y_gologit": flexible_simulation.y.reset_index(drop=True),
    }
    return data, designs, outcomes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_result(
    model: str,
    result: Any,
    *,
    expected_nobs: int,
    expected_groups: int | None = None,
    require_interior: bool = False,
) -> None:
    """Reject an invalid fit before it can become a parity reference."""
    _require(bool(getattr(result, "converged", False)), f"{model} did not converge")
    _require(
        bool(getattr(result, "inference_valid", False)),
        f"{model} did not produce valid inference",
    )

    actual_nobs = getattr(result, "nobs", None)
    _require(
        actual_nobs == expected_nobs,
        f"{model} nobs != {expected_nobs} (got {actual_nobs!r})",
    )
    if expected_groups is not None:
        actual_groups = getattr(result, "n_groups", None)
        _require(
            actual_groups == expected_groups,
            f"{model} n_groups != {expected_groups} (got {actual_groups!r})",
        )

    parameters = np.asarray(result.all_params, dtype=float)
    covariance = np.asarray(result.covariance, dtype=float)
    _require(parameters.size > 0, f"{model} has no parameter estimates")
    _require(
        np.isfinite(parameters).all(),
        f"{model} has nonfinite parameter estimates",
    )
    _require(
        covariance.shape == (parameters.size, parameters.size),
        f"{model} covariance shape does not match its parameter vector",
    )
    _require(
        np.isfinite(covariance).all(),
        f"{model} has nonfinite covariance entries",
    )
    _require(
        np.isfinite(float(result.loglike)),
        f"{model} has a nonfinite log likelihood",
    )

    if require_interior:
        constraint_slack = float(getattr(result, "constraint_slack", np.nan))
        _require(
            np.isfinite(constraint_slack) and constraint_slack > 0.0,
            f"{model} constraints are not strictly interior",
        )


def _record_model(
    *,
    model: str,
    dataset: str,
    result: Any,
    estimates: list[dict[str, Any]],
    covariance: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    probability_frame: pd.DataFrame | None = None,
    prediction_obs_ids: Any | None = None,
) -> None:
    all_params = result.all_params.astype(float)
    standard_errors = result.standard_errors.reindex(all_params.index).astype(float)
    for parameter, estimate in all_params.items():
        estimates.append(
            {
                "model": model,
                "dataset": dataset,
                "parameter": str(parameter),
                "estimate": float(estimate),
                "standard_error": float(standard_errors.loc[parameter]),
            }
        )

    covariance_frame = result.covariance.reindex(index=all_params.index, columns=all_params.index)
    for row_parameter in all_params.index:
        for column_parameter in all_params.index:
            covariance.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "row_parameter": str(row_parameter),
                    "column_parameter": str(column_parameter),
                    "covariance": float(covariance_frame.loc[row_parameter, column_parameter]),
                }
            )

    n_params = len(all_params)
    nobs = int(result.nobs)
    loglike = float(result.loglike)
    fits.append(
        {
            "model": model,
            "dataset": dataset,
            "nobs": nobs,
            "n_groups": getattr(result, "n_groups", np.nan),
            "n_params": n_params,
            "loglike": loglike,
            "aic": -2.0 * loglike + 2.0 * n_params,
            "bic": -2.0 * loglike + np.log(nobs) * n_params,
            "converged": bool(result.converged),
            "inference_valid": bool(result.inference_valid),
            "quadrature_points": getattr(result, "quadrature_points", np.nan),
            "constraint_slack": getattr(result, "constraint_slack", np.nan),
        }
    )

    if probability_frame is None:
        return
    if prediction_obs_ids is None:  # pragma: no cover - internal invariant
        raise ValueError("prediction_obs_ids are required with probability_frame")
    obs_ids = np.asarray(prediction_obs_ids)
    if probability_frame.shape[0] != obs_ids.size:
        raise ValueError("Prediction IDs and probability rows do not align.")
    for row in range(min(PREDICTION_ROWS, probability_frame.shape[0])):
        for category in probability_frame.columns:
            predictions.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "obs_id": int(obs_ids[row]),
                    "category": int(category),
                    "probability": float(probability_frame.iloc[row][category]),
                }
            )


def main() -> int:
    args = _parse_args()
    workdir = args.output.resolve()
    data_dir = workdir / "data"
    python_dir = workdir / "python"
    stata_dir = workdir / "stata"
    workdir.mkdir(parents=True, exist_ok=True)
    _remove_stale_evidence(workdir)
    data_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    stata_dir.mkdir(parents=True, exist_ok=True)

    estimates: list[dict[str, Any]] = []
    covariance: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    cross, designs, outcomes = _cross_section_data()
    file_hashes: dict[str, str] = {}
    for filename, digest in _write_dataset(cross, data_dir / "cross_section").items():
        file_hashes[f"data/{filename}"] = digest

    binary_logit = ldk.BinaryLogit().fit(designs["binary"], outcomes["y_logit"])
    _assert_result("binary_logit", binary_logit, expected_nobs=N_CROSS_SECTION)
    _record_model(
        model="binary_logit",
        dataset="cross_section",
        result=binary_logit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=binary_logit.predict_proba(designs["binary"]),
        prediction_obs_ids=cross["obs_id"],
    )

    binary_probit = ldk.BinaryProbit().fit(designs["binary"], outcomes["y_probit"])
    _assert_result("binary_probit", binary_probit, expected_nobs=N_CROSS_SECTION)
    _record_model(
        model="binary_probit",
        dataset="cross_section",
        result=binary_probit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=binary_probit.predict_proba(designs["binary"]),
        prediction_obs_ids=cross["obs_id"],
    )

    ordered_logit = ldk.OrderedLogit().fit(
        designs["ordinal"],
        outcomes["y_ologit"],
        category_order=[0, 1, 2],
        maxiter=ORDERED_OPTIMIZER_MAXITER,
        tolerance=ORDERED_OPTIMIZER_TOLERANCE,
    )
    _assert_result("ordered_logit", ordered_logit, expected_nobs=N_CROSS_SECTION)
    _record_model(
        model="ordered_logit",
        dataset="cross_section",
        result=ordered_logit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=ordered_logit.predict_proba(designs["ordinal"]),
        prediction_obs_ids=cross["obs_id"],
    )

    ordered_probit = ldk.OrderedProbit().fit(
        designs["ordinal"],
        outcomes["y_oprobit"],
        category_order=[0, 1, 2],
        maxiter=ORDERED_OPTIMIZER_MAXITER,
        tolerance=ORDERED_OPTIMIZER_TOLERANCE,
    )
    _assert_result("ordered_probit", ordered_probit, expected_nobs=N_CROSS_SECTION)
    _record_model(
        model="ordered_probit",
        dataset="cross_section",
        result=ordered_probit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=ordered_probit.predict_proba(designs["ordinal"]),
        prediction_obs_ids=cross["obs_id"],
    )

    generalized = ldk.GeneralizedOrderedLogit().fit(
        designs["flexible"], outcomes["y_gologit"], category_order=[0, 1, 2]
    )
    _assert_result(
        "generalized_ordered_logit",
        generalized,
        expected_nobs=N_CROSS_SECTION,
        require_interior=True,
    )
    _record_model(
        model="generalized_ordered_logit",
        dataset="cross_section",
        result=generalized,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=generalized.predict_proba(designs["flexible"]),
        prediction_obs_ids=cross["obs_id"],
    )

    partial = ldk.PartialProportionalOdds(varying=["gx1"]).fit(
        designs["flexible"], outcomes["y_gologit"], category_order=[0, 1, 2]
    )
    _assert_result(
        "partial_proportional_odds",
        partial,
        expected_nobs=N_CROSS_SECTION,
        require_interior=True,
    )
    _record_model(
        model="partial_proportional_odds",
        dataset="cross_section",
        result=partial,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=partial.predict_proba(designs["flexible"]),
        prediction_obs_ids=cross["obs_id"],
    )

    static_simulation = ldk.simulate_random_effects_ordered_logit(
        n_entities=80,
        n_periods=6,
        seed=8_821,
    )
    static_data = pd.DataFrame(
        {
            "obs_id": np.arange(1, static_simulation.nobs + 1),
            "entity": static_simulation.entity.to_numpy(dtype=np.int32) + 1,
            "time": static_simulation.time.to_numpy(dtype=np.int32),
            "x1": static_simulation.X["x1"].to_numpy(),
            "x2": static_simulation.X["x2"].to_numpy(),
            "y": static_simulation.y.to_numpy(dtype=np.int32),
        }
    )
    for filename, digest in _write_dataset(static_data, data_dir / "static_re").items():
        file_hashes[f"data/{filename}"] = digest
    static_result = ldk.RandomEffectsOrderedLogit().fit(
        static_simulation.X,
        static_simulation.y,
        entity=static_simulation.entity,
        category_order=[0, 1, 2],
        quadrature_points=QUADRATURE_POINTS,
        tolerance=PANEL_OPTIMIZER_TOLERANCE,
    )
    _assert_result(
        "random_effects_ordered_logit",
        static_result,
        expected_nobs=static_simulation.nobs,
        expected_groups=static_simulation.n_entities,
    )
    _record_model(
        model="random_effects_ordered_logit",
        dataset="static_re",
        result=static_result,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=static_result.predict_proba(static_simulation.X, random_effects=0.0),
        prediction_obs_ids=static_data["obs_id"],
    )

    dynamic_simulation = ldk.simulate_dynamic_random_effects_ordered_logit(
        n_entities=60,
        n_periods=6,
        seed=8_263,
    )
    dynamic_raw = pd.DataFrame(
        {
            "obs_id": np.arange(1, dynamic_simulation.nobs + 1),
            "entity": dynamic_simulation.entity.to_numpy(dtype=np.int32) + 1,
            "time": dynamic_simulation.time.to_numpy(dtype=np.int32),
            "x1": dynamic_simulation.X["x1"].to_numpy(),
            "y": dynamic_simulation.y.to_numpy(dtype=np.int32),
        }
    )
    for filename, digest in _write_dataset(dynamic_raw, data_dir / "dynamic_raw").items():
        file_hashes[f"data/{filename}"] = digest

    dynamic_result = ldk.DynamicRandomEffectsOrderedLogit().fit(
        dynamic_simulation.X,
        dynamic_simulation.y,
        entity=dynamic_simulation.entity,
        time=dynamic_simulation.time,
        category_order=[0, 1, 2],
        quadrature_points=QUADRATURE_POINTS,
        maxiter=800,
        tolerance=PANEL_OPTIMIZER_TOLERANCE,
    )
    _assert_result(
        "dynamic_random_effects_ordered_logit",
        dynamic_result,
        expected_nobs=dynamic_simulation.nobs - dynamic_simulation.n_entities,
        expected_groups=dynamic_simulation.n_entities,
    )
    dynamic_name_map = {
        "state[1]": "state_1",
        "state[2]": "state_2",
        "initial[1]": "initial_1",
        "initial[2]": "initial_2",
        "initial_x[x1]": "initial_x1",
        "mean[x1]": "mean_x1",
    }
    dynamic_design_X = dynamic_result.estimation_design.rename(columns=dynamic_name_map)
    dynamic_obs_ids = np.asarray(dynamic_result.estimation_index, dtype=int) + 1
    dynamic_design = pd.DataFrame(
        {
            "obs_id": dynamic_obs_ids,
            "entity": dynamic_result.estimation_entity.to_numpy(dtype=np.int32) + 1,
            "y": dynamic_result.estimation_outcome.to_numpy(dtype=np.int32),
        }
    )
    dynamic_design = pd.concat(
        [dynamic_design.reset_index(drop=True), dynamic_design_X.reset_index(drop=True)],
        axis=1,
    )
    for filename, digest in _write_dataset(dynamic_design, data_dir / "dynamic_design").items():
        file_hashes[f"data/{filename}"] = digest

    dynamic_fixed_probabilities = dynamic_result.base_result.predict_proba(
        dynamic_result.estimation_design,
        random_effects=0.0,
    )
    _record_model(
        model="dynamic_random_effects_ordered_logit",
        dataset="dynamic_design",
        result=dynamic_result,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=dynamic_fixed_probabilities,
        prediction_obs_ids=dynamic_obs_ids,
    )

    reference_frames = {
        "estimates.csv": pd.DataFrame(estimates),
        "covariance.csv": pd.DataFrame(covariance),
        "fit.csv": pd.DataFrame(fits),
        "predictions.csv": pd.DataFrame(predictions),
    }
    for filename, frame in reference_frames.items():
        path = python_dir / filename
        _write_csv(frame, path)
        file_hashes[f"python/{filename}"] = _sha256(path)

    manifest = {
        "schema_version": 1,
        "suite": "controlled_synthetic_certification",
        "limiteddepkit_version": ldk.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "quadrature_method": "ghermite",
        "quadrature_points": QUADRATURE_POINTS,
        "panel_optimizer_tolerance": PANEL_OPTIMIZER_TOLERANCE,
        "ordered_optimizer_tolerance": ORDERED_OPTIMIZER_TOLERANCE,
        "ordered_optimizer_maxiter": ORDERED_OPTIMIZER_MAXITER,
        "prediction_rows_per_model": PREDICTION_ROWS,
        "files": file_hashes,
        "models": {
            "binary_logit": {"reference": "logit", "required": True},
            "binary_probit": {"reference": "probit", "required": True},
            "ordered_logit": {"reference": "ologit", "required": True},
            "ordered_probit": {"reference": "oprobit", "required": True},
            "generalized_ordered_logit": {
                "reference": "gologit2, npl",
                "required": False,
                "constraint_slack": float(generalized.constraint_slack),
            },
            "partial_proportional_odds": {
                "reference": "gologit2, npl(gx1)",
                "required": False,
                "constraint_slack": float(partial.constraint_slack),
            },
            "random_effects_ordered_logit": {
                "reference": "meologit, intmethod(ghermite)",
                "required": True,
            },
            "dynamic_random_effects_ordered_logit": {
                "reference": "meologit on exported augmented design",
                "required": True,
                "stata_name_map": dynamic_name_map,
            },
        },
    }
    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Prepared Stata parity work directory: {workdir}")
    print(f"limiteddepkit version: {ldk.__version__}")
    print(f"Models prepared: {len(manifest['models'])}")
    print("Next: run limiteddepkit_parity.do in Stata with this work directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
