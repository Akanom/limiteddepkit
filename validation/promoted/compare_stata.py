"""Canonicalize promoted Stata exports and compare them with Python evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVIDENCE_CLASSES = {
    "firth_binary_logit": "optional_stata_command_with_aligned_fisher_covariance",
    "poisson": "exact_likelihood",
    "negative_binomial_nb2": "exact_likelihood",
    "tobit": "exact_likelihood",
    "truncated_regression": "exact_likelihood",
    "interval_regression": "exact_likelihood",
    "geometric_duration": "exact_person_period_likelihood_identity",
    "exponential_duration": "exact_likelihood",
    "weibull_duration": "exact_likelihood",
    "gamma_duration": "unsupported_exact_stata_match",
    "random_effects_ordered_probit": "numerical_quadrature_parity",
    "fixed_effects_ordered_logit": "conditional_buc_pseudo_sample",
}

RAW_ESTIMATE_COLUMNS = (
    "model",
    "dataset",
    "position",
    "stata_parameter",
    "estimate",
    "standard_error",
)
RAW_COVARIANCE_COLUMNS = (
    "model",
    "dataset",
    "row_position",
    "column_position",
    "row_parameter",
    "column_parameter",
    "covariance",
)
PREDICTION_COLUMNS = (
    "model",
    "dataset",
    "obs_id",
    "prediction",
    "category",
    "time",
    "value",
)
STATUS_COLUMNS = ("model", "dataset", "status", "reason")


@dataclass(frozen=True)
class ParameterTransform:
    """One scalar component of the raw-to-canonical Jacobian."""

    canonical_name: str
    raw_position: int
    raw_name: str
    estimate: float
    derivative: float
    formula: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workdir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "work" / "real_data",
        help="Prepared promoted suite containing manifest.json, python/, data/, and stata/.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _invalidate_promoted_evidence(workdir: Path) -> None:
    """Remove only derived promoted-comparison artifacts, never raw or legacy evidence."""

    candidates = (
        workdir / "comparison_report.csv",
        workdir / "comparison_summary.md",
        workdir / "parity_certificate.json",
        workdir / "stata" / "estimates_canonical.csv",
        workdir / "stata" / "covariance_canonical.csv",
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink()


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    if not frame.columns.is_unique:
        duplicate = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{label} has duplicate columns: {duplicate}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} has no rows")


def _assert_unique(frame: pd.DataFrame, keys: list[str], *, label: str) -> None:
    if frame[keys].isna().any(axis=None):
        raise ValueError(f"{label} has missing key values in {keys}")
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{label} has duplicate keys {keys}: {examples}")


def _load_manifest(workdir: Path) -> dict[str, Any]:
    path = workdir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing promoted manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Only promoted manifest schema_version=1 is supported")
    if not str(manifest.get("suite", "")).startswith("promoted_"):
        raise ValueError("Refusing to compare a non-promoted suite")
    model_order = manifest.get("model_order")
    model_specs = manifest.get("model_specs")
    if not isinstance(model_order, list) or not model_order:
        raise ValueError("Manifest model_order must be a nonempty list")
    if not isinstance(model_specs, dict) or set(model_order) != set(model_specs):
        raise ValueError("Manifest model_order and model_specs do not describe the same models")
    if set(model_order) != set(EVIDENCE_CLASSES):
        raise ValueError("Promoted Stata model inventory differs from the certified inventory")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict) or not declared_files:
        raise ValueError("Manifest files hash map is missing")
    mismatches: list[str] = []
    for relative, expected_hash in declared_files.items():
        artifact = workdir / str(relative)
        if not artifact.is_file():
            mismatches.append(f"missing {relative}")
        elif _sha256(artifact) != expected_hash:
            mismatches.append(f"hash mismatch {relative}")
    if mismatches:
        raise ValueError("Prepared evidence does not match manifest: " + "; ".join(mismatches))
    return manifest


def _parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"Malformed metadata line {line_number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in metadata:
            raise ValueError(f"Duplicate or empty metadata key on line {line_number}")
        metadata[key] = value
    return metadata


def _validate_metadata(metadata: dict[str, str], manifest: dict[str, Any]) -> None:
    if metadata.get("suite") != manifest.get("suite"):
        raise ValueError(
            f"Stata suite metadata {metadata.get('suite')!r} does not match manifest "
            f"{manifest.get('suite')!r}"
        )
    if metadata.get("run_completed") != "1":
        raise ValueError("Stata metadata does not certify a completed run")
    version_text = metadata.get("stata_version", "")
    match = re.search(r"\d+(?:\.\d+)?", version_text)
    if match is None or float(match.group()) < 17.0:
        raise ValueError(f"Promoted parity requires Stata 17 or newer; found {version_text!r}")
    method = metadata.get("random_effects_ordered_probit.intmethod", "").lower()
    if "hermite" not in method:
        raise ValueError(f"RE Ordered Probit did not record Gauss-Hermite quadrature: {method!r}")
    quadrature = metadata.get("random_effects_ordered_probit.n_quad", "")
    points = re.search(r"\d+", quadrature)
    if points is None or int(points.group()) != 20:
        raise ValueError(f"RE Ordered Probit must use exactly 20 quadrature points: {quadrature!r}")


def _validate_status(
    status: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, dict[str, str]]:
    _validate_columns(status, STATUS_COLUMNS, label="Stata model_status.csv")
    _assert_unique(status, ["model"], label="Stata model_status.csv")
    expected = set(manifest["model_order"])
    observed = set(status["model"].astype(str))
    if observed != expected:
        raise ValueError(
            "Stata model status inventory mismatch; missing="
            f"{sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )

    result: dict[str, dict[str, str]] = {}
    for row in status.itertuples(index=False):
        model = str(row.model)
        state = str(row.status).upper()
        reason = str(row.reason)
        dataset = str(row.dataset)
        expected_dataset = str(manifest["model_specs"][model]["dataset"])
        if dataset != expected_dataset:
            raise ValueError(f"{model} status dataset is {dataset!r}, expected {expected_dataset!r}")
        if model == "gamma_duration":
            allowed = state == "SKIP" and reason.startswith("unsupported_exact_match:")
        elif model == "firth_binary_logit":
            allowed = state == "RUN" or (
                state == "SKIP" and reason == "optional_command_not_installed"
            )
        else:
            allowed = state == "RUN"
        if not allowed:
            raise ValueError(f"Unapproved Stata status for {model}: {state} ({reason})")
        result[model] = {"status": state, "reason": reason, "dataset": dataset}
    return result


def _raw_name_parts(name: str) -> tuple[str, str, str, str]:
    normalized = re.sub(r"\s+", "", str(name).lower())
    if ":" in normalized:
        equation, term = normalized.rsplit(":", 1)
    else:
        equation, term = "", normalized
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    return normalized, equation.strip("/"), term.strip("/"), compact


def _parameter_transform(
    *,
    model: str,
    raw_name: str,
    raw_position: int,
    raw_estimate: float,
    expected_order: list[str],
    spec: dict[str, Any],
) -> ParameterTransform:
    normalized, equation, term, compact = _raw_name_parts(raw_name)

    if model == "negative_binomial_nb2" and "log_alpha" in expected_order:
        if "lnalpha" in compact or "logalpha" in compact:
            return ParameterTransform(
                "log_alpha", raw_position, raw_name, raw_estimate, 1.0, "identity(/lnalpha)"
            )
        if "alpha" in compact and term in {"alpha", "_cons"}:
            if raw_estimate <= 0:
                raise ValueError("NB2 natural alpha must be positive")
            return ParameterTransform(
                "log_alpha",
                raw_position,
                raw_name,
                math.log(raw_estimate),
                1.0 / raw_estimate,
                "log(alpha)",
            )

    if model == "weibull_duration" and "log_alpha" in expected_order:
        if "lnp" in compact or "logp" in compact:
            return ParameterTransform(
                "log_alpha", raw_position, raw_name, raw_estimate, 1.0, "identity(/ln_p)"
            )
        if equation == "p" or term == "p":
            if raw_estimate <= 0:
                raise ValueError("Weibull p must be positive")
            return ParameterTransform(
                "log_alpha",
                raw_position,
                raw_name,
                math.log(raw_estimate),
                1.0 / raw_estimate,
                "log(p)",
            )

    if model == "random_effects_ordered_probit" and "sigma_entity" in expected_order:
        if re.search(r"(?:^|[^a-z])lns", normalized) or compact.startswith("lns"):
            sigma = math.exp(raw_estimate)
            return ParameterTransform(
                "sigma_entity", raw_position, raw_name, sigma, sigma, "exp(log_sd)"
            )
        if "variance" in normalized or equation.startswith("var") or compact.startswith("var"):
            if raw_estimate <= 0:
                raise ValueError("Random-intercept variance must be positive")
            sigma = math.sqrt(raw_estimate)
            return ParameterTransform(
                "sigma_entity",
                raw_position,
                raw_name,
                sigma,
                0.5 / sigma,
                "sqrt(variance)",
            )
        if equation.startswith("sd") or "sigmaentity" in compact:
            return ParameterTransform(
                "sigma_entity", raw_position, raw_name, raw_estimate, 1.0, "identity(sd)"
            )

    if "sigma" in expected_order:
        if "varey" in compact or compact.startswith("var"):
            if raw_estimate <= 0:
                raise ValueError(f"{model} error variance must be positive")
            sigma = math.sqrt(raw_estimate)
            return ParameterTransform(
                "sigma",
                raw_position,
                raw_name,
                sigma,
                0.5 / sigma,
                "sqrt(variance)",
            )
        if "sigma" not in compact:
            pass
        elif "lnsigma" in compact or "logsigma" in compact:
            sigma = math.exp(raw_estimate)
            return ParameterTransform(
                "sigma", raw_position, raw_name, sigma, sigma, "exp(log_sigma)"
            )
        else:
            if raw_estimate <= 0:
                raise ValueError(f"{model} natural sigma must be positive")
            return ParameterTransform(
                "sigma", raw_position, raw_name, raw_estimate, 1.0, "identity(sigma)"
            )

    cut_match = re.search(r"cut(?:point)?(\d+)", normalized)
    if cut_match is not None:
        thresholds = [name for name in expected_order if name.startswith("threshold:")]
        cut_index = int(cut_match.group(1)) - 1
        if cut_index < 0 or cut_index >= len(thresholds):
            raise ValueError(f"Unexpected Stata cutpoint in {raw_name!r}")
        return ParameterTransform(
            thresholds[cut_index], raw_position, raw_name, raw_estimate, 1.0, "identity(cut)"
        )

    declared = {
        str(key).lower(): str(value)
        for key, value in spec.get("parameter_mappings", {})
        .get("stata_to_canonical", {})
        .items()
    }
    candidates = (term, normalized.strip("/"), equation)
    canonical = next((declared[item] for item in candidates if item in declared), None)
    if canonical is None and term in expected_order:
        canonical = term
    if canonical is None and term == "_cons":
        for constant_name in ("intercept", "const"):
            if constant_name in expected_order:
                canonical = constant_name
                break
    if canonical not in expected_order:
        raise ValueError(f"Cannot map raw Stata parameter {raw_name!r} for {model}")
    return ParameterTransform(
        str(canonical), raw_position, raw_name, raw_estimate, 1.0, "identity"
    )


def _raw_covariance_matrix(
    model: str, model_estimates: pd.DataFrame, model_covariance: pd.DataFrame
) -> np.ndarray:
    positions = model_estimates["position"].astype(int).to_numpy()
    expected_positions = np.arange(1, len(model_estimates) + 1)
    if not np.array_equal(positions, expected_positions):
        raise ValueError(f"{model} raw coefficient positions are not contiguous from 1")
    _assert_unique(
        model_covariance,
        ["row_position", "column_position"],
        label=f"{model} raw covariance",
    )
    expected_pairs = {
        (row, column)
        for row in expected_positions
        for column in expected_positions
    }
    actual_pairs = set(
        zip(
            model_covariance["row_position"].astype(int),
            model_covariance["column_position"].astype(int),
            strict=True,
        )
    )
    if actual_pairs != expected_pairs:
        raise ValueError(f"{model} raw covariance does not contain a complete square matrix")
    name_by_position = dict(
        zip(
            model_estimates["position"].astype(int),
            model_estimates["stata_parameter"].astype(str),
            strict=True,
        )
    )
    bad_labels = model_covariance.loc[
        model_covariance.apply(
            lambda row: (
                str(row["row_parameter"])
                != name_by_position[int(row["row_position"])]
                or str(row["column_parameter"])
                != name_by_position[int(row["column_position"])]
            ),
            axis=1,
        )
    ]
    if not bad_labels.empty:
        raise ValueError(f"{model} covariance parameter labels disagree with raw e(b)")
    ordered = model_covariance.sort_values(["row_position", "column_position"])
    matrix = ordered["covariance"].to_numpy(dtype=float).reshape(len(positions), len(positions))
    if not np.isfinite(matrix).all():
        raise ValueError(f"{model} raw covariance has non-finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{model} raw covariance is not symmetric")
    return (matrix + matrix.T) / 2.0


def _firth_aligned_covariance(
    *,
    workdir: Path,
    spec: dict[str, Any],
    estimates: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    data_file = workdir / str(spec["data_file"])
    data = pd.read_csv(data_file)
    features = list(spec["features"])
    outcome = str(spec["outcome"])
    beta = estimates.set_index("parameter").loc[features, "estimate"].to_numpy(dtype=float)
    design = data[features].to_numpy(dtype=float)
    outcomes = data[outcome].to_numpy(dtype=float)
    eta = design @ beta
    probabilities = 1.0 / (1.0 + np.exp(-eta))
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    sign, logdet = np.linalg.slogdet(information)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("Firth aligned ordinary Fisher information is not positive definite")
    covariance = np.linalg.inv(information)
    ordinary = float(np.sum(outcomes * eta - np.logaddexp(0.0, eta)))
    penalty = 0.5 * float(logdet)
    return covariance, {
        "loglike": ordinary,
        "penalized_loglike": ordinary + penalty,
        "jeffreys_penalty": penalty,
    }


def _canonicalize_model(
    *,
    model: str,
    workdir: Path,
    spec: dict[str, Any],
    raw_estimates: pd.DataFrame,
    raw_covariance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    model_estimates = raw_estimates.loc[raw_estimates["model"] == model].copy()
    model_covariance = raw_covariance.loc[raw_covariance["model"] == model].copy()
    if model_estimates.empty or model_covariance.empty:
        raise ValueError(f"{model} is RUN but raw coefficient/covariance output is missing")
    datasets = set(model_estimates["dataset"].astype(str)) | set(
        model_covariance["dataset"].astype(str)
    )
    if datasets != {str(spec["dataset"])}:
        raise ValueError(f"{model} raw output has incorrect dataset labels: {sorted(datasets)}")
    model_estimates["position"] = pd.to_numeric(model_estimates["position"], errors="raise")
    model_estimates = model_estimates.sort_values("position")
    _assert_unique(model_estimates, ["position"], label=f"{model} raw estimates")
    _assert_unique(model_estimates, ["stata_parameter"], label=f"{model} raw estimates")
    if not np.isfinite(model_estimates["estimate"].to_numpy(dtype=float)).all():
        raise ValueError(f"{model} raw estimates contain non-finite values")

    expected_order = [str(name) for name in spec["parameter_order"]]
    redundant_constants = []
    if "intercept" in expected_order:
        for row in model_estimates.itertuples(index=False):
            _, _, term, _ = _raw_name_parts(str(row.stata_parameter))
            if (
                term == "_cons"
                and abs(float(row.estimate)) <= 1e-15
                and abs(float(row.standard_error)) <= 1e-15
            ):
                redundant_constants.append(int(row.position))
    if redundant_constants:
        model_estimates = model_estimates.loc[
            ~model_estimates["position"].astype(int).isin(redundant_constants)
        ].copy()
        model_covariance = model_covariance.loc[
            ~model_covariance["row_position"].astype(int).isin(redundant_constants)
            & ~model_covariance["column_position"].astype(int).isin(redundant_constants)
        ].copy()
        new_positions = {
            old_position: new_position
            for new_position, old_position in enumerate(
                model_estimates["position"].astype(int).tolist(), start=1
            )
        }
        model_estimates["position"] = model_estimates["position"].astype(int).map(new_positions)
        model_covariance["row_position"] = (
            model_covariance["row_position"].astype(int).map(new_positions)
        )
        model_covariance["column_position"] = (
            model_covariance["column_position"].astype(int).map(new_positions)
        )

    transforms = [
        _parameter_transform(
            model=model,
            raw_name=str(row.stata_parameter),
            raw_position=int(row.position),
            raw_estimate=float(row.estimate),
            expected_order=expected_order,
            spec=spec,
        )
        for row in model_estimates.itertuples(index=False)
    ]
    mapped = [item.canonical_name for item in transforms]
    if len(mapped) != len(expected_order) or set(mapped) != set(expected_order):
        raise ValueError(
            f"{model} raw-to-canonical mapping mismatch; mapped={mapped}, expected={expected_order}"
        )
    if len(set(mapped)) != len(mapped):
        raise ValueError(f"{model} raw parameters map to duplicate canonical names")

    raw_v = _raw_covariance_matrix(model, model_estimates, model_covariance)
    transform_by_name = {item.canonical_name: item for item in transforms}
    ordered_transforms = [transform_by_name[name] for name in expected_order]
    jacobian = np.zeros((len(expected_order), len(model_estimates)), dtype=float)
    estimates = np.empty(len(expected_order), dtype=float)
    for row_index, item in enumerate(ordered_transforms):
        jacobian[row_index, item.raw_position - 1] = item.derivative
        estimates[row_index] = item.estimate
    canonical_v = jacobian @ raw_v @ jacobian.T
    aligned_fit: dict[str, float] = {}

    estimate_frame = pd.DataFrame(
        {
            "model": model,
            "dataset": str(spec["dataset"]),
            "parameter": expected_order,
            "estimate": estimates,
        }
    )
    if model == "firth_binary_logit":
        canonical_v, aligned_fit = _firth_aligned_covariance(
            workdir=workdir, spec=spec, estimates=estimate_frame
        )
    canonical_v = (canonical_v + canonical_v.T) / 2.0
    standard_errors = np.sqrt(np.clip(np.diag(canonical_v), 0.0, None))
    estimate_frame["standard_error"] = standard_errors
    covariance_frame = pd.DataFrame(
        [
            {
                "model": model,
                "dataset": str(spec["dataset"]),
                "row_parameter": row_name,
                "column_parameter": column_name,
                "covariance": float(canonical_v[row_index, column_index]),
            }
            for row_index, row_name in enumerate(expected_order)
            for column_index, column_name in enumerate(expected_order)
        ]
    )
    transform_audit = [
        {
            "model": model,
            "canonical_parameter": item.canonical_name,
            "raw_parameter": item.raw_name,
            "formula": (
                "aligned inverse ordinary Fisher"
                if model == "firth_binary_logit"
                else item.formula
            ),
        }
        for item in ordered_transforms
    ]
    return estimate_frame, covariance_frame, transform_audit, aligned_fit


def _comparison_tolerance(
    spec: dict[str, Any], statistic: str
) -> tuple[float, float]:
    configured = spec.get("comparison_tolerances", {})
    estimate_atol = float(configured.get("estimate_atol", 5e-5))
    estimate_rtol = float(configured.get("estimate_rtol", 0.0))
    covariance_atol = float(configured.get("covariance_atol", 1e-4))
    prediction_atol = float(configured.get("prediction_atol", 1e-4))
    if statistic == "estimate":
        return estimate_atol, estimate_rtol
    if statistic in {"standard_error", "covariance"}:
        return covariance_atol, 0.0
    if statistic == "prediction":
        return prediction_atol, estimate_rtol
    if statistic in {"loglike", "penalized_loglike", "jeffreys_penalty"}:
        return max(1e-5, 5.0 * estimate_atol), 1e-8
    if statistic in {"aic", "bic"}:
        return max(1e-5, 10.0 * estimate_atol), 1e-8
    return 0.0, 0.0


def _numeric_check(
    *,
    model: str,
    dataset: str,
    statistic: str,
    expected: np.ndarray,
    actual: np.ndarray,
    atol: float,
    rtol: float = 0.0,
    detail: str = "",
) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape or expected.size == 0:
        return {
            "model": model,
            "dataset": dataset,
            "statistic": statistic,
            "compared": 0,
            "max_abs_difference": np.nan,
            "max_allowed_difference": np.nan,
            "atol": atol,
            "rtol": rtol,
            "status": "FAIL",
            "detail": detail or f"shape mismatch {expected.shape} versus {actual.shape}",
        }
    finite = np.isfinite(expected) & np.isfinite(actual)
    difference = np.abs(expected - actual)
    allowance = atol + rtol * np.abs(expected)
    passed = bool(finite.all() and np.all(difference <= allowance))
    return {
        "model": model,
        "dataset": dataset,
        "statistic": statistic,
        "compared": int(expected.size),
        "max_abs_difference": float(np.max(difference)) if difference.size else np.nan,
        "max_allowed_difference": float(np.max(allowance)) if allowance.size else np.nan,
        "atol": atol,
        "rtol": rtol,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    }


def _missing_key_failure(
    *, model: str, dataset: str, statistic: str, missing: list[str]
) -> dict[str, Any]:
    return {
        "model": model,
        "dataset": dataset,
        "statistic": statistic,
        "compared": 0,
        "max_abs_difference": np.nan,
        "max_allowed_difference": np.nan,
        "atol": np.nan,
        "rtol": np.nan,
        "status": "FAIL",
        "detail": "missing/mismatched keys: " + ", ".join(missing[:10]),
    }


def _compare_parameters(
    *,
    model: str,
    spec: dict[str, Any],
    python_estimates: pd.DataFrame,
    python_covariance: pd.DataFrame,
    stata_estimates: pd.DataFrame,
    stata_covariance: pd.DataFrame,
) -> list[dict[str, Any]]:
    dataset = str(spec["dataset"])
    expected = python_estimates.loc[python_estimates["model"] == model]
    actual = stata_estimates.loc[stata_estimates["model"] == model]
    merged = expected.merge(
        actual,
        on=["model", "dataset", "parameter"],
        how="outer",
        suffixes=("_python", "_stata"),
        indicator=True,
        validate="one_to_one",
    )
    missing = merged.loc[merged["_merge"] != "both", "parameter"].astype(str).tolist()
    if missing:
        return [
            _missing_key_failure(
                model=model, dataset=dataset, statistic="parameter_keys", missing=missing
            )
        ]
    report: list[dict[str, Any]] = []
    for statistic in ("estimate", "standard_error"):
        atol, rtol = _comparison_tolerance(spec, statistic)
        report.append(
            _numeric_check(
                model=model,
                dataset=dataset,
                statistic=statistic,
                expected=merged[f"{statistic}_python"].to_numpy(),
                actual=merged[f"{statistic}_stata"].to_numpy(),
                atol=atol,
                rtol=rtol,
                detail=(
                    "Firth standard errors use inverse ordinary Fisher at the Stata "
                    "bias-reduced estimate."
                    if model == "firth_binary_logit" and statistic == "standard_error"
                    else ""
                ),
            )
        )

    expected_v = python_covariance.loc[python_covariance["model"] == model]
    actual_v = stata_covariance.loc[stata_covariance["model"] == model]
    merged_v = expected_v.merge(
        actual_v,
        on=["model", "dataset", "row_parameter", "column_parameter"],
        how="outer",
        suffixes=("_python", "_stata"),
        indicator=True,
        validate="one_to_one",
    )
    missing_v = [
        f"{row} x {column}"
        for row, column in merged_v.loc[
            merged_v["_merge"] != "both", ["row_parameter", "column_parameter"]
        ].itertuples(index=False, name=None)
    ]
    if missing_v:
        report.append(
            _missing_key_failure(
                model=model, dataset=dataset, statistic="covariance_keys", missing=missing_v
            )
        )
    else:
        atol, rtol = _comparison_tolerance(spec, "covariance")
        report.append(
            _numeric_check(
                model=model,
                dataset=dataset,
                statistic="covariance",
                expected=merged_v["covariance_python"].to_numpy(),
                actual=merged_v["covariance_stata"].to_numpy(),
                atol=atol,
                rtol=rtol,
                detail=(
                    "Full raw-to-canonical Jacobian applied; Firth uses aligned inverse "
                    "ordinary Fisher."
                    if model == "firth_binary_logit"
                    else "Full raw-to-canonical Jacobian applied."
                ),
            )
        )
    return report


def _single_model_row(frame: pd.DataFrame, model: str, *, label: str) -> pd.Series:
    selected = frame.loc[frame["model"] == model]
    if len(selected) != 1:
        raise ValueError(f"{label} must have one row for {model}; found {len(selected)}")
    return selected.iloc[0]


def _as_float(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else np.nan


def _fit_statistics_for_model(model: str) -> list[str]:
    if model == "firth_binary_logit":
        return ["nobs", "n_params", "loglike", "penalized_loglike", "jeffreys_penalty"]
    if model in {"exponential_duration", "weibull_duration"}:
        return ["nobs", "n_params", "n_events"]
    if model == "fixed_effects_ordered_logit":
        return [
            "nobs",
            "n_params",
            "loglike",
            "n_groups",
            "n_contributing_entities",
            "n_cutoff_clones",
            "n_pseudo_observations",
        ]
    return [
        "nobs",
        "n_params",
        "loglike",
        "aic",
        "bic",
        "n_groups",
        "n_events",
        "n_censored",
        "n_interval",
        "n_exact",
        "n_left_censored",
        "n_right_censored",
    ]


def _compare_fit(
    *,
    model: str,
    spec: dict[str, Any],
    python_fit: pd.DataFrame,
    stata_fit: pd.DataFrame,
    aligned_fit: dict[str, float],
) -> list[dict[str, Any]]:
    dataset = str(spec["dataset"])
    expected = _single_model_row(python_fit, model, label="Python fit")
    actual = _single_model_row(stata_fit, model, label="Stata fit")
    if str(actual["dataset"]) != dataset:
        raise ValueError(f"{model} Stata fit has incorrect dataset label")
    report: list[dict[str, Any]] = []
    for statistic in _fit_statistics_for_model(model):
        expected_value = _as_float(expected.get(statistic, np.nan))
        if not np.isfinite(expected_value):
            continue
        actual_value = (
            float(aligned_fit[statistic])
            if statistic in aligned_fit
            else _as_float(actual.get(statistic, np.nan))
        )
        atol, rtol = _comparison_tolerance(spec, statistic)
        report.append(
            _numeric_check(
                model=model,
                dataset=dataset,
                statistic=statistic,
                expected=np.array([expected_value]),
                actual=np.array([actual_value]),
                atol=atol,
                rtol=rtol,
                detail=(
                    "Recomputed at Stata Firth coefficients from the prepared sample."
                    if model == "firth_binary_logit" and statistic in aligned_fit
                    else ""
                ),
            )
        )
    for statistic in ("converged", "inference_valid"):
        actual_value = _as_float(actual.get(statistic, np.nan))
        report.append(
            _numeric_check(
                model=model,
                dataset=dataset,
                statistic=f"stata_{statistic}",
                expected=np.array([1.0]),
                actual=np.array([actual_value]),
                atol=0.0,
            )
        )
    return report


def _prediction_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("category", "time"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        result[f"__{column}_key"] = [
            "<NA>" if not np.isfinite(value) else format(float(value), ".17g")
            for value in numeric.to_numpy(dtype=float)
        ]
    result["__obs_key"] = pd.to_numeric(result["obs_id"], errors="raise").astype("int64")
    return result


def _compare_predictions(
    *,
    model: str,
    spec: dict[str, Any],
    python_predictions: pd.DataFrame,
    stata_predictions: pd.DataFrame,
) -> dict[str, Any]:
    dataset = str(spec["dataset"])
    if model == "fixed_effects_ordered_logit":
        return {
            "model": model,
            "dataset": dataset,
            "statistic": "prediction",
            "compared": 0,
            "max_abs_difference": np.nan,
            "max_allowed_difference": np.nan,
            "atol": np.nan,
            "rtol": np.nan,
            "status": "NOT_APPLICABLE",
            "detail": "BUC certificate is intentionally limited to slopes, composite LL, and counts.",
        }
    expected = _prediction_key_frame(
        python_predictions.loc[python_predictions["model"] == model]
    )
    actual = _prediction_key_frame(
        stata_predictions.loc[stata_predictions["model"] == model]
    )
    keys = ["model", "dataset", "__obs_key", "prediction", "__category_key", "__time_key"]
    _assert_unique(expected, keys, label=f"{model} Python predictions")
    _assert_unique(actual, keys, label=f"{model} Stata predictions")
    merged = expected.merge(
        actual[keys + ["value"]],
        on=keys,
        how="left",
        suffixes=("_python", "_stata"),
        indicator=True,
        validate="one_to_one",
    )
    missing = [
        f"obs={obs},pred={prediction},cat={category},time={time}"
        for obs, prediction, category, time in merged.loc[
            merged["_merge"] != "both",
            ["__obs_key", "prediction", "__category_key", "__time_key"],
        ].itertuples(index=False, name=None)
    ]
    if missing:
        return _missing_key_failure(
            model=model, dataset=dataset, statistic="prediction", missing=missing
        )
    atol, rtol = _comparison_tolerance(spec, "prediction")
    return _numeric_check(
        model=model,
        dataset=dataset,
        statistic="prediction",
        expected=merged["value_python"].to_numpy(),
        actual=merged["value_stata"].to_numpy(),
        atol=atol,
        rtol=rtol,
        detail="Only the manifest-declared Python prediction key grid is compared.",
    )


def _derive_buc_counts(workdir: Path, spec: dict[str, Any]) -> dict[str, int]:
    data = pd.read_csv(workdir / str(spec["data_file"]))
    entity = str(spec["entity"])
    outcome = str(spec["outcome"])
    valid_clones: list[tuple[Any, int, int]] = []
    contributing: set[Any] = set()
    pseudo = 0
    for entity_value, group in data.groupby(entity, sort=False):
        for cutoff in spec["buc_cutoffs"]:
            binary = (group[outcome].to_numpy() >= cutoff).astype(int)
            if binary.min() == binary.max():
                continue
            valid_clones.append((entity_value, int(cutoff), len(group)))
            contributing.add(entity_value)
            pseudo += len(group)
    return {
        "n_groups": int(data[entity].nunique()),
        "n_contributing_entities": len(contributing),
        "n_cutoff_clones": len(valid_clones),
        "n_pseudo_observations": pseudo,
    }


def _compare_buc_contract(
    *, workdir: Path, spec: dict[str, Any], stata_fit: pd.DataFrame
) -> list[dict[str, Any]]:
    model = "fixed_effects_ordered_logit"
    dataset = str(spec["dataset"])
    actual = _single_model_row(stata_fit, model, label="Stata fit")
    derived = _derive_buc_counts(workdir, spec)
    report = []
    for statistic, expected in derived.items():
        report.append(
            _numeric_check(
                model=model,
                dataset=dataset,
                statistic=f"prepared_contract.{statistic}",
                expected=np.array([expected]),
                actual=np.array([_as_float(actual.get(statistic, np.nan))]),
                atol=0.0,
                detail="Independently reconstructed from panel_nlswork and manifest cutoffs.",
            )
        )
    return report


def _availability_row(
    model: str, spec: dict[str, Any], state: str, reason: str
) -> dict[str, Any]:
    return {
        "model": model,
        "dataset": str(spec["dataset"]),
        "statistic": "availability",
        "compared": 0,
        "max_abs_difference": np.nan,
        "max_allowed_difference": np.nan,
        "atol": np.nan,
        "rtol": np.nan,
        "status": "PASS" if state == "RUN" else "SKIP",
        "detail": reason,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.10g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _write_evidence(
    *,
    workdir: Path,
    manifest: dict[str, Any],
    metadata: dict[str, str],
    statuses: dict[str, dict[str, str]],
    report: pd.DataFrame,
    canonical_estimates: pd.DataFrame,
    canonical_covariance: pd.DataFrame,
    transform_audit: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    stata_dir = workdir / "stata"
    canonical_estimates.to_csv(
        stata_dir / "estimates_canonical.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    canonical_covariance.to_csv(
        stata_dir / "covariance_canonical.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    report_path = workdir / "comparison_report.csv"
    report.to_csv(report_path, index=False, float_format="%.10g", lineterminator="\n")

    failures = report.loc[report["status"] == "FAIL"]
    result = "PASS" if failures.empty else "FAIL"
    skipped = [model for model, item in statuses.items() if item["status"] == "SKIP"]
    claim = (
        "All required promoted Stata checks passed within the manifest tolerances. "
        "The claim is model-, dataset-, estimand-, quadrature-, and covariance-specific."
        if result == "PASS"
        else "No promoted Stata parity claim is supported because required checks failed."
    )
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(stata_dir.iterdir())
        if path.is_file()
        and path.name
        in {
            "estimates_raw.csv",
            "covariance_raw.csv",
            "fit.csv",
            "predictions.csv",
            "model_status.csv",
            "metadata.txt",
            "stata_run.log",
            "estimates_canonical.csv",
            "covariance_canonical.csv",
        }
    }
    certificate = {
        "schema_version": 1,
        "suite": manifest["suite"],
        "result": result,
        "claim": claim,
        "manifest_sha256": _sha256(workdir / "manifest.json"),
        "dependency_versions": manifest.get("dependency_versions", {}),
        "stata_run_metadata": metadata,
        "model_evidence": {
            model: {
                "status": statuses[model]["status"],
                "reason": statuses[model]["reason"],
                "evidence_class": EVIDENCE_CLASSES[model],
            }
            for model in manifest["model_order"]
        },
        "required_models": [
            model
            for model in manifest["model_order"]
            if model not in {"firth_binary_logit", "gamma_duration"}
        ],
        "optional_models": ["firth_binary_logit"],
        "unsupported_models": ["gamma_duration"],
        "skipped_models": skipped,
        "raw_and_canonical_stata_sha256": artifacts,
        "python_reference_sha256": {
            path: digest
            for path, digest in manifest["files"].items()
            if str(path).startswith("python/")
        },
        "prepared_data_sha256": {
            path: digest
            for path, digest in manifest["files"].items()
            if str(path).startswith("data/")
        },
        "raw_to_canonical_transforms": transform_audit,
        "checks": int(len(report)),
        "failed_checks": int(len(failures)),
        "skipped_checks": int(report["status"].eq("SKIP").sum()),
        "not_applicable_checks": int(report["status"].eq("NOT_APPLICABLE").sum()),
    }
    certificate_path = workdir / "parity_certificate.json"
    certificate_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_lines = [
        "# Promoted Stata parity comparison",
        "",
        f"- Suite: `{manifest['suite']}`",
        f"- Result: **{result}**",
        f"- Failed checks: {len(failures)}",
        f"- Explicit skips: {', '.join(skipped) if skipped else 'none'}",
        "",
        claim,
        "",
        "Gamma duration is excluded because Stata `streg, distribution(ggamma)` is "
        "not the ordinary Gamma likelihood. Firth is optional and is compared only "
        "when the user installed `firthlogit` manually. Firth covariance is aligned "
        "to limiteddepkit as inverse ordinary Fisher at the bias-reduced estimate.",
        "",
        "Geometric duration and BUC Ordered Logit use declared pseudo-sample "
        "likelihood identities. Random-effects Ordered Probit is numerical "
        "Gauss-Hermite parity at Q=20, not an analytic identity.",
        "",
        "## Numerical checks",
        "",
        _markdown_table(report),
        "",
    ]
    summary_path = workdir / "comparison_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return report_path, summary_path, certificate_path


def main() -> int:
    args = _parse_args()
    workdir = args.workdir.resolve()
    _invalidate_promoted_evidence(workdir)
    manifest = _load_manifest(workdir)
    python_dir = workdir / "python"
    stata_dir = workdir / "stata"
    required_stata = {
        "estimates": stata_dir / "estimates_raw.csv",
        "covariance": stata_dir / "covariance_raw.csv",
        "fit": stata_dir / "fit.csv",
        "predictions": stata_dir / "predictions.csv",
        "status": stata_dir / "model_status.csv",
        "metadata": stata_dir / "metadata.txt",
        "log": stata_dir / "stata_run.log",
    }
    missing = [str(path) for path in required_stata.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stata outputs are incomplete. Run limiteddepkit_real_data.do first. Missing: "
            + ", ".join(missing)
        )

    python_estimates = pd.read_csv(python_dir / "estimates.csv")
    python_covariance = pd.read_csv(python_dir / "covariance.csv")
    python_fit = pd.read_csv(python_dir / "fit.csv")
    python_predictions = pd.read_csv(python_dir / "predictions.csv")
    raw_estimates = pd.read_csv(required_stata["estimates"])
    raw_covariance = pd.read_csv(required_stata["covariance"])
    stata_fit = pd.read_csv(required_stata["fit"])
    stata_predictions = pd.read_csv(required_stata["predictions"])
    status_frame = pd.read_csv(required_stata["status"], keep_default_na=False)

    schemas = manifest["output_schemas"]
    _validate_columns(
        python_estimates, schemas["python/estimates.csv"], label="Python estimates"
    )
    _validate_columns(
        python_covariance, schemas["python/covariance.csv"], label="Python covariance"
    )
    _validate_columns(python_fit, schemas["python/fit.csv"], label="Python fit")
    _validate_columns(
        python_predictions, schemas["python/predictions.csv"], label="Python predictions"
    )
    _assert_unique(
        python_estimates, ["model", "parameter"], label="Python estimates"
    )
    _assert_unique(
        python_covariance,
        ["model", "row_parameter", "column_parameter"],
        label="Python covariance",
    )
    _assert_unique(python_fit, ["model"], label="Python fit")
    _validate_columns(raw_estimates, RAW_ESTIMATE_COLUMNS, label="raw Stata estimates")
    _validate_columns(raw_covariance, RAW_COVARIANCE_COLUMNS, label="raw Stata covariance")
    _validate_columns(stata_fit, ("model", "dataset", "nobs", "n_params", "loglike"), label="Stata fit")
    _validate_columns(stata_predictions, PREDICTION_COLUMNS, label="Stata predictions")
    _assert_unique(stata_fit, ["model"], label="Stata fit")

    metadata = _parse_metadata(required_stata["metadata"])
    _validate_metadata(metadata, manifest)
    statuses = _validate_status(status_frame, manifest)
    run_models = {
        model for model, item in statuses.items() if item["status"] == "RUN"
    }
    for table_name, frame in {
        "raw estimates": raw_estimates,
        "raw covariance": raw_covariance,
        "Stata fit": stata_fit,
    }.items():
        observed = set(frame["model"].astype(str))
        if observed != run_models:
            raise ValueError(
                f"{table_name} model inventory differs from RUN statuses; "
                f"missing={sorted(run_models - observed)}, unexpected={sorted(observed - run_models)}"
            )
    prediction_models = set(stata_predictions["model"].astype(str))
    unexpected_prediction_models = prediction_models - run_models
    if unexpected_prediction_models:
        raise ValueError(
            "Stata predictions contain skipped or undeclared models: "
            + ", ".join(sorted(unexpected_prediction_models))
        )

    canonical_estimate_frames: list[pd.DataFrame] = []
    canonical_covariance_frames: list[pd.DataFrame] = []
    transform_audit: list[dict[str, Any]] = []
    aligned_fit_by_model: dict[str, dict[str, float]] = {}
    for model in manifest["model_order"]:
        if model not in run_models:
            continue
        estimates, covariance, audit, aligned_fit = _canonicalize_model(
            model=model,
            workdir=workdir,
            spec=manifest["model_specs"][model],
            raw_estimates=raw_estimates,
            raw_covariance=raw_covariance,
        )
        canonical_estimate_frames.append(estimates)
        canonical_covariance_frames.append(covariance)
        transform_audit.extend(audit)
        aligned_fit_by_model[model] = aligned_fit
    canonical_estimates = pd.concat(canonical_estimate_frames, ignore_index=True)
    canonical_covariance = pd.concat(canonical_covariance_frames, ignore_index=True)

    report_rows: list[dict[str, Any]] = []
    for model in manifest["model_order"]:
        spec = manifest["model_specs"][model]
        state = statuses[model]
        report_rows.append(
            _availability_row(model, spec, state["status"], state["reason"])
        )
        if state["status"] != "RUN":
            continue
        report_rows.extend(
            _compare_parameters(
                model=model,
                spec=spec,
                python_estimates=python_estimates,
                python_covariance=python_covariance,
                stata_estimates=canonical_estimates,
                stata_covariance=canonical_covariance,
            )
        )
        report_rows.extend(
            _compare_fit(
                model=model,
                spec=spec,
                python_fit=python_fit,
                stata_fit=stata_fit,
                aligned_fit=aligned_fit_by_model[model],
            )
        )
        report_rows.append(
            _compare_predictions(
                model=model,
                spec=spec,
                python_predictions=python_predictions,
                stata_predictions=stata_predictions,
            )
        )
    if "fixed_effects_ordered_logit" in run_models:
        report_rows.extend(
            _compare_buc_contract(
                workdir=workdir,
                spec=manifest["model_specs"]["fixed_effects_ordered_logit"],
                stata_fit=stata_fit,
            )
        )

    report = pd.DataFrame(report_rows)
    report_path, summary_path, certificate_path = _write_evidence(
        workdir=workdir,
        manifest=manifest,
        metadata=metadata,
        statuses=statuses,
        report=report,
        canonical_estimates=canonical_estimates,
        canonical_covariance=canonical_covariance,
        transform_audit=transform_audit,
    )
    print(report.to_string(index=False))
    print(f"\nComparison report: {report_path}")
    print(f"Comparison summary: {summary_path}")
    print(f"Parity certificate: {certificate_path}")
    failures = int(report["status"].eq("FAIL").sum())
    if failures:
        print(f"Promoted Stata parity: FAIL ({failures} failed checks)")
        return 1
    print("Promoted Stata parity: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Promoted Stata comparison cannot continue: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
