"""Compare raw Stata parity exports with limiteddepkit reference results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CanonicalParameter:
    """One canonical parameter derived from a raw Stata coefficient."""

    name: str
    source_position: int
    estimate: float
    derivative: float
    standard_error: float


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "binary_logit": {
        "kind": "binary",
        "features": ("intercept", "x1", "x2"),
        "required": True,
    },
    "binary_probit": {
        "kind": "binary",
        "features": ("intercept", "x1", "x2"),
        "required": True,
    },
    "ordered_logit": {
        "kind": "ordered",
        "features": ("ox1", "ox2"),
        "required": True,
    },
    "ordered_probit": {
        "kind": "ordered",
        "features": ("ox1", "ox2"),
        "required": True,
    },
    "generalized_ordered_logit": {
        "kind": "generalized",
        "features": ("gx1", "gx2"),
        "required": False,
    },
    "partial_proportional_odds": {
        "kind": "partial",
        "features": ("gx1", "gx2"),
        "varying": ("gx1",),
        "required": False,
    },
    "random_effects_ordered_logit": {
        "kind": "random_effects",
        "features": ("x1", "x2"),
        "required": True,
    },
    "dynamic_random_effects_ordered_logit": {
        "kind": "random_effects",
        "features": (
            "x1",
            "state_1",
            "state_2",
            "initial_1",
            "initial_2",
            "initial_x1",
            "mean_x1",
        ),
        "feature_map": {
            "x1": "x1",
            "state_1": "state[1]",
            "state_2": "state[2]",
            "initial_1": "initial[1]",
            "initial_2": "initial[2]",
            "initial_x1": "initial_x[x1]",
            "mean_x1": "mean[x1]",
        },
        "required": True,
    },
}

TOLERANCES: dict[str, dict[str, float]] = {
    "binary": {
        "estimate": 2e-6,
        "standard_error": 2e-6,
        "covariance": 3e-6,
        "loglike": 1e-7,
        "probability": 2e-6,
    },
    "ordered": {
        "estimate": 5e-5,
        "standard_error": 5e-5,
        "covariance": 1e-4,
        "loglike": 5e-6,
        "probability": 5e-5,
    },
    "generalized": {
        "estimate": 2e-4,
        "standard_error": 5e-4,
        "covariance": 1e-3,
        "loglike": 2e-5,
        "probability": 2e-4,
    },
    "partial": {
        "estimate": 2e-4,
        "standard_error": 5e-4,
        "covariance": 1e-3,
        "loglike": 2e-5,
        "probability": 2e-4,
    },
    "random_effects": {
        "estimate": 1e-3,
        "standard_error": 2e-3,
        "covariance": 3e-3,
        "loglike": 1e-3,
        "probability": 1e-3,
    },
}

PYTHON_REFERENCE_SCHEMAS = {
    "estimates": (
        ("model", "parameter", "estimate", "standard_error"),
        (("model", "parameter"),),
    ),
    "covariance": (
        ("model", "row_parameter", "column_parameter", "covariance"),
        (("model", "row_parameter", "column_parameter"),),
    ),
    "fit": (
        ("model", "nobs", "n_groups", "n_params", "loglike", "aic", "bic", "converged"),
        (("model",),),
    ),
    "predictions": (
        ("model", "obs_id", "category", "probability"),
        (("model", "obs_id", "category"),),
    ),
}

RAW_STATA_SCHEMAS = {
    "estimates": (
        ("model", "position", "stata_parameter", "estimate", "standard_error"),
        (("model", "position"), ("model", "stata_parameter")),
    ),
    "covariance": (
        (
            "model",
            "row_position",
            "column_position",
            "row_parameter",
            "column_parameter",
            "covariance",
        ),
        (("model", "row_position", "column_position"),),
    ),
    "fit": (
        ("model", "nobs", "n_groups", "n_params", "loglike", "aic", "bic", "converged"),
        (("model",),),
    ),
    "predictions": (
        ("model", "obs_id", "category", "probability"),
        (("model", "obs_id", "category"),),
    ),
}

CANONICAL_STATA_SCHEMAS = {
    "estimates": (
        ("model", "parameter", "estimate", "standard_error"),
        (("model", "parameter"),),
    ),
    "covariance": (
        ("model", "row_parameter", "column_parameter", "covariance"),
        (("model", "row_parameter", "column_parameter"),),
    ),
}


def _validate_table(
    frame: pd.DataFrame,
    *,
    name: str,
    required_columns: tuple[str, ...],
    unique_keys: tuple[tuple[str, ...], ...],
    require_rows: bool = True,
) -> None:
    """Reject malformed tables before selection or merging can hide bad rows."""

    if not frame.columns.is_unique:
        duplicated_columns = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{name} has duplicate columns: {duplicated_columns}")

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")
    if require_rows and frame.empty:
        raise ValueError(f"{name} has no rows")

    for key_columns in unique_keys:
        key_frame = frame.loc[:, list(key_columns)]
        null_keys = key_frame.isna().any(axis=1)
        if null_keys.any():
            raise ValueError(
                f"{name} has {int(null_keys.sum())} row(s) with null key values in "
                f"{list(key_columns)}"
            )
        duplicate_keys = frame.duplicated(subset=list(key_columns), keep=False)
        if duplicate_keys.any():
            examples = (
                frame.loc[duplicate_keys, list(key_columns)]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(f"{name} has duplicate keys for {list(key_columns)}: {examples}")


def _validate_schema_group(
    frames: dict[str, pd.DataFrame],
    schemas: dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]],
    *,
    label: str,
    require_rows: bool = True,
) -> None:
    for table_name, (required_columns, unique_keys) in schemas.items():
        _validate_table(
            frames[table_name],
            name=f"{label} {table_name}",
            required_columns=required_columns,
            unique_keys=unique_keys,
            require_rows=require_rows,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workdir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "work",
        help="Parity work directory containing python/ and stata/ outputs.",
    )
    parser.add_argument(
        "--require-flexible",
        action="store_true",
        help="Fail if the optional gologit2 models were not run.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_metadata(text: str) -> dict[str, str]:
    """Parse the strict key-value metadata emitted by the maintained do-files."""

    metadata: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Malformed Stata metadata line {line_number}: {raw_line!r}")
        if key in metadata:
            raise ValueError(f"Duplicate Stata metadata key: {key}")
        metadata[key] = value.strip()
    return metadata


def _verify_run_metadata(metadata: dict[str, str], manifest: dict[str, Any]) -> dict[str, str]:
    """Require a completed Stata run for the currently prepared suite."""

    expected_suite = str(manifest.get("suite", "controlled_synthetic_certification"))
    expected_values = {
        "suite": expected_suite,
        "run_completed": "1",
        "panel_prediction": "conditional_fixedonly",
    }
    mismatches = [
        f"{key}={metadata.get(key)!r}, expected {expected!r}"
        for key, expected in expected_values.items()
        if metadata.get(key) != expected
    ]

    stata_version = metadata.get("stata_version", "").strip()
    if not stata_version:
        mismatches.append("missing stata_version")
    gologit2_installed = metadata.get("gologit2_installed")
    if gologit2_installed not in {"0", "1"}:
        mismatches.append(
            f"gologit2_installed must be recorded as 0 or 1; got {gologit2_installed!r}"
        )

    if mismatches:
        raise RuntimeError(
            "Stata metadata does not identify a completed matching run: " + "; ".join(mismatches)
        )

    verified = {
        **expected_values,
        "stata_version": stata_version,
        "gologit2_installed": str(gologit2_installed),
    }
    if "gologit2_path" in metadata:
        verified["gologit2_path"] = metadata["gologit2_path"]
    if "source_release" in metadata:
        verified["source_release"] = metadata["source_release"]
    return verified


def _verify_panel_quadrature(
    metadata: dict[str, str],
    manifest: dict[str, Any],
    model_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Verify the actual ``e(intmethod)`` and ``e(n_quad)`` for every panel fit."""

    expected_method = str(manifest.get("quadrature_method", "ghermite")).lower()
    expected_points = int(manifest["quadrature_points"])
    panel_models = [
        model for model, spec in model_specs.items() if spec.get("kind") == "random_effects"
    ]
    mismatches: list[str] = []
    actual_settings: dict[str, dict[str, Any]] = {}

    for model in panel_models:
        method_key = f"{model}.intmethod"
        points_key = f"{model}.n_quad"
        actual_method = metadata.get(method_key)
        raw_points = metadata.get(points_key)

        if actual_method is None:
            mismatches.append(f"missing {method_key}")
        elif actual_method.lower() != expected_method:
            mismatches.append(f"{method_key}={actual_method!r}, expected {expected_method!r}")

        actual_points: int | None = None
        if raw_points is None:
            mismatches.append(f"missing {points_key}")
        else:
            try:
                numeric_points = float(raw_points)
                if not numeric_points.is_integer():
                    raise ValueError
                actual_points = int(numeric_points)
            except ValueError:
                mismatches.append(f"{points_key}={raw_points!r} is not an integer")
            else:
                if actual_points != expected_points:
                    mismatches.append(f"{points_key}={actual_points}, expected {expected_points}")

        if actual_method is not None and actual_points is not None:
            actual_settings[model] = {
                "intmethod": actual_method,
                "n_quad": actual_points,
            }

    if mismatches:
        raise RuntimeError(
            "Stata metadata does not confirm matching nonadaptive quadrature: "
            + "; ".join(mismatches)
        )
    return actual_settings


def _verify_manifest(workdir: Path) -> dict[str, Any]:
    manifest_path = workdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run the appropriate parity preparation script "
            "before comparison."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for relative_path, expected_hash in manifest["files"].items():
        path = workdir / relative_path
        if not path.exists():
            mismatches.append(f"missing {relative_path}")
        elif _sha256(path) != expected_hash:
            mismatches.append(f"hash mismatch for {relative_path}")
    if mismatches:
        raise RuntimeError("Parity inputs changed: " + "; ".join(mismatches))
    return manifest


def _split_stata_name(full_name: str) -> tuple[str, str]:
    text = str(full_name).strip()
    if ":" in text:
        equation, term = text.rsplit(":", 1)
    else:
        equation, term = "", text
    return equation.strip(), term.strip().lstrip("/")


def _cut_number(full_name: str) -> int | None:
    match = re.search(r"cut_?(\d+)", str(full_name), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _canonical_mapping(
    model: str,
    raw_estimates: pd.DataFrame,
    model_specs: dict[str, dict[str, Any]] | None = None,
) -> list[CanonicalParameter]:
    specs = MODEL_SPECS if model_specs is None else model_specs
    spec = specs[model]
    kind = spec["kind"]
    rows = raw_estimates.loc[raw_estimates["model"] == model].sort_values("position")
    rows = rows.reset_index(drop=True)
    mappings: list[CanonicalParameter] = []

    def add(
        row: pd.Series,
        name: str,
        *,
        transform: str = "identity",
    ) -> None:
        raw_estimate = float(row["estimate"])
        raw_standard_error = float(row["standard_error"])
        if transform == "identity":
            estimate = raw_estimate
            derivative = 1.0
        elif transform == "negative":
            estimate = -raw_estimate
            derivative = -1.0
        elif transform == "exp":
            estimate = float(np.exp(raw_estimate))
            derivative = estimate
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"Unknown transformation: {transform}")
        mappings.append(
            CanonicalParameter(
                name=name,
                source_position=int(row["position"]),
                estimate=estimate,
                derivative=derivative,
                standard_error=abs(derivative) * raw_standard_error,
            )
        )

    if kind == "binary":
        for _, row in rows.iterrows():
            _, term = _split_stata_name(row["stata_parameter"])
            if term in spec["features"]:
                add(row, term)
        return mappings

    if kind in {"ordered", "random_effects"}:
        feature_map = spec.get("feature_map", {feature: feature for feature in spec["features"]})
        cuts: list[tuple[int, pd.Series]] = []
        for _, row in rows.iterrows():
            full_name = str(row["stata_parameter"])
            lower_name = full_name.lower()
            _, term = _split_stata_name(full_name)
            cut = _cut_number(full_name)
            if cut is not None:
                cuts.append((cut, row))
            elif kind == "random_effects" and ("lns" in lower_name or "lnsig" in lower_name):
                add(row, "sigma_entity", transform="exp")
            elif term in feature_map:
                add(row, feature_map[term])
        for cut, row in sorted(cuts, key=lambda item: item[0]):
            add(row, f"threshold: {cut - 1} | {cut}")
        return mappings

    feature_names = set(spec["features"])
    equation_order: list[str] = []
    parsed_rows: list[tuple[pd.Series, str, str]] = []
    for _, row in rows.iterrows():
        equation, term = _split_stata_name(row["stata_parameter"])
        if term in feature_names or term == "_cons":
            parsed_rows.append((row, equation, term))
            if equation not in equation_order:
                equation_order.append(equation)
    equation_index = {equation: index for index, equation in enumerate(equation_order)}

    common_seen: set[str] = set()
    varying = set(spec.get("varying", ()))
    for row, equation, term in parsed_rows:
        split = equation_index[equation]
        split_name = f"{split} | {split + 1}"
        if term == "_cons":
            add(row, f"threshold: {split_name}", transform="negative")
        elif kind == "generalized":
            add(row, f"slope {split_name}: {term}")
        elif term in varying:
            add(row, f"varying {split_name}: {term}")
        elif term not in common_seen:
            add(row, f"common: {term}")
            common_seen.add(term)
    return mappings


def _canonical_stata_results(
    model: str,
    raw_estimates: pd.DataFrame,
    raw_covariance: pd.DataFrame,
    model_specs: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = _canonical_mapping(model, raw_estimates, model_specs)
    estimates = pd.DataFrame(
        {
            "model": model,
            "parameter": [item.name for item in mapping],
            "estimate": [item.estimate for item in mapping],
            "standard_error": [item.standard_error for item in mapping],
        }
    )
    if not mapping:
        covariance = pd.DataFrame(columns=CANONICAL_STATA_SCHEMAS["covariance"][0])
        _validate_schema_group(
            {"estimates": estimates, "covariance": covariance},
            CANONICAL_STATA_SCHEMAS,
            label=f"Canonical Stata results for {model}",
        )
        return estimates, covariance  # pragma: no cover - validation always raises

    model_covariance = raw_covariance.loc[raw_covariance["model"] == model]
    max_position = int(
        max(
            model_covariance["row_position"].max(),
            model_covariance["column_position"].max(),
        )
    )
    raw_matrix = np.full((max_position, max_position), np.nan)
    for _, row in model_covariance.iterrows():
        raw_matrix[
            int(row["row_position"]) - 1,
            int(row["column_position"]) - 1,
        ] = float(row["covariance"])

    transformed = np.empty((len(mapping), len(mapping)))
    for row_index, row_parameter in enumerate(mapping):
        for column_index, column_parameter in enumerate(mapping):
            transformed[row_index, column_index] = (
                row_parameter.derivative
                * raw_matrix[
                    row_parameter.source_position - 1,
                    column_parameter.source_position - 1,
                ]
                * column_parameter.derivative
            )
    covariance_rows = [
        {
            "model": model,
            "row_parameter": row_parameter.name,
            "column_parameter": column_parameter.name,
            "covariance": transformed[row_index, column_index],
        }
        for row_index, row_parameter in enumerate(mapping)
        for column_index, column_parameter in enumerate(mapping)
    ]
    covariance = pd.DataFrame(
        covariance_rows,
        columns=CANONICAL_STATA_SCHEMAS["covariance"][0],
    )
    _validate_schema_group(
        {"estimates": estimates, "covariance": covariance},
        CANONICAL_STATA_SCHEMAS,
        label=f"Canonical Stata results for {model}",
    )
    return estimates, covariance


def _single_fit_row(frame: pd.DataFrame, *, model: str, source: str) -> pd.Series:
    rows = frame.loc[frame["model"] == model]
    if len(rows) != 1:
        raise ValueError(
            f"{source} must contain exactly one fit row for {model!r}; found {len(rows)}"
        )
    return rows.iloc[0]


def _comparison_row(
    *,
    model: str,
    statistic: str,
    differences: pd.Series,
    tolerance: float,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    numeric_differences = pd.to_numeric(differences, errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    finite_mask = np.isfinite(numeric_differences)
    finite = numeric_differences[finite_mask]
    nonfinite_count = int((~finite_mask).sum())
    max_difference = float(finite.max()) if finite.size else np.nan
    missing = missing or []
    passed = (
        not missing
        and numeric_differences.size > 0
        and nonfinite_count == 0
        and max_difference <= tolerance
    )
    detail_parts: list[str] = []
    if missing:
        detail_parts.append("missing keys: " + ", ".join(missing[:8]))
    if nonfinite_count:
        detail_parts.append(f"non-finite differences: {nonfinite_count}")
    if numeric_differences.size == 0:
        detail_parts.append("no comparisons")
    return {
        "model": model,
        "statistic": statistic,
        "compared": int(finite.size),
        "nonfinite": nonfinite_count,
        "max_abs_difference": max_difference,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
        "detail": "; ".join(detail_parts),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe without adding an optional tabulate dependency."""

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
    report: pd.DataFrame,
    stata_estimates: pd.DataFrame,
    stata_covariance: pd.DataFrame,
    stata_run_metadata: dict[str, str],
    stata_panel_quadrature: dict[str, dict[str, Any]],
    require_optional: bool = False,
) -> tuple[Path, Path]:
    """Write human- and machine-readable evidence after a completed comparison."""

    stata_dir = workdir / "stata"
    stata_estimates.to_csv(
        stata_dir / "estimates_canonical.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    stata_covariance.to_csv(
        stata_dir / "covariance_canonical.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    failures = report.loc[report["status"] == "FAIL"]
    result = "PASS" if failures.empty else "FAIL"
    suite = str(manifest.get("suite", "controlled_synthetic_certification"))
    model_specs = manifest.get("comparison_model_specs", MODEL_SPECS)
    required_models = [
        model
        for model, spec in model_specs.items()
        if bool(spec.get("required", False)) or require_optional
    ]
    optional_models = [model for model in model_specs if model not in required_models]
    available_models = sorted(set(stata_estimates["model"].astype(str)))
    stata_artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(stata_dir.iterdir())
        if path.is_file()
        and path.name
        in {
            "estimates_raw.csv",
            "covariance_raw.csv",
            "fit.csv",
            "predictions.csv",
            "metadata.txt",
            "stata_run.log",
            "estimates_canonical.csv",
            "covariance_canonical.csv",
        }
    }
    if result == "PASS" and suite == "real_data_application":
        claim = (
            "The maintained real-data application checks passed within the declared "
            "benchmark tolerances; this does not broaden the controlled-fixture "
            "certification claim."
        )
    elif result == "PASS":
        claim = (
            "The maintained controlled fixtures passed the declared Stata parity "
            "checks within benchmark-specific tolerances."
        )
    else:
        claim = "No parity claim is supported because one or more required checks failed."

    certificate = {
        "schema_version": 1,
        "suite": suite,
        "result": result,
        "claim": claim,
        "limiteddepkit_version": manifest.get("limiteddepkit_version"),
        "panel_optimizer_tolerance": manifest.get("panel_optimizer_tolerance"),
        "quadrature_points": manifest.get("quadrature_points"),
        "stata_run_metadata": stata_run_metadata,
        "stata_panel_quadrature": stata_panel_quadrature,
        "required_models": required_models,
        "optional_models": optional_models,
        "available_models": available_models,
        "stata_artifact_sha256": stata_artifact_hashes,
        "python_reference_sha256": {
            relative_path: digest
            for relative_path, digest in manifest.get("files", {}).items()
            if str(relative_path).startswith("python/")
        },
        "checks": int(len(report)),
        "failed_checks": int(len(failures)),
        "skipped_checks": int(report["status"].eq("SKIP").sum()),
    }
    certificate_path = workdir / "parity_certificate.json"
    certificate_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_lines = [
        "# limiteddepkit Stata parity comparison",
        "",
        f"- Suite: `{suite}`",
        f"- Result: **{result}**",
        f"- Required models: {len(required_models)}",
        f"- Optional models: {len(optional_models)}",
        f"- Failed checks: {len(failures)}",
        "",
        claim,
        "",
        "## Numerical checks",
        "",
        _markdown_table(report),
        "",
        "The claim is benchmark-specific. It does not imply universal equality across "
        "datasets, optimizers, quadrature rules, covariance choices, or preprocessing.",
        "",
    ]
    summary_path = workdir / "comparison_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return summary_path, certificate_path


def _compare_model(
    model: str,
    python_estimates: pd.DataFrame,
    python_covariance: pd.DataFrame,
    python_fit: pd.DataFrame,
    python_predictions: pd.DataFrame,
    stata_estimates: pd.DataFrame,
    stata_covariance: pd.DataFrame,
    stata_fit: pd.DataFrame,
    stata_predictions: pd.DataFrame,
    model_specs: dict[str, dict[str, Any]] | None = None,
    tolerances: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    specs = MODEL_SPECS if model_specs is None else model_specs
    tolerance_table = TOLERANCES if tolerances is None else tolerances
    tolerance = tolerance_table[specs[model]["kind"]]
    report: list[dict[str, Any]] = []

    expected_estimates = python_estimates.loc[
        python_estimates["model"] == model,
        ["parameter", "estimate", "standard_error"],
    ]
    actual_estimates = stata_estimates.loc[
        stata_estimates["model"] == model,
        ["parameter", "estimate", "standard_error"],
    ]
    estimate_merge = expected_estimates.merge(
        actual_estimates,
        on="parameter",
        how="outer",
        suffixes=("_python", "_stata"),
        indicator=True,
        validate="one_to_one",
    )
    missing_parameters = (
        estimate_merge.loc[estimate_merge["_merge"] != "both", "parameter"].astype(str).tolist()
    )
    report.append(
        _comparison_row(
            model=model,
            statistic="estimate",
            differences=(
                estimate_merge["estimate_python"] - estimate_merge["estimate_stata"]
            ).abs(),
            tolerance=tolerance["estimate"],
            missing=missing_parameters,
        )
    )
    report.append(
        _comparison_row(
            model=model,
            statistic="standard_error",
            differences=(
                estimate_merge["standard_error_python"] - estimate_merge["standard_error_stata"]
            ).abs(),
            tolerance=tolerance["standard_error"],
            missing=missing_parameters,
        )
    )

    expected_covariance = python_covariance.loc[
        python_covariance["model"] == model,
        ["row_parameter", "column_parameter", "covariance"],
    ]
    actual_covariance = stata_covariance.loc[
        stata_covariance["model"] == model,
        ["row_parameter", "column_parameter", "covariance"],
    ]
    covariance_merge = expected_covariance.merge(
        actual_covariance,
        on=["row_parameter", "column_parameter"],
        how="outer",
        suffixes=("_python", "_stata"),
        indicator=True,
        validate="one_to_one",
    )
    missing_covariance = [
        f"{row_parameter} x {column_parameter}"
        for row_parameter, column_parameter in covariance_merge.loc[
            covariance_merge["_merge"] != "both",
            ["row_parameter", "column_parameter"],
        ].itertuples(index=False, name=None)
    ]
    report.append(
        _comparison_row(
            model=model,
            statistic="covariance",
            differences=(
                covariance_merge["covariance_python"] - covariance_merge["covariance_stata"]
            ).abs(),
            tolerance=tolerance["covariance"],
            missing=missing_covariance,
        )
    )

    expected_fit = _single_fit_row(python_fit, model=model, source="Python fit table")
    actual_fit = _single_fit_row(stata_fit, model=model, source="Stata fit table")
    nobs_difference = abs(float(expected_fit["nobs"]) - float(actual_fit["nobs"]))
    report.append(
        _comparison_row(
            model=model,
            statistic="nobs",
            differences=pd.Series([nobs_difference]),
            tolerance=0.0,
        )
    )
    for statistic, statistic_tolerance in (
        ("n_params", 0.0),
        ("loglike", tolerance["loglike"]),
        ("aic", 2.0 * tolerance["loglike"]),
        ("bic", 2.0 * tolerance["loglike"]),
    ):
        report.append(
            _comparison_row(
                model=model,
                statistic=statistic,
                differences=pd.Series(
                    [abs(float(expected_fit[statistic]) - float(actual_fit[statistic]))]
                ),
                tolerance=statistic_tolerance,
            )
        )

    expected_groups = pd.to_numeric(
        pd.Series([expected_fit.get("n_groups", np.nan)]), errors="coerce"
    ).iloc[0]
    if np.isfinite(expected_groups):
        actual_groups = pd.to_numeric(
            pd.Series([actual_fit.get("n_groups", np.nan)]), errors="coerce"
        ).iloc[0]
        report.append(
            _comparison_row(
                model=model,
                statistic="n_groups",
                differences=pd.Series([abs(expected_groups - actual_groups)]),
                tolerance=0.0,
            )
        )

    stata_converged = pd.to_numeric(
        pd.Series([actual_fit.get("converged", np.nan)]), errors="coerce"
    ).iloc[0]
    report.append(
        _comparison_row(
            model=model,
            statistic="stata_converged",
            differences=pd.Series([abs(1.0 - stata_converged)]),
            tolerance=0.0,
        )
    )

    expected_predictions = python_predictions.loc[
        python_predictions["model"] == model,
        ["obs_id", "category", "probability"],
    ].copy()
    actual_predictions = stata_predictions.loc[
        stata_predictions["model"] == model,
        ["obs_id", "category", "probability"],
    ].copy()
    # Python references intentionally select a small prediction subset.  Compare that
    # subset against the full Stata export without treating extra Stata rows as errors.
    prediction_merge = expected_predictions.merge(
        actual_predictions,
        on=["obs_id", "category"],
        how="left",
        suffixes=("_python", "_stata"),
        indicator=True,
        validate="one_to_one",
    )
    missing_predictions = [
        f"obs={int(obs_id)},cat={int(category)}"
        for obs_id, category in prediction_merge.loc[
            prediction_merge["_merge"] != "both", ["obs_id", "category"]
        ].itertuples(index=False, name=None)
    ]
    report.append(
        _comparison_row(
            model=model,
            statistic="probability",
            differences=(
                prediction_merge["probability_python"] - prediction_merge["probability_stata"]
            ).abs(),
            tolerance=tolerance["probability"],
            missing=missing_predictions,
        )
    )
    return report


def main() -> int:
    args = _parse_args()
    workdir = args.workdir.resolve()
    manifest = _verify_manifest(workdir)
    model_specs = manifest.get("comparison_model_specs", MODEL_SPECS)
    unknown_kinds = sorted(
        {
            str(spec.get("kind"))
            for spec in model_specs.values()
            if spec.get("kind") not in TOLERANCES
        }
    )
    if unknown_kinds:
        raise ValueError(
            "Manifest contains unsupported comparison model kinds: " + ", ".join(unknown_kinds)
        )

    python_dir = workdir / "python"
    stata_dir = workdir / "stata"
    required_stata_files = {
        "estimates": stata_dir / "estimates_raw.csv",
        "covariance": stata_dir / "covariance_raw.csv",
        "fit": stata_dir / "fit.csv",
        "predictions": stata_dir / "predictions.csv",
        "metadata": stata_dir / "metadata.txt",
        "log": stata_dir / "stata_run.log",
    }
    missing_files = [str(path) for path in required_stata_files.values() if not path.exists()]
    if missing_files:
        runner = (
            "limiteddepkit_real_data.do"
            if manifest.get("suite") == "real_data_application"
            else "limiteddepkit_parity.do"
        )
        raise FileNotFoundError(
            f"Stata outputs are incomplete. Run {runner} first. Missing: "
            + ", ".join(missing_files)
        )

    python_estimates = pd.read_csv(python_dir / "estimates.csv")
    python_covariance = pd.read_csv(python_dir / "covariance.csv")
    python_fit = pd.read_csv(python_dir / "fit.csv")
    python_predictions = pd.read_csv(python_dir / "predictions.csv")
    raw_estimates = pd.read_csv(required_stata_files["estimates"])
    raw_covariance = pd.read_csv(required_stata_files["covariance"])
    stata_fit = pd.read_csv(required_stata_files["fit"])
    stata_predictions = pd.read_csv(required_stata_files["predictions"])

    _validate_schema_group(
        {
            "estimates": python_estimates,
            "covariance": python_covariance,
            "fit": python_fit,
            "predictions": python_predictions,
        },
        PYTHON_REFERENCE_SCHEMAS,
        label="Python reference",
    )
    _validate_schema_group(
        {
            "estimates": raw_estimates,
            "covariance": raw_covariance,
            "fit": stata_fit,
            "predictions": stata_predictions,
        },
        RAW_STATA_SCHEMAS,
        label="Raw Stata export",
    )

    metadata = _parse_metadata(required_stata_files["metadata"].read_text(encoding="utf-8"))
    stata_run_metadata = _verify_run_metadata(metadata, manifest)
    stata_panel_quadrature = _verify_panel_quadrature(metadata, manifest, model_specs)

    canonical_estimates: list[pd.DataFrame] = []
    canonical_covariance: list[pd.DataFrame] = []
    for model in model_specs:
        if model not in set(raw_estimates["model"]):
            continue
        model_estimates, model_covariance = _canonical_stata_results(
            model, raw_estimates, raw_covariance, model_specs
        )
        canonical_estimates.append(model_estimates)
        canonical_covariance.append(model_covariance)
    if not canonical_estimates:
        raise ValueError("Raw Stata estimates contain no models declared by the manifest")
    stata_estimates = pd.concat(canonical_estimates, ignore_index=True)
    stata_covariance = pd.concat(canonical_covariance, ignore_index=True)
    _validate_schema_group(
        {"estimates": stata_estimates, "covariance": stata_covariance},
        CANONICAL_STATA_SCHEMAS,
        label="Combined canonical Stata results",
    )

    report_rows: list[dict[str, Any]] = []
    available_models = set(stata_fit["model"])
    for model, spec in model_specs.items():
        required = bool(spec["required"] or args.require_flexible)
        if model not in available_models:
            report_rows.append(
                {
                    "model": model,
                    "statistic": "availability",
                    "compared": 0,
                    "nonfinite": 0,
                    "max_abs_difference": np.nan,
                    "tolerance": np.nan,
                    "status": "FAIL" if required else "SKIP",
                    "detail": "Stata model output is absent",
                }
            )
            continue
        report_rows.extend(
            _compare_model(
                model,
                python_estimates,
                python_covariance,
                python_fit,
                python_predictions,
                stata_estimates,
                stata_covariance,
                stata_fit,
                stata_predictions,
                model_specs,
                TOLERANCES,
            )
        )

    report = pd.DataFrame(report_rows)
    report_path = workdir / "comparison_report.csv"
    report.to_csv(report_path, index=False, float_format="%.10g", lineterminator="\n")
    summary_path, certificate_path = _write_evidence(
        workdir=workdir,
        manifest=manifest,
        report=report,
        stata_estimates=stata_estimates,
        stata_covariance=stata_covariance,
        stata_run_metadata=stata_run_metadata,
        stata_panel_quadrature=stata_panel_quadrature,
        require_optional=args.require_flexible,
    )
    print(report.to_string(index=False))
    print(f"\nComparison report: {report_path}")
    print(f"Comparison summary: {summary_path}")
    print(f"Parity certificate: {certificate_path}")
    failures = report["status"].eq("FAIL").sum()
    if failures:
        print(f"Parity result: FAIL ({failures} failed checks)")
        return 1
    print("Parity result: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Parity comparison cannot continue: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
