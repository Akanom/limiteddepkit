"""Certify the separate promoted-family Python/R public-data parity suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_ORDER = (
    "firth_binary_logit",
    "poisson",
    "negative_binomial_nb2",
    "tobit",
    "truncated_regression",
    "interval_regression",
    "geometric_duration",
    "exponential_duration",
    "weibull_duration",
    "gamma_duration",
    "random_effects_ordered_probit",
    "fixed_effects_ordered_logit",
)

EVIDENCE_CLASSES = {
    "firth_binary_logit": "independent-likelihood",
    "poisson": "industrial-package",
    "negative_binomial_nb2": "industrial-package",
    "tobit": "industrial-package",
    "truncated_regression": "independent-likelihood",
    "interval_regression": "industrial-package",
    "geometric_duration": "likelihood-identity",
    "exponential_duration": "industrial-package",
    "weibull_duration": "industrial-package",
    "gamma_duration": "independent-likelihood",
    "random_effects_ordered_probit": "industrial-package",
    "fixed_effects_ordered_logit": "likelihood-identity",
}

# These limits are code-owned, not merely trusted from the mutable manifest.
# The comparator requires the manifest to declare exactly this contract.
TOLERANCES: dict[str, dict[str, float]] = {
    model: {
        "estimate_atol": 5e-5,
        "estimate_rtol": 5e-5,
        "covariance_atol": 1e-4,
        "prediction_atol": 1e-4,
        "standard_error_atol": 1e-4,
        "fit_atol": 1e-4,
    }
    for model in MODEL_ORDER
}
for model in ("weibull_duration", "gamma_duration"):
    TOLERANCES[model] = {
        "estimate_atol": 2e-4,
        "estimate_rtol": 2e-4,
        "covariance_atol": 5e-4,
        "prediction_atol": 3e-4,
        "standard_error_atol": 5e-4,
        "fit_atol": 3e-4,
    }
TOLERANCES["random_effects_ordered_probit"] = {
    "estimate_atol": 2e-3,
    "estimate_rtol": 2e-3,
    "covariance_atol": 3e-3,
    "prediction_atol": 2e-3,
    "standard_error_atol": 3e-3,
    "fit_atol": 2e-3,
}

EXPECTED_SCHEMAS = {
    "estimates": ["model", "dataset", "parameter", "estimate", "standard_error"],
    "covariance": [
        "model",
        "dataset",
        "row_parameter",
        "column_parameter",
        "covariance",
    ],
    "fit": [
        "model",
        "dataset",
        "nobs",
        "n_params",
        "loglike",
        "aic",
        "bic",
        "converged",
        "inference_valid",
        "n_groups",
        "n_contributing_entities",
        "n_cutoff_clones",
        "n_pseudo_observations",
        "n_events",
        "n_censored",
        "n_interval",
        "n_exact",
        "n_left_censored",
        "n_right_censored",
        "score_norm",
        "scaled_score_norm",
        "penalized_loglike",
        "jeffreys_penalty",
        "backend",
        "covariance_type",
    ],
    "predictions": [
        "model",
        "dataset",
        "obs_id",
        "prediction",
        "category",
        "time",
        "value",
    ],
}

METADATA_SCHEMA = [
    "model",
    "dataset",
    "evidence_class",
    "engine",
    "engine_version",
    "r_version",
    "details",
    "completed",
]

TABLE_KEYS = {
    "estimates": ["model", "dataset", "parameter"],
    "covariance": ["model", "dataset", "row_parameter", "column_parameter"],
    "fit": ["model", "dataset"],
    "predictions": [
        "model",
        "dataset",
        "obs_id",
        "prediction",
        "category",
        "time",
    ],
}

COMPARISON_OUTPUTS = (
    "comparison_report.csv",
    "comparison_summary.md",
    "parity_certificate.json",
)


class ContractError(ValueError):
    """Raised when the promoted evidence contract is structurally invalid."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workdir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "work" / "real_data",
        help="Prepared promoted suite containing manifest.json, python/, and r/.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _invalidate_outputs(workdir: Path) -> None:
    directory = workdir / "r"
    for filename in COMPARISON_OUTPUTS:
        path = directory / filename
        if path.is_file() or path.is_symlink():
            path.unlink()


def _load_and_verify_manifest(workdir: Path) -> tuple[dict[str, Any], str]:
    path = workdir / "manifest.json"
    if not path.is_file():
        raise ContractError(f"Missing promoted manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"Could not read promoted manifest: {error}") from error
    if manifest.get("schema_version") != 1:
        raise ContractError("Only promoted manifest schema_version=1 is supported")
    if manifest.get("suite") != "promoted_public_data_parity":
        raise ContractError(
            "Expected suite='promoted_public_data_parity'; found "
            f"{manifest.get('suite')!r}"
        )
    if tuple(manifest.get("model_order", ())) != MODEL_ORDER:
        raise ContractError("Manifest model_order does not match the certified 12-model suite")
    specs = manifest.get("model_specs")
    if not isinstance(specs, dict) or set(specs) != set(MODEL_ORDER):
        raise ContractError("Manifest model_specs must cover exactly the 12 promoted models")
    schemas = manifest.get("output_schemas")
    if not isinstance(schemas, dict):
        raise ContractError("Manifest output_schemas must be an object")
    for table, expected in EXPECTED_SCHEMAS.items():
        declared = schemas.get(f"python/{table}.csv")
        if declared != expected:
            raise ContractError(
                f"Manifest schema for python/{table}.csv changed: {declared!r}"
            )
    for model in MODEL_ORDER:
        declared = specs[model].get("comparison_tolerances")
        required = {
            key: TOLERANCES[model][key]
            for key in (
                "estimate_atol",
                "estimate_rtol",
                "covariance_atol",
                "prediction_atol",
            )
        }
        if not isinstance(declared, dict) or any(
            not np.isclose(float(declared.get(key, np.nan)), value, rtol=0, atol=0)
            for key, value in required.items()
        ):
            raise ContractError(
                f"Manifest comparison tolerances changed for {model}: {declared!r}"
            )
        parameter_order = specs[model].get("parameter_order")
        if not isinstance(parameter_order, list) or not parameter_order:
            raise ContractError(f"Manifest parameter_order is invalid for {model}")
    registered = manifest.get("files")
    if not isinstance(registered, dict) or not registered:
        raise ContractError("Manifest files must contain SHA-256 registrations")
    required_files = {
        *(f"python/{table}.csv" for table in EXPECTED_SCHEMAS),
        *(str(specs[model]["data_file"]).replace("\\", "/") for model in MODEL_ORDER),
    }
    missing = sorted(required_files - set(registered))
    if missing:
        raise ContractError(
            "Manifest is missing required registrations: " + ", ".join(missing)
        )
    root = workdir.resolve()
    errors: list[str] = []
    for relative_name, expected_digest in sorted(registered.items()):
        normalized = str(relative_name).replace("\\", "/")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_digest)):
            errors.append(f"invalid SHA-256 for {normalized}")
            continue
        relative = Path(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe manifest path {normalized}")
            continue
        artifact = (root / relative).resolve()
        if artifact != root and root not in artifact.parents:
            errors.append(f"manifest path escapes workdir: {normalized}")
        elif not artifact.is_file():
            errors.append(f"missing {normalized}")
        elif _sha256(artifact) != expected_digest:
            errors.append(f"hash mismatch for {normalized}")
    if errors:
        raise ContractError("Promoted inputs changed: " + "; ".join(errors))
    return manifest, _sha256(path)


def _model_sequence(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(frame["model"].drop_duplicates().astype(str))


def _read_table(directory: Path, table: str, label: str) -> pd.DataFrame:
    path = directory / f"{table}.csv"
    if not path.is_file():
        raise ContractError(f"Missing {label} artifact: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_SCHEMAS[table]:
        raise ContractError(
            f"{label} {table}.csv schema changed: {list(frame.columns)!r}"
        )
    if frame.empty:
        raise ContractError(f"{label} {table}.csv is empty")
    if _model_sequence(frame) != MODEL_ORDER:
        raise ContractError(f"{label} {table}.csv has wrong model order/coverage")
    keys = TABLE_KEYS[table]
    nullable_keys = {"category", "time"} if table == "predictions" else set()
    required_keys = [column for column in keys if column not in nullable_keys]
    if frame[required_keys].isna().any(axis=None):
        raise ContractError(f"{label} {table}.csv has null required keys")
    if _normalized_keys(frame, keys).duplicated().any():
        raise ContractError(f"{label} {table}.csv has duplicate canonical keys")
    finite_columns = {
        "estimates": ("estimate", "standard_error"),
        "covariance": ("covariance",),
        "fit": ("nobs", "n_params", "loglike"),
        "predictions": ("value",),
    }[table]
    for column in finite_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ContractError(f"{label} {table}.{column} contains non-finite values")
    return frame


def _normalized_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    normalized = pd.DataFrame(index=frame.index)
    for column in keys:
        values = frame[column]
        if column in {"obs_id", "category", "time"}:
            numeric = pd.to_numeric(values, errors="coerce")
            normalized[column] = [
                "<NA>" if pd.isna(value) else format(float(value), ".17g")
                for value in numeric
            ]
        else:
            normalized[column] = values.astype(str)
    return normalized


def _validate_coverage(
    tables: dict[str, pd.DataFrame], manifest: dict[str, Any], label: str
) -> None:
    specs = manifest["model_specs"]
    for model in MODEL_ORDER:
        spec = specs[model]
        dataset = str(spec["dataset"])
        parameter_order = list(map(str, spec["parameter_order"]))
        estimates = tables["estimates"].loc[tables["estimates"]["model"] == model]
        found_parameters = estimates["parameter"].astype(str).tolist()
        if found_parameters != parameter_order:
            raise ContractError(
                f"{label} parameter order for {model} changed: {found_parameters!r}"
            )
        for table, frame in tables.items():
            selected = frame.loc[frame["model"] == model]
            if set(selected["dataset"].astype(str)) != {dataset}:
                raise ContractError(f"{label} {table} dataset mismatch for {model}")
        covariance = tables["covariance"].loc[
            tables["covariance"]["model"] == model
        ]
        expected_pairs = {
            (row, column) for row in parameter_order for column in parameter_order
        }
        found_pairs = set(
            covariance[["row_parameter", "column_parameter"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if found_pairs != expected_pairs or len(covariance) != len(expected_pairs):
            raise ContractError(f"{label} covariance is not a full square for {model}")
        fit = tables["fit"].loc[tables["fit"]["model"] == model]
        if len(fit) != 1:
            raise ContractError(f"{label} fit must contain one row for {model}")
        predictions = tables["predictions"].loc[
            tables["predictions"]["model"] == model
        ]
        expected_types = set(map(str, spec["prediction_types"]))
        if set(predictions["prediction"].astype(str)) != expected_types:
            raise ContractError(f"{label} prediction types changed for {model}")


def _read_metadata(workdir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    path = workdir / "r" / "metadata.csv"
    if not path.is_file():
        raise ContractError(f"Missing R metadata: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != METADATA_SCHEMA:
        raise ContractError(f"R metadata.csv schema changed: {list(frame.columns)!r}")
    if _model_sequence(frame) != MODEL_ORDER or len(frame) != len(MODEL_ORDER):
        raise ContractError("R metadata must contain exactly one row per promoted model")
    for row in frame.itertuples(index=False):
        spec = manifest["model_specs"][row.model]
        if row.dataset != spec["dataset"]:
            raise ContractError(f"R metadata dataset mismatch for {row.model}")
        if row.evidence_class != EVIDENCE_CLASSES[row.model]:
            raise ContractError(f"R metadata evidence class mismatch for {row.model}")
        if not row.engine or not row.engine_version or not row.r_version or not row.details:
            raise ContractError(f"R metadata is incomplete for {row.model}")
        if str(row.completed).strip().lower() not in {"true", "1"}:
            raise ContractError(f"R metadata completion marker is false for {row.model}")
    return frame


def _merge_exact(
    python: pd.DataFrame, r: pd.DataFrame, keys: list[str], context: str
) -> pd.DataFrame:
    python = python.copy()
    r = r.copy()
    normalized_python = _normalized_keys(python, keys)
    normalized_r = _normalized_keys(r, keys)
    join_columns = [f"__key_{column}" for column in keys]
    normalized_python.columns = join_columns
    normalized_r.columns = join_columns
    python = pd.concat([normalized_python, python.drop(columns=keys)], axis=1)
    r = pd.concat([normalized_r, r.drop(columns=keys)], axis=1)
    merged = python.merge(
        r, on=join_columns, how="outer", suffixes=("_python", "_r"), indicator=True
    )
    missing = merged["_merge"].ne("both")
    if missing.any():
        counts = merged.loc[missing, "_merge"].value_counts().to_dict()
        raise ContractError(f"{context} key coverage differs: {counts}")
    return merged


def _report_row(
    model: str,
    dataset: str,
    statistic: str,
    differences: np.ndarray,
    tolerance: float,
    passed: np.ndarray | bool,
    details: str = "",
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    pass_values = np.asarray(passed, dtype=bool)
    return {
        "model": model,
        "dataset": dataset,
        "evidence_class": EVIDENCE_CLASSES.get(model, "suite-contract"),
        "statistic": statistic,
        "n_compared": int(values.size),
        "max_abs_difference": float(np.max(values)) if values.size else 0.0,
        "tolerance": float(tolerance),
        "status": "pass" if bool(np.all(pass_values)) else "fail",
        "details": details,
    }


def _compare_model(
    model: str,
    dataset: str,
    python: dict[str, pd.DataFrame],
    r: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    tolerance = TOLERANCES[model]
    reports: list[dict[str, Any]] = []

    py_estimates = python["estimates"].loc[python["estimates"]["model"] == model]
    r_estimates = r["estimates"].loc[r["estimates"]["model"] == model]
    estimates = _merge_exact(
        py_estimates,
        r_estimates,
        TABLE_KEYS["estimates"],
        f"{model} estimates",
    )
    reference = pd.to_numeric(estimates["estimate_python"]).to_numpy(float)
    candidate = pd.to_numeric(estimates["estimate_r"]).to_numpy(float)
    difference = np.abs(reference - candidate)
    allowed = tolerance["estimate_atol"] + tolerance["estimate_rtol"] * np.abs(
        reference
    )
    reports.append(
        _report_row(
            model,
            dataset,
            "estimate",
            difference,
            float(np.max(allowed)),
            difference <= allowed,
            "componentwise absolute + relative tolerance",
        )
    )
    reference_se = pd.to_numeric(estimates["standard_error_python"]).to_numpy(float)
    candidate_se = pd.to_numeric(estimates["standard_error_r"]).to_numpy(float)
    se_difference = np.abs(reference_se - candidate_se)
    reports.append(
        _report_row(
            model,
            dataset,
            "standard_error",
            se_difference,
            tolerance["standard_error_atol"],
            se_difference <= tolerance["standard_error_atol"],
        )
    )

    py_covariance = python["covariance"].loc[python["covariance"]["model"] == model]
    r_covariance = r["covariance"].loc[r["covariance"]["model"] == model]
    covariance = _merge_exact(
        py_covariance,
        r_covariance,
        TABLE_KEYS["covariance"],
        f"{model} covariance",
    )
    covariance_difference = np.abs(
        pd.to_numeric(covariance["covariance_python"]).to_numpy(float)
        - pd.to_numeric(covariance["covariance_r"]).to_numpy(float)
    )
    reports.append(
        _report_row(
            model,
            dataset,
            "covariance",
            covariance_difference,
            tolerance["covariance_atol"],
            covariance_difference <= tolerance["covariance_atol"],
            "full canonical covariance matrix",
        )
    )

    py_fit = python["fit"].loc[python["fit"]["model"] == model]
    r_fit = r["fit"].loc[r["fit"]["model"] == model]
    fit = _merge_exact(py_fit, r_fit, TABLE_KEYS["fit"], f"{model} fit")
    fit_numeric = ("loglike", "aic", "bic", "penalized_loglike", "jeffreys_penalty")
    for column in fit_numeric:
        reference_value = pd.to_numeric(fit[f"{column}_python"], errors="coerce").to_numpy(float)
        candidate_value = pd.to_numeric(fit[f"{column}_r"], errors="coerce").to_numpy(float)
        missing_match = np.isnan(reference_value) == np.isnan(candidate_value)
        finite = np.isfinite(reference_value) & np.isfinite(candidate_value)
        difference = np.abs(reference_value[finite] - candidate_value[finite])
        passed = bool(np.all(missing_match)) and bool(
            np.all(difference <= tolerance["fit_atol"])
        )
        reports.append(
            _report_row(
                model,
                dataset,
                column,
                difference,
                tolerance["fit_atol"],
                passed,
                "unsupported values must be absent on both sides",
            )
        )
    exact_fit_columns = (
        "nobs",
        "n_params",
        "converged",
        "inference_valid",
        "n_groups",
        "n_contributing_entities",
        "n_cutoff_clones",
        "n_pseudo_observations",
        "n_events",
        "n_censored",
        "n_interval",
        "n_exact",
        "n_left_censored",
        "n_right_censored",
    )
    exact_failures: list[str] = []
    for column in exact_fit_columns:
        left = fit[f"{column}_python"]
        right = fit[f"{column}_r"]
        left_missing = left.isna().to_numpy()
        right_missing = right.isna().to_numpy()
        equal = left.astype(str).to_numpy() == right.astype(str).to_numpy()
        if not np.all((left_missing & right_missing) | (~left_missing & ~right_missing & equal)):
            exact_failures.append(column)
    reports.append(
        _report_row(
            model,
            dataset,
            "fit_structure",
            np.array([0.0 if not exact_failures else 1.0]),
            0.0,
            not exact_failures,
            "exact counts/flags" if not exact_failures else "mismatch: " + ", ".join(exact_failures),
        )
    )

    py_predictions = python["predictions"].loc[
        python["predictions"]["model"] == model
    ]
    r_predictions = r["predictions"].loc[r["predictions"]["model"] == model]
    predictions = _merge_exact(
        py_predictions,
        r_predictions,
        TABLE_KEYS["predictions"],
        f"{model} predictions",
    )
    prediction_difference = np.abs(
        pd.to_numeric(predictions["value_python"]).to_numpy(float)
        - pd.to_numeric(predictions["value_r"]).to_numpy(float)
    )
    reports.append(
        _report_row(
            model,
            dataset,
            "prediction",
            prediction_difference,
            tolerance["prediction_atol"],
            prediction_difference <= tolerance["prediction_atol"],
        )
    )
    return reports


def _artifact_hashes(workdir: Path, filenames: tuple[str, ...]) -> dict[str, str]:
    output: dict[str, str] = {}
    for filename in filenames:
        path = workdir / filename
        if path.is_file():
            output[filename.replace("\\", "/")] = _sha256(path)
    return output


def _write_evidence(
    workdir: Path,
    report: pd.DataFrame,
    manifest: dict[str, Any] | None,
    manifest_hash: str | None,
    metadata: pd.DataFrame | None,
    *,
    contract_error: str | None = None,
) -> None:
    output_directory = workdir / "r"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "comparison_report.csv"
    report.to_csv(report_path, index=False)
    failed = report.loc[report["status"] != "pass"]
    status = "failed" if contract_error or not failed.empty else "passed"
    evidence_counts = {
        evidence_class: sum(
            1 for model in MODEL_ORDER if EVIDENCE_CLASSES[model] == evidence_class
        )
        for evidence_class in sorted(set(EVIDENCE_CLASSES.values()))
    }
    lines = [
        "# Promoted public-data Python/R parity",
        "",
        f"**Result: {status.upper()}**",
        "",
        (
            "This is a separate 12-family application suite. It supplies public-data "
            "evidence plus the official fictional interval-regression software example; "
            "it does not extend or replace the "
            "legacy eight-family certificates and is not a claim of universal numerical parity."
        ),
        "",
        f"- Models required: {len(MODEL_ORDER)}",
        f"- Comparison checks: {len(report)}",
        f"- Failed checks: {len(failed)}",
        "- Evidence classes: "
        + ", ".join(f"{name}={count}" for name, count in evidence_counts.items()),
    ]
    if contract_error:
        lines.extend(["", f"Contract failure: `{contract_error}`"])
    elif not failed.empty:
        lines.extend(["", "Failed checks:", ""])
        for row in failed.itertuples(index=False):
            lines.append(
                f"- `{row.model}` / `{row.statistic}`: max difference "
                f"{row.max_abs_difference:.8g}, tolerance {row.tolerance:.8g}"
            )
    else:
        maxima = (
            report.groupby("statistic", sort=False)["max_abs_difference"].max().to_dict()
        )
        lines.extend(["", "Largest observed differences:", ""])
        for statistic, value in maxima.items():
            lines.append(f"- `{statistic}`: {value:.8g}")
    summary_path = output_directory / "comparison_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    certificate: dict[str, Any] = {
        "schema_version": 1,
        "suite": manifest.get("suite") if manifest else None,
        "status": status,
        "claim": (
            "Separate promoted-family Python/R application parity on registered public "
            "datasets and the official fictional interval-regression software example."
        ),
        "scope_limit": (
            "This certificate is not an extension of the legacy eight-family certificates "
            "and does not establish universal equivalence across datasets or implementations."
        ),
        "manifest_sha256": manifest_hash,
        "required_models": list(MODEL_ORDER),
        "evidence_classes": EVIDENCE_CLASSES,
        "evidence_class_counts": evidence_counts,
        "comparison_tolerances": TOLERANCES,
        "summary": {
            "checks": int(len(report)),
            "passed": int((report["status"] == "pass").sum()),
            "failed": int(len(failed)),
        },
        "contract_error": contract_error,
        "software": (
            metadata[["model", "engine", "engine_version", "r_version"]].to_dict("records")
            if metadata is not None
            else []
        ),
        "artifact_sha256": _artifact_hashes(
            workdir,
            tuple(f"python/{table}.csv" for table in EXPECTED_SCHEMAS)
            + tuple(f"r/{table}.csv" for table in (*EXPECTED_SCHEMAS, "metadata"))
            + ("r/comparison_report.csv", "r/comparison_summary.md"),
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    certificate_path = output_directory / "parity_certificate.json"
    certificate_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    workdir = _parse_args().workdir.resolve()
    _invalidate_outputs(workdir)
    manifest: dict[str, Any] | None = None
    manifest_hash: str | None = None
    metadata: pd.DataFrame | None = None
    try:
        manifest, manifest_hash = _load_and_verify_manifest(workdir)
        python_tables = {
            table: _read_table(workdir / "python", table, "Python")
            for table in EXPECTED_SCHEMAS
        }
        r_tables = {
            table: _read_table(workdir / "r", table, "R")
            for table in EXPECTED_SCHEMAS
        }
        _validate_coverage(python_tables, manifest, "Python")
        _validate_coverage(r_tables, manifest, "R")
        metadata = _read_metadata(workdir, manifest)
        reports: list[dict[str, Any]] = []
        for model in MODEL_ORDER:
            reports.extend(
                _compare_model(
                    model,
                    str(manifest["model_specs"][model]["dataset"]),
                    python_tables,
                    r_tables,
                )
            )
        report = pd.DataFrame(reports)
        _write_evidence(workdir, report, manifest, manifest_hash, metadata)
        if report["status"].eq("pass").all():
            print(
                f"Promoted Python/R parity passed: {len(MODEL_ORDER)} models, "
                f"{len(report)} checks."
            )
            return 0
        print(
            f"Promoted Python/R parity failed: "
            f"{int(report['status'].eq('fail').sum())} numerical checks.",
            file=sys.stderr,
        )
        return 1
    except (ContractError, FileNotFoundError, OSError, KeyError, TypeError) as error:
        message = str(error)
        report = pd.DataFrame(
            [
                {
                    "model": "__suite__",
                    "dataset": "",
                    "evidence_class": "suite-contract",
                    "statistic": "contract",
                    "n_compared": 0,
                    "max_abs_difference": 1.0,
                    "tolerance": 0.0,
                    "status": "fail",
                    "details": message,
                }
            ]
        )
        _write_evidence(
            workdir,
            report,
            manifest,
            manifest_hash,
            metadata,
            contract_error=message,
        )
        print(f"Promoted parity contract failed: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
