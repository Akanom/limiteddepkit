"""Compare canonical R parity exports with frozen limiteddepkit references."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_KINDS = {
    "binary_logit": "binary",
    "binary_probit": "binary",
    "ordered_logit": "ordered",
    "ordered_probit": "ordered",
    "generalized_ordered_logit": "generalized",
    "partial_proportional_odds": "partial",
    "random_effects_ordered_logit": "random_effects",
    "dynamic_random_effects_ordered_logit": "random_effects",
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

SCHEMAS = {
    "estimates": (
        ("model", "dataset", "parameter", "estimate", "standard_error"),
        ("model", "parameter"),
        ("estimate", "standard_error"),
    ),
    "covariance": (
        ("model", "dataset", "row_parameter", "column_parameter", "covariance"),
        ("model", "row_parameter", "column_parameter"),
        ("covariance",),
    ),
    "fit": (
        (
            "model",
            "dataset",
            "nobs",
            "n_groups",
            "n_params",
            "loglike",
            "aic",
            "bic",
            "converged",
            "inference_valid",
            "quadrature_points",
            "constraint_slack",
        ),
        ("model",),
        ("nobs", "n_params", "loglike", "aic", "bic"),
    ),
    "predictions": (
        ("model", "dataset", "obs_id", "category", "probability"),
        ("model", "obs_id", "category"),
        ("obs_id", "category", "probability"),
    ),
}

EXPECTED_METADATA = {
    "schema_version": "1",
    "source_manifest_schema_version": "1",
    "runner": "validation/r/run_parity.R",
    "r_version": "4.5.1",
    "stats_version": "4.5.1",
    "MASS_version": "7.3-65",
    "jsonlite_version": "2.0.0",
    "VGAM_version": "1.1-14",
    "ordinal_version": "2025.12-29",
    "numDeriv_version": "2016.8-1.1",
    "ucminf_version": "1.2.3",
    "Matrix_version": "1.7-3",
    "nlme_version": "3.1-168",
    "binary_covariance": "observed-information",
    "binary_probit_covariance": "inverse-Mills observed-information",
    "ordered_estimator": "MASS::polr",
    "flexible_estimator": "VGAM::vglm cumulative logit",
    "flexible_covariance": "limiteddepkit-compatible central observed Hessian",
    "panel_estimator": "ordinal::clmm",
    "panel_prediction": "conditional fixed-only: random intercept = 0",
    "completion_marker": "R_PARITY_COMPLETE",
}

SOFTWARE_METADATA_KEYS = (
    "r_version",
    "stats_version",
    "MASS_version",
    "Matrix_version",
    "VGAM_version",
    "jsonlite_version",
    "nlme_version",
    "numDeriv_version",
    "ordinal_version",
    "ucminf_version",
)

COMPARISON_EVIDENCE = (
    "comparison_report.csv",
    "comparison_summary.md",
    "parity_certificate.json",
)

REQUIRED_MANIFEST_FILES = {
    "controlled_synthetic_certification": {
        "data/cross_section.csv",
        "data/static_re.csv",
        "data/dynamic_design.csv",
    },
    "real_data_application": {
        "data/binary_lbw.csv",
        "data/ordinal_tvsfpors.csv",
        "data/dynamic_nlswork_design.csv",
    },
}
REQUIRED_PYTHON_REFERENCES = {
    "python/estimates.csv",
    "python/covariance.csv",
    "python/fit.csv",
    "python/predictions.csv",
}

EXPECTED_MANIFEST_CONTROLS = {
    "limiteddepkit_version": "0.1.0a1",
    "prediction_rows_per_model": 25,
    "quadrature_method": "ghermite",
    "ordered_optimizer_maxiter": 5_000,
    "ordered_optimizer_tolerance": 1e-13,
    "panel_optimizer_tolerance": 1e-12,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workdir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "stata" / "work",
        help="Prepared parity directory containing python/ and r/ outputs.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _invalidate_comparison_evidence(workdir: Path) -> None:
    r_directory = workdir / "r"
    for name in COMPARISON_EVIDENCE:
        path = r_directory / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _verify_manifest(workdir: Path) -> dict[str, Any]:
    path = workdir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing parity manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Only parity manifest schema version 1 is supported")
    suite = str(manifest.get("suite"))
    if suite not in REQUIRED_MANIFEST_FILES:
        raise ValueError(f"Unsupported parity suite: {suite!r}")
    expected_controls = {
        **EXPECTED_MANIFEST_CONTROLS,
        "quadrature_points": 12 if suite == "controlled_synthetic_certification" else 20,
    }
    control_errors = [
        f"{key}: expected {expected!r}, found {manifest.get(key)!r}"
        for key, expected in expected_controls.items()
        if manifest.get(key) != expected
    ]
    if control_errors:
        raise ValueError("Parity manifest control mismatch: " + "; ".join(control_errors))
    registered = manifest.get("files")
    if not isinstance(registered, dict):
        raise ValueError("Parity manifest files must be an object of SHA-256 registrations")
    required = REQUIRED_PYTHON_REFERENCES | REQUIRED_MANIFEST_FILES[suite]
    missing_required = sorted(required - set(registered))
    if missing_required:
        raise ValueError(
            "Parity manifest is missing required file registrations: "
            + ", ".join(missing_required)
        )
    errors: list[str] = []
    root = workdir.resolve()
    for relative, expected in registered.items():
        relative = str(relative).replace("\\", "/")
        expected = str(expected)
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            errors.append(f"invalid SHA-256 registration for {relative}")
            continue
        artifact = (root / relative).resolve()
        if artifact != root and root not in artifact.parents:
            errors.append(f"path escapes work directory: {relative}")
            continue
        if not artifact.is_file():
            errors.append(f"missing {relative}")
        elif _sha256(artifact) != expected:
            errors.append(f"hash mismatch for {relative}")
    if errors:
        raise ValueError("Parity inputs changed: " + "; ".join(errors))
    return manifest


def _validate_table(frame: pd.DataFrame, name: str, label: str) -> None:
    required, keys, finite_columns = SCHEMAS[name]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} {name} is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{label} {name} is empty")
    if frame[list(keys)].isna().any(axis=None):
        raise ValueError(f"{label} {name} has null key values")
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{label} {name} has duplicate keys: {', '.join(keys)}")
    for column in finite_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} {name}.{column} contains non-finite values")


def _read_tables(directory: Path, label: str) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in SCHEMAS:
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")
        tables[name] = pd.read_csv(path)
        _validate_table(tables[name], name, label)
    return tables


def _read_metadata(r_directory: Path, manifest: dict[str, Any]) -> dict[str, str]:
    path = r_directory / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing R metadata: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(("key", "value")) - set(frame.columns):
        raise ValueError("R metadata must contain key and value columns")
    if frame["key"].eq("").any() or frame["key"].duplicated().any():
        raise ValueError("R metadata keys must be non-empty and unique")
    metadata = dict(frame[["key", "value"]].itertuples(index=False, name=None))
    expected = dict(EXPECTED_METADATA)
    expected["suite"] = str(manifest.get("suite"))
    expected["prediction_rows_per_model"] = str(manifest.get("prediction_rows_per_model"))
    expected["panel_quadrature"] = (
        "nonadaptive Gauss-Hermite, "
        f"nAGQ={-int(manifest.get('quadrature_points'))}"
    )
    errors = [
        f"{key}: expected {value!r}, found {metadata.get(key)!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    expected_models = sorted(MODEL_KINDS)
    actual_models = sorted(filter(None, metadata.get("models", "").split(";")))
    if actual_models != expected_models:
        errors.append(f"models: expected {expected_models}, found {actual_models}")
    if errors:
        raise ValueError("R run metadata mismatch: " + "; ".join(errors))
    return metadata


def _comparison_row(
    model: str,
    statistic: str,
    differences: pd.Series | list[float],
    tolerance: float,
    *,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    values = pd.to_numeric(pd.Series(differences, dtype=float), errors="coerce")
    nonfinite = int((~np.isfinite(values)).sum())
    finite = values[np.isfinite(values)]
    maximum = float(finite.max()) if not finite.empty else np.nan
    missing = [] if missing is None else missing
    passed = bool(
        len(values) > 0
        and nonfinite == 0
        and not missing
        and np.isfinite(maximum)
        and maximum <= tolerance
    )
    return {
        "model": model,
        "statistic": statistic,
        "compared": int(len(values)),
        "nonfinite": nonfinite,
        "max_abs_difference": maximum,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
        "detail": "missing keys: " + ", ".join(missing) if missing else "",
    }


def _single_fit(frame: pd.DataFrame, model: str, label: str) -> pd.Series:
    rows = frame.loc[frame["model"] == model]
    if len(rows) != 1:
        raise ValueError(f"{label} must contain exactly one fit row for {model}")
    return rows.iloc[0]


def _compare_model(
    model: str,
    python: dict[str, pd.DataFrame],
    r: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    tolerance = TOLERANCES[MODEL_KINDS[model]]
    report: list[dict[str, Any]] = []

    expected_estimates = python["estimates"].loc[
        python["estimates"]["model"] == model,
        ["dataset", "parameter", "estimate", "standard_error"],
    ]
    actual_estimates = r["estimates"].loc[
        r["estimates"]["model"] == model,
        ["dataset", "parameter", "estimate", "standard_error"],
    ]
    estimates = expected_estimates.merge(
        actual_estimates,
        on=["dataset", "parameter"],
        how="outer",
        suffixes=("_python", "_r"),
        indicator=True,
        validate="one_to_one",
    )
    missing_parameters = estimates.loc[estimates["_merge"] != "both", "parameter"].tolist()
    report.append(
        _comparison_row(
            model,
            "estimate",
            (estimates["estimate_python"] - estimates["estimate_r"]).abs(),
            tolerance["estimate"],
            missing=missing_parameters,
        )
    )
    report.append(
        _comparison_row(
            model,
            "standard_error",
            (estimates["standard_error_python"] - estimates["standard_error_r"]).abs(),
            tolerance["standard_error"],
            missing=missing_parameters,
        )
    )

    expected_covariance = python["covariance"].loc[
        python["covariance"]["model"] == model,
        ["dataset", "row_parameter", "column_parameter", "covariance"],
    ]
    actual_covariance = r["covariance"].loc[
        r["covariance"]["model"] == model,
        ["dataset", "row_parameter", "column_parameter", "covariance"],
    ]
    covariance = expected_covariance.merge(
        actual_covariance,
        on=["dataset", "row_parameter", "column_parameter"],
        how="outer",
        suffixes=("_python", "_r"),
        indicator=True,
        validate="one_to_one",
    )
    missing_covariance = [
        f"{row} x {column}"
        for row, column in covariance.loc[
            covariance["_merge"] != "both", ["row_parameter", "column_parameter"]
        ].itertuples(index=False, name=None)
    ]
    report.append(
        _comparison_row(
            model,
            "covariance",
            (covariance["covariance_python"] - covariance["covariance_r"]).abs(),
            tolerance["covariance"],
            missing=missing_covariance,
        )
    )

    expected_fit = _single_fit(python["fit"], model, "Python fit table")
    actual_fit = _single_fit(r["fit"], model, "R fit table")
    dataset_difference = 0.0 if str(expected_fit["dataset"]) == str(actual_fit["dataset"]) else 1.0
    report.append(_comparison_row(model, "dataset", [dataset_difference], 0.0))
    for statistic, statistic_tolerance in (
        ("nobs", 0.0),
        ("n_params", 0.0),
        ("loglike", tolerance["loglike"]),
        ("aic", 2.0 * tolerance["loglike"]),
        ("bic", 2.0 * tolerance["loglike"]),
    ):
        difference = abs(float(expected_fit[statistic]) - float(actual_fit[statistic]))
        report.append(
            _comparison_row(model, statistic, [difference], statistic_tolerance)
        )
    expected_groups = pd.to_numeric(pd.Series([expected_fit["n_groups"]]), errors="coerce").iloc[0]
    if np.isfinite(expected_groups):
        actual_groups = pd.to_numeric(pd.Series([actual_fit["n_groups"]]), errors="coerce").iloc[0]
        report.append(
            _comparison_row(model, "n_groups", [abs(expected_groups - actual_groups)], 0.0)
        )
    expected_quadrature = pd.to_numeric(
        pd.Series([expected_fit["quadrature_points"]]), errors="coerce"
    ).iloc[0]
    if np.isfinite(expected_quadrature):
        actual_quadrature = pd.to_numeric(
            pd.Series([actual_fit["quadrature_points"]]), errors="coerce"
        ).iloc[0]
        report.append(
            _comparison_row(
                model,
                "quadrature_points",
                [abs(expected_quadrature - actual_quadrature)],
                0.0,
            )
        )
    for statistic in ("converged", "inference_valid"):
        actual = str(actual_fit[statistic]).strip().lower() in {"true", "1", "1.0"}
        report.append(_comparison_row(model, f"r_{statistic}", [abs(1.0 - float(actual))], 0.0))
    expected_slack = pd.to_numeric(
        pd.Series([expected_fit["constraint_slack"]]), errors="coerce"
    ).iloc[0]
    if np.isfinite(expected_slack):
        actual_slack = pd.to_numeric(
            pd.Series([actual_fit["constraint_slack"]]), errors="coerce"
        ).iloc[0]
        report.append(
            _comparison_row(
                model,
                "constraint_slack",
                [abs(expected_slack - actual_slack)],
                tolerance["probability"],
            )
        )

    expected_predictions = python["predictions"].loc[
        python["predictions"]["model"] == model,
        ["dataset", "obs_id", "category", "probability"],
    ]
    actual_predictions = r["predictions"].loc[
        r["predictions"]["model"] == model,
        ["dataset", "obs_id", "category", "probability"],
    ]
    predictions = expected_predictions.merge(
        actual_predictions,
        on=["dataset", "obs_id", "category"],
        how="outer",
        suffixes=("_python", "_r"),
        indicator=True,
        validate="one_to_one",
    )
    missing_predictions = [
        f"obs={obs_id},cat={category}"
        for obs_id, category in predictions.loc[
            predictions["_merge"] != "both", ["obs_id", "category"]
        ].itertuples(index=False, name=None)
    ]
    report.append(
        _comparison_row(
            model,
            "probability",
            (predictions["probability_python"] - predictions["probability_r"]).abs(),
            tolerance["probability"],
            missing=missing_predictions,
        )
    )
    probability_sums = actual_predictions.groupby("obs_id", sort=False)["probability"].sum()
    report.append(
        _comparison_row(model, "probability_sum", (probability_sums - 1.0).abs(), 1e-10)
    )
    return report


def _markdown_table(report: pd.DataFrame) -> str:
    columns = ["model", "statistic", "max_abs_difference", "tolerance", "status"]
    display = report[columns].copy()
    for column in ("max_abs_difference", "tolerance"):
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def _write_evidence(
    workdir: Path,
    manifest: dict[str, Any],
    metadata: dict[str, str],
    report: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    r_directory = workdir / "r"
    report_path = r_directory / "comparison_report.csv"
    report.to_csv(report_path, index=False, float_format="%.10g", lineterminator="\n")
    unexpected_statuses = sorted(set(report["status"]) - {"PASS", "FAIL", "SKIP"})
    if unexpected_statuses:
        raise ValueError(
            "R comparison report contains unsupported statuses: "
            + ", ".join(map(str, unexpected_statuses))
        )
    failures = report.loc[report["status"] == "FAIL"]
    skips = report.loc[report["status"] == "SKIP"]
    result = "PASS" if report["status"].eq("PASS").all() else "FAIL"
    suite = str(manifest.get("suite"))
    if result == "PASS" and suite == "controlled_synthetic_certification":
        claim = (
            "The maintained controlled fixtures passed all eight declared R parity "
            "checks within benchmark-specific tolerances."
        )
    elif result == "PASS":
        claim = (
            "The maintained real-data R application checks passed for all eight families "
            "within benchmark-specific tolerances; this does not broaden the controlled "
            "certification claim."
        )
    else:
        claim = "No R parity claim is supported because one or more checks failed."
    r_inputs = {
        name: _sha256(r_directory / name)
        for name in ("estimates.csv", "covariance.csv", "fit.csv", "predictions.csv", "metadata.csv")
    }
    certificate = {
        "schema_version": 1,
        "suite": suite,
        "result": result,
        "claim": claim,
        "limiteddepkit_version": manifest.get("limiteddepkit_version"),
        "required_models": sorted(MODEL_KINDS),
        "available_models": sorted(report["model"].unique()),
        "software": {key: metadata[key] for key in SOFTWARE_METADATA_KEYS},
        "r_run_metadata": metadata,
        "quadrature_points": manifest.get("quadrature_points"),
        "ordered_optimizer_maxiter": manifest.get("ordered_optimizer_maxiter"),
        "ordered_optimizer_tolerance": manifest.get("ordered_optimizer_tolerance"),
        "panel_optimizer_tolerance": manifest.get("panel_optimizer_tolerance"),
        "tolerances": TOLERANCES,
        "manifest_sha256": _sha256(workdir / "manifest.json"),
        "prepared_data_sha256": {
            relative: digest
            for relative, digest in manifest.get("files", {}).items()
            if str(relative).startswith("data/")
        },
        "python_reference_sha256": {
            relative: digest
            for relative, digest in manifest.get("files", {}).items()
            if str(relative).startswith("python/")
        },
        "r_artifact_sha256": r_inputs,
        "checks": int(len(report)),
        "failed_checks": int(len(failures)),
        "skipped_checks": int(len(skips)),
    }
    summary_path = r_directory / "comparison_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Python–R parity summary",
                "",
                f"- Suite: `{suite}`",
                f"- Result: **{result}**",
                f"- Models: {len(MODEL_KINDS)}",
                f"- Failed checks: {len(failures)}",
                f"- Skipped checks: {len(skips)}",
                "",
                claim,
                "",
                "## Numerical checks",
                "",
                _markdown_table(report),
                "",
                "This is a benchmark-specific claim, not universal equality across data, "
                "optimizers, covariance estimands, quadrature implementations, or preprocessing.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    certificate_path = r_directory / "parity_certificate.json"
    certificate_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path, summary_path, certificate_path


def main() -> int:
    args = _parse_args()
    workdir = args.workdir.resolve()
    _invalidate_comparison_evidence(workdir)
    manifest = _verify_manifest(workdir)
    python = _read_tables(workdir / "python", "Python reference")
    r = _read_tables(workdir / "r", "R export")
    metadata = _read_metadata(workdir / "r", manifest)

    expected_models = set(MODEL_KINDS)
    for label, tables in (("Python", python), ("R", r)):
        for name, frame in tables.items():
            actual = set(frame["model"].astype(str))
            if actual != expected_models:
                raise ValueError(
                    f"{label} {name} model set differs from the eight-model matrix: "
                    f"missing={sorted(expected_models - actual)}, "
                    f"extra={sorted(actual - expected_models)}"
                )

    report_rows: list[dict[str, Any]] = []
    for model in MODEL_KINDS:
        report_rows.extend(_compare_model(model, python, r))
    report = pd.DataFrame(report_rows)
    report_path, summary_path, certificate_path = _write_evidence(
        workdir, manifest, metadata, report
    )
    print(report.to_string(index=False))
    print(f"\nComparison report: {report_path}")
    print(f"Comparison summary: {summary_path}")
    print(f"Parity certificate: {certificate_path}")
    failures = int(report["status"].eq("FAIL").sum())
    if failures:
        print(f"R parity result: FAIL ({failures} failed checks)")
        return 1
    print("R parity result: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"R parity comparison cannot continue: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
