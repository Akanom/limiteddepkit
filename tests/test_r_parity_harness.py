from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R_VALIDATION = PROJECT_ROOT / "validation" / "r"


def _load_comparator():
    path = R_VALIDATION / "compare_parity.py"
    spec = importlib.util.spec_from_file_location("limiteddepkit_r_comparator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["binary_logit"],
            "dataset": ["fixture"],
            "parameter": ["intercept"],
            "estimate": [0.2],
            "standard_error": [0.1],
        }
    )


def _minimal_report(status: str = "PASS") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["binary_logit"],
            "statistic": ["estimate"],
            "compared": [1],
            "nonfinite": [0],
            "max_abs_difference": [0.0],
            "tolerance": [2e-6],
            "status": [status],
            "detail": [""],
        }
    )


def test_r_harness_declares_and_fits_all_eight_models():
    comparator = _load_comparator()
    assert set(comparator.MODEL_KINDS) == {
        "binary_logit",
        "binary_probit",
        "ordered_logit",
        "ordered_probit",
        "generalized_ordered_logit",
        "partial_proportional_odds",
        "random_effects_ordered_logit",
        "dynamic_random_effects_ordered_logit",
    }

    runner = (R_VALIDATION / "run_parity.R").read_text(encoding="utf-8")
    assert "ldk_fit_flexible_models(" in runner
    assert "ldk_fit_panel_model(" in runner
    assert 'require_package("VGAM")' in runner
    assert 'require_package("ordinal")' in runner
    assert '"completion_marker"' in runner
    assert '"R_PARITY_COMPLETE"' in runner


def test_r_runner_invalidates_only_maintained_outputs_before_fitting():
    runner = (R_VALIDATION / "run_parity.R").read_text(encoding="utf-8")
    assert "maintained_outputs <- file.path(" in runner
    for artifact in (
        "estimates.csv",
        "covariance.csv",
        "fit.csv",
        "predictions.csv",
        "metadata.csv",
        "comparison_report.csv",
        "comparison_summary.md",
        "parity_certificate.json",
    ):
        assert f'"{artifact}"' in runner
    cleanup_position = runner.index("status <- unlink(artifact")
    assert cleanup_position < runner.index("sys.source(")
    assert cleanup_position < runner.index('require_package("jsonlite")')
    assert cleanup_position < runner.index("manifest_path <-")
    assert "Could not invalidate stale R parity artifact" in runner
    assert "unlink(work_directory" not in runner
    assert "unlink(output_directory" not in runner


def test_r_dependency_setup_pins_versions_urls_and_hashes():
    setup = (R_VALIDATION / "setup_dependencies.ps1").read_text(encoding="utf-8")
    expected = {
        "MASS_7.3-65.zip": "46f1a3d0991c8387411b23cc9faf657a5abfc5e93438546f8b042073d9988c14",
        "jsonlite_2.0.0.zip": "4b9418cff57f2357fbf5d24b1a618f082310cb9d5b63af051bd8dd7f570e188a",
        "numDeriv_2016.8-1.1.zip": "0df596925b695a2ba0bc327b71340921ba6550e8cbdc53e49024e41b50e2cdac",
        "ucminf_1.2.3.zip": "335437fae88c185ae31142e7828ba1855b45e50524a5ac0bca17175d53d673e0",
        "ordinal_2025.12-29.zip": "b27a83300c6664abe0b568fab39c962c4651e62d3be95bdfb552a15550789e9b",
        "VGAM_1.1-14.zip": "752dd0d4012731a0e7b37bdf4a443631850d8b0263100dae1a877afae3a61bed",
    }
    for archive, digest in expected.items():
        assert archive in setup
        assert digest in setup
    assert "https://" in setup
    assert "http://" not in setup
    assert "Get-FileHash -Algorithm SHA256" in setup
    assert "Unexpected package versions" in setup
    assert 'Matrix = \"1.7.3\"' in setup
    assert 'nlme = \"3.1.168\"' in setup
    assert "Pinned packages did not resolve from the project library" in setup

    runner = (R_VALIDATION / "run_parity.R").read_text(encoding="utf-8")
    assert ".libPaths(c(local_library, .Library))" in runner
    assert '"Matrix_version"' in runner
    assert '"nlme_version"' in runner


def test_r_comparison_row_rejects_missing_or_nonfinite_values():
    comparator = _load_comparator()
    passing = comparator._comparison_row("binary_logit", "estimate", [0.0], 1e-8)
    assert passing["status"] == "PASS"

    for bad in (np.nan, np.inf, -np.inf):
        failing = comparator._comparison_row(
            "binary_logit", "estimate", [0.0, bad], 1e-8
        )
        assert failing["status"] == "FAIL"
        assert failing["nonfinite"] == 1

    missing = comparator._comparison_row(
        "binary_logit", "estimate", [0.0], 1e-8, missing=["x1"]
    )
    assert missing["status"] == "FAIL"
    assert "x1" in missing["detail"]


def test_r_comparator_invalidates_prior_evidence_before_manifest_checks(tmp_path: Path):
    comparator = _load_comparator()
    r_directory = tmp_path / "r"
    r_directory.mkdir()
    for name in comparator.COMPARISON_EVIDENCE:
        (r_directory / name).write_text("stale\n", encoding="utf-8")
    unrelated = r_directory / "estimates.csv"
    unrelated.write_text("keep\n", encoding="utf-8")

    comparator._invalidate_comparison_evidence(tmp_path)

    assert all(not (r_directory / name).exists() for name in comparator.COMPARISON_EVIDENCE)
    assert unrelated.read_text(encoding="utf-8") == "keep\n"
    source = (R_VALIDATION / "compare_parity.py").read_text(encoding="utf-8")
    assert source.index("_invalidate_comparison_evidence(workdir)") < source.index(
        "manifest = _verify_manifest(workdir)"
    )


def test_r_certificate_is_strict_and_written_as_completion_artifact(tmp_path: Path):
    comparator = _load_comparator()
    r_directory = tmp_path / "r"
    r_directory.mkdir()
    for name in (
        "estimates.csv",
        "covariance.csv",
        "fit.csv",
        "predictions.csv",
        "metadata.csv",
    ):
        (r_directory / name).write_text("fixture\n", encoding="utf-8")
    manifest = {
        "suite": "controlled_synthetic_certification",
        "limiteddepkit_version": "0.1.0a1",
        "files": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    comparator._write_evidence(
        tmp_path,
        manifest,
        dict(comparator.EXPECTED_METADATA),
        _minimal_report(status="SKIP"),
    )
    certificate = json.loads(
        (r_directory / "parity_certificate.json").read_text(encoding="utf-8")
    )
    assert certificate["result"] == "FAIL"
    assert certificate["failed_checks"] == 0
    assert certificate["skipped_checks"] == 1
    assert set(certificate["software"]) == set(comparator.SOFTWARE_METADATA_KEYS)
    assert "schema_version" not in certificate["software"]
    assert "source_manifest_schema_version" not in certificate["software"]

    source = (R_VALIDATION / "compare_parity.py").read_text(encoding="utf-8")
    writer = source[source.index("def _write_evidence(") : source.index("def main()")]
    assert writer.index("summary_path =") < writer.index("certificate_path =")


def test_r_manifest_requires_frozen_inputs_and_safe_hash_registrations(tmp_path: Path):
    comparator = _load_comparator()
    manifest_path = tmp_path / "manifest.json"
    base_manifest = {
        "schema_version": 1,
        "suite": "controlled_synthetic_certification",
        **comparator.EXPECTED_MANIFEST_CONTROLS,
        "quadrature_points": 12,
    }
    manifest_path.write_text(
        json.dumps(
            {
                **base_manifest,
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required file registrations"):
        comparator._verify_manifest(tmp_path)

    required = (
        comparator.REQUIRED_PYTHON_REFERENCES
        | comparator.REQUIRED_MANIFEST_FILES["controlled_synthetic_certification"]
    )
    registrations = {name: "0" * 64 for name in required}
    registrations["../outside.csv"] = "0" * 64
    registrations["data/bad.csv"] = "not-a-sha256"
    manifest_path.write_text(
        json.dumps(
            {
                **base_manifest,
                "files": registrations,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path escapes work directory"):
        comparator._verify_manifest(tmp_path)


def test_r_schema_validation_rejects_duplicate_null_and_nonfinite_rows():
    comparator = _load_comparator()
    estimates = _minimal_estimates()
    comparator._validate_table(estimates, "estimates", "R export")

    duplicate = pd.concat([estimates, estimates], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate keys"):
        comparator._validate_table(duplicate, "estimates", "R export")

    null_key = estimates.copy()
    null_key.loc[0, "parameter"] = np.nan
    with pytest.raises(ValueError, match="null key"):
        comparator._validate_table(null_key, "estimates", "R export")

    nonfinite = estimates.copy()
    nonfinite.loc[0, "estimate"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        comparator._validate_table(nonfinite, "estimates", "R export")


def test_r_panel_and_flexible_helpers_document_independent_mappings():
    panel = (R_VALIDATION / "panel_models.R").read_text(encoding="utf-8")
    flexible = (R_VALIDATION / "flexible_models.R").read_text(encoding="utf-8")

    assert "nAGQ = -quadrature_points" in panel
    assert 'jacobian["sigma_entity", raw_tau_names] <- sigma_entity' in panel
    assert "fixed-only: random intercept = 0" in panel
    assert "VGAM::predictvglm(" in flexible
    assert "reverse = FALSE" in flexible
    assert ".ldk_central_hessian(objective, theta)" in flexible
    assert "theta <- unname(raw[sources]) * signs" in flexible
