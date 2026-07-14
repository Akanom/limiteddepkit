from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATA_DIR = PROJECT_ROOT / "validation" / "stata"

STATA_ARTIFACTS = {
    "stata_run.log",
    "estimates_raw.csv",
    "covariance_raw.csv",
    "fit.csv",
    "predictions.csv",
    "metadata.txt",
    "estimates_canonical.csv",
    "covariance_canonical.csv",
}
COMPARISON_ARTIFACTS = {
    "comparison_report.csv",
    "comparison_summary.md",
    "parity_certificate.json",
}


def _load_script(filename: str) -> ModuleType:
    path = STATA_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", ["prepare_parity.py", "prepare_real_data.py"])
def test_preparation_removes_only_known_stale_evidence(filename: str, tmp_path: Path) -> None:
    preparation = _load_script(filename)
    stata_dir = tmp_path / "stata"
    stata_dir.mkdir()

    for name in STATA_ARTIFACTS:
        (stata_dir / name).write_text("stale\n", encoding="utf-8")
    for name in COMPARISON_ARTIFACTS:
        (tmp_path / name).write_text("stale\n", encoding="utf-8")
    unrelated_stata = stata_dir / "manual_notes.txt"
    unrelated_root = tmp_path / "keep-me.csv"
    unrelated_stata.write_text("keep\n", encoding="utf-8")
    unrelated_root.write_text("keep\n", encoding="utf-8")

    preparation._remove_stale_evidence(tmp_path)

    assert all(not (stata_dir / name).exists() for name in STATA_ARTIFACTS)
    assert all(not (tmp_path / name).exists() for name in COMPARISON_ARTIFACTS)
    assert unrelated_stata.read_text(encoding="utf-8") == "keep\n"
    assert unrelated_root.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize(
    ("filename", "suite"),
    [
        ("limiteddepkit_parity.do", "controlled_synthetic_certification"),
        ("limiteddepkit_real_data.do", "real_data_application"),
    ],
)
def test_do_file_invalidates_old_evidence_and_marks_only_completed_runs(
    filename: str, suite: str
) -> None:
    source = (STATA_DIR / filename).read_text(encoding="utf-8")

    for name in STATA_ARTIFACTS | COMPARISON_ARTIFACTS:
        assert name in source
    cleanup_position = source.index("capture erase")
    assert cleanup_position < source.index("log using")

    suite_marker = f'"suite={suite}"'
    completion_marker = '"run_completed=1"'
    assert source.count(suite_marker) == 1
    assert source.count(completion_marker) == 1
    assert source.rindex("export delimited") < source.index(completion_marker)
    assert source.index(suite_marker) < source.index(completion_marker)
    assert source.index(completion_marker) < source.index("file close `metadata_handle'")
