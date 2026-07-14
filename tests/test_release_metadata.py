import re
from pathlib import Path

import limiteddepkit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "0.1.0a1"


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_release_version_is_synchronized_across_metadata():
    pyproject = _read("pyproject.toml")
    citation = _read("CITATION.cff")

    project_version = re.search(
        r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE
    )
    citation_version = re.search(
        r"^version: (\S+)$", citation, flags=re.MULTILINE
    )
    assert project_version is not None
    assert citation_version is not None
    assert project_version.group(1) == RELEASE_VERSION
    assert citation_version.group(1) == RELEASE_VERSION
    assert limiteddepkit.__version__ == RELEASE_VERSION


def test_alpha_classifier_changelog_and_publish_guard_match_the_freeze():
    pyproject = _read("pyproject.toml")
    citation = _read("CITATION.cff")
    changelog = _read("CHANGELOG.md")
    ci_workflow = _read(".github/workflows/ci.yml")
    publish_workflow = _read(".github/workflows/publish.yml")
    contributing = _read("CONTRIBUTING.md")
    manifest = _read("MANIFEST.in")

    assert "Development Status :: 3 - Alpha" in pyproject
    assert "Development Status :: 2 - Pre-Alpha" not in pyproject
    assert "currently released" not in citation.lower()
    assert f"## [{RELEASE_VERSION}] - Release pending" in changelog
    assert "python -m twine check --strict dist/*" in ci_workflow
    assert "python -m twine check --strict dist/*" in contributing
    assert "Verify release tag matches package version" in publish_workflow
    assert "prune _out_of_scope" in manifest
    assert "recursive-include validation/stata *.do *.md *.ps1 *.py" in manifest
    assert "prune validation/stata/work" in manifest
    assert "recursive-include validation/r *.R *.md *.ps1 *.py" in manifest
    assert "prune validation/r/work" in manifest
    assert "include validation/PARITY_EVIDENCE.md" in manifest


def test_documentation_records_completed_benchmark_specific_parity():
    readme = _read("README.md")
    parity_guide = _read("validation/stata/README.md")
    r_guide = _read("validation/r/README.md")
    validation = _read("docs/VALIDATION.md")
    evidence_index = _read("validation/PARITY_EVIDENCE.md")

    assert "PASS — 82/82" in readme
    assert "PASS — 110/110" in readme
    assert "PASS — ALL EIGHT FAMILIES" in parity_guide
    assert "PASS — ALL EIGHT FAMILIES" in r_guide
    assert "benchmark-specific" in parity_guide
    assert "benchmark-specific" in r_guide
    assert "gologit2` 3.2.8" in readme
    assert evidence_index.count("**PASS**") == 4
    assert "82 | 0 | 0" in evidence_index
    assert "110 | 0 | 0" in evidence_index
    assert "b74f790dac0d25c3d0ef872ed43c5941" in evidence_index
    assert "2780339b9e02d6b8917c9c33edad1042" in evidence_index
    combined = "\n".join([readme, parity_guide, r_guide, validation])
    assert "AWAITING MANUAL STATA" not in combined
    assert "PASS_STATA_PARITY" not in combined
    assert "PASS_STATA_COMPARISON" not in combined
