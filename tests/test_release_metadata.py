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
