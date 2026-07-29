import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ecosystem_comparison_page_is_linked_and_evidence_bounded() -> None:
    comparison = (PROJECT_ROOT / "docs" / "ECOSYSTEM_COMPARISON.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    assert "docs/ECOSYSTEM_COMPARISON.md" in readme
    assert "ECOSYSTEM_COMPARISON.md" in docs_index

    for heading in (
        "## Workflow 1: limited outcome with exposure",
        "## Workflow 2: dynamic-panel System GMM",
        "## Workflow 3: causal treatment effect",
    ):
        assert heading in comparison

    for package in (
        "limiteddepkit",
        "systemgmmkit",
        "CauseKit",
        "universal-output-hub",
        "Statsmodels",
        "pydynpd",
        "DoWhy",
    ):
        assert package in comparison

    assert "Runtime rows are not a league table" in comparison
    assert "Alternative runtime | Not measured" in comparison
    assert "not yet a released performance claim" in comparison
    assert "does **not** support claims" in comparison
    assert "140/140" in comparison
    assert "120/120" in comparison
    assert "CauseKit is the renamed successor" in comparison
    assert "add_to_outputhub(hub, causal_result" in comparison

    relative_links = re.findall(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)", comparison)
    missing_links = [
        target
        for target in relative_links
        if not (PROJECT_ROOT / "docs" / target.split("#", 1)[0]).resolve().exists()
    ]
    assert missing_links == []
