#!/usr/bin/env python3
"""Validate bounded dependency metadata and optionally run installed audits."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

LOWER_BOUND = re.compile(r"(?:>=|(?<![<>=])>|==|~=)")
UPPER_BOUND = re.compile(r"(?:<=|(?<![<>=])<|==|~=)")
DIRECT_URL = re.compile(r"(?:\s@\s|^(?:git\+|https?://))", re.I)


def _requirement_errors(requirement: object, group: str) -> list[str]:
    if not isinstance(requirement, str):
        return [f"{group} contains a non-string requirement: {requirement!r}"]
    errors: list[str] = []
    if DIRECT_URL.search(requirement):
        errors.append(f"{group} contains a direct URL: {requirement}")
    specifier = requirement.split(";", 1)[0]
    if not LOWER_BOUND.search(specifier):
        errors.append(f"{group} lacks a tested lower bound: {requirement}")
    if not UPPER_BOUND.search(specifier):
        errors.append(f"{group} lacks a compatibility upper bound: {requirement}")
    return errors


def dependency_errors(project: Path) -> list[str]:
    pyproject = project / "pyproject.toml"
    if not pyproject.is_file():
        return [f"missing {pyproject}"]
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"cannot parse {pyproject}: {exc}"]

    errors: list[str] = []
    project_table = data.get("project", {})
    groups: dict[str, object] = {
        "build-system.requires": data.get("build-system", {}).get("requires", []),
        "project.dependencies": project_table.get("dependencies", []),
    }
    optional = project_table.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        errors.append("project.optional-dependencies must be a table")
    else:
        groups.update(
            {f"project.optional-dependencies.{name}": value for name, value in optional.items()}
        )

    for group, requirements in groups.items():
        if not isinstance(requirements, list):
            errors.append(f"{group} must be an array")
            continue
        for requirement in requirements:
            errors.extend(_requirement_errors(requirement, group))
    return errors


def run(command: list[str], cwd: Path) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--run-tools", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    errors = dependency_errors(project)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    if args.run_tools:
        if run([sys.executable, "-m", "pip", "check"], project):
            return 1
        if run([sys.executable, "-m", "pip_audit", "--strict"], project):
            return 1
    print("Dependency policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
