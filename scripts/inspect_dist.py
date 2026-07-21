#!/usr/bin/env python3
"""Inventory wheel/sdist contents and reject common release-integrity risks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

DENIED_NAMES = re.compile(
    r"(^|/)(\.env(?:\..*)?|id_rsa|id_ed25519|.*\.(?:pem|key|p12|pfx|bak|orig|rej)|.*\.before_[^/]*)$",
    re.I,
)
DENIED_SUFFIXES = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".whl",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "OpenAI key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
}


def _safe_name(name: str) -> bool:
    pure = PurePosixPath(name.replace("\\", "/"))
    return not pure.is_absolute() and ".." not in pure.parts


def _scan(name: str, data: bytes, allowed_suffixes: set[str]) -> list[str]:
    errors: list[str] = []
    normalized = name.replace("\\", "/")
    if not _safe_name(normalized):
        errors.append(f"unsafe archive path: {name}")
    if DENIED_NAMES.search(normalized):
        errors.append(f"sensitive file name: {name}")
    suffix = PurePosixPath(normalized).suffix.lower()
    if suffix in DENIED_SUFFIXES and suffix not in allowed_suffixes:
        errors.append(f"unexpected binary or nested archive: {name}")
    if len(data) <= 2_000_000:
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                errors.append(f"possible {label} in: {name}")
    return errors


def inspect(
    path: Path, allowed_suffixes: set[str], max_bytes: int, max_members: int
) -> dict[str, object]:
    members: list[dict[str, object]] = []
    errors: list[str] = []
    total = 0
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            for item in infos:
                data = archive.read(item) if item.file_size <= 2_000_000 else b""
                total += item.file_size
                members.append({"path": item.filename, "size": item.file_size})
                errors.extend(_scan(item.filename, data, allowed_suffixes))
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            infos = [item for item in archive.getmembers() if item.isfile()]
            for item in infos:
                extracted = archive.extractfile(item)
                data = extracted.read() if extracted is not None and item.size <= 2_000_000 else b""
                total += item.size
                members.append({"path": item.name, "size": item.size})
                errors.extend(_scan(item.name, data, allowed_suffixes))
            for item in archive.getmembers():
                if (item.issym() or item.islnk()) and not _safe_name(item.linkname):
                    errors.append(f"unsafe archive link: {item.name} -> {item.linkname}")
    else:
        raise ValueError(f"unsupported distribution: {path}")
    if len(members) > max_members:
        errors.append(f"archive has {len(members)} members; limit is {max_members}")
    if total > max_bytes:
        errors.append(f"archive expands to {total} bytes; limit is {max_bytes}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "artifact": str(path),
        "sha256": digest.hexdigest(),
        "member_count": len(members),
        "uncompressed_bytes": total,
        "members": members,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow-suffix", action="append", default=[])
    parser.add_argument("--max-bytes", type=int, default=250_000_000)
    parser.add_argument("--max-members", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    target = args.path.resolve()
    paths = (
        sorted(target.glob("*.whl")) + sorted(target.glob("*.tar.gz"))
        if target.is_dir()
        else [target]
    )
    if not paths:
        print("ERROR: no wheel or sdist found", file=sys.stderr)
        return 2
    allowed = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in args.allow_suffix
    }
    reports = [inspect(path, allowed, args.max_bytes, args.max_members) for path in paths]
    rendered = json.dumps(reports, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 1 if any(report["errors"] for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
