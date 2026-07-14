from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATA_DIR = PROJECT_ROOT / "validation" / "stata"


def _load_script(filename: str) -> ModuleType:
    path = STATA_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse(io.BytesIO):
    def geturl(self) -> str:
        return "https://www.stata-press.com/data/r19/lbw.dta"


class _FakeOpener:
    def open(self, request: object, timeout: int) -> _FakeResponse:
        del request, timeout
        return _FakeResponse(b"not the pinned Stata dataset")


def test_real_data_download_rejects_bad_hash_before_cache_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparation = _load_script("prepare_real_data.py")
    destination = tmp_path / "lbw.dta"
    destination.write_bytes(b"existing verified cache")
    monkeypatch.setattr(preparation, "build_opener", lambda *args: _FakeOpener())

    with pytest.raises(RuntimeError, match="Could not download lbw.dta"):
        preparation._download_source("lbw.dta", destination)

    assert destination.read_bytes() == b"existing verified cache"
    assert not destination.with_name("lbw.dta.part").exists()


@pytest.mark.parametrize("filename", ["prepare_parity.py", "prepare_real_data.py"])
def test_panel_reference_fits_use_tight_recorded_tolerance(filename: str) -> None:
    preparation = _load_script(filename)
    source = (STATA_DIR / filename).read_text(encoding="utf-8")

    assert preparation.PANEL_OPTIMIZER_TOLERANCE == 1e-12
    assert source.count("tolerance=PANEL_OPTIMIZER_TOLERANCE") == 2
    assert '"quadrature_method": "ghermite"' in source
    assert '"panel_optimizer_tolerance": PANEL_OPTIMIZER_TOLERANCE' in source


@pytest.mark.parametrize("filename", ["prepare_parity.py", "prepare_real_data.py"])
def test_ordered_reference_fits_use_tight_recorded_tolerance(filename: str) -> None:
    preparation = _load_script(filename)
    source = (STATA_DIR / filename).read_text(encoding="utf-8")

    assert preparation.ORDERED_OPTIMIZER_TOLERANCE == 1e-13
    assert preparation.ORDERED_OPTIMIZER_MAXITER == 5_000
    assert source.count("tolerance=ORDERED_OPTIMIZER_TOLERANCE") == 2
    assert source.count("maxiter=ORDERED_OPTIMIZER_MAXITER") == 2
    assert '"ordered_optimizer_tolerance": ORDERED_OPTIMIZER_TOLERANCE' in source
    assert '"ordered_optimizer_maxiter": ORDERED_OPTIMIZER_MAXITER' in source
