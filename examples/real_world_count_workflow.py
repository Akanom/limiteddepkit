"""Fit real-world infant-mortality count models with verified public data.

The example uses Stata Press's ``rod93`` infant-mortality dataset. Network access
is opt-in, HTTPS-only, and followed by an exact SHA-256 check. The source data is
cached outside the repository by default and is never redistributed by this
project.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import numpy as np
import pandas as pd

from limiteddepkit import NegativeBinomialNB2, PoissonRegressor

SOURCE_URL = "https://www.stata-press.com/data/r19/rod93.dta"
SOURCE_SHA256 = "023d1676ef716e320b49ebf0e0b31d259439161d3268da5b0b93022d138ddeab"


class _HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        resolved = urljoin(req.full_url, newurl)
        if urlsplit(resolved).scheme.lower() != "https":
            raise RuntimeError("Source download redirected away from HTTPS.")
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_data_path() -> Path:
    local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return local_root / "limiteddepkit" / "examples" / "rod93.dta"


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = Request(SOURCE_URL, headers={"User-Agent": "limiteddepkit-real-world-example"})
    opener = build_opener(
        HTTPSHandler(context=ssl.create_default_context()), _HTTPSOnlyRedirectHandler()
    )
    try:
        with opener.open(request, timeout=90) as response:
            if urlsplit(response.geturl()).scheme.lower() != "https":
                raise RuntimeError("Source download redirected away from HTTPS.")
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = _sha256(temporary)
        if actual != SOURCE_SHA256:
            raise RuntimeError(
                f"Source SHA-256 mismatch: expected {SOURCE_SHA256}, received {actual}."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_verified(path: Path, *, allow_download: bool) -> pd.DataFrame:
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"Pinned source not found at {path}. Re-run with --download to acquire it."
            )
        _download(path)
    actual = _sha256(path)
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"Refusing unverified data at {path}: expected {SOURCE_SHA256}, received {actual}."
        )
    return pd.read_stata(path, convert_categoricals=False)


def _analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"deaths", "exposure", "age_mos", "cohort"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"rod93 data is missing required columns: {sorted(missing)}")
    data = pd.DataFrame(
        {
            "deaths": raw["deaths"].astype(int),
            "exposure": raw["exposure"].astype(float),
            "intercept": 1.0,
            "log_age": np.log(raw["age_mos"].astype(float)),
            "cohort_2": (raw["cohort"] == 2).astype(float),
            "cohort_3": (raw["cohort"] == 3).astype(float),
        }
    )
    if not np.isfinite(data.to_numpy()).all() or not (data["exposure"] > 0.0).all():
        raise ValueError("rod93 contains invalid numeric values or non-positive exposure.")
    return data


def _assert_poisson_engine_parity(reference: Any, accelerated: Any) -> None:
    for name in ("params", "covariance", "standard_errors", "fitted_values"):
        np.testing.assert_array_equal(
            np.asarray(getattr(reference, name)),
            np.asarray(getattr(accelerated, name)),
            err_msg=f"Poisson engines differ for {name}",
        )
    for name in ("loglike", "score_norm", "scaled_score_norm", "pearson_chi2", "deviance"):
        if getattr(reference, name) != getattr(accelerated, name):
            raise AssertionError(f"Poisson engines differ for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_default_data_path())
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the pinned Stata Press file over verified HTTPS when absent.",
    )
    parser.add_argument(
        "--poisson-engine",
        choices=("reference", "accelerated"),
        default="accelerated",
    )
    args = parser.parse_args()

    data = _analysis_frame(_load_verified(args.data, allow_download=args.download))
    features = ["intercept", "log_age", "cohort_2", "cohort_3"]
    X = data[features]
    y = data["deaths"]
    exposure = data["exposure"]

    poisson_results = {
        engine: PoissonRegressor().fit(
            X,
            y,
            exposure=exposure,
            tolerance=1e-10,
            maxiter=3_000,
            engine=engine,
        )
        for engine in ("reference", "accelerated")
    }
    _assert_poisson_engine_parity(poisson_results["reference"], poisson_results["accelerated"])
    poisson = poisson_results[args.poisson_engine]
    nb2 = NegativeBinomialNB2().fit(
        X,
        y,
        exposure=exposure,
        tolerance=1e-10,
        maxiter=3_000,
    )
    predictions = pd.DataFrame(
        {
            "observed_deaths": y,
            "poisson_mean": poisson.predict(X, exposure=exposure),
            "nb2_mean": nb2.predict(X, exposure=exposure),
        }
    )

    print("Verified source:", args.data.resolve())
    print("Observations:", len(data))
    print("Poisson reference/accelerated parity: exact")
    print("\nPoisson coefficients")
    print(poisson.summary_frame().to_string())
    print("\nNB2 coefficients and dispersion")
    print(nb2.summary_frame().to_string())
    print("\nFirst five observed and fitted deaths")
    print(predictions.head().to_string(index=False))
    print("\nInterpretation note: coefficients are conditional rate associations; this example")
    print("does not by itself identify causal effects of age or birth cohort.")


if __name__ == "__main__":
    main()
