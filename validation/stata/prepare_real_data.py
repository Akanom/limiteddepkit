"""Prepare public-data limiteddepkit references for manual Stata parity checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import ssl
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import numpy as np
import pandas as pd
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ldk = importlib.import_module("limiteddepkit")

QUADRATURE_POINTS = 20
PANEL_OPTIMIZER_TOLERANCE = 1e-12
PREDICTION_ROWS = 25
STATA_TIMESTAMP = datetime(2000, 1, 1)
EXPECTED_DYNAMIC_GROUPS = 335
EXPECTED_DYNAMIC_RAW_NOBS = 2_010
EXPECTED_DYNAMIC_ESTIMATION_NOBS = 1_675
STALE_STATA_ARTIFACTS = (
    "stata_run.log",
    "estimates_raw.csv",
    "covariance_raw.csv",
    "fit.csv",
    "predictions.csv",
    "metadata.txt",
    "estimates_canonical.csv",
    "covariance_canonical.csv",
)
STALE_COMPARISON_ARTIFACTS = (
    "comparison_report.csv",
    "comparison_summary.md",
    "parity_certificate.json",
)

SOURCES: dict[str, dict[str, str]] = {
    "lbw.dta": {
        "url": "https://www.stata-press.com/data/r19/lbw.dta",
        "sha256": "00204ef3586836e56e49598cd9850148aea9058090a607e5bf20e12a6b0a58ee",
        "role": "Binary logit and probit application using infant low-birthweight data.",
    },
    "tvsfpors.dta": {
        "url": "https://www.stata-press.com/data/r19/tvsfpors.dta",
        "sha256": "50197a3e7b15809ed816b2846ca9dc1a4bc6aecac06ba75f4ae0312d7ceebfc8",
        "role": "Pooled, flexible, and school-level random-effects ordinal application.",
    },
    "nlswork.dta": {
        "url": "https://www.stata-press.com/data/r19/nlswork.dta",
        "sha256": "b77bc182ac586205d769ad847e5e7cb0063c31be2c4bbef5f1ad16b74118c86f",
        "role": "Balanced six-period panel application for dynamic ordinal parity.",
    },
}


class _HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    """Reject any redirect that would leave verified HTTPS transport."""

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
            raise URLError("refusing a redirect away from HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "work" / "real_data",
        help="Working directory for source data, prepared data, and reference outputs.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Optional source-data directory. Missing pinned files are downloaded there and "
            "then copied into <output>/source."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_stale_evidence(workdir: Path) -> None:
    """Remove only maintained outputs that could be mistaken for this run."""
    candidates = (
        *(workdir / "stata" / name for name in STALE_STATA_ARTIFACTS),
        *(workdir / name for name in STALE_COMPARISON_ARTIFACTS),
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Real-data parity invariant failed: {message}")


def _download_source(filename: str, destination: Path) -> None:
    source = SOURCES[filename]
    request = Request(
        source["url"],
        headers={"User-Agent": f"limiteddepkit/{ldk.__version__} parity preparation"},
    )
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        context = ssl.create_default_context()
        opener = build_opener(HTTPSHandler(context=context), _HTTPSOnlyRedirectHandler())
        with opener.open(request, timeout=90) as response:
            final_url = response.geturl()
            if not final_url.lower().startswith("https://"):
                raise RuntimeError("the download was redirected away from HTTPS")
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual_hash = _sha256(temporary)
        expected_hash = source["sha256"]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA256 mismatch for downloaded {filename}; "
                f"expected {expected_hash}, got {actual_hash}"
            )
        temporary.replace(destination)
    except (HTTPError, URLError, OSError, RuntimeError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {filename} over verified HTTPS. Place the pinned file "
            f"in {destination.parent} and rerun."
        ) from exc


def _prepare_sources(workdir: Path, supplied_source_dir: Path | None) -> dict[str, Path]:
    cache_dir = workdir / "source"
    cache_dir.mkdir(parents=True, exist_ok=True)
    acquisition_dir = (
        supplied_source_dir.resolve() if supplied_source_dir is not None else cache_dir
    )
    acquisition_dir.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, Path] = {}
    for filename, source in SOURCES.items():
        acquired = acquisition_dir / filename
        if not acquired.exists():
            _download_source(filename, acquired)
        actual_hash = _sha256(acquired)
        _require(
            actual_hash == source["sha256"],
            f"SHA256 mismatch for {acquired}; expected {source['sha256']}, got {actual_hash}",
        )

        cached = cache_dir / filename
        if acquired.resolve() != cached.resolve():
            shutil.copyfile(acquired, cached)
        _require(
            _sha256(cached) == source["sha256"],
            f"cached source hash changed while copying {filename}",
        )
        prepared[filename] = cached
    return prepared


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _write_dataset(frame: pd.DataFrame, stem: Path) -> dict[str, str]:
    csv_path = stem.with_suffix(".csv")
    dta_path = stem.with_suffix(".dta")
    _write_csv(frame, csv_path)
    frame.to_stata(
        dta_path,
        write_index=False,
        version=118,
        time_stamp=STATA_TIMESTAMP,
    )
    return {
        csv_path.name: _sha256(csv_path),
        dta_path.name: _sha256(dta_path),
    }


def _assert_result(
    model: str,
    result: Any,
    *,
    expected_nobs: int,
    expected_groups: int | None = None,
) -> None:
    _require(bool(result.converged), f"{model} did not converge")
    _require(bool(result.inference_valid), f"{model} did not produce valid inference")
    _require(int(result.nobs) == expected_nobs, f"{model} nobs != {expected_nobs}")
    if expected_groups is not None:
        _require(
            int(result.n_groups) == expected_groups,
            f"{model} n_groups != {expected_groups}",
        )
    parameters = result.all_params.to_numpy(dtype=float)
    covariance = result.covariance.to_numpy(dtype=float)
    _require(np.isfinite(parameters).all(), f"{model} has nonfinite estimates")
    _require(np.isfinite(covariance).all(), f"{model} has nonfinite covariance entries")
    _require(np.isfinite(float(result.loglike)), f"{model} has a nonfinite log likelihood")


def _record_model(
    *,
    model: str,
    dataset: str,
    result: Any,
    estimates: list[dict[str, Any]],
    covariance: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    probability_frame: pd.DataFrame,
    prediction_obs_ids: Any,
) -> None:
    all_params = result.all_params.astype(float)
    standard_errors = result.standard_errors.reindex(all_params.index).astype(float)
    for parameter, estimate in all_params.items():
        estimates.append(
            {
                "model": model,
                "dataset": dataset,
                "parameter": str(parameter),
                "estimate": float(estimate),
                "standard_error": float(standard_errors.loc[parameter]),
            }
        )

    covariance_frame = result.covariance.reindex(index=all_params.index, columns=all_params.index)
    for row_parameter in all_params.index:
        for column_parameter in all_params.index:
            covariance.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "row_parameter": str(row_parameter),
                    "column_parameter": str(column_parameter),
                    "covariance": float(covariance_frame.loc[row_parameter, column_parameter]),
                }
            )

    n_params = len(all_params)
    nobs = int(result.nobs)
    loglike = float(result.loglike)
    fits.append(
        {
            "model": model,
            "dataset": dataset,
            "nobs": nobs,
            "n_groups": getattr(result, "n_groups", np.nan),
            "n_params": n_params,
            "loglike": loglike,
            "aic": -2.0 * loglike + 2.0 * n_params,
            "bic": -2.0 * loglike + np.log(nobs) * n_params,
            "converged": bool(result.converged),
            "inference_valid": bool(result.inference_valid),
            "quadrature_points": getattr(result, "quadrature_points", np.nan),
            "constraint_slack": getattr(result, "constraint_slack", np.nan),
        }
    )

    obs_ids = np.asarray(prediction_obs_ids)
    _require(
        probability_frame.shape[0] == obs_ids.size,
        f"{model} prediction IDs and probability rows do not align",
    )
    for row in range(min(PREDICTION_ROWS, probability_frame.shape[0])):
        for category in probability_frame.columns:
            predictions.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "obs_id": int(obs_ids[row]),
                    "category": int(category),
                    "probability": float(probability_frame.iloc[row][category]),
                }
            )


def _binary_lbw(source_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    source = pd.read_stata(source_path, convert_categoricals=False)
    required = ["low", "age", "lwt", "smoke", "ht"]
    _require(source.shape[0] == 189, "lbw.dta must contain 189 observations")
    _require(not source[required].isna().any().any(), "lbw variables contain missing values")

    data = pd.DataFrame(
        {
            "obs_id": np.arange(1, source.shape[0] + 1, dtype=np.int32),
            "y": source["low"].to_numpy(dtype=np.int32),
            "intercept": np.ones(source.shape[0]),
            "x1": source["age"].to_numpy(dtype=float) / 10.0,
            "x2": source["lwt"].to_numpy(dtype=float) / 100.0,
            "x3": source["smoke"].to_numpy(dtype=float),
            "x4": source["ht"].to_numpy(dtype=float),
        }
    )
    _require(set(data["y"]) == {0, 1}, "binary_lbw outcome support must be {0, 1}")
    features = data[["intercept", "x1", "x2", "x3", "x4"]]
    return data, features, data["y"]


def _ordinal_tvsfpors(
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series]:
    source = pd.read_stata(source_path, convert_categoricals=False)
    required = ["school", "thk", "prethk", "cc", "tv"]
    _require(source.shape[0] == 1_600, "tvsfpors.dta must contain 1,600 observations")
    _require(
        not source[required].isna().any().any(),
        "tvsfpors analysis variables contain missing values",
    )
    base = np.column_stack(
        [
            source["prethk"].to_numpy(dtype=float),
            source["cc"].to_numpy(dtype=float),
            source["tv"].to_numpy(dtype=float),
            (source["cc"].to_numpy(dtype=float) * source["tv"].to_numpy(dtype=float)),
        ]
    )
    y = source["thk"].to_numpy(dtype=np.int32) - 1
    data = pd.DataFrame(
        {
            "obs_id": np.arange(1, source.shape[0] + 1, dtype=np.int32),
            "entity": source["school"].to_numpy(dtype=np.int32),
            "y": y,
        }
    )
    for prefix in ("ox", "gx", "x"):
        for column in range(base.shape[1]):
            data[f"{prefix}{column + 1}"] = base[:, column]
    _require(set(data["y"]) == {0, 1, 2, 3}, "ordinal outcome support must be 0..3")
    _require(data["entity"].nunique() == 28, "tvsfpors must contain 28 schools")
    designs = {
        "ordered": data[["ox1", "ox2", "ox3", "ox4"]],
        "generalized": data[["gx1", "gx2", "gx3", "gx4"]],
        "random_effects": data[["x1", "x2", "x3", "x4"]],
    }
    return data, designs, data["y"]


def _dynamic_nlswork(source_path: Path) -> pd.DataFrame:
    source = pd.read_stata(source_path, convert_categoricals=False)
    required = ["idcode", "year", "ln_wage", "tenure"]
    _require(all(column in source for column in required), "nlswork variables are missing")
    restricted = source.loc[source["year"].between(68, 73), required].dropna().copy()
    group_sizes = restricted.groupby("idcode", sort=False).size()
    eligible = group_sizes.index[group_sizes.eq(6)]
    restricted = restricted.loc[restricted["idcode"].isin(eligible)].copy()
    restricted = restricted.sort_values(["idcode", "year"], kind="stable").reset_index(drop=True)

    _require(
        restricted.shape[0] == EXPECTED_DYNAMIC_RAW_NOBS,
        "balanced nlswork sample must contain 2,010 observations",
    )
    _require(
        restricted["idcode"].nunique() == EXPECTED_DYNAMIC_GROUPS,
        "balanced nlswork sample must contain 335 entities",
    )
    _require(
        not restricted.duplicated(["idcode", "year"]).any(),
        "balanced nlswork sample contains duplicate entity-years",
    )
    expected_years = (68, 69, 70, 71, 72, 73)
    year_sequences = restricted.groupby("idcode", sort=False)["year"].apply(
        lambda values: tuple(int(value) for value in values)
    )
    _require(
        bool(year_sequences.map(lambda values: values == expected_years).all()),
        "every retained nlswork entity must have years 68 through 73",
    )

    wage = restricted["ln_wage"].to_numpy(dtype=float)
    outcome = np.select([wage < 1.4, wage < 1.8], [0, 1], default=2).astype(np.int32)
    data = pd.DataFrame(
        {
            "obs_id": np.arange(1, restricted.shape[0] + 1, dtype=np.int32),
            "entity": restricted["idcode"].to_numpy(dtype=np.int32),
            "time": restricted["year"].to_numpy(dtype=np.int32),
            "x1": restricted["tenure"].to_numpy(dtype=float),
            "y": outcome,
        }
    )
    _require(set(data["y"]) == {0, 1, 2}, "dynamic outcome support must be {0, 1, 2}")
    return data


def main() -> int:
    args = _parse_args()
    workdir = args.output.resolve()
    data_dir = workdir / "data"
    python_dir = workdir / "python"
    stata_dir = workdir / "stata"
    workdir.mkdir(parents=True, exist_ok=True)
    _remove_stale_evidence(workdir)
    data_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    stata_dir.mkdir(parents=True, exist_ok=True)

    sources = _prepare_sources(workdir, args.source_dir)
    file_hashes = {f"source/{filename}": _sha256(path) for filename, path in sources.items()}
    estimates: list[dict[str, Any]] = []
    covariance: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    binary_data, binary_X, binary_y = _binary_lbw(sources["lbw.dta"])
    for filename, digest in _write_dataset(binary_data, data_dir / "binary_lbw").items():
        file_hashes[f"data/{filename}"] = digest

    binary_logit = ldk.BinaryLogit().fit(binary_X, binary_y)
    _assert_result("binary_logit", binary_logit, expected_nobs=189)
    _record_model(
        model="binary_logit",
        dataset="binary_lbw",
        result=binary_logit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=binary_logit.predict_proba(binary_X),
        prediction_obs_ids=binary_data["obs_id"],
    )

    binary_probit = ldk.BinaryProbit().fit(binary_X, binary_y)
    _assert_result("binary_probit", binary_probit, expected_nobs=189)
    _record_model(
        model="binary_probit",
        dataset="binary_lbw",
        result=binary_probit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=binary_probit.predict_proba(binary_X),
        prediction_obs_ids=binary_data["obs_id"],
    )

    ordinal_data, ordinal_designs, ordinal_y = _ordinal_tvsfpors(sources["tvsfpors.dta"])
    for filename, digest in _write_dataset(ordinal_data, data_dir / "ordinal_tvsfpors").items():
        file_hashes[f"data/{filename}"] = digest

    ordered_logit = ldk.OrderedLogit().fit(
        ordinal_designs["ordered"], ordinal_y, category_order=[0, 1, 2, 3]
    )
    _assert_result("ordered_logit", ordered_logit, expected_nobs=1_600)
    _record_model(
        model="ordered_logit",
        dataset="ordinal_tvsfpors",
        result=ordered_logit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=ordered_logit.predict_proba(ordinal_designs["ordered"]),
        prediction_obs_ids=ordinal_data["obs_id"],
    )

    ordered_probit = ldk.OrderedProbit().fit(
        ordinal_designs["ordered"], ordinal_y, category_order=[0, 1, 2, 3]
    )
    _assert_result("ordered_probit", ordered_probit, expected_nobs=1_600)
    _record_model(
        model="ordered_probit",
        dataset="ordinal_tvsfpors",
        result=ordered_probit,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=ordered_probit.predict_proba(ordinal_designs["ordered"]),
        prediction_obs_ids=ordinal_data["obs_id"],
    )

    generalized = ldk.GeneralizedOrderedLogit().fit(
        ordinal_designs["generalized"], ordinal_y, category_order=[0, 1, 2, 3]
    )
    _assert_result("generalized_ordered_logit", generalized, expected_nobs=1_600)
    _require(
        float(generalized.constraint_slack) > 0.0,
        "generalized_ordered_logit constraints are not strictly interior",
    )
    _record_model(
        model="generalized_ordered_logit",
        dataset="ordinal_tvsfpors",
        result=generalized,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=generalized.predict_proba(ordinal_designs["generalized"]),
        prediction_obs_ids=ordinal_data["obs_id"],
    )

    partial = ldk.PartialProportionalOdds(varying=["gx4"]).fit(
        ordinal_designs["generalized"], ordinal_y, category_order=[0, 1, 2, 3]
    )
    _assert_result("partial_proportional_odds", partial, expected_nobs=1_600)
    _require(
        float(partial.constraint_slack) > 0.0,
        "partial_proportional_odds constraints are not strictly interior",
    )
    _record_model(
        model="partial_proportional_odds",
        dataset="ordinal_tvsfpors",
        result=partial,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=partial.predict_proba(ordinal_designs["generalized"]),
        prediction_obs_ids=ordinal_data["obs_id"],
    )

    random_effects = ldk.RandomEffectsOrderedLogit().fit(
        ordinal_designs["random_effects"],
        ordinal_y,
        entity=ordinal_data["entity"],
        category_order=[0, 1, 2, 3],
        quadrature_points=QUADRATURE_POINTS,
        maxiter=1_500,
        tolerance=PANEL_OPTIMIZER_TOLERANCE,
    )
    _assert_result(
        "random_effects_ordered_logit",
        random_effects,
        expected_nobs=1_600,
        expected_groups=28,
    )
    _record_model(
        model="random_effects_ordered_logit",
        dataset="ordinal_tvsfpors",
        result=random_effects,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=random_effects.predict_proba(
            ordinal_designs["random_effects"], random_effects=0.0
        ),
        prediction_obs_ids=ordinal_data["obs_id"],
    )

    dynamic_raw = _dynamic_nlswork(sources["nlswork.dta"])
    for filename, digest in _write_dataset(dynamic_raw, data_dir / "dynamic_nlswork_raw").items():
        file_hashes[f"data/{filename}"] = digest

    dynamic_X = dynamic_raw[["x1"]]
    dynamic_result = ldk.DynamicRandomEffectsOrderedLogit().fit(
        dynamic_X,
        dynamic_raw["y"],
        entity=dynamic_raw["entity"],
        time=dynamic_raw["time"],
        category_order=[0, 1, 2],
        quadrature_points=QUADRATURE_POINTS,
        maxiter=2_000,
        tolerance=PANEL_OPTIMIZER_TOLERANCE,
    )
    _assert_result(
        "dynamic_random_effects_ordered_logit",
        dynamic_result,
        expected_nobs=EXPECTED_DYNAMIC_ESTIMATION_NOBS,
        expected_groups=EXPECTED_DYNAMIC_GROUPS,
    )
    _require(
        dynamic_result.n_original_obs == EXPECTED_DYNAMIC_RAW_NOBS,
        "dynamic model n_original_obs != 2,010",
    )
    _require(
        dynamic_result.dropped_initial == EXPECTED_DYNAMIC_GROUPS,
        "dynamic model must drop exactly one initial observation per entity",
    )
    _require(
        dynamic_result.dropped_nonconsecutive == 0,
        "balanced dynamic sample must not drop nonconsecutive observations",
    )

    dynamic_name_map = {
        "state[1]": "state_1",
        "state[2]": "state_2",
        "initial[1]": "initial_1",
        "initial[2]": "initial_2",
        "initial_x[x1]": "initial_x1",
        "mean[x1]": "mean_x1",
    }
    estimation_index = dynamic_result.estimation_index
    dynamic_obs_ids = dynamic_raw.loc[estimation_index, "obs_id"].to_numpy(dtype=np.int32)
    dynamic_times = dynamic_raw.loc[estimation_index, "time"].to_numpy(dtype=np.int32)
    dynamic_design_X = dynamic_result.estimation_design.rename(columns=dynamic_name_map)
    expected_dynamic_columns = [
        "x1",
        "state_1",
        "state_2",
        "initial_1",
        "initial_2",
        "initial_x1",
        "mean_x1",
    ]
    _require(
        dynamic_design_X.columns.tolist() == expected_dynamic_columns,
        "dynamic augmented design columns do not match the documented aliases",
    )
    dynamic_design = pd.DataFrame(
        {
            "obs_id": dynamic_obs_ids,
            "entity": dynamic_result.estimation_entity.to_numpy(dtype=np.int32),
            "time": dynamic_times,
            "y": dynamic_result.estimation_outcome.to_numpy(dtype=np.int32),
        }
    )
    dynamic_design = pd.concat([dynamic_design, dynamic_design_X.reset_index(drop=True)], axis=1)
    _require(
        bool((dynamic_design["obs_id"].to_numpy() == dynamic_obs_ids).all()),
        "dynamic design obs_id values do not trace to raw rows",
    )
    for filename, digest in _write_dataset(
        dynamic_design, data_dir / "dynamic_nlswork_design"
    ).items():
        file_hashes[f"data/{filename}"] = digest

    dynamic_probabilities = dynamic_result.base_result.predict_proba(
        dynamic_result.estimation_design, random_effects=0.0
    )
    _record_model(
        model="dynamic_random_effects_ordered_logit",
        dataset="dynamic_nlswork_design",
        result=dynamic_result,
        estimates=estimates,
        covariance=covariance,
        fits=fits,
        predictions=predictions,
        probability_frame=dynamic_probabilities,
        prediction_obs_ids=dynamic_obs_ids,
    )

    reference_frames = {
        "estimates.csv": pd.DataFrame(estimates),
        "covariance.csv": pd.DataFrame(covariance),
        "fit.csv": pd.DataFrame(fits),
        "predictions.csv": pd.DataFrame(predictions),
    }
    for filename, frame in reference_frames.items():
        path = python_dir / filename
        _write_csv(frame, path)
        file_hashes[f"python/{filename}"] = _sha256(path)

    comparison_model_specs = {
        "binary_logit": {
            "kind": "binary",
            "features": ["intercept", "x1", "x2", "x3", "x4"],
            "required": True,
        },
        "binary_probit": {
            "kind": "binary",
            "features": ["intercept", "x1", "x2", "x3", "x4"],
            "required": True,
        },
        "ordered_logit": {
            "kind": "ordered",
            "features": ["ox1", "ox2", "ox3", "ox4"],
            "required": True,
        },
        "ordered_probit": {
            "kind": "ordered",
            "features": ["ox1", "ox2", "ox3", "ox4"],
            "required": True,
        },
        "generalized_ordered_logit": {
            "kind": "generalized",
            "features": ["gx1", "gx2", "gx3", "gx4"],
            "required": False,
            "stata_reference": "gologit2, npl",
            "optional_reason": "The user-written gologit2 command may not be installed.",
            "constraint_slack": float(generalized.constraint_slack),
        },
        "partial_proportional_odds": {
            "kind": "partial",
            "features": ["gx1", "gx2", "gx3", "gx4"],
            "varying": ["gx4"],
            "required": False,
            "stata_reference": "gologit2, npl(gx4)",
            "optional_reason": "The user-written gologit2 command may not be installed.",
            "constraint_slack": float(partial.constraint_slack),
        },
        "random_effects_ordered_logit": {
            "kind": "random_effects",
            "features": ["x1", "x2", "x3", "x4"],
            "required": True,
        },
        "dynamic_random_effects_ordered_logit": {
            "kind": "random_effects",
            "features": expected_dynamic_columns,
            "feature_map": {
                "x1": "x1",
                "state_1": "state[1]",
                "state_2": "state[2]",
                "initial_1": "initial[1]",
                "initial_2": "initial[2]",
                "initial_x1": "initial_x[x1]",
                "mean_x1": "mean[x1]",
            },
            "required": True,
        },
    }
    source_manifest = {
        filename: {
            "url": source["url"],
            "sha256": source["sha256"],
            "role": source["role"],
            "redistribution": (
                "Download for local validation only; source datasets are not distributed "
                "with limiteddepkit."
            ),
            "no_redistribution_note": (
                "Do not commit or redistribute this third-party source dataset."
            ),
        }
        for filename, source in SOURCES.items()
    }
    manifest = {
        "schema_version": 1,
        "suite": "real_data_application",
        "limiteddepkit_version": ldk.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "dependency_versions": {
            "limiteddepkit": ldk.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "quadrature_method": "ghermite",
        "quadrature_points": QUADRATURE_POINTS,
        "panel_optimizer_tolerance": PANEL_OPTIMIZER_TOLERANCE,
        "prediction_rows_per_model": PREDICTION_ROWS,
        "files": file_hashes,
        "source_datasets": source_manifest,
        "transformations": {
            "binary_lbw": (
                "Keep all 189 rows; y=low, intercept=1, x1=age/10, x2=lwt/100, x3=smoke, x4=ht."
            ),
            "ordinal_tvsfpors": (
                "Keep all 1,600 rows; y=thk-1, entity=school, and each ox/gx/x design "
                "uses prethk, cc, tv, and cc*tv."
            ),
            "dynamic_nlswork_raw": (
                "Keep complete idcode/year/ln_wage/tenure rows in years 68..73, retain "
                "entities with exactly six rows, set x1=tenure, and bin ln_wage at "
                "1.4 and 1.8."
            ),
            "dynamic_nlswork_design": (
                "Export the exact post-initial augmented design returned by "
                "DynamicRandomEffectsOrderedLogit with Stata-safe aliases."
            ),
        },
        "sample_assertions": {
            "binary_lbw": {"nobs": 189, "outcome_support": [0, 1]},
            "ordinal_tvsfpors": {
                "nobs": 1_600,
                "n_groups": 28,
                "outcome_support": [0, 1, 2, 3],
            },
            "dynamic_nlswork_raw": {
                "nobs": EXPECTED_DYNAMIC_RAW_NOBS,
                "n_groups": EXPECTED_DYNAMIC_GROUPS,
                "periods_per_group": 6,
                "years": [68, 69, 70, 71, 72, 73],
            },
            "dynamic_nlswork_design": {
                "nobs": EXPECTED_DYNAMIC_ESTIMATION_NOBS,
                "n_groups": EXPECTED_DYNAMIC_GROUPS,
                "dropped_initial": EXPECTED_DYNAMIC_GROUPS,
                "dropped_nonconsecutive": 0,
            },
        },
        "comparison_model_specs": comparison_model_specs,
    }
    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Prepared public-data Stata parity work directory: {workdir}")
    print(f"limiteddepkit version: {ldk.__version__}")
    print(f"Models prepared: {len(comparison_model_specs)}")
    print("Source datasets are cached locally and must not be committed or redistributed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
