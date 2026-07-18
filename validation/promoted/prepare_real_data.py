"""Prepare pinned public-data applications for promoted limiteddepkit families.

The script is deliberately separate from ``validation/stata``.  It validates
and copies the already-maintained LBW and ordinal/panel extracts, acquires four
pinned Stata Press datasets over verified HTTPS (or from ``--source-dir``),
creates cross-language analysis files, fits the twelve promoted estimators, and
writes canonical Python evidence plus a machine-readable manifest.
"""

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

PREDICTION_ROWS = 10
SURVIVAL_TIMES = (5.0, 15.0)
QUADRATURE_POINTS = 20
OPTIMIZER_TOLERANCE = 1e-10
OPTIMIZER_MAXITER = 3_000
STATA_TIMESTAMP = datetime(2000, 1, 1)

SOURCES: dict[str, dict[str, str]] = {
    "rod93.dta": {
        "url": "https://www.stata-press.com/data/r19/rod93.dta",
        "sha256": "023d1676ef716e320b49ebf0e0b31d259439161d3268da5b0b93022d138ddeab",
        "role": "Poisson and NB2 infant-mortality count reference.",
    },
    "mroz87.dta": {
        "url": "https://www.stata-press.com/data/r19/mroz87.dta",
        "sha256": "2dbdabaad3f1c1a1239c1db4c01ca58b26bbfbbd7555bef5e2b07089bbda7c1c",
        "role": "Left-censored and positive-truncated hours-worked reference.",
    },
    "womenwage2.dta": {
        "url": "https://www.stata-press.com/data/r19/womenwage2.dta",
        "sha256": "0ef5bad643a9b7562056d026fd1d2781c5bccb9d4c68a12fdc1b7ea216d9ea5a",
        "role": (
            "Official Stata fictional wage example for open-ended interval-regression "
            "software parity."
        ),
    },
    "cancer.dta": {
        "url": "https://www.stata-press.com/data/r19/cancer.dta",
        "sha256": "928e4449356bdd8d1466709c599a88f68e0aad2f60b3a3932869e3732fb962fb",
        "role": "Right-censored parametric-duration reference.",
    },
}

LEGACY_FILES = {
    "binary_lbw.csv": "binary_lbw.csv",
    "ordinal_tvsfpors.csv": "ordinal_tvsfpors.csv",
    "dynamic_nlswork_raw.csv": "panel_nlswork.csv",
}

MODEL_ORDER = (
    "firth_binary_logit",
    "poisson",
    "negative_binomial_nb2",
    "tobit",
    "truncated_regression",
    "interval_regression",
    "geometric_duration",
    "exponential_duration",
    "weibull_duration",
    "gamma_duration",
    "random_effects_ordered_probit",
    "fixed_effects_ordered_logit",
)


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
        help="Destination for source, prepared-data, and Python-reference files.",
    )
    parser.add_argument(
        "--legacy-work",
        type=Path,
        default=PROJECT_ROOT / "validation" / "stata" / "work" / "real_data" / "data",
        help=(
            "Legacy real-data directory (the data directory by default). Its sibling "
            "manifest.json is required and every reused CSV is hash-validated."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing all four pinned .dta files. This bypasses "
            "network acquisition but never bypasses SHA256 validation."
        ),
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Promoted parity invariant failed: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_source(filename: str, destination: Path) -> None:
    source = SOURCES[filename]
    request = Request(
        source["url"],
        headers={"User-Agent": f"limiteddepkit/{ldk.__version__} promoted-parity"},
    )
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        context = ssl.create_default_context()
        opener = build_opener(HTTPSHandler(context=context), _HTTPSOnlyRedirectHandler())
        with opener.open(request, timeout=90) as response:
            if urlsplit(response.geturl()).scheme.lower() != "https":
                raise RuntimeError("the source download left verified HTTPS")
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = _sha256(temporary)
        expected = source["sha256"]
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {filename}: expected {expected}, got {actual}")
        temporary.replace(destination)
    except (HTTPError, URLError, OSError, RuntimeError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download pinned {filename} over verified HTTPS. Place it in a "
            "directory supplied through --source-dir and rerun."
        ) from exc


def _prepare_sources(workdir: Path, supplied_source_dir: Path | None) -> dict[str, Path]:
    cache_dir = workdir / "source"
    cache_dir.mkdir(parents=True, exist_ok=True)
    supplied = supplied_source_dir.resolve() if supplied_source_dir is not None else None
    if supplied is not None:
        _require(supplied.is_dir(), f"--source-dir does not exist: {supplied}")

    prepared: dict[str, Path] = {}
    for filename, source in SOURCES.items():
        cached = cache_dir / filename
        if supplied is not None:
            acquired = supplied / filename
            _require(acquired.is_file(), f"--source-dir is missing {filename}")
            _require(
                _sha256(acquired) == source["sha256"],
                f"SHA256 mismatch for supplied {filename}",
            )
            if acquired.resolve() != cached.resolve():
                shutil.copyfile(acquired, cached)
        elif not cached.exists():
            _download_source(filename, cached)

        _require(cached.is_file(), f"source cache is missing {filename}")
        _require(
            _sha256(cached) == source["sha256"],
            f"SHA256 mismatch for cached {filename}",
        )
        prepared[filename] = cached
    return prepared


def _resolve_legacy(legacy_argument: Path) -> tuple[Path, Path, dict[str, Any]]:
    candidate = legacy_argument.resolve()
    if (candidate / "manifest.json").is_file() and (candidate / "data").is_dir():
        legacy_root = candidate
        data_dir = candidate / "data"
    else:
        data_dir = candidate
        legacy_root = candidate.parent
    manifest_path = legacy_root / "manifest.json"
    _require(manifest_path.is_file(), f"legacy source manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest.get("files"), dict), "legacy manifest has no files map")
    for source_name in LEGACY_FILES:
        path = data_dir / source_name
        expected = manifest["files"].get(f"data/{source_name}")
        _require(path.is_file(), f"legacy data is missing {path}")
        _require(isinstance(expected, str), f"legacy manifest has no hash for {source_name}")
        _require(_sha256(path) == expected, f"legacy hash mismatch for {source_name}")
    return data_dir, manifest_path, manifest


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="",
        lineterminator="\n",
    )


def _write_dta(frame: pd.DataFrame, path: Path) -> None:
    stata_frame = frame.replace([np.inf, -np.inf], np.nan)
    stata_frame.to_stata(
        path,
        write_index=False,
        version=118,
        time_stamp=STATA_TIMESTAMP,
    )


def _write_dataset(frame: pd.DataFrame, stem: Path) -> dict[str, str]:
    csv_path = stem.with_suffix(".csv")
    dta_path = stem.with_suffix(".dta")
    _write_csv(frame, csv_path)
    _write_dta(frame, dta_path)
    return {
        csv_path.name: _sha256(csv_path),
        dta_path.name: _sha256(dta_path),
    }


def _copy_legacy_dataset(
    source: Path, destination_stem: Path, frame: pd.DataFrame
) -> dict[str, str]:
    csv_path = destination_stem.with_suffix(".csv")
    dta_path = destination_stem.with_suffix(".dta")
    shutil.copyfile(source, csv_path)
    _write_dta(frame, dta_path)
    return {
        csv_path.name: _sha256(csv_path),
        dta_path.name: _sha256(dta_path),
    }


def _zscore(values: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    numeric = values.astype(float)
    center = float(numeric.mean())
    scale = float(numeric.std(ddof=1))
    _require(np.isfinite(scale) and scale > 0.0, f"cannot scale {values.name}")
    return (numeric - center) / scale, {"center": center, "scale": scale, "ddof": 1}


def _assert_result(model: str, result: Any, expected_nobs: int) -> None:
    _require(bool(result.converged), f"{model} did not converge")
    _require(bool(result.inference_valid), f"{model} did not produce valid inference")
    _require(int(result.nobs) == expected_nobs, f"{model} nobs != {expected_nobs}")
    parameters = result.all_params.to_numpy(dtype=float)
    covariance = result.covariance.to_numpy(dtype=float)
    _require(np.isfinite(parameters).all(), f"{model} has nonfinite estimates")
    _require(np.isfinite(covariance).all(), f"{model} has nonfinite covariance")
    _require(np.isfinite(float(result.loglike)), f"{model} has nonfinite log likelihood")


def _safe_numeric(result: Any, name: str) -> float | int | None:
    value = getattr(result, name, None)
    if value is None or callable(value):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _record_result(
    *,
    model: str,
    dataset: str,
    result: Any,
    estimates: list[dict[str, Any]],
    covariance: list[dict[str, Any]],
    fits: list[dict[str, Any]],
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

    n_groups = _safe_numeric(result, "n_groups")
    if n_groups is None:
        n_groups = _safe_numeric(result, "n_entities")
    fits.append(
        {
            "model": model,
            "dataset": dataset,
            "nobs": int(result.nobs),
            "n_params": len(all_params),
            "loglike": float(result.loglike),
            "aic": _safe_numeric(result, "aic"),
            "bic": _safe_numeric(result, "bic"),
            "converged": bool(result.converged),
            "inference_valid": bool(result.inference_valid),
            "n_groups": n_groups,
            "n_contributing_entities": _safe_numeric(result, "n_contributing_entities"),
            "n_cutoff_clones": _safe_numeric(result, "n_cutoff_clones"),
            "n_pseudo_observations": _safe_numeric(result, "n_pseudo_observations"),
            "n_events": _safe_numeric(result, "n_events"),
            "n_censored": _safe_numeric(result, "n_censored"),
            "n_interval": _safe_numeric(result, "n_interval"),
            "n_exact": _safe_numeric(result, "n_exact"),
            "n_left_censored": _safe_numeric(result, "n_left_censored"),
            "n_right_censored": _safe_numeric(result, "n_right_censored"),
            "score_norm": _safe_numeric(result, "score_norm"),
            "scaled_score_norm": _safe_numeric(result, "scaled_score_norm"),
            "penalized_loglike": _safe_numeric(result, "penalized_loglike"),
            "jeffreys_penalty": _safe_numeric(result, "jeffreys_penalty"),
            "backend": str(getattr(result, "backend", "")),
            "covariance_type": str(getattr(result, "covariance_type", "")),
        }
    )


def _append_series_prediction(
    predictions: list[dict[str, Any]],
    *,
    model: str,
    dataset: str,
    obs_ids: Any,
    prediction: str,
    values: Any,
    category: Any = None,
    time: Any = None,
) -> None:
    ids = np.asarray(obs_ids)
    numeric = np.asarray(values, dtype=float)
    _require(ids.shape == numeric.shape, f"{model} prediction shape mismatch")
    for obs_id, value in zip(ids, numeric, strict=True):
        predictions.append(
            {
                "model": model,
                "dataset": dataset,
                "obs_id": int(obs_id),
                "prediction": prediction,
                "category": category,
                "time": time,
                "value": float(value),
            }
        )


def _append_probability_predictions(
    predictions: list[dict[str, Any]],
    *,
    model: str,
    dataset: str,
    obs_ids: Any,
    probabilities: pd.DataFrame,
) -> None:
    for category in probabilities.columns:
        _append_series_prediction(
            predictions,
            model=model,
            dataset=dataset,
            obs_ids=obs_ids,
            prediction="probability",
            category=str(category),
            values=probabilities[category],
        )


def _append_duration_predictions(
    predictions: list[dict[str, Any]],
    *,
    model: str,
    dataset: str,
    result: Any,
    X: pd.DataFrame,
    obs_ids: pd.Series,
) -> None:
    X_prediction = X.iloc[:PREDICTION_ROWS]
    ids = obs_ids.iloc[:PREDICTION_ROWS]
    _append_series_prediction(
        predictions,
        model=model,
        dataset=dataset,
        obs_ids=ids,
        prediction="mean",
        values=result.predict_mean(X_prediction),
    )
    survival = result.predict_survival(X_prediction, times=SURVIVAL_TIMES)
    _require(isinstance(survival, pd.DataFrame), f"{model} survival must be a DataFrame")
    for time in survival.columns:
        _append_series_prediction(
            predictions,
            model=model,
            dataset=dataset,
            obs_ids=ids,
            prediction="survival",
            time=float(time),
            values=survival[time],
        )


def _comparison_tolerance(model: str) -> dict[str, float]:
    if model == "random_effects_ordered_probit":
        return {
            "estimate_atol": 2e-3,
            "estimate_rtol": 2e-3,
            "covariance_atol": 3e-3,
            "prediction_atol": 2e-3,
        }
    if model in {"weibull_duration", "gamma_duration"}:
        return {
            "estimate_atol": 2e-4,
            "estimate_rtol": 2e-4,
            "covariance_atol": 5e-4,
            "prediction_atol": 3e-4,
        }
    return {
        "estimate_atol": 5e-5,
        "estimate_rtol": 5e-5,
        "covariance_atol": 1e-4,
        "prediction_atol": 1e-4,
    }


def _common_parameter_maps(
    features: list[str], *, intercept_name: str = "intercept"
) -> dict[str, dict[str, str]]:
    r_map = {feature: feature for feature in features}
    stata_map = {feature: feature for feature in features}
    if intercept_name in features:
        r_map["(Intercept)"] = intercept_name
        stata_map["_cons"] = intercept_name
    return {"r_to_canonical": r_map, "stata_to_canonical": stata_map}


def main() -> int:
    args = _parse_args()
    workdir = args.output.resolve()
    data_dir = workdir / "data"
    python_dir = workdir / "python"
    data_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)

    legacy_dir, legacy_manifest_path, legacy_manifest = _resolve_legacy(args.legacy_work)
    sources = _prepare_sources(workdir, args.source_dir)
    file_hashes = {f"source/{name}": _sha256(path) for name, path in sources.items()}

    binary_data = pd.read_csv(legacy_dir / "binary_lbw.csv")
    ordinal_data = pd.read_csv(legacy_dir / "ordinal_tvsfpors.csv")
    panel_data = pd.read_csv(legacy_dir / "dynamic_nlswork_raw.csv")
    _require(binary_data.shape[0] == 189, "binary_lbw must contain 189 rows")
    _require(set(binary_data["y"]) == {0, 1}, "binary_lbw outcome support changed")
    _require(ordinal_data.shape[0] == 1_600, "ordinal_tvsfpors must contain 1,600 rows")
    _require(ordinal_data["entity"].nunique() == 28, "ordinal entity count changed")
    _require(set(ordinal_data["y"]) == {0, 1, 2, 3}, "ordinal support changed")
    _require(panel_data.shape[0] == 2_010, "panel_nlswork must contain 2,010 rows")
    _require(panel_data["entity"].nunique() == 335, "panel entity count changed")
    _require(set(panel_data["y"]) == {0, 1, 2}, "panel outcome support changed")

    legacy_outputs = {
        "binary_lbw.csv": (binary_data, "binary_lbw"),
        "ordinal_tvsfpors.csv": (ordinal_data, "ordinal_tvsfpors"),
        "dynamic_nlswork_raw.csv": (panel_data, "panel_nlswork"),
    }
    for source_name, (frame, destination_name) in legacy_outputs.items():
        hashes = _copy_legacy_dataset(
            legacy_dir / source_name,
            data_dir / destination_name,
            frame,
        )
        file_hashes.update({f"data/{name}": digest for name, digest in hashes.items()})

    rod = pd.read_stata(sources["rod93.dta"], convert_categoricals=False)
    count_data = pd.DataFrame(
        {
            "obs_id": np.arange(1, len(rod) + 1, dtype=np.int32),
            "deaths": rod["deaths"].astype(np.int32),
            "exposure": rod["exposure"].astype(float),
            "intercept": 1.0,
            "log_age": np.log(rod["age_mos"].astype(float)),
            "cohort_2": (rod["cohort"] == 2).astype(np.int8),
            "cohort_3": (rod["cohort"] == 3).astype(np.int8),
        }
    )
    _require((count_data["exposure"] > 0.0).all(), "rod93 exposure must be positive")
    hashes = _write_dataset(count_data, data_dir / "count_rod93")
    file_hashes.update({f"data/{name}": digest for name, digest in hashes.items()})

    mroz = pd.read_stata(sources["mroz87.dta"], convert_categoricals=False)
    age_z, age_scaling = _zscore(mroz["wifeage"])
    education_z, education_scaling = _zscore(mroz["wedyrs"])
    experience_z, experience_scaling = _zscore(mroz["wexper"])
    censoring_data = pd.DataFrame(
        {
            "obs_id": np.arange(1, len(mroz) + 1, dtype=np.int32),
            "y": mroz["whrs75"].astype(float) / 1_000.0,
            "intercept": 1.0,
            "age_z": age_z,
            "education_z": education_z,
            "experience_z": experience_z,
            "young_children": mroz["kl6"].astype(float),
        }
    )
    censoring_data["positive"] = (censoring_data["y"] > 0.0).astype(np.int8)
    _require(censoring_data["positive"].sum() == 428, "mroz positive-hours count changed")
    hashes = _write_dataset(censoring_data, data_dir / "censoring_mroz87")
    file_hashes.update({f"data/{name}": digest for name, digest in hashes.items()})

    wages = pd.read_stata(sources["womenwage2.dta"], convert_categoricals=False)
    wage_age_z, wage_age_scaling = _zscore(wages["age"])
    school_z, school_scaling = _zscore(wages["school"])
    tenure_z, tenure_scaling = _zscore(wages["tenure"])
    interval_data = pd.DataFrame(
        {
            "obs_id": np.arange(1, len(wages) + 1, dtype=np.int32),
            "lower": wages["wage1"].astype(float) / 10.0,
            "upper": wages["wage2"].astype(float) / 10.0,
            "intercept": 1.0,
            "age_z": wage_age_z,
            "school_z": school_z,
            "tenure_z": tenure_z,
            "never_married": wages["nev_mar"].astype(float),
            "rural": wages["rural"].astype(float),
        }
    )
    _require(interval_data["lower"].isna().sum() == 14, "open lower-bound count changed")
    _require(interval_data["upper"].isna().sum() == 6, "open upper-bound count changed")
    hashes = _write_dataset(interval_data, data_dir / "censoring_womenwage2")
    file_hashes.update({f"data/{name}": digest for name, digest in hashes.items()})

    cancer = pd.read_stata(sources["cancer.dta"], convert_categoricals=False)
    cancer_age_z, cancer_age_scaling = _zscore(cancer["age"])
    duration_data = pd.DataFrame(
        {
            "obs_id": np.arange(1, len(cancer) + 1, dtype=np.int32),
            "duration": cancer["studytime"].astype(float),
            "event": cancer["died"].astype(np.int8),
            "intercept": 1.0,
            "age_z": cancer_age_z,
            "drug_2": (cancer["drug"] == 2).astype(np.int8),
            "drug_3": (cancer["drug"] == 3).astype(np.int8),
        }
    )
    _require(duration_data["event"].sum() == 31, "cancer death count changed")
    hashes = _write_dataset(duration_data, data_dir / "duration_cancer")
    file_hashes.update({f"data/{name}": digest for name, digest in hashes.items()})
    duration_deaths = duration_data.loc[duration_data["event"] == 1].reset_index(drop=True)
    hashes = _write_dataset(duration_deaths, data_dir / "duration_cancer_deaths")
    file_hashes.update({f"data/{name}": digest for name, digest in hashes.items()})

    estimates: list[dict[str, Any]] = []
    covariance: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    binary_features = ["intercept", "x1", "x2", "x3", "x4"]
    binary_X = binary_data[binary_features]
    result = ldk.FirthBinaryLogit().fit(
        binary_X,
        binary_data["y"],
        tolerance=OPTIMIZER_TOLERANCE,
        maxiter=OPTIMIZER_MAXITER,
    )
    _assert_result("firth_binary_logit", result, 189)
    results["firth_binary_logit"] = result
    probability = result.predict_proba(binary_X.iloc[:PREDICTION_ROWS])
    _append_probability_predictions(
        predictions,
        model="firth_binary_logit",
        dataset="binary_lbw",
        obs_ids=binary_data["obs_id"].iloc[:PREDICTION_ROWS],
        probabilities=probability,
    )

    count_features = ["intercept", "log_age", "cohort_2", "cohort_3"]
    count_X = count_data[count_features]
    for model, estimator in (
        ("poisson", ldk.PoissonRegressor()),
        ("negative_binomial_nb2", ldk.NegativeBinomialNB2()),
    ):
        result = estimator.fit(
            count_X,
            count_data["deaths"],
            exposure=count_data["exposure"],
            tolerance=OPTIMIZER_TOLERANCE,
            maxiter=OPTIMIZER_MAXITER,
        )
        _assert_result(model, result, 21)
        results[model] = result
        _append_series_prediction(
            predictions,
            model=model,
            dataset="count_rod93",
            obs_ids=count_data["obs_id"].iloc[:PREDICTION_ROWS],
            prediction="mean",
            values=result.predict(
                count_X.iloc[:PREDICTION_ROWS],
                exposure=count_data["exposure"].iloc[:PREDICTION_ROWS],
            ),
        )

    censoring_features = [
        "intercept",
        "age_z",
        "education_z",
        "experience_z",
        "young_children",
    ]
    censoring_X = censoring_data[censoring_features]
    result = ldk.Tobit(censoring_point=0.0, side="left").fit(
        censoring_X,
        censoring_data["y"],
        tolerance=OPTIMIZER_TOLERANCE,
        maxiter=OPTIMIZER_MAXITER,
    )
    _assert_result("tobit", result, 753)
    results["tobit"] = result
    _append_series_prediction(
        predictions,
        model="tobit",
        dataset="censoring_mroz87",
        obs_ids=censoring_data["obs_id"].iloc[:PREDICTION_ROWS],
        prediction="mean",
        values=result.predict(censoring_X.iloc[:PREDICTION_ROWS], which="observed"),
    )

    positive = censoring_data["positive"] == 1
    truncated_X = censoring_X.loc[positive].reset_index(drop=True)
    truncated_y = censoring_data.loc[positive, "y"].reset_index(drop=True)
    truncated_ids = censoring_data.loc[positive, "obs_id"].reset_index(drop=True)
    result = ldk.TruncatedRegression(truncation_point=0.0, side="left").fit(
        truncated_X,
        truncated_y,
        tolerance=OPTIMIZER_TOLERANCE,
        maxiter=OPTIMIZER_MAXITER,
    )
    _assert_result("truncated_regression", result, 428)
    results["truncated_regression"] = result
    _append_series_prediction(
        predictions,
        model="truncated_regression",
        dataset="censoring_mroz87",
        obs_ids=truncated_ids.iloc[:PREDICTION_ROWS],
        prediction="mean",
        values=result.predict(truncated_X.iloc[:PREDICTION_ROWS], which="conditional"),
    )

    interval_features = [
        "intercept",
        "age_z",
        "school_z",
        "tenure_z",
        "never_married",
        "rural",
    ]
    interval_X = interval_data[interval_features]
    lower = interval_data["lower"].fillna(-np.inf)
    upper = interval_data["upper"].fillna(np.inf)
    result = ldk.IntervalRegression().fit(
        interval_X,
        lower,
        upper,
        tolerance=OPTIMIZER_TOLERANCE,
        maxiter=OPTIMIZER_MAXITER,
    )
    _assert_result("interval_regression", result, 488)
    results["interval_regression"] = result
    _append_series_prediction(
        predictions,
        model="interval_regression",
        dataset="censoring_womenwage2",
        obs_ids=interval_data["obs_id"].iloc[:PREDICTION_ROWS],
        prediction="mean",
        values=result.predict(interval_X.iloc[:PREDICTION_ROWS]),
    )

    duration_features = ["intercept", "age_z", "drug_2", "drug_3"]
    duration_X = duration_data[duration_features]
    for model, estimator in (
        ("geometric_duration", ldk.GeometricDuration()),
        ("exponential_duration", ldk.ExponentialDuration()),
        ("weibull_duration", ldk.WeibullDuration()),
    ):
        result = estimator.fit(
            duration_X,
            duration_data["duration"],
            duration_data["event"],
            tolerance=OPTIMIZER_TOLERANCE,
            maxiter=OPTIMIZER_MAXITER,
        )
        _assert_result(model, result, 48)
        results[model] = result
        _append_duration_predictions(
            predictions,
            model=model,
            dataset="duration_cancer",
            result=result,
            X=duration_X,
            obs_ids=duration_data["obs_id"],
        )

    gamma_X = duration_deaths[duration_features]
    result = ldk.GammaDuration().fit(
        gamma_X,
        duration_deaths["duration"],
        duration_deaths["event"],
        tolerance=OPTIMIZER_TOLERANCE,
        maxiter=OPTIMIZER_MAXITER,
    )
    _assert_result("gamma_duration", result, 31)
    results["gamma_duration"] = result
    _append_duration_predictions(
        predictions,
        model="gamma_duration",
        dataset="duration_cancer_deaths",
        result=result,
        X=gamma_X,
        obs_ids=duration_deaths["obs_id"],
    )

    ordinal_features = ["x1", "x2", "x3", "x4"]
    ordinal_X = ordinal_data[ordinal_features]
    result = ldk.RandomEffectsOrderedProbit().fit(
        ordinal_X,
        ordinal_data["y"],
        entity=ordinal_data["entity"],
        category_order=[0, 1, 2, 3],
        quadrature_points=QUADRATURE_POINTS,
        tolerance=OPTIMIZER_TOLERANCE,
        maxiter=OPTIMIZER_MAXITER,
    )
    _assert_result("random_effects_ordered_probit", result, 1_600)
    _require(result.n_groups == 28, "random-effects Probit group count changed")
    results["random_effects_ordered_probit"] = result
    probability = result.predict_proba(ordinal_X.iloc[:PREDICTION_ROWS], random_effects=0.0)
    _append_probability_predictions(
        predictions,
        model="random_effects_ordered_probit",
        dataset="ordinal_tvsfpors",
        obs_ids=ordinal_data["obs_id"].iloc[:PREDICTION_ROWS],
        probabilities=probability,
    )

    panel_features = ["x1"]
    panel_X = panel_data[panel_features]
    result = ldk.FixedEffectsOrderedLogit().fit(
        panel_X,
        panel_data["y"],
        entity=panel_data["entity"],
        category_order=[0, 1, 2],
        tolerance=OPTIMIZER_TOLERANCE,
        maxiter=OPTIMIZER_MAXITER,
    )
    _assert_result("fixed_effects_ordered_logit", result, 2_010)
    results["fixed_effects_ordered_logit"] = result
    _append_series_prediction(
        predictions,
        model="fixed_effects_ordered_logit",
        dataset="panel_nlswork",
        obs_ids=panel_data["obs_id"].iloc[:PREDICTION_ROWS],
        prediction="linear_index",
        values=result.linear_index(panel_X.iloc[:PREDICTION_ROWS]),
    )

    dataset_by_model = {
        "firth_binary_logit": "binary_lbw",
        "poisson": "count_rod93",
        "negative_binomial_nb2": "count_rod93",
        "tobit": "censoring_mroz87",
        "truncated_regression": "censoring_mroz87",
        "interval_regression": "censoring_womenwage2",
        "geometric_duration": "duration_cancer",
        "exponential_duration": "duration_cancer",
        "weibull_duration": "duration_cancer",
        "gamma_duration": "duration_cancer_deaths",
        "random_effects_ordered_probit": "ordinal_tvsfpors",
        "fixed_effects_ordered_logit": "panel_nlswork",
    }
    for model in MODEL_ORDER:
        _record_result(
            model=model,
            dataset=dataset_by_model[model],
            result=results[model],
            estimates=estimates,
            covariance=covariance,
            fits=fits,
        )

    output_frames = {
        "estimates.csv": pd.DataFrame(estimates),
        "covariance.csv": pd.DataFrame(covariance),
        "fit.csv": pd.DataFrame(fits),
        "predictions.csv": pd.DataFrame(predictions),
    }
    for filename, frame in output_frames.items():
        path = python_dir / filename
        _write_csv(frame, path)
        file_hashes[f"python/{filename}"] = _sha256(path)

    model_specs: dict[str, dict[str, Any]] = {
        "firth_binary_logit": {
            "kind": "firth_binary",
            "dataset": "binary_lbw",
            "data_file": "data/binary_lbw.csv",
            "features": binary_features,
            "outcome": "y",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["probability"],
            "parameter_mappings": _common_parameter_maps(
                binary_features, intercept_name="intercept"
            ),
        },
        "poisson": {
            "kind": "count",
            "dataset": "count_rod93",
            "data_file": "data/count_rod93.csv",
            "features": count_features,
            "outcome": "deaths",
            "exposure": "exposure",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["mean"],
            "parameter_mappings": _common_parameter_maps(count_features),
        },
        "negative_binomial_nb2": {
            "kind": "count_nb2",
            "dataset": "count_rod93",
            "data_file": "data/count_rod93.csv",
            "features": count_features,
            "outcome": "deaths",
            "exposure": "exposure",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["mean"],
            "parameter_mappings": {
                **_common_parameter_maps(count_features),
                "transforms_to_canonical": {
                    "r": {"log_alpha": "-log(theta)"},
                    "stata": {"log_alpha": "/lnalpha"},
                },
            },
        },
        "tobit": {
            "kind": "left_censored_gaussian",
            "dataset": "censoring_mroz87",
            "data_file": "data/censoring_mroz87.csv",
            "features": censoring_features,
            "outcome": "y",
            "censoring_point": 0.0,
            "side": "left",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["mean"],
            "prediction_mean": "observed",
            "parameter_mappings": {
                **_common_parameter_maps(censoring_features),
                "transforms_to_canonical": {
                    "r": {"sigma": "exp(Log(scale))"},
                    "stata": {"sigma": "sigma"},
                },
            },
        },
        "truncated_regression": {
            "kind": "left_truncated_gaussian",
            "dataset": "censoring_mroz87",
            "data_file": "data/censoring_mroz87.csv",
            "features": censoring_features,
            "outcome": "y",
            "subset": "positive == 1",
            "subset_field": "positive",
            "subset_value": 1,
            "truncation_point": 0.0,
            "side": "left",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["mean"],
            "prediction_mean": "conditional",
            "parameter_mappings": {
                **_common_parameter_maps(censoring_features),
                "transforms_to_canonical": {
                    "r": {"sigma": "exp(logSigma)"},
                    "stata": {"sigma": "sigma"},
                },
            },
        },
        "interval_regression": {
            "kind": "interval_gaussian",
            "dataset": "censoring_womenwage2",
            "data_file": "data/censoring_womenwage2.csv",
            "features": interval_features,
            "lower": "lower",
            "upper": "upper",
            "open_bound_encoding": "missing; map missing lower to -Inf and missing upper to +Inf",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["mean"],
            "prediction_mean": "latent",
            "parameter_mappings": {
                **_common_parameter_maps(interval_features),
                "transforms_to_canonical": {
                    "r": {"sigma": "exp(Log(scale))"},
                    "stata": {"sigma": "sigma"},
                },
            },
        },
        "geometric_duration": {
            "kind": "geometric_duration",
            "dataset": "duration_cancer",
            "data_file": "data/duration_cancer.csv",
            "features": duration_features,
            "duration": "duration",
            "event": "event",
            "time_scale": "integer study periods",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_times": list(SURVIVAL_TIMES),
            "prediction_types": ["mean", "survival"],
            "parameter_mappings": _common_parameter_maps(duration_features),
        },
        "exponential_duration": {
            "kind": "exponential_aft",
            "dataset": "duration_cancer",
            "data_file": "data/duration_cancer.csv",
            "features": duration_features,
            "duration": "duration",
            "event": "event",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_times": list(SURVIVAL_TIMES),
            "prediction_types": ["mean", "survival"],
            "parameter_mappings": _common_parameter_maps(duration_features),
        },
        "weibull_duration": {
            "kind": "weibull_aft",
            "dataset": "duration_cancer",
            "data_file": "data/duration_cancer.csv",
            "features": duration_features,
            "duration": "duration",
            "event": "event",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_times": list(SURVIVAL_TIMES),
            "prediction_types": ["mean", "survival"],
            "parameter_mappings": {
                **_common_parameter_maps(duration_features),
                "transforms_to_canonical": {
                    "r": {"log_alpha": "-Log(scale)"},
                    "stata": {"log_alpha": "/ln_p"},
                },
            },
        },
        "gamma_duration": {
            "kind": "gamma_aft_uncensored",
            "dataset": "duration_cancer_deaths",
            "data_file": "data/duration_cancer_deaths.csv",
            "features": duration_features,
            "duration": "duration",
            "event": "event",
            "subset": "event == 1",
            "prediction_rows": PREDICTION_ROWS,
            "prediction_times": list(SURVIVAL_TIMES),
            "prediction_types": ["mean", "survival"],
            "parameter_mappings": {
                **_common_parameter_maps(duration_features),
                "transforms_to_canonical": {
                    "r": {"log_k": "log(shape)"},
                    "stata": {"log_k": "log(shape)"},
                },
            },
        },
        "random_effects_ordered_probit": {
            "kind": "random_intercept_ordered_probit",
            "dataset": "ordinal_tvsfpors",
            "data_file": "data/ordinal_tvsfpors.csv",
            "features": ordinal_features,
            "outcome": "y",
            "entity": "entity",
            "category_order": [0, 1, 2, 3],
            "quadrature_method": "nonadaptive_gauss_hermite",
            "quadrature_points": QUADRATURE_POINTS,
            "prediction_random_effect": 0.0,
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["probability"],
            "parameter_mappings": {
                **_common_parameter_maps(ordinal_features),
                "r_to_canonical": {
                    **_common_parameter_maps(ordinal_features)["r_to_canonical"],
                    "0|1": "threshold: 0 | 1",
                    "1|2": "threshold: 1 | 2",
                    "2|3": "threshold: 2 | 3",
                    "sigma_entity": "sigma_entity",
                },
                "stata_to_canonical": {
                    **_common_parameter_maps(ordinal_features)["stata_to_canonical"],
                    "/cut1": "threshold: 0 | 1",
                    "/cut2": "threshold: 1 | 2",
                    "/cut3": "threshold: 2 | 3",
                    "sigma_entity": "sigma_entity",
                },
            },
        },
        "fixed_effects_ordered_logit": {
            "kind": "buc_fixed_effects_ordered_logit",
            "dataset": "panel_nlswork",
            "data_file": "data/panel_nlswork.csv",
            "features": panel_features,
            "outcome": "y",
            "entity": "entity",
            "time": "time",
            "category_order": [0, 1, 2],
            "buc_cutoffs": [1, 2],
            "drop_nonvarying_entity_cutoff_clones": True,
            "cluster": "entity",
            "expected_n_contributing_entities": int(
                results["fixed_effects_ordered_logit"].n_contributing_entities
            ),
            "expected_n_cutoff_clones": int(results["fixed_effects_ordered_logit"].n_cutoff_clones),
            "expected_n_pseudo_observations": int(
                results["fixed_effects_ordered_logit"].n_pseudo_observations
            ),
            "prediction_rows": PREDICTION_ROWS,
            "prediction_types": ["linear_index"],
            "parameter_mappings": _common_parameter_maps(panel_features, intercept_name="unused"),
        },
    }

    for model, spec in model_specs.items():
        spec["parameter_order"] = [str(value) for value in results[model].all_params.index]
        spec["fit_options"] = {
            "maxiter": OPTIMIZER_MAXITER,
            "tolerance": OPTIMIZER_TOLERANCE,
        }
        spec["comparison_tolerances"] = _comparison_tolerance(model)

    source_manifest = {
        filename: {
            **source,
            "redistribution": (
                "Download for local validation only; source datasets are not distributed "
                "with limiteddepkit releases."
            ),
        }
        for filename, source in SOURCES.items()
    }
    manifest = {
        "schema_version": 1,
        "suite": "promoted_public_data_parity",
        "model_order": list(MODEL_ORDER),
        "dependency_versions": {
            "limiteddepkit": ldk.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "preparation": {
            "prediction_rows_per_model": PREDICTION_ROWS,
            "survival_prediction_times": list(SURVIVAL_TIMES),
            "optimizer_tolerance": OPTIMIZER_TOLERANCE,
            "optimizer_maxiter": OPTIMIZER_MAXITER,
            "quadrature_points": QUADRATURE_POINTS,
        },
        "source_datasets": source_manifest,
        "legacy_source": {
            "manifest_path": str(legacy_manifest_path),
            "manifest_sha256": _sha256(legacy_manifest_path),
            "suite": legacy_manifest.get("suite"),
            "reused_files": {
                source: {
                    "output": destination,
                    "sha256": legacy_manifest["files"][f"data/{source}"],
                }
                for source, destination in LEGACY_FILES.items()
            },
        },
        "transformations": {
            "count_rod93": {
                "outcome": "deaths",
                "exposure": "exposure",
                "log_age": "log(age_mos)",
                "cohort_coding": "treatment indicators for cohorts 2 and 3; cohort 1 reference",
            },
            "censoring_mroz87": {
                "outcome": "whrs75 / 1000",
                "positive": "1[outcome > 0]",
                "age_z": {"source": "wifeage", **age_scaling},
                "education_z": {"source": "wedyrs", **education_scaling},
                "experience_z": {"source": "wexper", **experience_scaling},
                "young_children": "kl6",
            },
            "censoring_womenwage2": {
                "lower": "wage1 / 10; missing remains an open lower endpoint",
                "upper": "wage2 / 10; missing remains an open upper endpoint",
                "age_z": {"source": "age", **wage_age_scaling},
                "school_z": {"source": "school", **school_scaling},
                "tenure_z": {"source": "tenure", **tenure_scaling},
            },
            "duration_cancer": {
                "duration": "studytime",
                "event": "died",
                "age_z": {"source": "age", **cancer_age_scaling},
                "drug_coding": "treatment indicators for drugs 2 and 3; drug 1 reference",
                "gamma_subset": "event == 1 (31 uncensored deaths)",
            },
            "binary_lbw": legacy_manifest.get("transformations", {}).get("binary_lbw"),
            "ordinal_tvsfpors": legacy_manifest.get("transformations", {}).get("ordinal_tvsfpors"),
            "panel_nlswork": legacy_manifest.get("transformations", {}).get("dynamic_nlswork_raw"),
        },
        "sample_assertions": {
            "binary_lbw": {"nobs": 189, "outcome_support": [0, 1]},
            "count_rod93": {"nobs": 21, "cohorts": [1, 2, 3]},
            "censoring_mroz87": {"nobs": 753, "positive_nobs": 428},
            "censoring_womenwage2": {
                "nobs": 488,
                "open_lower": 14,
                "open_upper": 6,
            },
            "duration_cancer": {"nobs": 48, "events": 31},
            "duration_cancer_deaths": {"nobs": 31, "events": 31},
            "ordinal_tvsfpors": {"nobs": 1_600, "n_groups": 28},
            "panel_nlswork": {
                "nobs": 2_010,
                "n_groups": 335,
                "n_contributing_entities": int(
                    results["fixed_effects_ordered_logit"].n_contributing_entities
                ),
                "n_cutoff_clones": int(results["fixed_effects_ordered_logit"].n_cutoff_clones),
                "n_pseudo_observations": int(
                    results["fixed_effects_ordered_logit"].n_pseudo_observations
                ),
            },
        },
        "model_specs": model_specs,
        "output_schemas": {
            "python/estimates.csv": [
                "model",
                "dataset",
                "parameter",
                "estimate",
                "standard_error",
            ],
            "python/covariance.csv": [
                "model",
                "dataset",
                "row_parameter",
                "column_parameter",
                "covariance",
            ],
            "python/fit.csv": list(output_frames["fit.csv"].columns),
            "python/predictions.csv": [
                "model",
                "dataset",
                "obs_id",
                "prediction",
                "category",
                "time",
                "value",
            ],
        },
        "files": dict(sorted(file_hashes.items())),
    }
    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared promoted-family public-data parity suite: {workdir}")
    print(f"limiteddepkit version: {ldk.__version__}")
    print(f"Models fitted: {len(results)}")
    print("All twelve models converged and produced finite canonical inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
