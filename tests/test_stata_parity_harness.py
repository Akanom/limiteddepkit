"""Fast structural tests for the manual Stata parity harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = PROJECT_ROOT / "validation" / "stata" / "compare_parity.py"


def _load_comparator():
    spec = importlib.util.spec_from_file_location("limiteddepkit_stata_compare", COMPARE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_rows(model: str, parameters: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "position": position,
                "stata_parameter": name,
                "estimate": estimate,
                "standard_error": standard_error,
            }
            for position, (name, estimate, standard_error) in enumerate(parameters, start=1)
        ]
    )


def _minimal_comparison_tables() -> dict[str, pd.DataFrame]:
    model = "binary_logit"
    estimates = pd.DataFrame(
        [
            {
                "model": model,
                "parameter": "intercept",
                "estimate": -0.25,
                "standard_error": 0.1,
            }
        ]
    )
    covariance = pd.DataFrame(
        [
            {
                "model": model,
                "row_parameter": "intercept",
                "column_parameter": "intercept",
                "covariance": 0.01,
            }
        ]
    )
    fit = pd.DataFrame(
        [
            {
                "model": model,
                "nobs": 20,
                "n_groups": np.nan,
                "n_params": 1,
                "loglike": -10.0,
                "aic": 22.0,
                "bic": 23.0,
                "converged": 1,
            }
        ]
    )
    predictions = pd.DataFrame([{"model": model, "obs_id": 1, "category": 0, "probability": 0.6}])
    return {
        "estimates": estimates,
        "covariance": covariance,
        "fit": fit,
        "predictions": predictions,
    }


def _minimal_raw_stata_tables() -> dict[str, pd.DataFrame]:
    model = "binary_logit"
    tables = _minimal_comparison_tables()
    return {
        "estimates": pd.DataFrame(
            [
                {
                    "model": model,
                    "position": 1,
                    "stata_parameter": "y:intercept",
                    "estimate": -0.25,
                    "standard_error": 0.1,
                }
            ]
        ),
        "covariance": pd.DataFrame(
            [
                {
                    "model": model,
                    "row_position": 1,
                    "column_position": 1,
                    "row_parameter": "y:intercept",
                    "column_parameter": "y:intercept",
                    "covariance": 0.01,
                }
            ]
        ),
        "fit": tables["fit"],
        "predictions": tables["predictions"],
    }


def test_stata_parameter_mappings_cover_sign_and_scale_transforms():
    comparator = _load_comparator()

    binary = _raw_rows(
        "binary_logit",
        [
            ("y_logit:intercept", -0.3, 0.1),
            ("y_logit:x1", 0.7, 0.1),
            ("y_logit:x2", -0.5, 0.1),
        ],
    )
    binary_mapping = {
        item.name: item for item in comparator._canonical_mapping("binary_logit", binary)
    }
    assert binary_mapping["intercept"].estimate == -0.3
    assert binary_mapping["x1"].derivative == 1.0

    ordered = _raw_rows(
        "ordered_logit",
        [
            ("y_ologit:ox1", 0.6, 0.1),
            ("y_ologit:ox2", -0.4, 0.1),
            ("/cut1:_cons", -0.7, 0.1),
            ("/cut2:_cons", 0.8, 0.1),
        ],
    )
    ordered_mapping = {
        item.name: item for item in comparator._canonical_mapping("ordered_logit", ordered)
    }
    assert ordered_mapping["threshold: 0 | 1"].estimate == -0.7
    assert ordered_mapping["threshold: 1 | 2"].estimate == 0.8

    generalized = _raw_rows(
        "generalized_ordered_logit",
        [
            ("split1:gx1", 0.8, 0.1),
            ("split1:gx2", -0.4, 0.1),
            ("split1:_cons", 0.9, 0.2),
            ("split2:gx1", 0.3, 0.1),
            ("split2:gx2", -0.4, 0.1),
            ("split2:_cons", -0.9, 0.2),
        ],
    )
    generalized_mapping = {
        item.name: item
        for item in comparator._canonical_mapping("generalized_ordered_logit", generalized)
    }
    assert generalized_mapping["threshold: 0 | 1"].estimate == -0.9
    assert generalized_mapping["threshold: 0 | 1"].derivative == -1.0
    assert generalized_mapping["slope 1 | 2: gx1"].estimate == 0.3

    partial = _raw_rows(
        "partial_proportional_odds",
        [
            ("split1:gx1", 0.8, 0.1),
            ("split1:gx2", -0.4, 0.1),
            ("split1:_cons", 0.9, 0.2),
            ("split2:gx1", 0.3, 0.1),
            ("split2:gx2", -0.4, 0.1),
            ("split2:_cons", -0.9, 0.2),
        ],
    )
    partial_mapping = {
        item.name: item
        for item in comparator._canonical_mapping("partial_proportional_odds", partial)
    }
    assert partial_mapping["common: gx2"].estimate == -0.4
    assert partial_mapping["varying 0 | 1: gx1"].estimate == 0.8
    assert partial_mapping["varying 1 | 2: gx1"].estimate == 0.3
    assert len([name for name in partial_mapping if name == "common: gx2"]) == 1

    random_effects = _raw_rows(
        "random_effects_ordered_logit",
        [
            ("y:x1", 0.8, 0.1),
            ("y:x2", -0.5, 0.1),
            ("/cut1:_cons", -0.8, 0.1),
            ("/cut2:_cons", 0.9, 0.1),
            ("lns1_1_1:_cons", np.log(0.7), 0.2),
        ],
    )
    random_mapping = {
        item.name: item
        for item in comparator._canonical_mapping("random_effects_ordered_logit", random_effects)
    }
    assert np.isclose(random_mapping["sigma_entity"].estimate, 0.7)
    assert np.isclose(random_mapping["sigma_entity"].standard_error, 0.14)
    assert random_mapping["threshold: 0 | 1"].estimate == -0.8

    dynamic = _raw_rows(
        "dynamic_random_effects_ordered_logit",
        [
            ("y:x1", 0.5, 0.1),
            ("y:state_1", 0.3, 0.1),
            ("y:state_2", 0.7, 0.1),
            ("y:initial_1", 0.2, 0.1),
            ("y:initial_2", 0.6, 0.1),
            ("y:initial_x1", 0.2, 0.1),
            ("y:mean_x1", 0.4, 0.1),
            ("/cut1:_cons", -0.8, 0.1),
            ("/cut2:_cons", 0.9, 0.1),
            ("lns1_1_1:_cons", np.log(0.55), 0.2),
        ],
    )
    dynamic_mapping = {
        item.name: item
        for item in comparator._canonical_mapping("dynamic_random_effects_ordered_logit", dynamic)
    }
    assert dynamic_mapping["state[1]"].estimate == 0.3
    assert dynamic_mapping["initial_x[x1]"].estimate == 0.2
    assert dynamic_mapping["mean[x1]"].estimate == 0.4


def test_stata_covariance_transform_uses_parameter_jacobian():
    comparator = _load_comparator()
    estimates = _raw_rows(
        "random_effects_ordered_logit",
        [
            ("y:x1", 0.8, 1.0),
            ("/cut1:_cons", -0.8, 1.0),
            ("/cut2:_cons", 0.9, 1.0),
            ("lns1_1_1:_cons", np.log(0.7), 1.0),
        ],
    )
    covariance = pd.DataFrame(
        [
            {
                "model": "random_effects_ordered_logit",
                "row_position": row,
                "column_position": column,
                "row_parameter": "unused",
                "column_parameter": "unused",
                "covariance": float(row == column),
            }
            for row in range(1, 5)
            for column in range(1, 5)
        ]
    )
    _, transformed = comparator._canonical_stata_results(
        "random_effects_ordered_logit", estimates, covariance
    )
    sigma_variance = transformed.loc[
        (transformed["row_parameter"] == "sigma_entity")
        & (transformed["column_parameter"] == "sigma_entity"),
        "covariance",
    ].iloc[0]
    assert np.isclose(sigma_variance, 0.7**2)


def test_comparison_row_fails_for_any_nonfinite_difference_or_missing_key():
    comparator = _load_comparator()

    for nonfinite in (np.nan, np.inf, -np.inf):
        row = comparator._comparison_row(
            model="binary_logit",
            statistic="estimate",
            differences=pd.Series([0.0, nonfinite]),
            tolerance=1e-8,
        )
        assert row["status"] == "FAIL"
        assert row["compared"] == 1
        assert row["nonfinite"] == 1
        assert row["max_abs_difference"] == 0.0
        assert row["detail"] == "non-finite differences: 1"

    missing_row = comparator._comparison_row(
        model="binary_logit",
        statistic="estimate",
        differences=pd.Series([0.0]),
        tolerance=1e-8,
        missing=["x2"],
    )
    assert missing_row["status"] == "FAIL"
    assert missing_row["nonfinite"] == 0
    assert missing_row["detail"] == "missing keys: x2"


def test_comparator_validates_required_columns_and_keys_at_ingestion():
    comparator = _load_comparator()

    python_tables = _minimal_comparison_tables()
    python_tables["estimates"] = python_tables["estimates"].drop(columns="standard_error")
    with pytest.raises(ValueError, match=r"missing required columns.*standard_error"):
        comparator._validate_schema_group(
            python_tables,
            comparator.PYTHON_REFERENCE_SCHEMAS,
            label="Python reference",
        )

    raw_tables = _minimal_raw_stata_tables()
    conflicting_fit = raw_tables["fit"].copy()
    conflicting_fit.loc[0, "loglike"] = -9.0
    raw_tables["fit"] = pd.concat([raw_tables["fit"], conflicting_fit], ignore_index=True)
    with pytest.raises(ValueError, match=r"Raw Stata export fit has duplicate keys.*model"):
        comparator._validate_schema_group(
            raw_tables,
            comparator.RAW_STATA_SCHEMAS,
            label="Raw Stata export",
        )

    canonical_tables = _minimal_comparison_tables()
    canonical_tables.pop("fit")
    canonical_tables.pop("predictions")
    canonical_tables["estimates"] = pd.concat(
        [canonical_tables["estimates"], canonical_tables["estimates"]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match=r"canonical.*duplicate keys.*parameter"):
        comparator._validate_schema_group(
            canonical_tables,
            comparator.CANONICAL_STATA_SCHEMAS,
            label="canonical",
        )


def test_comparator_rejects_null_or_duplicate_prediction_keys():
    comparator = _load_comparator()

    for bad_obs_ids in ([1, 1], [1, np.nan]):
        tables = _minimal_comparison_tables()
        tables["predictions"] = pd.DataFrame(
            {
                "model": ["binary_logit", "binary_logit"],
                "obs_id": bad_obs_ids,
                "category": [0, 0],
                "probability": [0.6, 0.6],
            }
        )
        with pytest.raises(ValueError, match=r"prediction.*(duplicate keys|null key values)"):
            comparator._validate_schema_group(
                tables,
                comparator.PYTHON_REFERENCE_SCHEMAS,
                label="Python reference",
            )


def test_model_comparison_uses_one_to_one_merges_and_left_prediction_subset():
    comparator = _load_comparator()
    python_tables = _minimal_comparison_tables()
    stata_tables = {name: frame.copy() for name, frame in python_tables.items()}
    stata_tables["predictions"] = pd.concat(
        [
            stata_tables["predictions"],
            pd.DataFrame(
                [
                    {
                        "model": "binary_logit",
                        "obs_id": 99,
                        "category": 0,
                        "probability": 0.2,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    report = comparator._compare_model(
        "binary_logit",
        python_tables["estimates"],
        python_tables["covariance"],
        python_tables["fit"],
        python_tables["predictions"],
        stata_tables["estimates"],
        stata_tables["covariance"],
        stata_tables["fit"],
        stata_tables["predictions"],
    )
    probability = next(row for row in report if row["statistic"] == "probability")
    assert probability["status"] == "PASS"
    assert probability["compared"] == 1

    for table_name in ("estimates", "covariance", "predictions"):
        duplicated_tables = {name: frame.copy() for name, frame in stata_tables.items()}
        duplicated_tables[table_name] = pd.concat(
            [duplicated_tables[table_name], duplicated_tables[table_name].iloc[[0]]],
            ignore_index=True,
        )
        with pytest.raises(pd.errors.MergeError, match="one-to-one"):
            comparator._compare_model(
                "binary_logit",
                python_tables["estimates"],
                python_tables["covariance"],
                python_tables["fit"],
                python_tables["predictions"],
                duplicated_tables["estimates"],
                duplicated_tables["covariance"],
                duplicated_tables["fit"],
                duplicated_tables["predictions"],
            )

    duplicate_fit = pd.concat(
        [stata_tables["fit"], stata_tables["fit"].assign(loglike=-9.0)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="exactly one fit row.*found 2"):
        comparator._compare_model(
            "binary_logit",
            python_tables["estimates"],
            python_tables["covariance"],
            python_tables["fit"],
            python_tables["predictions"],
            stata_tables["estimates"],
            stata_tables["covariance"],
            duplicate_fit,
            stata_tables["predictions"],
        )


def test_comparator_verifies_actual_panel_quadrature_for_each_model():
    comparator = _load_comparator()
    panel_models = (
        "random_effects_ordered_logit",
        "dynamic_random_effects_ordered_logit",
    )
    model_specs = {model: {"kind": "random_effects", "required": True} for model in panel_models}
    metadata = {f"{model}.intmethod": "ghermite" for model in panel_models} | {
        f"{model}.n_quad": "12" for model in panel_models
    }

    actual = comparator._verify_panel_quadrature(metadata, {"quadrature_points": 12}, model_specs)
    assert actual == {model: {"intmethod": "ghermite", "n_quad": 12} for model in panel_models}

    metadata["dynamic_random_effects_ordered_logit.n_quad"] = "7"
    with pytest.raises(RuntimeError, match=r"dynamic.*n_quad=7, expected 12"):
        comparator._verify_panel_quadrature(metadata, {"quadrature_points": 12}, model_specs)


def test_comparator_rejects_malformed_or_duplicate_metadata():
    comparator = _load_comparator()

    with pytest.raises(ValueError, match="Malformed Stata metadata"):
        comparator._parse_metadata("stata_version=17\nmalformed")
    with pytest.raises(ValueError, match="Duplicate Stata metadata key"):
        comparator._parse_metadata("stata_version=17\nstata_version=18")


def test_comparator_requires_completed_matching_stata_run_metadata():
    comparator = _load_comparator()
    manifest = {"suite": "controlled_synthetic_certification"}
    metadata = {
        "suite": "controlled_synthetic_certification",
        "run_completed": "1",
        "panel_prediction": "conditional_fixedonly",
        "stata_version": "17",
        "gologit2_installed": "0",
    }

    assert comparator._verify_run_metadata(metadata, manifest) == {
        **metadata,
    }

    for key, bad_value in (
        ("suite", "real_data_application"),
        ("run_completed", "0"),
        ("panel_prediction", "marginal"),
        ("stata_version", ""),
        ("gologit2_installed", "yes"),
    ):
        bad_metadata = {**metadata, key: bad_value}
        with pytest.raises(RuntimeError, match="completed matching run"):
            comparator._verify_run_metadata(bad_metadata, manifest)


def test_do_file_pins_required_stata_estimands_and_quadrature():
    do_file = (PROJECT_ROOT / "validation" / "stata" / "limiteddepkit_parity.do").read_text(
        encoding="utf-8"
    )
    assert "logit y_logit intercept x1 x2, noconstant vce(oim)" in do_file
    assert "probit y_probit intercept x1 x2, noconstant vce(oim)" in do_file
    assert "ologit y_ologit ox1 ox2, vce(oim)" in do_file
    assert "oprobit y_oprobit ox1 ox2, vce(oim)" in do_file
    assert "intmethod(ghermite) intpoints(12)" in do_file
    assert "conditional(fixedonly)" in do_file
    assert do_file.count("iterate(2000) tolerance(1e-10) ltolerance(1e-12) nrtolerance(1e-8)") == 2
    assert "random_effects_ordered_logit.intmethod=`static_re_intmethod'" in do_file
    assert "random_effects_ordered_logit.n_quad=`static_re_n_quad'" in do_file
    assert "dynamic_random_effects_ordered_logit.intmethod=`dynamic_re_intmethod'" in do_file
    assert "dynamic_random_effects_ordered_logit.n_quad=`dynamic_re_n_quad'" in do_file
    assert "ssc install gologit2" in do_file
    assert "double n_groups double n_params" in do_file


def test_real_data_do_file_uses_pinned_application_specs_without_installing():
    do_file = (PROJECT_ROOT / "validation" / "stata" / "limiteddepkit_real_data.do").read_text(
        encoding="utf-8"
    )
    assert "logit y intercept x1 x2 x3 x4, noconstant vce(oim)" in do_file
    assert "probit y intercept x1 x2 x3 x4, noconstant vce(oim)" in do_file
    assert "ologit y ox1 ox2 ox3 ox4, vce(oim)" in do_file
    assert "oprobit y ox1 ox2 ox3 ox4, vce(oim)" in do_file
    assert "gologit2 y gx1 gx2 gx3 gx4, npl(gx4)" in do_file
    assert "intmethod(ghermite) intpoints(20)" in do_file
    assert "conditional(fixedonly)" in do_file
    assert do_file.count("iterate(2000) tolerance(1e-10) ltolerance(1e-12) nrtolerance(1e-8)") == 2
    assert "random_effects_ordered_logit.intmethod=`static_re_intmethod'" in do_file
    assert "random_effects_ordered_logit.n_quad=`static_re_n_quad'" in do_file
    assert "dynamic_random_effects_ordered_logit.intmethod=`dynamic_re_intmethod'" in do_file
    assert "dynamic_random_effects_ordered_logit.n_quad=`dynamic_re_n_quad'" in do_file
    assert "ssc install gologit2" in do_file
    assert "ssc install gologit2," not in do_file
    assert "suite=real_data_application" in do_file


def test_comparator_accepts_manifest_supplied_feature_contracts():
    comparator = _load_comparator()
    real_specs = {
        "binary_logit": {
            "kind": "binary",
            "features": ["intercept", "x1", "x2", "x3", "x4"],
            "required": True,
        }
    }
    estimates = _raw_rows(
        "binary_logit",
        [
            ("y:intercept", 0.5, 0.1),
            ("y:x1", -0.1, 0.1),
            ("y:x2", -0.2, 0.1),
            ("y:x3", 0.7, 0.1),
            ("y:x4", 1.8, 0.2),
        ],
    )
    mapping = comparator._canonical_mapping("binary_logit", estimates, real_specs)
    assert [item.name for item in mapping] == [
        "intercept",
        "x1",
        "x2",
        "x3",
        "x4",
    ]


def test_real_data_downloader_pins_official_sources_and_hashes():
    downloader = (PROJECT_ROOT / "validation" / "stata" / "download_real_data.ps1").read_text(
        encoding="utf-8"
    )
    assert "https://www.stata-press.com/data/r19/lbw.dta" in downloader
    assert "https://www.stata-press.com/data/r19/tvsfpors.dta" in downloader
    assert "https://www.stata-press.com/data/r19/nlswork.dta" in downloader
    assert "00204ef3586836e56e49598cd9850148aea9058090a607e5bf20e12a6b0a58ee" in downloader
    assert "50197a3e7b15809ed816b2846ca9dc1a4bc6aecac06ba75f4ae0312d7ceebfc8" in downloader
    assert "b77bc182ac586205d769ad847e5e7cb0063c31be2c4bbef5f1ad16b74118c86f" in downloader
    assert "Get-FileHash" in downloader


def test_real_data_preparation_pins_samples_and_comparison_contract():
    preparation = (PROJECT_ROOT / "validation" / "stata" / "prepare_real_data.py").read_text(
        encoding="utf-8"
    )
    assert '"suite": "real_data_application"' in preparation
    assert "EXPECTED_DYNAMIC_GROUPS = 335" in preparation
    assert "EXPECTED_DYNAMIC_RAW_NOBS = 2_010" in preparation
    assert "EXPECTED_DYNAMIC_ESTIMATION_NOBS = 1_675" in preparation
    assert '"comparison_model_specs": comparison_model_specs' in preparation
    assert 'PartialProportionalOdds(varying=["gx4"])' in preparation
    assert "quadrature_points=QUADRATURE_POINTS" in preparation
    assert "ssl.create_default_context()" in preparation
    assert "Do not commit or redistribute" in preparation
