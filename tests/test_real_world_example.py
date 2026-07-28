from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "real_world_count_workflow.py"
    spec = importlib.util.spec_from_file_location("real_world_count_workflow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_world_example_prepares_the_registered_rod93_contract() -> None:
    module = _example_module()
    raw = pd.DataFrame(
        {
            "deaths": [2, 4, 3],
            "exposure": [100.0, 120.0, 95.0],
            "age_mos": [1.0, 6.0, 12.0],
            "cohort": [1, 2, 3],
        }
    )

    prepared = module._analysis_frame(raw)

    assert prepared.columns.tolist() == [
        "deaths",
        "exposure",
        "intercept",
        "log_age",
        "cohort_2",
        "cohort_3",
    ]
    np.testing.assert_array_equal(prepared["log_age"], np.log([1.0, 6.0, 12.0]))
    np.testing.assert_array_equal(prepared["cohort_2"], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(prepared["cohort_3"], [0.0, 0.0, 1.0])


def test_real_world_example_rejects_invalid_source_schema_and_exposure() -> None:
    module = _example_module()
    with pytest.raises(ValueError, match="missing required"):
        module._analysis_frame(pd.DataFrame({"deaths": [1]}))

    invalid = pd.DataFrame({"deaths": [1], "exposure": [0.0], "age_mos": [1.0], "cohort": [1]})
    with pytest.raises(ValueError, match="non-positive exposure"):
        module._analysis_frame(invalid)
