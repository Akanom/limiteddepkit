"""Acceptance-gate tests for controlled Stata parity references."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_PATH = PROJECT_ROOT / "validation" / "stata" / "prepare_parity.py"


def _load_preparation():
    spec = importlib.util.spec_from_file_location("limiteddepkit_stata_prepare", PREPARATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_result() -> SimpleNamespace:
    parameters = pd.Series([0.25, -0.5], index=["x1", "x2"])
    return SimpleNamespace(
        converged=True,
        inference_valid=True,
        nobs=12,
        n_groups=3,
        all_params=parameters,
        covariance=pd.DataFrame(np.eye(2), index=parameters.index, columns=parameters.index),
        loglike=-8.5,
        constraint_slack=0.2,
    )


def test_assert_result_accepts_only_a_complete_valid_fit():
    preparation = _load_preparation()

    preparation._assert_result(
        "valid_model",
        _valid_result(),
        expected_nobs=12,
        expected_groups=3,
        require_interior=True,
    )


@pytest.mark.parametrize(
    ("attribute", "bad_value", "message"),
    [
        ("converged", False, "did not converge"),
        ("inference_valid", False, "did not produce valid inference"),
        ("loglike", np.nan, "nonfinite log likelihood"),
    ],
)
def test_assert_result_rejects_invalid_fit_state(attribute, bad_value, message):
    preparation = _load_preparation()
    result = _valid_result()
    setattr(result, attribute, bad_value)

    with pytest.raises(RuntimeError, match=message):
        preparation._assert_result("invalid_model", result, expected_nobs=12)


def test_assert_result_rejects_wrong_observation_and_group_counts():
    preparation = _load_preparation()
    result = _valid_result()

    with pytest.raises(RuntimeError, match=r"nobs != 13"):
        preparation._assert_result("invalid_model", result, expected_nobs=13)
    with pytest.raises(RuntimeError, match=r"n_groups != 4"):
        preparation._assert_result("invalid_model", result, expected_nobs=12, expected_groups=4)


def test_assert_result_rejects_nonfinite_parameters_and_covariance():
    preparation = _load_preparation()

    nonfinite_parameters = _valid_result()
    nonfinite_parameters.all_params.iloc[1] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite parameter estimates"):
        preparation._assert_result("invalid_model", nonfinite_parameters, expected_nobs=12)

    nonfinite_covariance = _valid_result()
    nonfinite_covariance.covariance.iloc[0, 1] = np.inf
    with pytest.raises(RuntimeError, match="nonfinite covariance entries"):
        preparation._assert_result("invalid_model", nonfinite_covariance, expected_nobs=12)


@pytest.mark.parametrize("slack", [0.0, -0.1, np.nan, np.inf])
def test_assert_result_requires_finite_strictly_positive_constraint_slack(slack):
    preparation = _load_preparation()
    result = _valid_result()
    result.constraint_slack = slack

    with pytest.raises(RuntimeError, match="constraints are not strictly interior"):
        preparation._assert_result(
            "flexible_model",
            result,
            expected_nobs=12,
            require_interior=True,
        )


def test_every_synthetic_fit_is_followed_immediately_by_an_acceptance_gate():
    tree = ast.parse(PREPARATION_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    expected = {
        "binary_logit": "binary_logit",
        "binary_probit": "binary_probit",
        "ordered_logit": "ordered_logit",
        "ordered_probit": "ordered_probit",
        "generalized": "generalized_ordered_logit",
        "partial": "partial_proportional_odds",
        "static_result": "random_effects_ordered_logit",
        "dynamic_result": "dynamic_random_effects_ordered_logit",
    }
    gated: dict[str, str] = {}

    for index, statement in enumerate(main.body[:-1]):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "fit"
        ):
            continue

        target = statement.targets[0].id
        next_statement = main.body[index + 1]
        assert isinstance(next_statement, ast.Expr), target
        assert isinstance(next_statement.value, ast.Call), target
        assert isinstance(next_statement.value.func, ast.Name), target
        assert next_statement.value.func.id == "_assert_result", target
        gated[target] = ast.literal_eval(next_statement.value.args[0])

    assert gated == expected
