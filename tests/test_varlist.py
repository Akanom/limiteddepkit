import numpy as np
import pandas as pd
import pytest

import limiteddepkit as ldk


@pytest.fixture
def table():
    return pd.DataFrame(
        columns=[
            "entity_id",
            "outcome",
            "age",
            "income_low",
            "income_high",
            "controls_a",
            "controls_b",
            "control_long",
        ]
    )


def test_varlist_expands_in_token_and_dataframe_order_without_duplicates(table):
    columns = ldk.varlist(
        table,
        ["age", "income_*", "controls_?", "income_low"],
    )

    assert columns == [
        "age",
        "income_low",
        "income_high",
        "controls_a",
        "controls_b",
    ]


def test_varlist_splits_a_stata_style_string_and_applies_strict_exclusions(table):
    columns = ldk.varlist(
        table,
        "*",
        exclude="entity_id outcome control_long",
    )

    assert columns == [
        "age",
        "income_low",
        "income_high",
        "controls_a",
        "controls_b",
    ]


def test_varlist_question_mark_matches_exactly_one_character(table):
    assert ldk.varlist(table, "controls_?") == ["controls_a", "controls_b"]


def test_varlist_sequence_supports_an_exact_column_name_containing_spaces():
    table = pd.DataFrame(columns=["outcome", "household income"])

    assert ldk.varlist(table, ["household income"]) == ["household income"]


def test_varlist_treats_regex_metacharacters_as_literal_column_characters():
    table = pd.DataFrame(columns=["x[1]", "x11", "x.1"])

    assert ldk.varlist(table, ["x[1]"]) == ["x[1]"]
    assert ldk.varlist(table, "x[*]") == ["x[1]"]
    assert ldk.varlist(table, "x?1") == ["x11", "x.1"]


def test_varlist_wildcards_also_match_columns_named_with_wildcard_characters():
    table = pd.DataFrame(columns=["*", "?", "plain"])

    assert ldk.varlist(table, "*") == ["*", "?", "plain"]
    assert ldk.varlist(table, "?") == ["*", "?"]


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ("missing_*", "matched no columns"),
        ("", "at least one variable"),
        ([], "at least one variable"),
    ],
)
def test_varlist_rejects_empty_or_unmatched_includes(table, variables, message):
    with pytest.raises(ValueError, match=message):
        ldk.varlist(table, variables)


def test_varlist_rejects_unmatched_exclusions(table):
    with pytest.raises(ValueError, match="exclude pattern 'missing' matched no columns"):
        ldk.varlist(table, "income_*", exclude="missing")


def test_varlist_rejects_selection_fully_removed_by_exclusions(table):
    with pytest.raises(ValueError, match="No columns remain"):
        ldk.varlist(table, "income_*", exclude="income_*")


def test_varlist_rejects_ambiguous_or_non_string_columns():
    duplicated = pd.DataFrame(np.empty((0, 2)), columns=["x", "x"])
    with pytest.raises(ValueError, match="duplicates"):
        ldk.varlist(duplicated, "x")

    non_string = pd.DataFrame(columns=["x", 1])
    with pytest.raises(TypeError, match="non-string columns"):
        ldk.varlist(non_string, "x")


@pytest.mark.parametrize("variables", [{"x"}, b"x", ["x", 1]])
def test_varlist_requires_an_ordered_string_contract(table, variables):
    with pytest.raises(TypeError):
        ldk.varlist(table, variables)


def test_varlist_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ldk.varlist(["x1", "x2"], "x*")


def test_varlist_output_integrates_with_existing_fit_and_prediction_contracts():
    rng = np.random.default_rng(604)
    table = pd.DataFrame(
        {
            "entity_id": np.arange(300),
            "outcome": np.zeros(300, dtype=int),
            "const": np.ones(300),
            "income_low": rng.normal(size=300),
            "income_high": rng.normal(size=300),
            "controls_a": rng.normal(size=300),
        }
    )
    features = ldk.varlist(
        table,
        "const income_* controls_?",
        exclude=None,
    )
    linear_index = table[features].to_numpy() @ np.array([-0.2, 0.6, -0.3, 0.25])
    probabilities = 1.0 / (1.0 + np.exp(-linear_index))
    table["outcome"] = rng.binomial(1, probabilities)

    result = ldk.BinaryLogit().fit(table[features], table["outcome"])
    predictions = result.predict_proba(table.loc[:9, features])

    assert list(result.feature_names) == features
    assert predictions.shape == (10, 2)
    assert np.isfinite(predictions.to_numpy()).all()


def test_varlist_is_a_stable_root_export():
    assert "varlist" in ldk.__all__
    assert callable(ldk.varlist)
