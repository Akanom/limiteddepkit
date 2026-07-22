import pickle

import numpy as np
import pandas as pd
import pytest

import limiteddepkit as ldk


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "income_2": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
            "income_1": [1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
            "education": [
                "secondary",
                "college",
                "graduate",
                "secondary",
                "college",
                "graduate",
            ],
            "region": ["north", "south", "north", "south", "north", "south"],
        },
        index=pd.Index([10, 11, 12, 13, 14, 15], name="observation"),
    )


def test_compiler_builds_continuous_and_treatment_coded_categorical_columns(table):
    compiler = ldk.FactorVariableCompiler(
        "c.age i.education",
        category_orders={"education": ["secondary", "college", "graduate"]},
    )

    design = compiler.fit_transform(table)

    assert list(design) == ["age", "college.education", "graduate.education"]
    assert design.index.equals(table.index)
    assert design["age"].tolist() == table["age"].tolist()
    assert design["college.education"].tolist() == [0, 1, 0, 0, 1, 0]
    assert design["graduate.education"].tolist() == [0, 0, 1, 0, 0, 1]
    assert compiler.input_columns_ == ("age", "education")
    assert compiler.category_levels_["education"] == (
        "secondary",
        "college",
        "graduate",
    )
    assert compiler.base_categories_["education"] == "secondary"


def test_compiler_honors_an_explicit_base_category(table):
    compiler = ldk.FactorVariableCompiler(
        "i.education",
        category_orders={"education": ["secondary", "college", "graduate"]},
        base_categories={"education": "college"},
    )

    design = compiler.fit_transform(table)

    assert list(design) == ["secondary.education", "graduate.education"]
    assert compiler.base_categories_ == {"education": "college"}


def test_compiler_uses_declared_pandas_categorical_order():
    data = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["middle", "high", "low", "middle", "high", "low"],
                categories=["low", "middle", "high"],
                ordered=True,
            )
        }
    )

    compiler = ldk.FactorVariableCompiler("i.group")
    design = compiler.fit_transform(data)

    assert compiler.category_levels_["group"] == ("low", "middle", "high")
    assert list(design) == ["middle.group", "high.group"]


def test_hash_emits_only_the_interaction_and_double_hash_adds_main_effects(table):
    interaction_only = ldk.FactorVariableCompiler(
        "i.region#c.age",
        category_orders={"region": ["north", "south"]},
    ).fit_transform(table)
    factorial = ldk.FactorVariableCompiler(
        "i.region##c.age",
        category_orders={"region": ["north", "south"]},
    ).fit_transform(table)

    assert list(interaction_only) == ["south.region#c.age"]
    assert list(factorial) == ["south.region", "age", "south.region#c.age"]
    assert interaction_only.iloc[:, 0].tolist() == [0, 30, 0, 50, 0, 70]


def test_double_hash_supports_a_continuous_quadratic(table):
    design = ldk.FactorVariableCompiler("c.age##c.age").fit_transform(table)

    assert list(design) == ["age", "c.age#c.age"]
    np.testing.assert_allclose(design["c.age#c.age"], table["age"] ** 2)


def test_multiway_double_hash_emits_all_lower_order_terms(table):
    design = ldk.FactorVariableCompiler("c.age##c.income_1##c.income_2").fit_transform(table)

    assert list(design) == [
        "age",
        "income_1",
        "income_2",
        "c.age#c.income_1",
        "c.age#c.income_2",
        "c.income_1#c.income_2",
        "c.age#c.income_1#c.income_2",
    ]


def test_reversed_or_repeated_terms_are_deduplicated_by_semantics(table):
    design = ldk.FactorVariableCompiler(
        "c.age#c.income_1 c.income_1#c.age c.age#c.income_1"
    ).fit_transform(table)

    assert list(design) == ["c.age#c.income_1"]


def test_compiler_expands_varlist_wildcards_in_dataframe_order(table):
    compiler = ldk.FactorVariableCompiler("c.income_*#c.age")
    design = compiler.fit_transform(table)

    assert compiler.input_columns_ == ("income_2", "income_1", "age")
    assert list(design) == ["c.income_2#c.age", "c.income_1#c.age"]


def test_ordered_term_entries_support_exact_source_names_with_spaces():
    data = pd.DataFrame({"household income": [1.0, 2.0, 3.0]})

    design = ldk.FactorVariableCompiler(["c.household income"]).fit_transform(data)

    assert list(design) == ["household income"]


def test_transform_reuses_schema_allows_extra_columns_and_preserves_index(table):
    compiler = ldk.FactorVariableCompiler(
        "i.education##c.age",
        category_orders={"education": ["secondary", "college", "graduate"]},
    )
    compiler.fit(table)
    new_data = pd.DataFrame(
        {
            "age": [81.0, 82.0],
            "education": ["graduate", "secondary"],
            "unused": [1, 2],
        },
        index=[101, 103],
    )

    transformed = compiler.transform(new_data)

    assert transformed.index.tolist() == [101, 103]
    assert tuple(transformed) == compiler.feature_names_
    assert transformed.loc[101, "graduate.education#c.age"] == 81.0
    assert transformed.loc[103, "graduate.education#c.age"] == 0.0


def test_transform_rejects_unknown_categories_and_missing_source_columns(table):
    compiler = ldk.FactorVariableCompiler("i.region").fit(table)

    with pytest.raises(ValueError, match="unknown levels.*east"):
        compiler.transform(pd.DataFrame({"region": ["east"]}))
    with pytest.raises(ValueError, match="matched no columns"):
        compiler.transform(pd.DataFrame({"other": [1]}))


def test_compiler_does_not_add_a_constant_unless_requested(table):
    without = ldk.FactorVariableCompiler("c.age").fit_transform(table)
    with_constant = ldk.FactorVariableCompiler("c.age", add_constant=True).fit_transform(table)

    assert list(without) == ["age"]
    assert list(with_constant) == ["const", "age"]
    assert with_constant["const"].eq(1.0).all()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["1", "2", "3"], "numeric, non-boolean"),
        ([True, False, True], "numeric, non-boolean"),
        ([1.0, np.nan, 3.0], "missing values"),
        ([1.0, np.inf, 3.0], "finite values"),
        ([1.0 + 1.0j, 2.0 + 0.0j, 3.0 + 0.0j], "real-valued"),
    ],
)
def test_continuous_components_reject_invalid_values(values, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ldk.FactorVariableCompiler("c.x").fit(pd.DataFrame({"x": values}))


def test_categorical_components_reject_missing_single_or_unused_levels():
    with pytest.raises(ValueError, match="missing values"):
        ldk.FactorVariableCompiler("i.group").fit(pd.DataFrame({"group": ["a", None, "b"]}))
    with pytest.raises(ValueError, match="at least two levels"):
        ldk.FactorVariableCompiler("i.group").fit(pd.DataFrame({"group": ["a", "a", "a"]}))

    categorical = pd.DataFrame(
        {"group": pd.Categorical(["a", "b"], categories=["a", "b", "unused"])}
    )
    with pytest.raises(ValueError, match="unobserved=.*unused"):
        ldk.FactorVariableCompiler("i.group").fit(categorical)


def test_category_order_and_base_configuration_are_strict(table):
    with pytest.raises(ValueError, match="unlisted=.*graduate"):
        ldk.FactorVariableCompiler(
            "i.education",
            category_orders={"education": ["secondary", "college"]},
        ).fit(table)
    with pytest.raises(ValueError, match="Base category.*doctoral"):
        ldk.FactorVariableCompiler("i.education", base_categories={"education": "doctoral"}).fit(
            table
        )
    with pytest.raises(ValueError, match="not used as categorical"):
        ldk.FactorVariableCompiler(
            "c.age", category_orders={"education": ["secondary", "college"]}
        ).fit(table)


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ("c.age#", "empty component"),
        ("c.age###c.income_1", "mixes # and ##"),
        ("i.", "no variable name"),
        ("ib2.education", "unsupported inline"),
        ("i.education##i.education", "cannot interact with itself"),
    ],
)
def test_malformed_or_unsupported_factor_syntax_fails_strictly(table, variables, message):
    with pytest.raises(ValueError, match=message):
        ldk.FactorVariableCompiler(variables).fit(table)


@pytest.mark.parametrize("variables", [{"c.age"}, b"c.age", ["c.age", 1]])
def test_factor_terms_require_an_ordered_string_contract(variables):
    with pytest.raises(TypeError):
        ldk.FactorVariableCompiler(variables)


def test_a_source_cannot_be_declared_continuous_and_categorical(table):
    with pytest.raises(ValueError, match="both continuous and categorical"):
        ldk.FactorVariableCompiler("c.age i.age").fit(table)


def test_feature_name_collisions_are_rejected():
    data = pd.DataFrame(
        {
            "2.group": [1.0, 2.0, 3.0, 4.0],
            "group": [1, 2, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="Compiled feature name.*ambiguous"):
        ldk.FactorVariableCompiler("c.2.group i.group").fit(data)


def test_max_columns_caps_categorical_interaction_expansion():
    data = pd.DataFrame(
        {
            "first": ["a", "b", "c"] * 3,
            "second": ["x", "y", "z"] * 3,
        }
    )

    with pytest.raises(ValueError, match="exceeds max_columns=7"):
        ldk.FactorVariableCompiler("i.first##i.second", max_columns=7).fit(data)


def test_max_columns_caps_wildcard_source_expansion_before_design_construction():
    data = pd.DataFrame({f"x{index}": [1.0, 2.0] for index in range(5)})

    with pytest.raises(ValueError, match="source expansion exceeds max_columns=4"):
        ldk.FactorVariableCompiler("c.x*#c.x*", max_columns=4).fit(data)


def test_compiled_interactions_reject_numeric_overflow():
    data = pd.DataFrame({"x": [1e308, 1e308]})

    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="Compiled feature.*non-finite"),
    ):
        ldk.FactorVariableCompiler("c.x##c.x").fit(data)


def test_fitted_compiler_is_pickle_serializable(table):
    compiler = ldk.FactorVariableCompiler(
        "i.education##c.age",
        category_orders={"education": ["secondary", "college", "graduate"]},
    ).fit(table)

    restored = pickle.loads(pickle.dumps(compiler))

    pd.testing.assert_frame_equal(restored.transform(table), compiler.transform(table))
    assert restored.feature_names_ == compiler.feature_names_
    assert restored.category_levels_ == compiler.category_levels_


def test_public_fitted_mappings_cannot_mutate_the_prediction_schema(table):
    compiler = ldk.FactorVariableCompiler(
        "i.education",
        category_orders={"education": ["secondary", "college", "graduate"]},
    ).fit(table)
    exposed_levels = compiler.category_levels_
    exposed_bases = compiler.base_categories_
    exposed_levels["education"] = ("corrupted", "levels")
    exposed_bases["education"] = "corrupted"

    transformed = compiler.transform(table)

    assert compiler.category_levels_["education"] == (
        "secondary",
        "college",
        "graduate",
    )
    assert compiler.base_categories_["education"] == "secondary"
    assert list(transformed) == ["college.education", "graduate.education"]


def test_fitted_design_integrates_with_binary_fit_and_prediction():
    rng = np.random.default_rng(726)
    data = pd.DataFrame(
        {
            "age": rng.normal(size=500),
            "education": np.resize(["secondary", "college", "graduate"], 500),
        }
    )
    compiler = ldk.FactorVariableCompiler(
        "c.age i.education i.education#c.age",
        category_orders={"education": ["secondary", "college", "graduate"]},
        add_constant=True,
    )
    design = compiler.fit_transform(data)
    coefficients = np.array([-0.3, 0.7, -0.2, 0.4, 0.25, -0.15])
    probabilities = 1.0 / (1.0 + np.exp(-(design.to_numpy() @ coefficients)))
    outcome = rng.binomial(1, probabilities)

    result = ldk.BinaryLogit().fit(design, outcome)
    new_design = compiler.transform(data.iloc[:12])
    predictions = result.predict_proba(new_design)

    assert result.feature_names == compiler.feature_names_
    assert predictions.shape == (12, 2)
    assert np.isfinite(predictions.to_numpy()).all()


def test_fitted_metadata_and_root_export_are_available(table):
    compiler = ldk.FactorVariableCompiler("age")
    with pytest.raises(RuntimeError, match="not fitted"):
        compiler.transform(table)
    with pytest.raises(RuntimeError, match="not fitted"):
        compiler.get_feature_names_out()

    compiler.fit(table)
    assert compiler.get_feature_names_out().tolist() == ["age"]
    assert "FactorVariableCompiler" in ldk.__all__
    assert ldk.__version__ == "0.1.0a2"
