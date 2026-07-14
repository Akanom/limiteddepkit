import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import (
    GeneralizedOrderedLogit,
    OrderedLogit,
    OrderedProbit,
    PartialProportionalOdds,
    confint,
    lincom,
    marginal_effects,
    margins,
    predict,
    predict_proba,
    summary_frame,
    vcov,
    wald_test,
)


@pytest.fixture(scope="module")
def ecosystem_results():
    rng = np.random.default_rng(734)
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, 500), "x2": rng.uniform(-1, 1, 500)})
    eta = X.to_numpy() @ np.array([0.7, -0.4])
    cumulative = expit(np.array([-0.8, 0.9])[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, [
        OrderedLogit().fit(X, y),
        OrderedProbit().fit(X, y),
        PartialProportionalOdds(varying=["x1"]).fit(X, y),
        GeneralizedOrderedLogit().fit(X, y),
    ]


def test_result_objects_follow_shared_inference_contract(ecosystem_results):
    _, results = ecosystem_results
    for result in results:
        table = result.summary_frame()
        assert list(table.columns) == ["coef", "std_err", "z", "p_value"]
        assert list(table.index) == list(result.all_params.index)
        assert result.vcov().equals(result.covariance)
        assert result.vcov() is not result.covariance
        assert confint(result).equals(result.conf_int())
        assert summary_frame(result).equals(table)
        assert vcov(result).equals(result.covariance)


def test_package_level_prediction_and_effect_wrappers(ecosystem_results):
    X, results = ecosystem_results
    for result in results:
        sample = X.iloc[:10]
        assert predict(result, sample).equals(result.predict(sample))
        assert predict_proba(result, sample).equals(result.predict_proba(sample))
        assert marginal_effects(result, sample).equals(result.marginal_effects(sample))
        assert margins(result, X).equals(result.margins(X))


def test_generic_linear_hypotheses_cover_all_ordinal_results(ecosystem_results):
    _, results = ecosystem_results
    for result in results:
        first_parameter = result.all_params.index[0]
        combination = lincom(result, {first_parameter: 1.0})
        test = wald_test(result, {first_parameter: 1.0})
        assert combination["estimate"] == pytest.approx(result.all_params.iloc[0])
        assert 0 <= combination["p_value"] <= 1
        assert test["df"] == 1
        assert 0 <= test["p_value"] <= 1
