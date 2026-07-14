import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from limiteddepkit import (
    BinaryLogit,
    BinaryProbit,
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


@pytest.fixture(params=[("logit", BinaryLogit), ("probit", BinaryProbit)])
def stable_binary_result(request):
    link, estimator = request.param
    rng = np.random.default_rng(730 if link == "logit" else 731)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=500)})
    index = X.to_numpy() @ np.array([-0.25, 0.7])
    probability = expit(index) if link == "logit" else norm.cdf(index)
    y = rng.binomial(1, probability)
    return X, estimator().fit(X, y)


def test_stable_binary_results_support_package_postestimation(stable_binary_result):
    X, result = stable_binary_result

    pd.testing.assert_frame_equal(vcov(result), result.vcov())
    pd.testing.assert_frame_equal(confint(result), result.conf_int())
    pd.testing.assert_frame_equal(summary_frame(result), result.summary_frame())
    pd.testing.assert_frame_equal(predict_proba(result, X), result.predict_proba(X))
    pd.testing.assert_series_equal(predict(result, X), result.predict(X))
    pd.testing.assert_frame_equal(
        marginal_effects(result, X), result.marginal_effects(X)
    )

    probability_margin = margins(result, X, at="overall", kind="probability")
    assert probability_margin.sum() == pytest.approx(1.0)
    pd.testing.assert_series_equal(
        probability_margin,
        result.margins(X, at="overall", kind="probability"),
    )
    effect_margin = margins(result, X, at="mean", kind="marginal_effect")
    assert effect_margin.index.tolist() == ["x"]


def test_stable_binary_inference_helpers_and_ame_delta_method(stable_binary_result):
    X, result = stable_binary_result

    ame = result.average_marginal_effects(X)
    inference = result.average_marginal_effects_inference(X)
    assert inference.index.tolist() == ["x"]
    assert inference.loc["x", "estimate"] == pytest.approx(ame["x"])
    assert np.isfinite(inference.to_numpy()).all()
    assert inference.loc["x", "lower"] < ame["x"] < inference.loc["x", "upper"]

    method_lincom = result.lincom({"x": 1.0})
    package_lincom = lincom(result, {"x": 1.0})
    pd.testing.assert_series_equal(method_lincom, package_lincom)
    method_wald = result.wald_test({"x": 1.0})
    package_wald = wald_test(result, {"x": 1.0})
    pd.testing.assert_series_equal(method_wald, package_wald)
