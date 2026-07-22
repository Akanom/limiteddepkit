import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import OrderedLogit, OrderedProbit


def make_ordinal_data(seed=4102, nobs=1_500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)})
    beta = np.array([0.9, -0.6])
    thresholds = np.array([-0.8, 0.7])
    cumulative = expit(thresholds[None, :] - X.to_numpy() @ beta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], 1 - cumulative[:, 1]]
    )
    draws = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, draws


def test_ordered_logit_fits_and_recovers_direction():
    X, y = make_ordinal_data()
    result = OrderedLogit().fit(X, y)

    assert result.converged
    assert result.nobs == len(X)
    assert result.params["x1"] > 0
    assert result.params["x2"] < 0
    assert np.all(np.diff(result.thresholds.to_numpy()) > 0)


def test_predicted_probabilities_are_valid():
    X, y = make_ordinal_data()
    result = OrderedLogit().fit(X, y)
    probabilities = result.predict_proba(X.iloc[:20])

    assert probabilities.shape == (20, 3)
    assert np.all(probabilities.to_numpy() >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(result.predict(X.iloc[:20]).unique()).issubset(set(result.categories))


def test_ordered_logit_rejects_binary_outcome():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least three"):
        OrderedLogit().fit(X, np.array([0, 0, 1, 1]))


def test_prediction_checks_dataframe_schema():
    X, y = make_ordinal_data(nobs=300)
    result = OrderedLogit().fit(X, y)

    with pytest.raises(ValueError, match="columns must match"):
        result.predict_proba(X[["x2", "x1"]])


def test_ordered_model_rejects_unidentified_constant_regressor():
    X, y = make_ordinal_data(nobs=300)
    with pytest.raises(ValueError, match="constant regressors"):
        OrderedLogit().fit(X.assign(constant=1.0), y)


def test_inference_is_aligned_and_finite():
    X, y = make_ordinal_data(nobs=800)
    result = OrderedLogit().fit(X, y)

    assert list(result.covariance.index) == list(result.all_params.index)
    assert list(result.covariance.columns) == list(result.all_params.index)
    assert np.allclose(result.covariance, result.covariance.T)
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(result.standard_errors > 0)
    assert np.all((result.pvalues >= 0) & (result.pvalues <= 1))


def test_confidence_intervals_contain_estimates():
    X, y = make_ordinal_data(nobs=800)
    result = OrderedLogit().fit(X, y)
    intervals = result.conf_int()

    assert np.all(intervals["lower"] < result.all_params)
    assert np.all(result.all_params < intervals["upper"])

    with pytest.raises(ValueError, match="strictly between"):
        result.conf_int(level=1.0)


def test_ordered_probit_uses_shared_result_contract():
    X, y = make_ordinal_data(nobs=800)
    result = OrderedProbit().fit(X, y)
    probabilities = result.predict_proba(X.iloc[:10])

    assert result.converged
    assert result.link == "probit"
    assert np.all(np.diff(result.thresholds) > 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("estimator", [OrderedLogit, OrderedProbit])
def test_loose_tolerance_cannot_certify_ordered_starting_values(estimator):
    X, y = make_ordinal_data(nobs=500)
    result = estimator().fit(X, y, tolerance=1e6)

    assert result.converged
    assert result.inference_valid
    assert result.scaled_score_norm <= 1e-4
    assert result.optimizer_result.nit > 0


@pytest.mark.parametrize("estimator", [OrderedLogit, OrderedProbit])
def test_marginal_effects_respect_probability_identity(estimator):
    X, y = make_ordinal_data(nobs=800)
    result = estimator().fit(X, y)
    effects = result.marginal_effects(X.iloc[:25])

    assert effects.shape == (25, len(result.categories) * X.shape[1])
    summed_over_categories = effects.T.groupby(level="feature").sum().T
    assert np.allclose(summed_over_categories, 0.0, atol=1e-12)


@pytest.mark.parametrize("estimator", [OrderedLogit, OrderedProbit])
def test_average_marginal_effects_equal_observation_mean(estimator):
    X, y = make_ordinal_data(nobs=800)
    result = estimator().fit(X, y)
    effects = result.marginal_effects(X)
    average = result.average_marginal_effects(X)

    expected = effects.mean(axis=0).unstack("feature")
    assert average.to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-12)
    assert list(average.index) == list(result.categories)
    assert list(average.columns) == list(X.columns)


@pytest.mark.parametrize("estimator", [OrderedLogit, OrderedProbit])
def test_analytical_marginal_effects_match_finite_differences(estimator):
    X, y = make_ordinal_data(nobs=800)
    result = estimator().fit(X, y)
    point = X.iloc[[0]].copy()
    analytical = result.marginal_effects(point)
    step = 1e-6

    for feature in X.columns:
        upper = point.copy()
        lower = point.copy()
        upper[feature] += step
        lower[feature] -= step
        numerical = (
            result.predict_proba(upper).to_numpy()
            - result.predict_proba(lower).to_numpy()
        ) / (2.0 * step)
        assert analytical.xs(feature, axis=1, level="feature").to_numpy() == pytest.approx(
            numerical, abs=1e-7
        )


@pytest.mark.parametrize("estimator", [OrderedLogit, OrderedProbit])
def test_average_marginal_effects_inference(estimator):
    X, y = make_ordinal_data(nobs=800)
    result = estimator().fit(X, y)
    inference = result.average_marginal_effects_inference(X)
    average_effects = result.average_marginal_effects(X)
    try:
        average = average_effects.stack(future_stack=True)
    except TypeError:  # pandas < 2.1 does not expose future_stack
        average = average_effects.stack()

    assert inference.index.names == ["category", "feature"]
    assert inference["estimate"].to_numpy() == pytest.approx(
        average.to_numpy(), abs=1e-12
    )
    assert np.all(np.isfinite(inference["standard_error"]))
    assert np.all(inference["standard_error"] > 0)
    assert np.all((inference["p_value"] >= 0) & (inference["p_value"] <= 1))
    assert np.all(inference["lower"] < inference["estimate"])
    assert np.all(inference["estimate"] < inference["upper"])

    with pytest.raises(ValueError, match="strictly between"):
        result.average_marginal_effects_inference(X, level=0.0)


@pytest.mark.parametrize("estimator", [OrderedLogit, OrderedProbit])
def test_margins_overall_and_at_mean(estimator):
    X, y = make_ordinal_data(nobs=800)
    result = estimator().fit(X, y)

    overall_probabilities = result.margins(X, kind="probability")
    mean_probabilities = result.margins(X, at="mean", kind="probability")
    overall_effects = result.margins(X, kind="marginal_effect")

    assert overall_probabilities.to_numpy() == pytest.approx(
        result.predict_proba(X).mean(axis=0).to_numpy(), abs=1e-12
    )
    assert mean_probabilities.sum() == pytest.approx(1.0)
    assert overall_effects.to_numpy() == pytest.approx(
        result.average_marginal_effects(X).to_numpy(), abs=1e-12
    )


def test_margins_at_user_specified_values_and_validation():
    X, y = make_ordinal_data(nobs=800)
    result = OrderedLogit().fit(X, y)
    custom = result.margins(X, at={"x1": 1.5}, kind="probability")
    representative = pd.DataFrame([{"x1": 1.5, "x2": X["x2"].mean()}])

    assert custom.to_numpy() == pytest.approx(
        result.predict_proba(representative).iloc[0].to_numpy(), abs=1e-12
    )
    with pytest.raises(ValueError, match="Unknown covariates"):
        result.margins(X, at={"not_a_feature": 0.0})
    with pytest.raises(ValueError, match="kind must be"):
        result.margins(X, kind="elasticity")
    with pytest.raises(ValueError, match="at must be"):
        result.margins(X, at="median")


@pytest.mark.parametrize("estimator", [OrderedLogit, OrderedProbit])
def test_lincom_uses_full_parameter_covariance(estimator):
    X, y = make_ordinal_data(nobs=800)
    result = estimator().fit(X, y)
    threshold_name = result.all_params.index[-1]
    weights = {"x1": 1.0, "x2": -0.5, threshold_name: 0.25}
    output = result.lincom(weights)

    contrast = np.array([1.0, -0.5, 0.0, 0.25])
    expected_estimate = contrast @ result.all_params.to_numpy()
    expected_se = np.sqrt(contrast @ result.covariance.to_numpy() @ contrast)
    assert output["estimate"] == pytest.approx(expected_estimate)
    assert output["standard_error"] == pytest.approx(expected_se)
    assert 0 <= output["p_value"] <= 1
    assert output["lower"] < output["estimate"] < output["upper"]


def test_wald_single_and_joint_restrictions():
    X, y = make_ordinal_data(nobs=800)
    result = OrderedLogit().fit(X, y)
    single = result.wald_test({"x1": 1.0})
    joint = result.wald_test([{"x1": 1.0}, {"x2": 1.0}])

    assert single["statistic"] == pytest.approx(result.zstats["x1"] ** 2)
    assert single["df"] == 1
    assert joint["df"] == 2
    assert 0 <= joint["p_value"] <= 1


def test_linear_hypothesis_validation():
    X, y = make_ordinal_data(nobs=400)
    result = OrderedLogit().fit(X, y)

    with pytest.raises(ValueError, match="Unknown parameters"):
        result.lincom({"not_a_parameter": 1.0})
    with pytest.raises(ValueError, match="one null value"):
        result.wald_test([{"x1": 1.0}, {"x2": 1.0}], values=[0.0])
    with pytest.raises(ValueError, match="at least one restriction"):
        result.wald_test([])


def test_proportional_odds_diagnostic_contract():
    X, y = make_ordinal_data(nobs=1_200)
    result = OrderedLogit().fit(X, y)
    diagnostic = result.proportional_odds_test(X, y)

    assert diagnostic.statistic >= 0
    assert diagnostic.df == X.shape[1]
    assert 0 <= diagnostic.p_value <= 1
    assert diagnostic.threshold_coefficients.shape == (2, X.shape[1])
    assert list(diagnostic.threshold_coefficients.columns) == list(X.columns)


def test_proportional_odds_diagnostic_is_logit_only():
    X, y = make_ordinal_data(nobs=400)
    result = OrderedProbit().fit(X, y)

    with pytest.raises(ValueError, match="available for Logit only"):
        result.proportional_odds_test(X, y)
