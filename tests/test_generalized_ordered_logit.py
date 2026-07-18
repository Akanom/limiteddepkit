from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import (
    GeneralizedOrderedLogit,
    OrderedLogit,
    OrderedProbit,
    PartialProportionalOdds,
    likelihood_ratio_test,
)


def make_ordinal_data(seed=6104, nobs=700):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, nobs), "x2": rng.uniform(-1, 1, nobs)})
    beta = np.array([0.7, -0.4])
    thresholds = np.array([-0.8, 0.9])
    cumulative = expit(thresholds[None, :] - X.to_numpy() @ beta[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], 1 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    return X, y


def test_generalized_ordered_logit_fit_and_probabilities():
    X, y = make_ordinal_data()
    result = GeneralizedOrderedLogit().fit(X, y)
    probabilities = result.predict_proba(X.iloc[:30])

    assert result.converged
    assert result.threshold_slopes.shape == (2, 2)
    assert result.minimum_index_gap > 0
    assert probabilities.shape == (30, 3)
    assert np.all(probabilities.to_numpy() >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.inference_valid
    assert list(result.covariance.index) == list(result.all_params.index)
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(result.standard_errors > 0)


def test_generalized_prediction_rejects_crossing_region():
    X, y = make_ordinal_data()
    result = GeneralizedOrderedLogit().fit(X, y)
    direction = result.threshold_slopes.iloc[1] - result.threshold_slopes.iloc[0]
    extreme = pd.DataFrame(
        [100_000 * direction.to_numpy()], columns=result.feature_names
    )

    if np.linalg.norm(direction) < 1e-10:
        pytest.skip("Fitted threshold slopes are numerically identical.")
    with pytest.raises(ValueError, match="cross"):
        result.predict_proba(extreme)


def test_generalized_ordered_logit_validates_minimum_gap():
    X, y = make_ordinal_data(nobs=300)
    with pytest.raises(ValueError, match="positive"):
        GeneralizedOrderedLogit().fit(X, y, minimum_gap=0.0)


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedOrderedLogit(), PartialProportionalOdds(varying=["x1"])],
)
def test_loose_tolerance_cannot_certify_flexible_ordinal_starting_values(estimator):
    X, y = make_ordinal_data(nobs=400)
    result = estimator.fit(X, y, tolerance=1e6)

    assert result.converged
    assert result.scaled_kkt_residual <= 1e-4
    assert result.optimizer_result.nit > 1


def test_partial_proportional_odds_keeps_unselected_slope_common():
    X, y = make_ordinal_data()
    result = PartialProportionalOdds(varying=["x1"]).fit(X, y)
    probabilities = result.predict_proba(X.iloc[:25])

    assert result.converged
    assert list(result.common_params.index) == ["x2"]
    assert list(result.varying_params.columns) == ["x1"]
    assert result.threshold_slopes["x2"].nunique() == 1
    assert result.minimum_index_gap > 0
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.inference_valid
    assert list(result.covariance.index) == list(result.all_params.index)
    assert np.all(np.isfinite(result.standard_errors))


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedOrderedLogit(), PartialProportionalOdds(varying=["x1"])],
)
def test_generalized_confidence_intervals(estimator):
    X, y = make_ordinal_data(nobs=500)
    result = estimator.fit(X, y)
    intervals = result.conf_int()

    assert np.all(intervals["lower"] < result.all_params)
    assert np.all(result.all_params < intervals["upper"])
    with pytest.raises(ValueError, match="strictly between"):
        result.conf_int(level=1.0)


def test_partial_proportional_odds_validates_varying_features():
    X, y = make_ordinal_data(nobs=300)
    with pytest.raises(ValueError, match="at least one"):
        PartialProportionalOdds(varying=[])
    with pytest.raises(ValueError, match="unique"):
        PartialProportionalOdds(varying=["x1", "x1"])
    with pytest.raises(ValueError, match="Unknown varying"):
        PartialProportionalOdds(varying=["unknown"]).fit(X, y)


def test_likelihood_ratio_comparisons_use_correct_parameter_counts():
    X, y = make_ordinal_data()
    ordered = OrderedLogit().fit(X, y)
    partial = PartialProportionalOdds(varying=["x1"]).fit(X, y)
    generalized = GeneralizedOrderedLogit().fit(X, y)

    ordered_vs_partial = likelihood_ratio_test(ordered, partial)
    partial_vs_generalized = likelihood_ratio_test(partial, generalized)

    assert ordered.n_params == 4
    assert partial.n_params == 5
    assert generalized.n_params == 6
    assert ordered_vs_partial.df == 1
    assert partial_vs_generalized.df == 1
    assert ordered_vs_partial.statistic >= 0
    assert 0 <= ordered_vs_partial.p_value <= 1
    assert ordered_vs_partial.regular_chi2_reference


def test_likelihood_ratio_test_rejects_non_nested_order():
    X, y = make_ordinal_data(nobs=400)
    ordered = OrderedLogit().fit(X, y)
    generalized = GeneralizedOrderedLogit().fit(X, y)

    with pytest.raises(ValueError, match="not a supported"):
        likelihood_ratio_test(generalized, ordered)


def test_likelihood_ratio_test_rejects_link_mismatch():
    X, y = make_ordinal_data(nobs=400)
    probit = OrderedProbit().fit(X, y)
    generalized = GeneralizedOrderedLogit().fit(X, y)

    with pytest.raises(ValueError, match="must also be Ordered Logit"):
        likelihood_ratio_test(probit, generalized)


def test_likelihood_ratio_test_suppresses_invalid_boundary_pvalue():
    X, y = make_ordinal_data(nobs=400)
    ordered = OrderedLogit().fit(X, y)
    partial = PartialProportionalOdds(varying=["x1"]).fit(X, y)
    boundary_fit = replace(partial, constraint_slack=1e-7)
    comparison = likelihood_ratio_test(ordered, boundary_fit)

    assert not comparison.regular_chi2_reference
    assert np.isnan(comparison.p_value)
    assert "constrained bootstrap" in comparison.note


def test_active_non_crossing_constraint_suppresses_hessian_inference():
    X, y = make_ordinal_data(nobs=400)
    result = GeneralizedOrderedLogit().fit(X, y, minimum_gap=1.5)

    assert result.converged
    assert result.constraint_slack == pytest.approx(0.0, abs=1e-6)
    assert not result.inference_valid
    assert result.standard_errors.isna().all()


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedOrderedLogit(), PartialProportionalOdds(varying=["x1"])],
)
def test_flexible_ordinal_marginal_effects_probability_identity(estimator):
    X, y = make_ordinal_data(nobs=500)
    result = estimator.fit(X, y)
    effects = result.marginal_effects(X.iloc[:20])

    summed_over_categories = effects.T.groupby(level="feature").sum().T
    assert effects.shape == (20, len(result.categories) * X.shape[1])
    assert np.allclose(summed_over_categories, 0.0, atol=1e-12)
    assert result.average_marginal_effects(X).to_numpy() == pytest.approx(
        result.marginal_effects(X).mean(axis=0).unstack("feature").to_numpy(),
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedOrderedLogit(), PartialProportionalOdds(varying=["x1"])],
)
def test_flexible_ordinal_marginal_effects_match_finite_differences(estimator):
    X, y = make_ordinal_data(nobs=500)
    result = estimator.fit(X, y)
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


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedOrderedLogit(), PartialProportionalOdds(varying=["x1"])],
)
def test_flexible_average_marginal_effects_inference(estimator):
    X, y = make_ordinal_data(nobs=500)
    result = estimator.fit(X, y)
    inference = result.average_marginal_effects_inference(X)
    average = result.average_marginal_effects(X).to_numpy().reshape(-1)

    assert inference.attrs["inference_valid"]
    assert inference["estimate"].to_numpy() == pytest.approx(average, abs=1e-12)
    assert np.all(np.isfinite(inference["standard_error"]))
    assert np.all(inference["standard_error"] > 0)
    assert np.all((inference["p_value"] >= 0) & (inference["p_value"] <= 1))


def test_boundary_fit_reports_ame_without_invalid_inference():
    X, y = make_ordinal_data(nobs=400)
    result = GeneralizedOrderedLogit().fit(X, y, minimum_gap=1.5)
    inference = result.average_marginal_effects_inference(X)

    assert not inference.attrs["inference_valid"]
    assert np.all(np.isfinite(inference["estimate"]))
    assert inference["standard_error"].isna().all()
    assert inference["p_value"].isna().all()


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedOrderedLogit(), PartialProportionalOdds(varying=["x1"])],
)
def test_flexible_model_margins_overall_mean_and_custom(estimator):
    X, y = make_ordinal_data(nobs=500)
    result = estimator.fit(X, y)
    overall = result.margins(X)
    at_mean = result.margins(X, at="mean")
    effects = result.margins(X, kind="marginal_effect")
    custom = result.margins(X, at={"x1": 0.25})
    representative = pd.DataFrame([{"x1": 0.25, "x2": X["x2"].mean()}])

    assert overall.to_numpy() == pytest.approx(
        result.predict_proba(X).mean(axis=0).to_numpy(), abs=1e-12
    )
    assert at_mean.sum() == pytest.approx(1.0)
    assert effects.to_numpy() == pytest.approx(
        result.average_marginal_effects(X).to_numpy(), abs=1e-12
    )
    assert custom.to_numpy() == pytest.approx(
        result.predict_proba(representative).iloc[0].to_numpy(), abs=1e-12
    )


def test_flexible_model_margins_validation():
    X, y = make_ordinal_data(nobs=300)
    result = GeneralizedOrderedLogit().fit(X, y)

    with pytest.raises(ValueError, match="Unknown covariates"):
        result.margins(X, at={"unknown": 0.0})
    with pytest.raises(ValueError, match="kind must be"):
        result.margins(X, kind="elasticity")
    with pytest.raises(ValueError, match="at must be"):
        result.margins(X, at="median")
