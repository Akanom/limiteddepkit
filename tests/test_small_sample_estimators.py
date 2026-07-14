"""Tests for experimental small-sample and separation-resistant estimators."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit.experimental.small_sample import (
    FirthBinaryLogit,
    RidgeBinaryLogit,
    RidgeOrderedLogit,
)


def make_binary_data(seed: int = 917, nobs: int = 240) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=nobs)
    x2 = rng.normal(size=nobs)
    design = pd.DataFrame(
        {"const": np.ones(nobs), "x1": x1, "x2": x2},
        index=pd.Index(np.arange(2_000, 2_000 + nobs), name="row"),
    )
    probabilities = expit(-0.65 + 1.25 * x1 - 0.8 * x2)
    outcomes = rng.binomial(1, probabilities)
    return design, outcomes


def make_ordered_data(
    seed: int = 4102, nobs: int = 500
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    design = pd.DataFrame(
        {"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)},
        index=pd.Index(np.arange(4_000, 4_000 + nobs), name="row"),
    )
    coefficients = np.array([1.05, -0.75])
    thresholds = np.array([-0.85, 0.65])
    cumulative = expit(thresholds[None, :] - design.to_numpy() @ coefficients[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], 1.0 - cumulative[:, 1]]
    )
    outcomes = np.array([rng.choice(3, p=row) for row in probabilities])
    return design, outcomes


def test_firth_matches_half_cell_correction_under_complete_separation() -> None:
    design = pd.DataFrame(
        {
            "const": np.ones(9),
            "x": np.r_[np.zeros(4), np.ones(5)],
        },
        index=pd.Index(np.arange(100, 109), name="case"),
    )
    outcomes = np.r_[np.zeros(4), np.ones(5)]

    result = FirthBinaryLogit().fit(design, outcomes, tolerance=1e-10)

    group_zero_log_odds = np.log(0.5 / 4.5)
    group_one_log_odds = np.log(5.5 / 0.5)
    expected = np.array(
        [group_zero_log_odds, group_one_log_odds - group_zero_log_odds]
    )
    probabilities = result.predict_proba(design)

    assert result.converged
    assert result.optimizer_result.success
    assert result.params.to_numpy() == pytest.approx(expected, abs=3e-8)
    assert np.isfinite(result.params).all()
    assert result.score_norm < 1e-8
    assert result.n_iter > 0
    assert result.step_halvings >= 0
    assert result.backend == "native-firth-adjusted-score"
    assert result.covariance_type == "inverse-ordinary-fisher-at-bias-reduced-estimate"
    assert "profile penalized-likelihood inference is not computed" in result.inference_note
    assert result.penalized_loglike == pytest.approx(
        result.loglike + result.jeffreys_penalty
    )
    assert np.isfinite(result.vcov()).all().all()
    assert (np.diag(result.vcov()) > 0).all()
    assert probabilities.index.equals(design.index)
    assert probabilities.to_numpy().sum(axis=1) == pytest.approx(np.ones(len(design)))
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all().all()


def test_firth_diagnostics_inference_and_prediction_contract() -> None:
    design = pd.DataFrame(
        {
            "const": np.ones(14),
            "x": np.linspace(-2.0, 2.0, 14),
        }
    )
    outcomes = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])

    result = FirthBinaryLogit().fit(design, outcomes)

    assert result.inference_valid
    assert result.constant_features == ("const",)
    assert result.summary_frame().columns.tolist() == ["coef", "std_err", "z", "p_value"]
    assert result.conf_int().shape == (2, 2)
    assert result.predict(design).isin([0, 1]).all()
    with pytest.raises(ValueError, match="columns must match"):
        result.predict_proba(design[["x", "const"]])
    with pytest.raises(ValueError, match="strictly between"):
        result.predict(design, threshold=1.0)
    with pytest.raises(ValueError, match="strictly between"):
        result.conf_int(level=0.0)


def test_ridge_binary_matches_sklearn_with_aligned_penalty_scaling() -> None:
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    design, outcomes = make_binary_data()
    penalty = 2.0

    native = RidgeBinaryLogit().fit(design, outcomes, penalty=penalty)
    reference = sklearn_linear.LogisticRegression(
        C=1.0 / penalty,
        fit_intercept=True,
        solver="lbfgs",
        tol=1e-12,
        max_iter=10_000,
    ).fit(design[["x1", "x2"]], outcomes)
    reference_parameters = np.r_[reference.intercept_, reference.coef_.ravel()]

    assert native.converged
    assert native.params.to_numpy() == pytest.approx(reference_parameters, abs=2e-6)
    assert native.penalty_mask.to_dict() == {"const": 0.0, "x1": 1.0, "x2": 1.0}
    assert native.constant_features == ("const",)
    assert native.covariance_type == "penalized-estimating-equation-sandwich"
    assert 0.0 < native.effective_df < native.n_params
    assert native.penalized_loglike <= native.loglike


def test_ridge_binary_shrinks_slopes_and_can_penalize_constant() -> None:
    design, outcomes = make_binary_data(seed=291)

    weak = RidgeBinaryLogit().fit(design, outcomes, penalty=0.05)
    strong = RidgeBinaryLogit().fit(design, outcomes, penalty=25.0)
    penalized_constant = RidgeBinaryLogit().fit(
        design, outcomes, penalty=25.0, penalize_intercept=True
    )
    predictions = strong.predict_proba(design.iloc[:13])

    assert np.linalg.norm(strong.params[["x1", "x2"]]) < np.linalg.norm(
        weak.params[["x1", "x2"]]
    )
    assert abs(penalized_constant.params["const"]) < abs(strong.params["const"])
    assert penalized_constant.penalty_mask.eq(1.0).all()
    assert predictions.index.equals(design.index[:13])
    assert predictions.to_numpy().sum(axis=1) == pytest.approx(np.ones(13))
    assert ((predictions >= 0.0) & (predictions <= 1.0)).all().all()


def test_ridge_ordered_has_ordered_thresholds_predictions_and_shrinkage() -> None:
    design, outcomes = make_ordered_data()

    weak = RidgeOrderedLogit().fit(design, outcomes, penalty=0.05)
    strong = RidgeOrderedLogit().fit(design, outcomes, penalty=25.0)
    probabilities = strong.predict_proba(design.iloc[:17])
    predictions = strong.predict(design.iloc[:17])

    assert weak.converged and strong.converged
    assert np.linalg.norm(strong.params) < np.linalg.norm(weak.params)
    assert np.all(np.diff(strong.thresholds.to_numpy()) > 0.0)
    assert strong.penalty_target == "slopes-only"
    assert strong.backend == "native-ridge-ordered-logit"
    assert strong.covariance_type == "penalized-observed-information-sandwich"
    assert "thresholds are not directly penalized" in strong.inference_note
    assert 0.0 < strong.effective_df < strong.n_params
    assert probabilities.index.equals(design.index[:17])
    assert predictions.index.equals(design.index[:17])
    assert probabilities.to_numpy().sum(axis=1) == pytest.approx(np.ones(17))
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all().all()
    assert set(predictions).issubset(set(strong.categories))
    assert np.isfinite(strong.vcov()).all().all()
    assert (np.diag(strong.vcov()) > 0.0).all()


def test_nearly_unpenalized_ordered_logit_matches_statsmodels() -> None:
    statsmodels_ordinal = pytest.importorskip("statsmodels.miscmodels.ordinal_model")
    design, outcomes = make_ordered_data(seed=789, nobs=650)

    native = RidgeOrderedLogit().fit(
        design, outcomes, penalty=1e-8, tolerance=1e-10, maxiter=2_000
    )
    reference_model = statsmodels_ordinal.OrderedModel(outcomes, design, distr="logit")
    reference = reference_model.fit(method="bfgs", disp=False, maxiter=2_000)
    reference_thresholds = reference_model.transform_threshold_params(
        reference.params.iloc[design.shape[1] :]
    )[1:-1]

    assert reference.mle_retvals["converged"]
    assert native.params.to_numpy() == pytest.approx(
        reference.params.iloc[: design.shape[1]].to_numpy(), abs=3e-4
    )
    assert native.thresholds.to_numpy() == pytest.approx(reference_thresholds, abs=3e-4)
    assert native.loglike == pytest.approx(reference.llf, abs=1e-5)


def test_small_sample_estimators_reject_invalid_identification_and_penalties() -> None:
    binary_design, outcomes = make_binary_data(nobs=40)
    ordered_design, ordered_outcomes = make_ordered_data(nobs=60)

    with pytest.raises(ValueError, match="strictly positive"):
        RidgeBinaryLogit().fit(binary_design, outcomes, penalty=0.0)
    with pytest.raises(ValueError, match="rank deficient"):
        FirthBinaryLogit().fit(
            binary_design.assign(duplicate=binary_design["x1"]), outcomes
        )
    with pytest.raises(ValueError, match="constant regressors"):
        RidgeOrderedLogit().fit(
            ordered_design.assign(const=1.0), ordered_outcomes, penalty=1.0
        )
    with pytest.raises(ValueError, match="strictly positive"):
        RidgeOrderedLogit().fit(ordered_design, ordered_outcomes, penalty=-1.0)

