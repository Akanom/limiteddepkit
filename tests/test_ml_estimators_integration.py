"""Real-estimator integration tests for the experimental validation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from limiteddepkit import BinaryLogit, OrderedLogit, RandomEffectsOrderedLogit
from limiteddepkit.experimental import (
    CensoredQuantileRegression,
    ConditionalLogit,
    ExponentialDuration,
    MultinomialLogit,
    SampleSelection,
    Tobit,
    ZeroInflatedPoisson,
)
from limiteddepkit.ml import (
    EntityHoldoutSplit,
    ForwardPanelSplit,
    GroupKFold,
    KFold,
    StratifiedKFold,
    cross_validate,
)


def _assert_complete_validation(result, nobs, *, metrics, prediction_columns):
    assert result.successful_folds == 2
    assert result.eligible_folds == 2
    assert result.eligible

    folds = result.fold_frame()
    assert set(metrics).issubset(folds.columns)
    assert np.isfinite(folds[list(metrics)].to_numpy(dtype=float)).all()

    predictions = result.out_of_fold_predictions()
    assert len(predictions) == nobs
    assert predictions["row_index"].nunique() == nobs
    assert set(prediction_columns).issubset(predictions.columns)
    assert np.isfinite(
        predictions[list(prediction_columns)].to_numpy(dtype=float)
    ).all()


def test_binary_logit_cross_validation_uses_real_probability_result():
    rng = np.random.default_rng(7)
    nobs = 400
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=nobs),
            "x2": rng.normal(size=nobs),
        },
        index=pd.Index(np.arange(10_000, 10_000 + nobs), name="row"),
    )
    probability = expit(X.to_numpy() @ np.array([-0.4, 0.8, -0.3]))
    y = rng.binomial(1, probability)

    result = cross_validate(
        BinaryLogit,
        X,
        y,
        splitter=StratifiedKFold(2, shuffle=True, random_state=44),
        outcome="auto",
    )

    assert result.outcome == "binary"
    _assert_complete_validation(
        result,
        nobs,
        metrics={"log_loss", "brier_score", "accuracy", "balanced_accuracy"},
        prediction_columns={"prediction_0", "prediction_1"},
    )
    assert set(result.out_of_fold_predictions()["row_index"]) == set(X.index)


def test_ordered_logit_cross_validation_scores_ordered_probabilities():
    rng = np.random.default_rng(1702)
    nobs = 400
    X = pd.DataFrame(
        {"x1": rng.normal(size=nobs), "x2": rng.normal(size=nobs)}
    )
    linear_index = 0.8 * X["x1"] - 0.4 * X["x2"]
    lower = expit(-0.7 - linear_index)
    upper = expit(0.8 - linear_index)
    probabilities = np.column_stack([lower, upper - lower, 1.0 - upper])
    y = np.array([rng.choice(3, p=row) for row in probabilities])

    result = cross_validate(
        OrderedLogit,
        X,
        y,
        splitter=StratifiedKFold(2, shuffle=True, random_state=2),
        outcome="ordinal",
    )

    _assert_complete_validation(
        result,
        nobs,
        metrics={
            "log_loss",
            "brier_score",
            "ranked_probability_score",
            "ordinal_mae",
        },
        prediction_columns={"prediction_0", "prediction_1", "prediction_2"},
    )


def test_multinomial_logit_cross_validation_scores_category_probabilities():
    rng = np.random.default_rng(1703)
    nobs = 240
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    utilities = np.column_stack(
        [
            np.zeros(nobs),
            X.to_numpy() @ np.array([0.2, 0.7]),
            X.to_numpy() @ np.array([-0.1, -0.5]),
        ]
    )
    probabilities = softmax(utilities, axis=1)
    y = np.array([rng.choice(3, p=row) for row in probabilities])

    result = cross_validate(
        MultinomialLogit,
        X,
        y,
        splitter=StratifiedKFold(2, shuffle=True, random_state=9),
        outcome="auto",
    )

    assert result.outcome == "multiclass"
    _assert_complete_validation(
        result,
        nobs,
        metrics={"log_loss", "brier_score"},
        prediction_columns={"prediction_0", "prediction_1", "prediction_2"},
    )


def test_conditional_logit_cross_validation_keeps_choice_sets_together():
    rng = np.random.default_rng(2718)
    n_choice_sets = 60
    n_alternatives = 3
    nrows = n_choice_sets * n_alternatives
    groups = np.repeat(
        [f"choice-set-{index}" for index in range(n_choice_sets)],
        n_alternatives,
    )
    X = pd.DataFrame(
        {
            "price": rng.normal(size=nrows),
            "quality": rng.normal(size=nrows),
        },
        index=pd.Index([f"choice-row-{index}" for index in range(nrows)]),
    )
    utilities = (X.to_numpy() @ np.array([-0.7, 0.45])).reshape(
        n_choice_sets, n_alternatives
    )
    probabilities = softmax(utilities, axis=1)
    selected_alternative = np.array(
        [rng.choice(n_alternatives, p=row) for row in probabilities]
    )
    choice = np.zeros(nrows, dtype=int)
    choice[np.arange(n_choice_sets) * n_alternatives + selected_alternative] = 1

    result = cross_validate(
        lambda: ConditionalLogit(n_alternatives),
        X,
        choice,
        splitter=GroupKFold(2, shuffle=True, random_state=8),
        entity=groups,
        outcome="choice",
        fit_context={"groups": groups},
        predict_context={"groups": groups},
        score_context={"groups": groups},
    )

    assert result.outcome == "choice"
    assert set(result.fold_frame()["prediction_target"]) == {"new_entity"}
    _assert_complete_validation(
        result,
        nrows,
        metrics={"log_loss", "brier_score", "accuracy"},
        prediction_columns={"prediction"},
    )
    out_of_fold = result.out_of_fold_predictions()
    group_by_row = pd.Series(groups, index=X.index)
    out_of_fold["group"] = out_of_fold["row_index"].map(group_by_row)
    assert (out_of_fold.groupby("group")["fold"].nunique() == 1).all()
    assert np.allclose(
        out_of_fold.groupby("group")["prediction"].sum().to_numpy(),
        1.0,
    )


def test_random_effects_ordinal_distinguishes_new_and_known_entity_targets():
    rng = np.random.default_rng(992)
    n_entities, n_periods = 30, 6
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=len(entity)),
            "x2": rng.normal(size=len(entity)),
        }
    )
    random_intercept = rng.normal(scale=0.5, size=n_entities)
    index = X.to_numpy() @ np.array([0.6, -0.35]) + random_intercept[entity]
    cumulative = expit(np.array([-0.6, 0.8])[None, :] - index[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], 1.0 - cumulative[:, 1]]
    )
    y = np.array([rng.choice(3, p=row) for row in probabilities])

    new_entities = cross_validate(
        RandomEffectsOrderedLogit,
        X,
        y,
        splitter=EntityHoldoutSplit(2),
        entity=entity,
        time=time,
        outcome="ordinal",
        prediction_target="new_entity",
        fit_kwargs={"quadrature_points": 5},
    )
    known_entities = cross_validate(
        RandomEffectsOrderedLogit,
        X,
        y,
        splitter=ForwardPanelSplit(1, min_train_periods=4),
        entity=entity,
        time=time,
        outcome="ordinal",
        prediction_target="known_entity_future",
        fit_kwargs={"quadrature_points": 5},
    )

    assert new_entities.eligible and known_entities.eligible
    assert set(new_entities.fold_frame()["prediction_target"]) == {"new_entity"}
    assert set(known_entities.fold_frame()["prediction_target"]) == {
        "known_entity_future"
    }
    assert len(new_entities.out_of_fold_predictions()) == len(X)
    assert len(known_entities.out_of_fold_predictions()) == n_entities
    probability_columns = ["prediction_0", "prediction_1", "prediction_2"]
    assert np.allclose(
        known_entities.out_of_fold_predictions()[probability_columns].sum(axis=1),
        1.0,
    )


def test_tobit_cross_validation_scores_the_default_observed_mean():
    rng = np.random.default_rng(1706)
    nobs = 240
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    latent = 0.5 + 0.8 * X["x"].to_numpy() + rng.normal(scale=0.7, size=nobs)
    observed = np.maximum(0.0, latent)

    result = cross_validate(
        Tobit,
        X,
        observed,
        splitter=KFold(2, shuffle=True, random_state=10),
        outcome="auto",
    )

    assert result.outcome == "continuous"
    _assert_complete_validation(
        result,
        nobs,
        metrics={"mae", "rmse", "bias"},
        prediction_columns={"prediction"},
    )


def test_censored_quantile_cross_validation_inherits_fitted_quantile():
    rng = np.random.default_rng(1707)
    nobs = 240
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    latent = 0.4 + 0.6 * X["x"].to_numpy() + rng.laplace(scale=0.5, size=nobs)
    observed = np.maximum(0.0, latent)

    result = cross_validate(
        lambda: CensoredQuantileRegression(quantile=0.5, lower=0.0),
        X,
        observed,
        splitter=KFold(2, shuffle=True, random_state=11),
        outcome="auto",
        fit_kwargs={"n_starts": 2, "maxiter": 600},
        require_inference_valid=False,
    )

    assert result.outcome == "quantile"
    _assert_complete_validation(
        result,
        nobs,
        metrics={"check_loss"},
        prediction_columns={"prediction"},
    )


def test_zero_inflated_poisson_reports_mean_and_zero_probability():
    rng = np.random.default_rng(19)
    nobs = 600
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    Z = pd.DataFrame({"const": 1.0, "z": rng.normal(size=nobs)})
    count_mean = np.exp(0.15 + 0.4 * X["x"])
    inflation_probability = expit(-0.9 - 0.45 * Z["z"])
    y = rng.poisson(count_mean)
    y[rng.uniform(size=nobs) < inflation_probability] = 0

    result = cross_validate(
        ZeroInflatedPoisson,
        X,
        y,
        splitter=StratifiedKFold(2, shuffle=True, random_state=3),
        split_y=y == 0,
        outcome="count",
        fit_context={"X_inflation": Z},
        predict_context={"X_inflation": Z},
    )

    _assert_complete_validation(
        result,
        nobs,
        metrics={"mae", "rmse", "poisson_deviance", "zero_rate_calibration_error"},
        prediction_columns={"prediction_mean", "prediction_zero_probability"},
    )
    zero_probability = result.out_of_fold_predictions()[
        "prediction_zero_probability"
    ]
    assert zero_probability.between(0.0, 1.0).all()


def test_exponential_duration_passes_event_context_to_fit_and_score():
    rng = np.random.default_rng(1704)
    nobs = 300
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=nobs)})
    latent_duration = rng.exponential(scale=np.exp(0.3 - 0.35 * X["x"]))
    censoring_duration = rng.exponential(scale=2.0, size=nobs)
    duration = np.minimum(latent_duration, censoring_duration)
    event = (latent_duration <= censoring_duration).astype(int)

    result = cross_validate(
        ExponentialDuration,
        X,
        duration,
        splitter=StratifiedKFold(2, shuffle=True, random_state=4),
        split_y=event,
        outcome="duration",
        fit_context={"event": event},
        score_context={"event": event},
    )

    _assert_complete_validation(
        result,
        nobs,
        metrics={"concordance_index"},
        prediction_columns={"prediction_expected_duration"},
    )
    predictions = result.out_of_fold_predictions()["prediction_expected_duration"]
    assert (predictions > 0.0).all()


def test_sample_selection_cross_validation_scores_both_equations():
    rng = np.random.default_rng(1705)
    nobs = 500
    x = rng.normal(size=nobs)
    excluded_instrument = rng.normal(size=nobs)
    X = pd.DataFrame({"const": 1.0, "x": x})
    Z = pd.DataFrame(
        {"const": 1.0, "x": x, "instrument": excluded_instrument}
    )

    selection_error = rng.normal(size=nobs)
    outcome_error = 0.35 * selection_error + np.sqrt(1.0 - 0.35**2) * rng.normal(
        size=nobs
    )
    selected = (
        0.1 + 0.3 * x + 0.55 * excluded_instrument + selection_error > 0.0
    ).astype(int)
    latent_outcome = 1.0 + 0.6 * x + outcome_error
    y = np.where(selected == 1, latent_outcome, np.nan)

    result = cross_validate(
        SampleSelection,
        X,
        y,
        splitter=StratifiedKFold(2, shuffle=True, random_state=5),
        split_y=selected,
        outcome="selection",
        fit_context={"Z": Z, "selection": selected},
        predict_context={"Z": Z},
        score_context={"selection": selected},
        fit_kwargs={"maxiter": 350},
    )

    _assert_complete_validation(
        result,
        nobs,
        metrics={
            "selection_brier_score",
            "selection_log_loss",
            "observed_outcome_rmse",
        },
        prediction_columns={
            "prediction_selection_probability",
            "prediction_outcome",
            "prediction_observed_outcome",
        },
    )
    selection_probability = result.out_of_fold_predictions()[
        "prediction_selection_probability"
    ]
    assert selection_probability.between(0.0, 1.0).all()
