"""Tests for dependency-light outcome-aware prediction metrics."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.ml.metrics import (
    binary_accuracy,
    binary_balanced_accuracy,
    binary_brier_score,
    binary_log_loss,
    binary_roc_auc,
    choice_accuracy,
    choice_brier_score,
    choice_log_loss,
    continuous_bias,
    continuous_mean_absolute_error,
    continuous_root_mean_squared_error,
    count_mean_absolute_error,
    count_root_mean_squared_error,
    duration_brier_score,
    duration_concordance_index,
    multiclass_brier_score,
    multiclass_log_loss,
    observed_outcome_rmse,
    ordinal_mean_absolute_error,
    poisson_deviance,
    quantile_check_loss,
    ranked_probability_score,
    score_predictions,
    selection_scores,
    zero_rate_calibration_error,
)


def test_binary_metrics_match_direct_calculations():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.7, 0.8, 0.4])

    expected_log_loss = -np.mean(
        y_true * np.log(probability) + (1 - y_true) * np.log1p(-probability)
    )
    assert binary_log_loss(y_true, probability) == pytest.approx(expected_log_loss)
    assert binary_brier_score(y_true, probability) == pytest.approx(
        np.mean((probability - y_true) ** 2)
    )
    assert binary_accuracy(y_true, probability) == pytest.approx(0.5)
    assert binary_balanced_accuracy(y_true, probability) == pytest.approx(0.5)
    assert binary_roc_auc(y_true, probability) == pytest.approx(0.75)

    scores = score_predictions(y_true, probability, outcome="binary")
    assert scores == pytest.approx(
        {
            "log_loss": expected_log_loss,
            "brier_score": np.mean((probability - y_true) ** 2),
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "roc_auc": 0.75,
        }
    )


def test_binary_metrics_accept_two_column_predict_proba_output():
    y_true = np.array([0, 1, 1])
    probability = pd.DataFrame(
        [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]], columns=[0, 1]
    )

    assert binary_log_loss(y_true, probability) == pytest.approx(
        binary_log_loss(y_true, probability[1])
    )
    assert score_predictions(
        y_true, probability, outcome="binary"
    )["brier_score"] == pytest.approx(binary_brier_score(y_true, probability[1]))
    reversed_columns = probability[[1, 0]]
    assert binary_log_loss(y_true, reversed_columns) == pytest.approx(
        binary_log_loss(y_true, probability[1])
    )


@pytest.mark.parametrize(
    ("function", "y_true", "probability", "message"),
    [
        (binary_log_loss, [0, 2], [0.2, 0.8], "binary values"),
        (binary_brier_score, [0, 1], [0.2, 1.2], "between 0 and 1"),
        (binary_accuracy, [0, 1], [0.2], "same number"),
        (binary_log_loss, [0, 1], [0.2, np.nan], "finite"),
    ],
)
def test_binary_metrics_reject_invalid_inputs(function, y_true, probability, message):
    with pytest.raises(ValueError, match=message):
        function(y_true, probability)


def test_binary_balanced_accuracy_requires_both_classes():
    with pytest.raises(ValueError, match="both outcome classes"):
        binary_balanced_accuracy([1, 1], [0.7, 0.8])
    scores = score_predictions([1, 1], [0.7, 0.8], outcome="binary")
    assert np.isnan(scores["balanced_accuracy"])
    assert np.isnan(scores["roc_auc"])


def test_binary_roc_auc_handles_ties_and_rejects_one_class():
    assert binary_roc_auc([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)
    with pytest.raises(ValueError, match="both outcome classes"):
        binary_roc_auc([1, 1], [0.2, 0.8])


def test_grouped_choice_metrics_score_complete_choice_sets():
    choice = np.array([0, 1, 0, 1, 0])
    probability = np.array([0.2, 0.7, 0.1, 0.6, 0.4])
    groups = np.array(["a", "a", "a", "b", "b"])

    assert choice_log_loss(choice, probability, groups) == pytest.approx(
        -np.mean(np.log([0.7, 0.6]))
    )
    assert choice_brier_score(choice, probability, groups) == pytest.approx(
        ((0.2**2 + 0.3**2 + 0.1**2) + (0.4**2 + 0.4**2)) / 2
    )
    assert choice_accuracy(choice, probability, groups) == pytest.approx(1.0)
    assert score_predictions(
        choice, probability, outcome="choice", groups=groups
    ) == pytest.approx(
        {
            "log_loss": choice_log_loss(choice, probability, groups),
            "brier_score": choice_brier_score(choice, probability, groups),
            "accuracy": 1.0,
        }
    )


@pytest.mark.parametrize(
    ("choice", "probability", "groups", "message"),
    [
        ([0, 0], [0.5, 0.5], [1, 1], "exactly one"),
        ([1, 0], [0.4, 0.4], [1, 1], "sum to 1"),
        ([1], [1.0], [1], "at least two"),
    ],
)
def test_grouped_choice_metrics_reject_invalid_choice_sets(
    choice, probability, groups, message
):
    with pytest.raises(ValueError, match=message):
        choice_log_loss(choice, probability, groups)


def test_multiclass_and_ordinal_metrics_use_dataframe_column_order():
    labels = ["low", "middle", "high"]
    y_true = np.array(["low", "middle", "high"], dtype=object)
    probability = pd.DataFrame(
        [[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]],
        columns=labels,
    )
    expected_log_loss = -np.mean(np.log([0.8, 0.6, 0.7]))
    observed = np.eye(3)
    expected_brier = np.mean(np.sum((probability.to_numpy() - observed) ** 2, axis=1))

    assert multiclass_log_loss(y_true, probability) == pytest.approx(expected_log_loss)
    assert multiclass_brier_score(y_true, probability) == pytest.approx(expected_brier)
    assert ranked_probability_score(y_true, probability) == pytest.approx(
        np.mean(
            np.sum(
                (
                    np.cumsum(probability.to_numpy(), axis=1)[:, :-1]
                    - np.array([[1, 1], [0, 1], [0, 0]])
                )
                ** 2,
                axis=1,
            )
        )
    )
    assert ordinal_mean_absolute_error(
        ["low", "high"], ["middle", "middle"], labels=labels
    ) == pytest.approx(1.0)

    scores = score_predictions(y_true, probability, outcome_kind="ordinal")
    assert set(scores) == {
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "ordinal_mae",
    }
    assert scores["ordinal_mae"] == pytest.approx(0.0)


def test_perfect_multiclass_predictions_have_zero_probability_scores():
    y_true = np.array([0, 1, 2])
    probability = np.eye(3)

    assert multiclass_log_loss(y_true, probability) == pytest.approx(0.0)
    assert multiclass_brier_score(y_true, probability) == pytest.approx(0.0)
    assert ranked_probability_score(y_true, probability) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("y_true", "probability", "labels", "message"),
    [
        ([0, 1], [[0.7, 0.4], [0.2, 0.8]], None, "sum to 1"),
        ([0, 2], [[0.7, 0.3], [0.2, 0.8]], None, "absent"),
        ([0, 1], [[0.7], [0.2]], None, "at least two"),
        ([0, 1], [[0.7, 0.3]], None, "same number"),
        ([0, 1], [[0.7, 0.3], [0.2, 0.8]], [0, 0], "unique"),
    ],
)
def test_multiclass_metrics_reject_invalid_contracts(y_true, probability, labels, message):
    with pytest.raises(ValueError, match=message):
        multiclass_log_loss(y_true, probability, labels=labels)


def test_probability_dataframe_rejects_conflicting_explicit_labels():
    probability = pd.DataFrame([[0.7, 0.3], [0.2, 0.8]], columns=["no", "yes"])

    with pytest.raises(ValueError, match="match probability DataFrame columns"):
        multiclass_log_loss(["no", "yes"], probability, labels=["yes", "no"])


def test_continuous_metrics_and_dispatcher():
    y_true = np.array([1.0, 2.0, 4.0])
    prediction = np.array([2.0, 2.0, 2.0])

    assert continuous_mean_absolute_error(y_true, prediction) == pytest.approx(1.0)
    assert continuous_root_mean_squared_error(y_true, prediction) == pytest.approx(
        np.sqrt(5.0 / 3.0)
    )
    assert continuous_bias(y_true, prediction) == pytest.approx(-1.0 / 3.0)
    assert score_predictions(
        y_true, prediction, outcome_kind="continuous"
    ) == pytest.approx(
        {"mae": 1.0, "rmse": np.sqrt(5.0 / 3.0), "bias": -1.0 / 3.0}
    )


def test_count_metrics_match_definitions_and_dispatcher():
    y_true = np.array([0, 1, 4])
    predicted_mean = np.array([0.2, 1.4, 3.5])
    zero_probability = np.array([0.8, 0.2, 0.05])
    positive = y_true > 0
    terms = predicted_mean.copy()
    terms[positive] = (
        y_true[positive] * np.log(y_true[positive] / predicted_mean[positive])
        - y_true[positive]
        + predicted_mean[positive]
    )

    assert count_mean_absolute_error(y_true, predicted_mean) == pytest.approx(
        np.mean(np.abs(y_true - predicted_mean))
    )
    assert count_root_mean_squared_error(y_true, predicted_mean) == pytest.approx(
        np.sqrt(np.mean((y_true - predicted_mean) ** 2))
    )
    assert poisson_deviance(y_true, predicted_mean) == pytest.approx(2.0 * np.mean(terms))
    assert zero_rate_calibration_error(y_true, zero_probability) == pytest.approx(
        abs(1.0 / 3.0 - np.mean(zero_probability))
    )

    scores = score_predictions(
        y_true,
        predicted_mean,
        outcome_kind="count",
        zero_probability=zero_probability,
    )
    assert set(scores) == {"mae", "rmse", "poisson_deviance", "zero_rate_calibration_error"}

    mapped_scores = score_predictions(
        y_true,
        {"mean": predicted_mean, "zero_probability": zero_probability},
        outcome="count",
    )
    assert mapped_scores == pytest.approx(scores)


def test_count_metrics_reject_invalid_counts_and_means():
    with pytest.raises(ValueError, match="integer counts"):
        count_mean_absolute_error([0, 1.5], [0.2, 1.2])
    with pytest.raises(ValueError, match="non-negative predicted means"):
        count_root_mean_squared_error([0, 1], [0.2, -1.2])
    with pytest.raises(ValueError, match="strictly positive"):
        poisson_deviance([0, 1], [0.0, 1.0])


def test_quantile_check_loss_matches_asymmetric_definition():
    assert quantile_check_loss([0.0, 2.0], [1.0, 1.0], quantile=0.25) == pytest.approx(
        0.5
    )
    assert score_predictions(
        [0.0, 2.0], [1.0, 1.0], outcome_kind="quantile", quantile=0.25
    ) == pytest.approx({"check_loss": 0.5})


@pytest.mark.parametrize("quantile", [0.0, 1.0, np.nan])
def test_quantile_check_loss_rejects_invalid_quantile(quantile):
    with pytest.raises(ValueError, match="strictly between"):
        quantile_check_loss([0.0], [1.0], quantile=quantile)


def test_duration_concordance_handles_censoring_and_score_direction():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])

    assert duration_concordance_index(time, event, risk) == pytest.approx(1.0)
    assert duration_concordance_index(
        time, event, -risk, higher_risk=False
    ) == pytest.approx(1.0)


def test_duration_concordance_orders_equal_time_event_before_censoring():
    assert duration_concordance_index([2.0, 2.0], [1, 0], [0.8, 0.2]) == pytest.approx(
        1.0
    )
    assert duration_concordance_index([2.0, 2.0], [1, 0], [0.2, 0.8]) == pytest.approx(
        0.0
    )
    with pytest.raises(ValueError, match="no comparable pairs"):
        duration_concordance_index([2.0, 2.0], [1, 1], [0.8, 0.2])


def test_duration_brier_score_excludes_unknown_censored_status():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 1, 0])
    survival_probability = np.array([0.1, 0.9, 0.2, 0.8])

    # At horizon 2.5, row 2 is censored before the horizon and excluded.
    expected = np.mean([0.1**2, (0.2 - 1.0) ** 2, (0.8 - 1.0) ** 2])
    assert duration_brier_score(
        time, event, survival_probability, horizon=2.5
    ) == pytest.approx(expected)

    scores = score_predictions(
        time,
        [4.0, 3.0, 2.0, 1.0],
        outcome_kind="duration",
        event=event,
        horizon=2.5,
        survival_probability=survival_probability,
    )
    assert scores == pytest.approx(
        {"concordance_index": 1.0, "brier_score_at_horizon": expected}
    )

    mapped_scores = score_predictions(
        time,
        {
            "expected_duration": [1.0, 2.0, 3.0, 4.0],
            "survival_probability": survival_probability,
        },
        outcome="duration",
        event=event,
        horizon=2.5,
    )
    assert mapped_scores == pytest.approx(
        {"concordance_index": 1.0, "brier_score_at_horizon": expected}
    )


def test_duration_metrics_reject_unidentified_or_incomplete_scoring():
    with pytest.raises(ValueError, match="no comparable pairs"):
        duration_concordance_index([1.0, 2.0], [0, 0], [2.0, 1.0])
    with pytest.raises(ValueError, match="known survival state"):
        duration_brier_score([1.0, 2.0], [0, 0], [0.8, 0.7], horizon=3.0)
    with pytest.raises(ValueError, match="supplied together"):
        score_predictions(
            [1.0, 2.0],
            [2.0, 1.0],
            outcome_kind="duration",
            event=[1, 1],
            horizon=1.5,
        )


def test_selection_scores_combine_incidence_and_observed_outcome_metrics():
    selected = np.array([1, 0, 1])
    selection_probability = np.array([0.8, 0.2, 0.6])
    outcome_true = np.array([2.0, np.nan, 5.0])
    outcome_predicted = np.array([1.0, np.nan, 7.0])

    expected_rmse = np.sqrt(2.5)
    assert observed_outcome_rmse(
        selected, outcome_true, outcome_predicted
    ) == pytest.approx(expected_rmse)
    scores = selection_scores(
        selected, selection_probability, outcome_true, outcome_predicted
    )
    assert scores == pytest.approx(
        {
            "selection_brier_score": binary_brier_score(
                selected, selection_probability
            ),
            "selection_log_loss": binary_log_loss(selected, selection_probability),
            "observed_outcome_rmse": expected_rmse,
        }
    )
    assert score_predictions(
        selected,
        selection_probability,
        outcome_kind="selection",
        outcome_true=outcome_true,
        outcome_predicted=outcome_predicted,
    ) == pytest.approx(scores)


def test_selection_dispatcher_accepts_adapter_prediction_mapping():
    selected = np.array([1, 0, 1])
    outcome_true = np.array([2.0, np.nan, 5.0])
    prediction = {
        "selection_probability": np.array([0.8, 0.2, 0.6]),
        "outcome": np.array([99.0, 99.0, 99.0]),
        "observed_outcome": np.array([1.0, np.nan, 7.0]),
    }

    scores = score_predictions(
        outcome_true,
        prediction,
        outcome="selection",
        selection=selected,
    )
    assert scores["observed_outcome_rmse"] == pytest.approx(np.sqrt(2.5))
    assert scores["selection_brier_score"] == pytest.approx(
        binary_brier_score(selected, prediction["selection_probability"])
    )


def test_selection_mapping_requires_predictions_and_selection_context():
    with pytest.raises(ValueError, match="selection_probability"):
        score_predictions(
            [1.0],
            {"outcome": [1.0]},
            outcome_kind="selection",
            score_context=[1],
        )
    with pytest.raises(ValueError, match="score_context"):
        score_predictions(
            [1.0],
            {"selection_probability": [0.8], "outcome": [1.0]},
            outcome_kind="selection",
        )


def test_selection_outcome_requires_selected_finite_observations():
    with pytest.raises(ValueError, match="at least one selected"):
        observed_outcome_rmse([0, 0], [np.nan, np.nan], [np.nan, np.nan])
    with pytest.raises(ValueError, match="must be finite"):
        observed_outcome_rmse([1, 0], [np.nan, np.nan], [1.0, np.nan])


def test_dispatcher_rejects_missing_family_arguments_and_unknown_kind():
    with pytest.raises(ValueError, match="quantile is required"):
        score_predictions([1.0], [1.0], outcome_kind="quantile")
    with pytest.raises(ValueError, match="event is required"):
        score_predictions([1.0], [1.0], outcome_kind="duration")
    with pytest.raises(ValueError, match="outcome_true and outcome_predicted"):
        score_predictions([1], [0.8], outcome_kind="selection")
    with pytest.raises(ValueError, match="Unsupported outcome"):
        score_predictions([1.0], [1.0], outcome_kind="other")  # type: ignore[arg-type]
