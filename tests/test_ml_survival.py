"""Tests for training-censoring-aware survival metrics."""

import numpy as np
import pytest

from limiteddepkit.ml.survival import (
    CensoringDistribution,
    cumulative_dynamic_auc,
    fit_censoring_distribution,
    integrated_brier_score,
    ipcw_concordance_index,
    time_dependent_brier_score,
    time_dependent_brier_scores,
)


def test_reverse_kaplan_meier_distribution_and_support_contract():
    censoring = fit_censoring_distribution([1, 2, 3, 4], [1, 0, 1, 0])

    assert censoring.n_samples == 4
    assert censoring.survival_at([0.5, 1.0, 2.5, 3.5]) == pytest.approx(
        [1.0, 1.0, 2.0 / 3.0, 2.0 / 3.0]
    )
    assert censoring.survival_at(4.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="exceeds.*support"):
        censoring.survival_at(4.1)


def test_reverse_kaplan_meier_orders_events_before_tied_censoring():
    censoring = fit_censoring_distribution([1.0, 1.0, 2.0], [1, 0, 1])

    # At t=1, the event is removed before the censoring hazard: 1 - 1/(3-1).
    assert censoring.survival_at(1.0) == pytest.approx(0.5)


def test_ipcw_concordance_matches_uncensored_pair_ranking():
    censoring = fit_censoring_distribution([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    duration = [1, 2, 3, 4]
    event = [1, 1, 1, 1]

    assert ipcw_concordance_index(
        duration,
        event,
        [4, 3, 2, 1],
        censoring=censoring,
    ) == pytest.approx(1.0)
    assert ipcw_concordance_index(
        duration,
        event,
        [1, 2, 3, 4],
        censoring=censoring,
    ) == pytest.approx(0.0)


def test_ipcw_concordance_includes_equal_time_event_censor_pair():
    censoring = fit_censoring_distribution([1, 2, 3], [1, 1, 1])
    assert ipcw_concordance_index(
        [2.0, 2.0],
        [1, 0],
        [0.8, 0.2],
        censoring=censoring,
    ) == pytest.approx(1.0)


def test_ipcw_concordance_uses_strict_tau_boundary():
    censoring = fit_censoring_distribution([1, 2, 3], [1, 1, 1])

    assert ipcw_concordance_index(
        [1, 2, 3],
        [1, 1, 1],
        [3.0, 1.0, 2.0],
        tau=2.0,
        censoring=censoring,
    ) == pytest.approx(1.0)


def test_time_dependent_brier_score_matches_manual_ipcw_formula():
    censoring = fit_censoring_distribution([1, 2, 3, 4], [1, 0, 1, 0])
    duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 1, 0])
    survival = np.array([0.1, 0.9, 0.2, 0.8])

    # At t=2.5: event at 1 has G=1; later controls have G(t)=2/3;
    # censoring at 2 contributes zero under IPCW rather than changing n.
    expected = (0.1**2 + 0.0 + 0.8**2 / (2 / 3) + 0.2**2 / (2 / 3)) / 4
    assert time_dependent_brier_score(
        duration,
        event,
        survival,
        horizon=2.5,
        censoring=censoring,
    ) == pytest.approx(expected)


def test_brier_does_not_require_horizon_weight_without_dynamic_controls():
    censoring = fit_censoring_distribution([1, 2], [1, 0])
    assert censoring.survival_at(2.0) == pytest.approx(0.0)

    assert time_dependent_brier_score(
        [1.0],
        [1],
        [0.2],
        horizon=2.0,
        censoring=censoring,
    ) == pytest.approx(0.04)


def test_brier_curve_and_integrated_brier_use_same_pointwise_scores():
    censoring = fit_censoring_distribution([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4, dtype=int)
    times = np.array([1.5, 2.5])
    probabilities = np.array(
        [
            [0.1, 0.05],
            [0.8, 0.2],
            [0.9, 0.8],
            [0.95, 0.9],
        ]
    )

    curve = time_dependent_brier_scores(
        duration,
        event,
        probabilities,
        times=times,
        censoring=censoring,
    )
    assert curve.to_frame()["time"].tolist() == times.tolist()
    assert curve.scores[0] == pytest.approx(
        time_dependent_brier_score(
            duration,
            event,
            probabilities[:, 0],
            horizon=times[0],
            censoring=censoring,
        )
    )
    assert integrated_brier_score(
        duration,
        event,
        probabilities,
        times=times,
        censoring=censoring,
    ) == pytest.approx(np.mean(curve.scores))


def test_cumulative_dynamic_auc_scores_cases_against_dynamic_controls():
    censoring = fit_censoring_distribution([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    result = cumulative_dynamic_auc(
        [1, 2, 3, 4],
        [1, 0, 1, 0],
        [4.0, 3.0, 2.0, 1.0],
        times=[1.5, 3.5],
        censoring=censoring,
    )

    assert result.auc == pytest.approx([1.0, 1.0])
    assert result.to_frame()["auc"].tolist() == pytest.approx([1.0, 1.0])


def test_dynamic_auc_counts_tied_risk_as_half_concordant():
    censoring = fit_censoring_distribution([1, 2, 3], [1, 1, 1])
    result = cumulative_dynamic_auc(
        [1, 2, 3],
        [1, 1, 0],
        [0.5, 0.5, 0.5],
        times=[1.5],
        censoring=censoring,
    )
    assert result.auc[0] == pytest.approx(0.5)


def test_survival_metrics_require_explicit_training_distribution():
    with pytest.raises(TypeError, match="explicit CensoringDistribution"):
        time_dependent_brier_score(
            [1, 3],
            [1, 1],
            [0.2, 0.8],
            horizon=1.5,
            censoring=None,  # type: ignore[arg-type]
        )


def test_survival_metrics_reject_zero_weights_bad_shapes_and_unidentified_auc():
    zero = fit_censoring_distribution([1, 2], [1, 0])
    with pytest.raises(ValueError, match="censoring survival is zero"):
        time_dependent_brier_score(
            [1, 2],
            [1, 1],
            [0.2, 0.8],
            horizon=2.0,
            censoring=zero,
        )

    full = fit_censoring_distribution([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError, match="shape"):
        integrated_brier_score(
            [1, 2],
            [1, 1],
            [[0.2], [0.8]],
            times=[1.0, 1.5],
            censoring=full,
        )
    with pytest.raises(ValueError, match="case and control"):
        cumulative_dynamic_auc(
            [1, 2],
            [1, 1],
            [2, 1],
            times=[2.0],
            censoring=full,
        )


def test_censoring_distribution_type_is_not_replaceable_by_test_fold_arrays():
    fake = CensoringDistribution(
        times=np.array([1.0]), survival=np.array([1.0]), n_samples=1
    )
    assert fake.survival_at(0.5) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="non-increasing"):
        CensoringDistribution(
            times=np.array([1.0, 2.0]),
            survival=np.array([0.5, 0.7]),
            n_samples=2,
        )
