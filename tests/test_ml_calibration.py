"""Tests for probability calibration diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit.ml.calibration import (
    binary_brier_decomposition,
    binary_calibration_intercept_slope,
    binary_reliability_table,
    ordinal_cumulative_calibration,
)


def test_binary_calibration_recovers_known_logistic_recalibration():
    rng = np.random.default_rng(20260714)
    raw_log_odds = rng.normal(size=12_000)
    probability = expit(raw_log_odds)
    calibrated_probability = expit(-0.3 + 0.7 * raw_log_odds)
    target = rng.binomial(1, calibrated_probability)

    result = binary_calibration_intercept_slope(target, probability)

    assert result.converged
    assert result.nobs == len(target)
    assert result.intercept == pytest.approx(-0.3, abs=0.04)
    assert result.slope == pytest.approx(0.7, abs=0.04)
    assert np.all(result.standard_errors > 0.0)
    assert result.log_loss_after < result.log_loss_before


def test_binary_calibration_matches_statsmodels_logistic_recalibration():
    statsmodels_api = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(811)
    raw_log_odds = rng.normal(size=800)
    probability = expit(raw_log_odds)
    target = rng.binomial(1, expit(0.2 + 0.85 * raw_log_odds))

    observed = binary_calibration_intercept_slope(target, probability)
    reference = statsmodels_api.Logit(
        target,
        np.column_stack((np.ones(len(target)), raw_log_odds)),
    ).fit(disp=False)

    assert [observed.intercept, observed.slope] == pytest.approx(
        reference.params,
        abs=1e-7,
    )
    assert observed.covariance == pytest.approx(reference.cov_params(), abs=1e-7)


def test_reliability_table_assigns_endpoints_and_preserves_counts():
    target = np.array([0, 0, 1, 1, 1])
    probability = np.array([0.0, 0.19, 0.51, 0.8, 1.0])

    table = binary_reliability_table(target, probability, n_bins=2)

    assert table["count"].sum() == len(target)
    assert table["fraction"].sum() == pytest.approx(1.0)
    assert table.iloc[-1]["upper"] == pytest.approx(1.0)
    assert table.iloc[-1]["count"] == 3
    assert table.iloc[-1]["event_rate"] == pytest.approx(1.0)


def test_quantile_reliability_table_collapses_duplicate_edges_safely():
    table = binary_reliability_table(
        [0, 1, 0, 1],
        [0.25, 0.25, 0.25, 0.25],
        n_bins=10,
        strategy="quantile",
    )

    assert len(table) == 1
    assert table.loc[0, "count"] == 4
    assert table.loc[0, "mean_probability"] == pytest.approx(0.25)
    assert table.loc[0, "event_rate"] == pytest.approx(0.5)


def test_brier_decomposition_retains_exact_grouping_residual_identity():
    target = np.array([0, 1, 0, 1, 1, 0])
    probability = np.array([0.05, 0.2, 0.35, 0.7, 0.85, 0.95])

    result = binary_brier_decomposition(target, probability, n_bins=3)

    assert result.score == pytest.approx(
        result.reliability
        - result.resolution
        + result.uncertainty
        + result.residual
    )
    assert result.uncertainty == pytest.approx(0.25)
    assert result.n_bins == 3


def test_ordinal_calibration_uses_every_cumulative_threshold():
    rng = np.random.default_rng(1907)
    linear = rng.normal(size=15_000)
    lower = expit(-0.6 - linear)
    upper = expit(0.8 - linear)
    probabilities = pd.DataFrame(
        {
            "low": lower,
            "middle": upper - lower,
            "high": 1.0 - upper,
        }
    )
    uniforms = rng.random(len(linear))
    codes = np.where(uniforms <= lower, 0, np.where(uniforms <= upper, 1, 2))
    labels = np.array(["low", "middle", "high"], dtype=object)
    target = labels[codes]

    result = ordinal_cumulative_calibration(target, probabilities)

    assert result.labels == ("low", "middle", "high")
    assert result.valid_thresholds == 2
    assert result.table["threshold"].tolist() == ["low", "middle"]
    assert result.table["intercept"].to_numpy() == pytest.approx([0.0, 0.0], abs=0.05)
    assert result.table["slope"].to_numpy() == pytest.approx([1.0, 1.0], abs=0.05)
    assert result.mean_absolute_intercept < 0.05
    assert result.mean_absolute_slope_deviation < 0.05


def test_ordinal_calibration_retains_unidentified_sparse_boundary():
    probabilities = pd.DataFrame(
        [[0.1, 0.6, 0.3], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
        columns=["low", "middle", "high"],
    )
    result = ordinal_cumulative_calibration(
        ["middle", "middle", "high"],
        probabilities,
    )

    first = result.table.iloc[0]
    assert not bool(first["valid"])
    assert "both outcome classes" in first["reason"]
    assert result.table.iloc[1]["valid"]


@pytest.mark.parametrize(
    ("target", "probability", "message"),
    [
        ([1, 1], [0.2, 0.8], "both outcome classes"),
        ([0, 1], [0.5, 0.5], "constant predictions"),
        ([0, 1], [0.2, 1.2], "between 0 and 1"),
        ([0, 1], [0.2], "same length"),
    ],
)
def test_binary_calibration_rejects_unidentified_or_invalid_inputs(
    target, probability, message
):
    with pytest.raises(ValueError, match=message):
        binary_calibration_intercept_slope(target, probability)


def test_ordinal_calibration_rejects_conflicting_dataframe_labels():
    probabilities = pd.DataFrame(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
        columns=["low", "middle", "high"],
    )
    with pytest.raises(ValueError, match="match probability DataFrame columns"):
        ordinal_cumulative_calibration(
            ["low", "high"],
            probabilities,
            labels=["high", "middle", "low"],
        )
