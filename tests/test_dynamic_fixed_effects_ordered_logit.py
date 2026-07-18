from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit.dynamic_fixed_effects_ordinal import (
    DynamicFixedEffectsOrderedLogit,
)


def _simulate_panel(
    *,
    n_entities: int = 2_500,
    seed: int = 773,
    beta: float = 0.55,
    state_dependence: float = 0.8,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    thresholds = np.array([-1.0, 0.0, 1.15])
    covariate = np.empty((n_entities, 4), dtype=float)
    covariate[:, :3] = rng.integers(0, 3, size=(n_entities, 3))
    covariate[:, 3] = covariate[:, 2]
    effects = rng.normal(scale=1.1, size=n_entities) + 0.35 * covariate.mean(axis=1)

    outcome = np.empty((n_entities, 4), dtype=int)
    initial_latent = effects + 0.2 * covariate[:, 0] + rng.logistic(size=n_entities)
    outcome[:, 0] = np.digitize(initial_latent, thresholds)
    for period in range(1, 4):
        latent = (
            effects
            + beta * covariate[:, period]
            + state_dependence * (outcome[:, period - 1] >= 2)
            + rng.logistic(size=n_entities)
        )
        outcome[:, period] = np.digitize(latent, thresholds)

    X = pd.DataFrame({"x": covariate.ravel()})
    entity = np.repeat(np.arange(n_entities), 4)
    time = np.tile(np.arange(4), n_entities)
    return X, outcome.ravel(), entity, time


def _fit_simulation(n_entities: int = 2_500):
    X, y, entity, time = _simulate_panel(n_entities=n_entities)
    return DynamicFixedEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        time=time,
        state_cutoff=2,
        category_order=[0, 1, 2, 3],
    )


def test_result_contract_and_conditioned_sample_are_auditable() -> None:
    result = _fit_simulation()

    assert result.converged
    assert result.inference_valid
    assert result.scaled_kkt_residual < 1e-7
    assert result.backend == "mrv-fixed-t-ccmle"
    assert result.covariance_type == "entity-cluster-godambe"
    assert not result.entity_effects_identified
    assert result.thresholds_identified
    assert not result.probability_prediction_available
    assert result.n_groups == result.n_entities
    assert result.n_inference_groups == result.n_contributing_entities
    pd.testing.assert_series_equal(result.structural_params, result.params)
    assert result.state_dependence_params.iloc[0] == result.state_dependence
    assert result.nobs == 10_000
    assert result.n_stayer_entities == 2_500
    assert result.n_contributing_entities <= result.n_stayer_entities
    assert result.n_conditional_contributions > result.n_contributing_entities
    assert result.thresholds.loc[result.normalized_threshold] == 0.0
    assert np.all(np.diff(result.thresholds.to_numpy()) > 0.0)
    assert result.covariance.index.tolist() == result.all_params.index.tolist()
    assert result.summary_frame().index.tolist() == result.all_params.index.tolist()
    assert np.isfinite(result.conf_int()).all().all()
    assert result.state_odds_ratio == pytest.approx(np.exp(result.state_dependence))

    conditional = result.conditional_sample_frame()
    assert len(conditional) == result.n_conditional_contributions
    assert set(result.all_params.index).issubset(conditional.columns)
    assert set(conditional["_response"].unique()) == {0.0, 1.0}
    assert conditional["_entity"].nunique() == result.n_contributing_entities
    assert np.linalg.norm(result.entity_score_frame().sum(), ord=np.inf) < 2e-3


def test_common_index_is_schema_preserving_but_probabilities_are_rejected() -> None:
    result = _fit_simulation(n_entities=1_800)
    prediction_X = pd.DataFrame({"x": [0.0, 2.0]}, index=["a", "b"])
    common = result.common_index(prediction_X, lagged_y=[1, 3])

    assert common.index.tolist() == ["a", "b"]
    assert common.iloc[0] == pytest.approx(0.0)
    assert common.iloc[1] == pytest.approx(
        2.0 * result.params["x"] + result.state_dependence
    )
    with pytest.raises(NotImplementedError, match="does not estimate entity effects"):
        result.predict_proba(prediction_X)
    with pytest.raises(NotImplementedError, match="cannot predict categories"):
        result.predict(prediction_X)
    with pytest.raises(ValueError, match="outside the fitted order"):
        result.common_index(prediction_X, lagged_y=[1, 99])
    with pytest.raises(ValueError, match="columns must match"):
        result.common_index(prediction_X.rename(columns={"x": "z"}), lagged_y=[1, 3])


def test_semantic_labels_row_order_and_entity_labels_do_not_change_estimates() -> None:
    X, y, entity, time = _simulate_panel(n_entities=2_000, seed=818)
    baseline = DynamicFixedEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        time=time,
        state_cutoff=2,
        category_order=[0, 1, 2, 3],
    )
    labels = np.array(["low", "fair", "good", "high"], dtype=object)[y]
    string_entities = np.array([f"person-{value}" for value in entity], dtype=object)
    permutation = np.random.default_rng(92).permutation(len(y))
    relabeled = DynamicFixedEffectsOrderedLogit().fit(
        X.iloc[permutation].reset_index(drop=True),
        labels[permutation],
        entity=string_entities[permutation],
        time=time[permutation] + 20,
        state_cutoff="good",
        category_order=["low", "fair", "good", "high"],
    )

    np.testing.assert_allclose(relabeled.all_params, baseline.all_params, atol=1e-7)
    np.testing.assert_allclose(relabeled.covariance, baseline.covariance, atol=1e-7)


def test_conditional_odds_identity_eliminates_the_fixed_effect() -> None:
    # One (j, l) history pair, evaluated directly from the full transition
    # probabilities.  The conditional B probability must not depend on alpha.
    beta = 0.6
    rho = 0.75
    gamma_j = -0.9
    gamma_l = 1.2
    x1, x2, x3 = 2.0, 0.5, 0.5
    d0 = 1.0

    def conditional_b(alpha: float, d3: float) -> float:
        q1 = alpha + x1 * beta + rho * d0
        q2_j = alpha + x2 * beta + rho - gamma_j
        q2_l = alpha + x2 * beta - gamma_l
        q3_j = alpha + x3 * beta + rho - gamma_j
        q3_l = alpha + x3 * beta - gamma_l
        probability_b = (
            expit(q1) * (1.0 - expit(q2_j)) * expit(q3_l) ** d3
            * (1.0 - expit(q3_l)) ** (1.0 - d3)
        )
        probability_a = (
            (1.0 - expit(q1)) * expit(q2_l) * expit(q3_j) ** d3
            * (1.0 - expit(q3_j)) ** (1.0 - d3)
        )
        return float(probability_b / (probability_a + probability_b))

    for d3 in (0.0, 1.0):
        conditional_index = (
            (x1 - x2) * beta
            + rho * (d0 - d3)
            + (1.0 - d3) * gamma_l
            + d3 * gamma_j
        )
        expected = expit(conditional_index)
        assert conditional_b(-2.3, d3) == pytest.approx(expected, abs=1e-13)
        assert conditional_b(1.7, d3) == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("too_few_periods", "exactly four observations"),
        ("time_gap", "four consecutive observations"),
        ("no_stayers", r"No exact X\[2\] == X\[3\]"),
        ("lowest_cutoff", "above the lowest category"),
    ],
)
def test_identification_guardrails(mutation: str, message: str) -> None:
    X, y, entity, time = _simulate_panel(n_entities=300, seed=456)
    cutoff = 2
    if mutation == "too_few_periods":
        keep = time != 3
        X, y, entity, time = X.loc[keep], y[keep], entity[keep], time[keep]
    elif mutation == "time_gap":
        time = time.copy()
        time[time == 3] = 4
    elif mutation == "no_stayers":
        X = X.copy()
        X.loc[time == 3, "x"] += 0.25
    else:
        cutoff = 0

    with pytest.raises(ValueError, match=message):
        DynamicFixedEffectsOrderedLogit().fit(
            X,
            y,
            entity=entity,
            time=time,
            state_cutoff=cutoff,
            category_order=[0, 1, 2, 3],
        )


def test_rank_deficient_conditional_design_is_rejected() -> None:
    X, y, entity, time = _simulate_panel(n_entities=500, seed=511)
    X["constant"] = 1.0
    with pytest.raises(ValueError, match="rank deficient"):
        DynamicFixedEffectsOrderedLogit().fit(
            X,
            y,
            entity=entity,
            time=time,
            state_cutoff=2,
            category_order=[0, 1, 2, 3],
        )


def test_full_rank_separated_conditional_sample_is_rejected() -> None:
    histories = np.array(
        [
            (1, 0, 2, 2),
            (0, 1, 0, 0),
            (0, 0, 1, 2),
            (2, 0, 1, 0),
        ]
    )
    panel_design = np.zeros_like(histories, dtype=float)
    panel_design[:, 1] = np.where(histories[:, 1] >= 1, 1.0, -1.0)

    with pytest.raises(ValueError, match="completely or quasi-completely separated"):
        DynamicFixedEffectsOrderedLogit().fit(
            pd.DataFrame({"x": panel_design.ravel()}),
            histories.ravel(),
            entity=np.repeat(np.arange(4), 4),
            time=np.tile(np.arange(4), 4),
            state_cutoff=1,
            category_order=[0, 1, 2],
        )


def test_confidence_level_validation() -> None:
    result = _fit_simulation(n_entities=1_600)
    with pytest.raises(ValueError, match="strictly between"):
        result.conf_int(level=1.0)


def test_loose_optimizer_tolerance_cannot_certify_an_unstationary_fit() -> None:
    X, y, entity, time = _simulate_panel(n_entities=800)
    result = DynamicFixedEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        time=time,
        state_cutoff=2,
        category_order=[0, 1, 2, 3],
        tolerance=1e6,
    )

    assert result.optimizer_result.success
    assert result.scaled_kkt_residual > 1e-3
    assert not result.converged
    assert not result.inference_valid
    assert result.standard_errors.isna().all()
