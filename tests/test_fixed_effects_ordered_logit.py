"""Tests for BUC fixed-effects Ordered Logit."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.fixed_effects_ordinal import (
    FixedEffectsOrderedLogit,
    _conditional_logit_clone,
)


def _panel(seed=7193, n_entities=160, periods=6):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), periods)
    entity_level = rng.normal(size=n_entities)
    x1 = rng.normal(size=n_entities * periods) + 0.35 * entity_level[entity]
    x2 = rng.normal(size=n_entities * periods)
    X = pd.DataFrame(
        {"x1": x1, "x2": x2},
        index=pd.Index([f"row-{row}" for row in range(len(entity))]),
    )
    beta = np.array([0.75, -0.45])
    fixed_effect = 0.8 * entity_level + 0.25 * np.bincount(
        entity, weights=x1, minlength=n_entities
    ) / periods
    latent = X.to_numpy() @ beta + fixed_effect[entity] + rng.logistic(size=len(entity))
    y = np.select(
        [latent <= -0.6, latent <= 0.35, latent <= 1.2],
        ["poor", "fair", "good"],
        default="excellent",
    )
    return X, y, entity, beta


def test_conditional_clone_matches_brute_force_probability_and_score():
    X = np.array(
        [
            [-1.0, 0.2],
            [-0.1, 0.7],
            [0.5, -0.4],
            [1.2, 0.1],
        ]
    )
    outcome = np.array([0.0, 1.0, 0.0, 1.0])
    beta = np.array([0.4, -0.25])
    observed, score = _conditional_logit_clone(beta, X, outcome)

    eta = X @ beta
    terms = [float(np.sum(eta[list(rows)])) for rows in combinations(range(4), 2)]
    maximum = max(terms)
    denominator = maximum + np.log(np.sum(np.exp(np.asarray(terms) - maximum)))
    expected = float(outcome @ eta - denominator)
    assert observed == pytest.approx(expected, abs=1e-14)

    numerical = np.empty(2)
    for column in range(2):
        shift = np.zeros(2)
        shift[column] = 1e-6
        upper = _conditional_logit_clone(beta + shift, X, outcome)[0]
        lower = _conditional_logit_clone(beta - shift, X, outcome)[0]
        numerical[column] = (upper - lower) / 2e-6
    np.testing.assert_allclose(score, numerical, rtol=0.0, atol=2e-9)


def test_buc_recovers_slopes_and_exposes_only_identified_objects():
    X, y, entity, beta = _panel(n_entities=240)
    result = FixedEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        category_order=["poor", "fair", "good", "excellent"],
    )

    assert result.converged
    assert result.inference_valid
    assert result.scaled_score_norm <= 1e-8
    np.testing.assert_allclose(result.params, beta, atol=0.17)
    assert not result.thresholds_identified
    assert not result.entity_effects_identified
    assert result.n_contributing_entities < result.n_entities
    assert result.n_cutoff_clones >= result.n_contributing_entities
    assert result.covariance_type == "entity-cluster-sandwich"
    assert result.backend == "conditional-buc"
    assert result.summary_frame().index.equals(result.params.index)
    assert np.all(result.odds_ratios() > 0.0)
    assert result.linear_index(X.iloc[:5]).index.equals(X.index[:5])
    assert np.linalg.norm(result.entity_score_frame().sum(axis=0)) < 1e-5

    assert not hasattr(result, "thresholds")
    assert not hasattr(result, "predict_proba")
    assert not hasattr(result, "aic")


def test_buc_loose_tolerance_cannot_certify_starting_values():
    X, y, entity, _ = _panel(n_entities=80)
    result = FixedEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        category_order=["poor", "fair", "good", "excellent"],
        tolerance=1e6,
    )

    assert result.converged
    assert result.inference_valid
    assert result.scaled_score_norm <= 1e-6
    assert not np.allclose(result.params.to_numpy(), 0.0)


def test_buc_is_invariant_to_row_order_entity_labels_and_category_labels():
    X, y, entity, _ = _panel(n_entities=55)
    baseline = FixedEffectsOrderedLogit().fit(
        X,
        y,
        entity=entity,
        category_order=["poor", "fair", "good", "excellent"],
    )
    rng = np.random.default_rng(88)
    order = rng.permutation(len(X))
    relabel = np.array([f"person-{code * 17 + 3}" for code in entity])
    category_mapping = {
        "poor": 20,
        "fair": 10,
        "good": 40,
        "excellent": 30,
    }
    numeric_y = np.array([category_mapping[value] for value in y])
    reordered = FixedEffectsOrderedLogit().fit(
        X.iloc[order],
        numeric_y[order],
        entity=relabel[order],
        category_order=[20, 10, 40, 30],
    )

    np.testing.assert_allclose(reordered.params, baseline.params, atol=2e-7)
    np.testing.assert_allclose(reordered.covariance, baseline.covariance, atol=2e-6)


def test_buc_cluster_covariance_matches_exposed_entity_scores_and_bread():
    X, y, entity, _ = _panel(n_entities=65)
    result = FixedEffectsOrderedLogit().fit(X, y, entity=entity)
    scores = result.entity_score_frame().to_numpy()
    meat = scores.T @ scores
    covariance = result.covariance.to_numpy()
    assert np.isfinite(covariance).all()
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    assert np.linalg.eigvalsh(meat).min() >= -1e-12


def test_buc_rejects_time_invariant_or_nonvarying_panel_designs():
    X, y, entity, _ = _panel(n_entities=35)
    with pytest.raises(ValueError, match="within-entity design is rank deficient"):
        FixedEffectsOrderedLogit().fit(
            X.assign(const=1.0),
            y,
            entity=entity,
        )
    with pytest.raises(ValueError, match="at least three observed categories"):
        FixedEffectsOrderedLogit().fit(
            X,
            (y == "excellent").astype(int),
            entity=entity,
        )
    with pytest.raises(ValueError, match="No entity crosses"):
        FixedEffectsOrderedLogit().fit(
            X,
            np.repeat(np.resize(np.array(["a", "b", "c", "d"]), 35), 6),
            entity=entity,
            category_order=["a", "b", "c", "d"],
        )


def test_buc_prediction_schema_is_strict_for_the_identified_linear_index():
    X, y, entity, _ = _panel(n_entities=35)
    result = FixedEffectsOrderedLogit().fit(X, y, entity=entity)
    with pytest.raises(ValueError, match="columns must match"):
        result.linear_index(X[["x2", "x1"]])
    with pytest.raises(ValueError, match="expected"):
        result.linear_index(np.ones((3, 1)))
