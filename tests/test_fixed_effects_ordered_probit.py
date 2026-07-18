"""Tests for split-panel-jackknife fixed-effects Ordered Probit."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.fixed_effects_ordinal import FixedEffectsOrderedProbit


def _balanced_panel(seed=20260715, n_entities=36, periods=8):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), periods)
    time = np.tile(np.arange(periods), n_entities)
    entity_level = rng.normal(scale=0.65, size=n_entities)
    x = rng.normal(size=n_entities * periods) + 0.30 * entity_level[entity]
    beta = np.array([0.8])
    y = np.empty(len(entity), dtype=int)
    for code in range(n_entities):
        rows = entity == code
        for _ in range(1_000):
            latent = beta[0] * x[rows] + entity_level[code] + rng.normal(size=periods)
            candidate = np.digitize(latent, [-0.8, 0.15, 0.95])
            halves = (candidate, candidate[: periods // 2], candidate[periods // 2 :])
            if not any(np.all(part == 0) or np.all(part == 3) for part in halves):
                y[rows] = candidate
                break
        else:  # pragma: no cover - deterministic fixture guard
            raise RuntimeError("Could not simulate a finite-effect panel entity.")
    X = pd.DataFrame(
        {"x": x},
        index=pd.Index([f"row-{row}" for row in range(len(entity))]),
    )
    return X, y, entity, time, beta


def test_fe_probit_reports_exact_split_panel_correction_and_recovers_slope():
    X, y, entity, time, beta = _balanced_panel(n_entities=44)
    result = FixedEffectsOrderedProbit().fit(
        X,
        y,
        entity=entity,
        time=time,
        category_order=[0, 1, 2, 3],
    )

    half_average = result.half_panel_common_parameters.mean(axis=0)
    full = np.r_[
        result.uncorrected_params.to_numpy(),
        result.uncorrected_thresholds.to_numpy(),
    ]
    np.testing.assert_allclose(result.all_params, 2.0 * full - half_average, atol=1e-12)
    np.testing.assert_allclose(result.params, beta, atol=0.25)
    assert result.converged
    assert not result.inference_valid
    assert result.backend == "experimental-fe-probit-spj"
    assert result.bias_correction == "split-panel-jackknife"
    assert result.covariance_type == "none"
    assert result.n_periods == 8
    assert result.bootstrap_repetitions == result.bootstrap_successes == 0
    assert result.score_norms.max() < 1e-6
    assert np.isnan(result.standard_errors).all()
    assert np.diff(result.thresholds).min() > 0.0
    assert result.entity_effects.mean() == pytest.approx(0.0, abs=1e-12)
    assert result.summary_frame().index.equals(result.all_params.index)
    with pytest.raises(RuntimeError, match="successful entity bootstrap"):
        result.conf_int()


def test_fe_probit_loose_tolerance_cannot_certify_starting_values():
    X, y, entity, time, _ = _balanced_panel(n_entities=24)
    result = FixedEffectsOrderedProbit().fit(
        X,
        y,
        entity=entity,
        time=time,
        category_order=[0, 1, 2, 3],
        tolerance=1e6,
    )

    assert result.converged
    assert result.score_norms.max() <= 1e-5
    assert not np.allclose(result.all_params.to_numpy(), 0.0)


def test_fe_probit_known_entity_plugin_probabilities_preserve_schema():
    X, y, entity, time, _ = _balanced_panel(n_entities=28)
    labels = np.array([f"person-{value}" for value in entity])
    result = FixedEffectsOrderedProbit().fit(
        X,
        y,
        entity=labels,
        time=time + 2001,
        category_order=[0, 1, 2, 3],
    )

    probabilities = result.predict_proba(X.iloc[:6], entity=labels[:6])
    assert probabilities.index.equals(X.index[:6])
    assert probabilities.columns.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-13)
    assert np.all(probabilities.to_numpy() >= 0.0)
    assert result.linear_index(X.iloc[:6]).index.equals(X.index[:6])

    with pytest.raises(ValueError, match="No fitted fixed effect"):
        result.predict_proba(X.iloc[:1], entity=["new-person"])
    with pytest.raises(ValueError, match="columns must match"):
        result.predict_proba(X.rename(columns={"x": "z"}).iloc[:1], entity=labels[:1])
    with pytest.raises(ValueError, match="one label per prediction row"):
        result.predict_proba(X.iloc[:2], entity=labels[:1])


def test_fe_probit_is_invariant_to_row_order_and_entity_relabeling():
    X, y, entity, time, _ = _balanced_panel(n_entities=24)
    baseline = FixedEffectsOrderedProbit().fit(
        X,
        y,
        entity=entity,
        time=time,
        category_order=[0, 1, 2, 3],
    )
    rng = np.random.default_rng(913)
    order = rng.permutation(len(X))
    relabeled = np.array([f"unit-{13 * value + 7}" for value in entity])
    reordered = FixedEffectsOrderedProbit().fit(
        X.iloc[order],
        y[order],
        entity=relabeled[order],
        time=time[order] + 17,
        category_order=[0, 1, 2, 3],
    )

    np.testing.assert_allclose(reordered.params, baseline.params, atol=2e-6)
    np.testing.assert_allclose(reordered.thresholds, baseline.thresholds, atol=5e-4)
    np.testing.assert_allclose(
        np.diff(reordered.thresholds), np.diff(baseline.thresholds), atol=2e-6
    )


def test_fe_probit_entity_bootstrap_enables_inference_only_after_successes():
    X, y, entity, time, _ = _balanced_panel(n_entities=16)
    result = FixedEffectsOrderedProbit().fit(
        X,
        y,
        entity=entity,
        time=time,
        category_order=[0, 1, 2, 3],
        bootstrap_repetitions=20,
        random_state=1,
    )

    assert result.bootstrap_successes == 20
    assert result.inference_valid
    assert result.covariance_type == "entity-bootstrap"
    assert np.isfinite(result.vcov()).all().all()
    assert np.isfinite(result.conf_int()).all().all()


def test_fe_probit_rejects_panels_that_do_not_support_the_correction():
    X, y, entity, time, _ = _balanced_panel(n_entities=18)

    with pytest.raises(ValueError, match="even common panel length"):
        keep = time < 7
        FixedEffectsOrderedProbit().fit(
            X.loc[keep], y[keep], entity=entity[keep], time=time[keep]
        )
    with pytest.raises(ValueError, match="balanced common time grid"):
        keep = ~((entity == 0) & (time == 0))
        FixedEffectsOrderedProbit().fit(
            X.loc[keep], y[keep], entity=entity[keep], time=time[keep]
        )
    with pytest.raises(ValueError, match="duplicate time"):
        duplicate_time = time.copy()
        duplicate_time[1] = duplicate_time[0]
        FixedEffectsOrderedProbit().fit(
            X, y, entity=entity, time=duplicate_time
        )
    with pytest.raises(ValueError, match="within-entity design is rank deficient"):
        FixedEffectsOrderedProbit().fit(
            X.assign(constant=1.0), y, entity=entity, time=time
        )
    extreme = y.copy()
    extreme[entity == 0] = 3
    with pytest.raises(ValueError, match="finite entity-effect MLE"):
        FixedEffectsOrderedProbit().fit(
            X,
            extreme,
            entity=entity,
            time=time,
            category_order=[0, 1, 2, 3],
        )
    with pytest.raises(ValueError, match="zero or at least 20"):
        FixedEffectsOrderedProbit().fit(
            X,
            y,
            entity=entity,
            time=time,
            bootstrap_repetitions=10,
        )
