"""Numerical-stability checks for dynamic random-effects Ordered Logit."""

import numpy as np
import pytest

from limiteddepkit import (
    DynamicRandomEffectsOrderedLogit,
    simulate_dynamic_random_effects_ordered_logit,
)

pytestmark = pytest.mark.simulation


@pytest.fixture(scope="module")
def deterministic_dynamic_panel():
    return simulate_dynamic_random_effects_ordered_logit(
        n_entities=60,
        n_periods=6,
        seed=8_263,
    )


@pytest.fixture(scope="module")
def reference_fit(deterministic_dynamic_panel):
    simulation = deterministic_dynamic_panel
    return DynamicRandomEffectsOrderedLogit().fit(
        simulation.X,
        simulation.y,
        entity=simulation.entity,
        time=simulation.time,
        quadrature_points=12,
        maxiter=800,
    )


def test_gauss_hermite_quadrature_converges(deterministic_dynamic_panel, reference_fit):
    simulation = deterministic_dynamic_panel
    high_order = DynamicRandomEffectsOrderedLogit().fit(
        simulation.X,
        simulation.y,
        entity=simulation.entity,
        time=simulation.time,
        quadrature_points=20,
        maxiter=800,
    )

    assert reference_fit.converged
    assert high_order.converged
    assert high_order.loglike == pytest.approx(reference_fit.loglike, abs=1e-6)
    assert high_order.all_params.to_numpy() == pytest.approx(
        reference_fit.all_params.to_numpy(), abs=2e-5
    )
    assert high_order.fitted_probabilities.iloc[:30].to_numpy() == pytest.approx(
        reference_fit.fitted_probabilities.iloc[:30].to_numpy(), abs=5e-6
    )


def test_fit_is_invariant_to_row_order(deterministic_dynamic_panel, reference_fit):
    simulation = deterministic_dynamic_panel
    permutation = np.random.default_rng(2_026).permutation(len(simulation.y))
    reordered = DynamicRandomEffectsOrderedLogit().fit(
        simulation.X.iloc[permutation].reset_index(drop=True),
        simulation.y.iloc[permutation].reset_index(drop=True),
        entity=simulation.entity.iloc[permutation].reset_index(drop=True),
        time=simulation.time.iloc[permutation].reset_index(drop=True),
        quadrature_points=12,
        maxiter=800,
    )

    assert reordered.converged
    assert reordered.loglike == pytest.approx(reference_fit.loglike, abs=1e-7)
    assert reordered.all_params.to_numpy() == pytest.approx(
        reference_fit.all_params.to_numpy(), abs=2e-6
    )


def test_fit_is_invariant_to_arbitrary_string_entity_labels(
    deterministic_dynamic_panel, reference_fit
):
    simulation = deterministic_dynamic_panel
    shuffled_codes = np.random.default_rng(99).permutation(simulation.n_entities)
    entity_labels = np.asarray(
        [f"panel-{shuffled_codes[group]:03d}-z" for group in simulation.entity]
    )
    relabelled = DynamicRandomEffectsOrderedLogit().fit(
        simulation.X,
        simulation.y,
        entity=entity_labels,
        time=simulation.time,
        quadrature_points=12,
        maxiter=800,
    )

    assert relabelled.converged
    assert relabelled.loglike == pytest.approx(reference_fit.loglike, abs=1e-7)
    assert relabelled.all_params.to_numpy() == pytest.approx(
        reference_fit.all_params.to_numpy(), abs=1e-5
    )


def test_fit_is_invariant_to_shifted_time_origin(deterministic_dynamic_panel, reference_fit):
    simulation = deterministic_dynamic_panel
    shifted = DynamicRandomEffectsOrderedLogit().fit(
        simulation.X,
        simulation.y,
        entity=simulation.entity,
        time=2_000 + simulation.time,
        quadrature_points=12,
        maxiter=800,
    )

    assert shifted.converged
    assert shifted.loglike == pytest.approx(reference_fit.loglike, abs=1e-7)
    assert shifted.all_params.to_numpy() == pytest.approx(
        reference_fit.all_params.to_numpy(), abs=2e-6
    )
