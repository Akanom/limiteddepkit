import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr, ndtr
from statsmodels.miscmodels.ordinal_model import OrderedModel

from limiteddepkit import (
    RandomEffectsOrderedProbit,
    RandomEffectsOrderedProbitResult,
    posterior_predict_proba,
    posterior_random_effects,
    simulate_random_effects_ordered_probit,
)
from limiteddepkit.panel_ordinal import _selected_log_probabilities


@pytest.fixture(scope="module")
def fitted_probit_panel():
    simulated = simulate_random_effects_ordered_probit(
        n_entities=90,
        n_periods=5,
        seed=4_119,
        sigma_entity=0.75,
    )
    result = RandomEffectsOrderedProbit().fit(
        simulated.X,
        simulated.y,
        entity=simulated.entity,
        quadrature_points=12,
    )
    return simulated, result


def test_random_effects_ordered_probit_contract(fitted_probit_panel):
    simulated, result = fitted_probit_panel

    assert isinstance(result, RandomEffectsOrderedProbitResult)
    assert result.link == "probit"
    assert result.converged
    assert result.inference_valid
    assert result.scaled_score_norm <= 1e-5
    assert result.nobs == simulated.nobs
    assert result.n_entities == simulated.n_entities
    assert result.backend == "native-ghq"
    assert result.covariance_type == "observed-information"
    assert result.sigma_entity > 0
    assert result.params["x1"] > 0
    assert result.params["x2"] < 0
    assert np.all(np.isfinite(result.standard_errors))
    assert np.isnan(result.zstats["sigma_entity"])
    assert np.isnan(result.pvalues["sigma_entity"])


def test_loose_tolerance_cannot_certify_a_nonstationary_re_probit_fit():
    simulated = simulate_random_effects_ordered_probit(
        n_entities=60,
        n_periods=4,
        seed=4_119,
        sigma_entity=0.75,
    )
    result = RandomEffectsOrderedProbit().fit(
        simulated.X,
        simulated.y,
        entity=simulated.entity,
        quadrature_points=8,
        tolerance=1e6,
    )

    assert result.converged
    assert result.inference_valid
    assert result.scaled_score_norm <= 1e-4
    assert result.optimizer_result.nit > 1


def test_population_averaged_probit_has_closed_form_identity(fitted_probit_panel):
    simulated, result = fitted_probit_panel
    sample = simulated.X.iloc[:25]
    actual = result.predict_proba(sample).to_numpy()

    eta = sample.to_numpy() @ result.params.to_numpy()
    scale = np.sqrt(1.0 + result.sigma_entity**2)
    cumulative = ndtr(
        (result.thresholds.to_numpy()[None, :] - eta[:, None]) / scale
    )
    expected = np.diff(
        np.column_stack([np.zeros(len(sample)), cumulative, np.ones(len(sample))]),
        axis=1,
    )

    assert actual == pytest.approx(expected, abs=2e-5)


def test_conditional_kernel_matches_statsmodels_ordered_probit_likelihood(
    fitted_probit_panel,
):
    simulated, result = fitted_probit_panel
    probabilities = result.predict_proba(simulated.X, random_effects=0.0).to_numpy()
    y = simulated.y.to_numpy(dtype=int)
    limiteddepkit_loglike = np.log(probabilities[np.arange(len(y)), y]).sum()

    raw_thresholds = np.r_[
        result.thresholds.iloc[0],
        np.log(np.diff(result.thresholds.to_numpy())),
    ]
    statsmodels = OrderedModel(y, simulated.X, distr="probit")
    statsmodels_loglike = statsmodels.loglike(
        np.r_[result.params.to_numpy(), raw_thresholds]
    )

    assert limiteddepkit_loglike == pytest.approx(statsmodels_loglike, abs=1e-9)


def test_conditional_probabilities_match_normal_cdf(fitted_probit_panel):
    simulated, result = fitted_probit_panel
    sample = simulated.X.iloc[:20]
    random_effects = np.linspace(-0.6, 0.6, len(sample))
    actual = result.predict_proba(
        sample, random_effects=random_effects
    ).to_numpy()

    eta = sample.to_numpy() @ result.params.to_numpy() + random_effects
    cumulative = ndtr(result.thresholds.to_numpy()[None, :] - eta[:, None])
    expected = np.diff(
        np.column_stack([np.zeros(len(sample)), cumulative, np.ones(len(sample))]),
        axis=1,
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_probit_posterior_likelihood_and_prediction_identity(fitted_probit_panel):
    simulated, result = fitted_probit_panel
    posterior = posterior_random_effects(
        result,
        simulated.X,
        simulated.y,
        entity=simulated.entity,
    )
    predicted = posterior_predict_proba(
        result,
        simulated.X.iloc[:20],
        entity=simulated.entity.iloc[:20],
        posterior=posterior,
    )

    assert posterior["log_marginal_likelihood"].sum() == pytest.approx(
        result.loglike, abs=1e-7
    )
    assert all(np.sum(weights) == pytest.approx(1.0) for weights in posterior["posterior_weights"])
    assert np.all(predicted.to_numpy() >= 0.0)
    assert np.allclose(predicted.sum(axis=1), 1.0)


def test_probit_simulation_supports_unbalanced_panels_and_labels():
    simulated = simulate_random_effects_ordered_probit(
        n_entities=12,
        n_periods=6,
        minimum_periods=2,
        unbalanced=True,
        seed=911,
        feature_names=("income",),
        coefficients=(0.4,),
    )

    assert not simulated.is_balanced
    assert simulated.n_entities == 12
    assert list(simulated.X.columns) == ["income"]
    assert simulated.group_sizes.between(2, 6).all()
    assert set(pd.unique(simulated.y)) == {0, 1, 2}


def test_probit_middle_category_log_probability_is_stable_in_extreme_tail():
    actual = _selected_log_probabilities(
        np.array([[0.0]]),
        np.array([1]),
        np.array([0.0]),
        np.array([40.0, 42.0]),
        0.0,
        "probit",
    )[0]
    log_survival_lower = log_ndtr(-40.0)
    log_survival_upper = log_ndtr(-42.0)
    expected = log_survival_lower + np.log1p(
        -np.exp(log_survival_upper - log_survival_lower)
    )

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-12)
