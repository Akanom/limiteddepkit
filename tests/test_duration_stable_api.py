"""Stable duration API, delayed-entry, weighting, and covariance tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaincc
from scipy.stats import gamma as gamma_distribution
from scipy.stats import weibull_min

from limiteddepkit._duration import log_gammaincc
from limiteddepkit.discrete_time_duration import (
    DiscreteTimeDuration,
    GeometricDuration,
)
from limiteddepkit.exponential_duration import ExponentialDuration
from limiteddepkit.gamma_duration import GammaDuration
from limiteddepkit.weibull_duration import WeibullDuration


def _continuous_duration_sample(seed=9142, nobs=260):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"const": 1.0, "x": rng.normal(size=nobs)},
        index=pd.Index([f"spell-{row}" for row in range(nobs)], name="spell"),
    )
    scale = np.exp(X.to_numpy() @ np.array([0.25, -0.3]))
    latent = weibull_min.rvs(1.4, scale=scale, random_state=rng)
    censoring = rng.exponential(2.5, size=nobs)
    duration = np.minimum(latent, censoring)
    event = (latent <= censoring).astype(int)
    return X, duration, event


@pytest.mark.parametrize(
    "model_type",
    [ExponentialDuration, WeibullDuration, GammaDuration],
)
def test_continuous_duration_prediction_surface_is_labeled_and_coherent(model_type):
    X, duration, event = _continuous_duration_sample()
    result = model_type().fit(X, duration, event)
    subset = X.iloc[:8]
    times = np.array([0.25, 0.75, 1.5])

    mean = result.predict_mean(subset)
    survival = result.predict_survival(subset, times)
    cumulative = result.predict_cumulative_hazard(subset, times)
    hazard = result.predict_hazard(subset, times)
    median = result.predict_quantile(subset, 0.5)

    assert mean.index.equals(subset.index)
    assert survival.index.equals(subset.index)
    assert survival.columns.equals(pd.Index(times, name="time"))
    np.testing.assert_allclose(survival, np.exp(-cumulative), rtol=2e-12, atol=2e-12)
    assert np.all(np.diff(survival.to_numpy(), axis=1) <= 0.0)
    assert np.all(hazard.to_numpy() >= 0.0)
    assert np.all(median.to_numpy() > 0.0)
    assert result.all_params.index.equals(result.covariance.index)
    assert result.summary_frame().index.equals(result.all_params.index)
    assert result.vcov().equals(result.covariance)
    assert np.isfinite([result.aic, result.bic]).all()

    with pytest.raises(ValueError, match="columns must match"):
        result.predict_mean(subset[["x", "const"]])
    with pytest.raises(ValueError, match="strictly between"):
        result.predict_quantile(subset, 1.0)


def test_exponential_delayed_entry_and_frequency_weights_match_manual_likelihood():
    rng = np.random.default_rng(4407)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=180)})
    scale = np.exp(X.to_numpy() @ np.array([0.3, -0.25]))
    entry = rng.uniform(0.0, 0.6, size=len(X))
    residual = rng.exponential(scale=scale)
    latent = entry + residual
    censoring = entry + rng.exponential(2.0, size=len(X))
    duration = np.minimum(latent, censoring)
    event = (latent <= censoring).astype(int)
    frequency = rng.integers(1, 4, size=len(X))

    result = ExponentialDuration().fit(
        X,
        duration,
        event,
        entry=entry,
        frequency_weights=frequency,
    )
    eta = X.to_numpy() @ result.params.to_numpy()
    expected = -np.sum(
        frequency * (event * eta + (duration - entry) * np.exp(-eta))
    )

    assert result.loglike == pytest.approx(expected, abs=1e-9)
    assert result.frequency_weight_sum == pytest.approx(float(frequency.sum()))
    assert result.n_delayed_entry == len(X)


def test_exponential_robust_and_cluster_covariance_match_sandwich_identities():
    X, duration, event = _continuous_duration_sample(seed=5508, nobs=240)
    robust = ExponentialDuration().fit(
        X,
        duration,
        event,
        covariance_type="robust",
    )
    clusters = np.repeat(np.arange(40), 6)
    clustered = ExponentialDuration().fit(
        X,
        duration,
        event,
        covariance_type="cluster",
        clusters=clusters,
    )

    design = X.to_numpy()
    eta = design @ robust.params.to_numpy()
    exposure = duration * np.exp(-eta)
    bread = np.linalg.inv(design.T @ (exposure[:, None] * design))
    scores = design * (exposure - event)[:, None]
    hc1 = len(X) / (len(X) - design.shape[1])
    expected_robust = hc1 * bread @ (scores.T @ scores) @ bread
    np.testing.assert_allclose(robust.covariance, expected_robust, rtol=2e-7, atol=2e-9)

    cluster_scores = np.vstack([scores[clusters == group].sum(axis=0) for group in range(40)])
    correction = 40 / 39 * (len(X) - 1) / (len(X) - design.shape[1])
    expected_cluster = correction * bread @ (cluster_scores.T @ cluster_scores) @ bread
    np.testing.assert_allclose(
        clustered.covariance,
        expected_cluster,
        rtol=2e-7,
        atol=2e-9,
    )
    assert robust.covariance_type == "sandwich"
    assert clustered.covariance_type == "cluster-sandwich"
    assert clustered.n_clusters == 40


def test_weibull_and_gamma_delayed_entry_likelihoods_condition_on_survival():
    rng = np.random.default_rng(9074)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=320)})
    entry = rng.uniform(0.01, 0.4, size=len(X))
    scale = np.exp(0.2 - 0.25 * X["x"].to_numpy())

    weibull_latent = scale * rng.weibull(1.5, size=len(X))
    weibull_latent = np.maximum(weibull_latent, entry + 0.01)
    censor = entry + rng.exponential(2.0, size=len(X))
    weibull_duration = np.minimum(weibull_latent, censor)
    weibull_event = (weibull_latent <= censor).astype(int)
    weibull = WeibullDuration().fit(
        X,
        weibull_duration,
        weibull_event,
        entry=entry,
    )
    fitted_scale = np.exp(X.to_numpy() @ weibull.params.to_numpy())
    expected_weibull = np.sum(
        weibull_event
        * weibull_min.logpdf(
            weibull_duration,
            weibull.shape_param,
            scale=fitted_scale,
        )
        + (1 - weibull_event)
        * weibull_min.logsf(
            weibull_duration,
            weibull.shape_param,
            scale=fitted_scale,
        )
        - weibull_min.logsf(entry, weibull.shape_param, scale=fitted_scale)
    )
    assert weibull.loglike == pytest.approx(expected_weibull, abs=2e-7)

    gamma_latent = entry + rng.gamma(1.8, scale=scale)
    censor = entry + rng.exponential(3.0, size=len(X))
    gamma_duration = np.minimum(gamma_latent, censor)
    gamma_event = (gamma_latent <= censor).astype(int)
    gamma_result = GammaDuration().fit(
        X,
        gamma_duration,
        gamma_event,
        entry=entry,
    )
    fitted_scale = np.exp(X.to_numpy() @ gamma_result.params.to_numpy())
    scaled_duration = gamma_duration / fitted_scale
    scaled_entry = entry / fitted_scale
    expected_gamma = np.sum(
        gamma_event
        * gamma_distribution.logpdf(
            gamma_duration,
            gamma_result.shape_param,
            scale=fitted_scale,
        )
        + (1 - gamma_event)
        * np.log(gammaincc(gamma_result.shape_param, scaled_duration))
        - np.log(gammaincc(gamma_result.shape_param, scaled_entry))
    )
    assert gamma_result.loglike == pytest.approx(expected_gamma, abs=2e-7)


def test_discrete_duration_entry_and_prediction_grid_contract():
    rng = np.random.default_rng(6084)
    X = pd.DataFrame(
        {"const": 1.0, "x": rng.normal(size=220)},
        index=pd.Index([f"id-{row}" for row in range(220)]),
    )
    entry = rng.integers(0, 3, size=len(X))
    hazard = 1.0 / (1.0 + np.exp(-(0.15 - 0.3 * X["x"].to_numpy())))
    residual = rng.geometric(hazard)
    duration = entry + np.minimum(residual, 8)
    event = (residual <= 8).astype(int)

    result = DiscreteTimeDuration().fit(
        X,
        duration,
        event,
        entry_period=entry,
    )
    subset = X.iloc[:7]
    survival = result.predict_survival(subset, [0, 1, 3])
    fitted_hazard = result.predict_hazard(subset)

    np.testing.assert_allclose(
        survival.to_numpy(),
        (1.0 - fitted_hazard.to_numpy()[:, None]) ** np.array([0, 1, 3]),
        atol=1e-14,
    )
    assert result.predict_quantile(subset, 0.5).dtype.kind in "iu"
    assert result.n_delayed_entry == int(np.count_nonzero(entry))
    assert result.predict_mean(subset).index.equals(subset.index)


def test_duration_covariance_and_entry_options_reject_invalid_inputs():
    X, duration, event = _continuous_duration_sample(nobs=60)
    with pytest.raises(ValueError, match="strictly smaller"):
        ExponentialDuration().fit(X, duration, event, entry=duration)
    with pytest.raises(ValueError, match="clusters is required"):
        ExponentialDuration().fit(X, duration, event, covariance_type="cluster")
    with pytest.raises(ValueError, match="only with"):
        ExponentialDuration().fit(
            X,
            duration,
            event,
            covariance_type="robust",
            clusters=np.arange(len(X)),
        )
    with pytest.raises(ValueError, match="non-negative"):
        ExponentialDuration().fit(
            X,
            duration,
            event,
            frequency_weights=np.r_[-1.0, np.ones(len(X) - 1)],
        )


@pytest.mark.parametrize(
    "model_type",
    [GeometricDuration, ExponentialDuration, WeibullDuration, GammaDuration],
)
@pytest.mark.parametrize("covariance_type", ["observed", "robust", "cluster"])
def test_duration_frequency_weights_match_literal_row_replication(
    model_type, covariance_type
):
    X, duration, event = _continuous_duration_sample(seed=781, nobs=48)
    rng = np.random.default_rng(401)
    frequency = rng.integers(1, 4, size=len(X))
    clusters = np.repeat(np.arange(12), 4)
    if model_type is GeometricDuration:
        hazard = 1.0 / (1.0 + np.exp(-(0.2 - 0.3 * X["x"].to_numpy())))
        latent = rng.geometric(hazard)
        duration = np.minimum(latent, 8)
        event = (latent <= 8).astype(int)

    expanded = np.repeat(np.arange(len(X)), frequency)
    weighted_kwargs = {"covariance_type": covariance_type}
    expanded_kwargs = {"covariance_type": covariance_type}
    if covariance_type == "cluster":
        weighted_kwargs["clusters"] = clusters
        expanded_kwargs["clusters"] = clusters[expanded]
    weighted = model_type().fit(
        X,
        duration,
        event,
        frequency_weights=frequency,
        **weighted_kwargs,
    )
    repeated = model_type().fit(
        X.iloc[expanded].reset_index(drop=True),
        duration[expanded],
        event[expanded],
        **expanded_kwargs,
    )

    np.testing.assert_allclose(weighted.all_params, repeated.all_params, atol=8e-7)
    assert weighted.loglike == pytest.approx(repeated.loglike, abs=2e-8)
    np.testing.assert_allclose(weighted.covariance, repeated.covariance, atol=3e-6)
    assert weighted.n_events == repeated.n_events
    assert weighted.effective_nobs == len(expanded)
    assert weighted.df_resid == repeated.df_resid
    assert weighted.bic == pytest.approx(repeated.bic, abs=5e-8)


@pytest.mark.parametrize(
    "model_type",
    [GeometricDuration, ExponentialDuration, WeibullDuration, GammaDuration],
)
def test_zero_weight_events_cannot_supply_duration_identification(model_type):
    X = pd.DataFrame({"const": 1.0, "x": np.linspace(-1.0, 1.0, 8)})
    duration = np.arange(1, 9, dtype=float)
    event = np.zeros(8, dtype=int)
    event[0] = 1
    frequency = np.ones(8, dtype=int)
    frequency[0] = 0

    with pytest.raises(ValueError, match="positive frequency weight"):
        model_type().fit(X, duration, event, frequency_weights=frequency)


def test_duration_frequency_and_cluster_validation_respects_active_rows():
    X, duration, event = _continuous_duration_sample(seed=291, nobs=48)
    frequency = np.ones(len(X), dtype=int)
    frequency[0] = 0
    clusters = np.repeat(np.arange(8), 6).astype(object)
    clusters[0] = None
    result = ExponentialDuration().fit(
        X,
        duration,
        event,
        frequency_weights=frequency,
        covariance_type="cluster",
        clusters=clusters,
    )
    assert result.n_clusters == 8

    with pytest.raises(ValueError, match="non-negative integers"):
        ExponentialDuration().fit(
            X,
            duration,
            event,
            frequency_weights=np.full(len(X), 1.5),
        )


def test_duration_reserved_and_duplicate_feature_names_are_rejected():
    duration = np.arange(1.0, 9.0)
    event = np.resize([1, 0], 8)
    duplicate = pd.DataFrame(
        np.column_stack([np.linspace(-1.0, 1.0, 8), np.ones(8)]),
        columns=["x", "x"],
    )
    with pytest.raises(ValueError, match="feature names must be unique"):
        ExponentialDuration().fit(duplicate, duration, event)
    with pytest.raises(ValueError, match="reserved for Weibull shape"):
        WeibullDuration().fit(
            pd.DataFrame({"const": 1.0, "log_alpha": np.linspace(-1.0, 1.0, 8)}),
            duration,
            event,
        )
    with pytest.raises(ValueError, match="reserved for Gamma shape"):
        GammaDuration().fit(
            pd.DataFrame({"const": 1.0, "log_k": np.linspace(-1.0, 1.0, 8)}),
            duration,
            event,
        )


def test_geometric_extreme_quantile_and_period_compatibility_alias():
    rng = np.random.default_rng(19)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=40)})
    result = GeometricDuration().fit(
        X,
        rng.integers(1, 6, size=len(X)),
        np.resize([1, 0], len(X)),
    )
    pd.testing.assert_series_equal(
        result.predict_survival(X.iloc[:4], 3),
        result.predict_survival(X.iloc[:4], period=3),
    )
    extreme = replace(
        result,
        params=pd.Series([-1_000.0, 0.0], index=result.params.index, name="coef"),
    )
    assert np.isinf(extreme.predict_quantile(X.iloc[:1], 0.5).iloc[0])


def test_gamma_upper_tail_predictions_remain_log_scale_accurate():
    X, duration, event = _continuous_duration_sample(seed=821, nobs=60)
    fitted = GammaDuration().fit(X, duration, event)
    standardized = replace(
        fitted,
        params=pd.Series([0.0, 0.0], index=fitted.params.index, name="coef"),
        shape_param=2.0,
    )
    row = X.iloc[:1]
    cumulative = standardized.predict_cumulative_hazard(row, 1_000.0).iloc[0]
    hazard = standardized.predict_hazard(row, 1_000.0).iloc[0]
    assert cumulative == pytest.approx(1_000.0 - np.log(1_001.0), abs=2e-11)
    assert hazard == pytest.approx(1_000.0 / 1_001.0, abs=2e-12)
    assert log_gammaincc(1.0, np.array([1_000.0]))[0] == pytest.approx(
        -1_000.0, abs=2e-12
    )


@pytest.mark.parametrize("model_type", [WeibullDuration, GammaDuration])
def test_coarse_optimizer_tolerance_cannot_certify_nonstationary_inference(model_type):
    X, duration, event = _continuous_duration_sample(seed=33, nobs=120)
    result = model_type().fit(X, duration, event, tolerance=0.2)
    assert result.scaled_score_norm > 1e-4
    assert not result.converged
    assert not result.inference_valid
