import numpy as np
import pandas as pd
import pytest

import limiteddepkit.count as count
from limiteddepkit.experimental import HurdlePoisson, ZeroInflatedPoisson


def _poisson_data(seed=410, nobs=260):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"row-{i}" for i in range(nobs)], name="observation")
    X = pd.DataFrame(
        {"const": 1.0, "x": rng.normal(size=nobs)},
        index=index,
    )
    offset = pd.Series(rng.normal(scale=0.12, size=nobs), index=index)
    exposure = pd.Series(rng.uniform(0.5, 2.0, size=nobs), index=index)
    mean = np.exp(X.to_numpy() @ np.array([-0.15, 0.45]) + offset) * exposure
    y = pd.Series(rng.poisson(mean), index=index)
    return X, y, offset, exposure


def _nb2_data(seed=771, nobs=340):
    X, _, offset, exposure = _poisson_data(seed, nobs)
    rng = np.random.default_rng(seed + 1)
    alpha = 0.65
    mean = np.exp(X.to_numpy() @ np.array([0.1, -0.3]) + offset) * exposure
    y = pd.Series(
        rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean)),
        index=X.index,
    )
    return X, y, offset, exposure


def test_count_namespace_contains_only_promoted_foundations():
    assert set(count.__all__) == {
        "NegativeBinomial",
        "NegativeBinomialNB2",
        "NegativeBinomialResult",
        "PoissonRegressor",
        "PoissonResult",
    }
    assert count.NegativeBinomialNB2 is count.NegativeBinomial
    assert "ZeroInflatedPoisson" not in count.__all__
    assert "HurdlePoisson" not in count.__all__
    assert ZeroInflatedPoisson is not None
    assert HurdlePoisson is not None


@pytest.mark.parametrize("estimator,data_factory", [
    (count.PoissonRegressor(), _poisson_data),
    (count.NegativeBinomial(), _nb2_data),
])
def test_count_result_contract_and_schema_safe_indexed_prediction(estimator, data_factory):
    X, y, offset, exposure = data_factory()
    result = estimator.fit(X, y, offset=offset, exposure=exposure)

    assert result.converged
    assert result.inference_valid
    assert result.all_params.index.equals(result.standard_errors.index)
    assert result.covariance.index.equals(result.all_params.index)
    assert result.covariance.columns.equals(result.all_params.index)
    pd.testing.assert_frame_equal(result.vcov(), result.covariance)
    assert result.summary_frame().columns.tolist() == ["coef", "std_err", "z", "p_value"]
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert result.df_resid > 0
    assert result.backend == "native-mle"
    assert result.diagnostics().loc["converged"]
    assert np.isfinite(result.scaled_score_norm)
    assert result.scaled_score_norm <= 1e-5

    new_index = pd.Index(["case-c", "case-a", "case-b"])
    new_X = pd.DataFrame(
        {"const": 1.0, "x": [0.2, -0.3, 0.7]},
        index=new_index,
    )
    predictions = result.predict(
        new_X,
        offset=pd.Series([0.1, -0.1, 0.0], index=new_index),
        exposure=pd.Series([2.0, 1.5, 0.75], index=new_index),
    )
    assert predictions.index.equals(new_index)
    assert np.all(predictions > 0.0)

    with pytest.raises(ValueError, match="columns must match"):
        result.predict(new_X[["x", "const"]])
    with pytest.raises(ValueError, match="offset index"):
        result.predict(new_X, offset=pd.Series([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        result.predict(new_X, exposure=[1.0, 0.0, 1.0])


def test_poisson_frequency_weights_are_exact_row_replication():
    X, y, offset, exposure = _poisson_data(nobs=180)
    rng = np.random.default_rng(6)
    weights = pd.Series(rng.integers(0, 4, size=len(X)), index=X.index)
    weighted = count.PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
    )
    repeated = np.repeat(np.arange(len(X)), weights.to_numpy())
    expanded = count.PoissonRegressor().fit(
        X.iloc[repeated].reset_index(drop=True),
        y.iloc[repeated].reset_index(drop=True),
        offset=offset.iloc[repeated].reset_index(drop=True),
        exposure=exposure.iloc[repeated].reset_index(drop=True),
    )

    np.testing.assert_allclose(weighted.params, expanded.params, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(weighted.covariance, expanded.covariance, rtol=2e-7, atol=1e-9)
    assert weighted.loglike == pytest.approx(expanded.loglike, abs=1e-9)
    assert weighted.weighted_nobs == len(repeated)
    assert weighted.df_resid == expanded.df_resid


@pytest.mark.parametrize("estimator,data_factory", [
    (count.PoissonRegressor(), _poisson_data),
    (count.NegativeBinomial(), _nb2_data),
])
def test_frequency_weights_cannot_remove_every_positive_count(estimator, data_factory):
    X, y, _, _ = data_factory(nobs=100)
    weights = pd.Series((y == 0).astype(int), index=X.index)
    with pytest.raises(ValueError, match="positive count must have positive weight"):
        estimator.fit(X, y, freq_weights=weights)


def test_nb2_frequency_weights_match_expansion_and_preserve_dispersion():
    X, y, offset, exposure = _nb2_data(nobs=220)
    rng = np.random.default_rng(9)
    weights = pd.Series(rng.integers(1, 4, size=len(X)), index=X.index)
    weighted = count.NegativeBinomial().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
    )
    repeated = np.repeat(np.arange(len(X)), weights.to_numpy())
    expanded = count.NegativeBinomial().fit(
        X.iloc[repeated].reset_index(drop=True),
        y.iloc[repeated].reset_index(drop=True),
        offset=offset.iloc[repeated].reset_index(drop=True),
        exposure=exposure.iloc[repeated].reset_index(drop=True),
    )

    np.testing.assert_allclose(weighted.all_params, expanded.all_params, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        weighted.covariance,
        expanded.covariance,
        rtol=2e-5,
        atol=2e-7,
    )
    assert weighted.loglike == pytest.approx(expanded.loglike, abs=2e-6)


def test_analytic_weights_have_explicit_pseudolikelihood_information_criterion_boundary():
    X, y, offset, exposure = _poisson_data()
    weights = pd.Series(np.linspace(0.5, 1.5, len(X)), index=X.index)
    result = count.PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        analytic_weights=weights,
    )

    assert result.weight_type == "analytic"
    assert np.isnan(result.aic)
    assert np.isnan(result.bic)
    assert np.isfinite(result.loglike)


@pytest.mark.parametrize("estimator,data_factory", [
    (count.PoissonRegressor(), _poisson_data),
    (count.NegativeBinomial(), _nb2_data),
])
def test_count_models_validate_weights_covariance_and_alignment(estimator, data_factory):
    X, y, offset, exposure = data_factory(nobs=80)
    ones = pd.Series(1, index=X.index)

    with pytest.raises(ValueError, match="only one"):
        estimator.fit(X, y, freq_weights=ones, analytic_weights=ones)
    with pytest.raises(ValueError, match="non-negative integers"):
        estimator.fit(X, y, freq_weights=np.full(len(X), 1.5))
    with pytest.raises(ValueError, match="strictly positive"):
        estimator.fit(X, y, analytic_weights=np.zeros(len(X)))
    with pytest.raises(ValueError, match="clusters is required"):
        estimator.fit(X, y, cov_type="cluster")
    with pytest.raises(ValueError, match="only when"):
        estimator.fit(X, y, cov_type="HC0", clusters=np.arange(len(X)))
    with pytest.raises(ValueError, match="y index"):
        estimator.fit(X, y.reset_index(drop=True), offset=offset, exposure=exposure)


def test_cluster_and_hc_covariances_report_their_contracts():
    X, y, offset, exposure = _poisson_data(nobs=240)
    clusters = pd.Series(np.repeat(np.arange(60), 4), index=X.index)
    clustered = count.PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        cov_type="cluster",
        clusters=clusters,
    )
    hc1 = count.PoissonRegressor().fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        cov_type="HC1",
    )

    assert clustered.covariance_type == "cluster"
    assert clustered.n_clusters == 60
    assert hc1.covariance_type == "HC1"
    assert np.isfinite(clustered.covariance.to_numpy()).all()
    assert np.isfinite(hc1.covariance.to_numpy()).all()


@pytest.mark.parametrize("estimator,data_factory", [
    (count.PoissonRegressor(), _poisson_data),
    (count.NegativeBinomial(), _nb2_data),
])
def test_zero_frequency_rows_do_not_require_cluster_labels(estimator, data_factory):
    X, y, offset, exposure = data_factory(nobs=240)
    weights = pd.Series(1, index=X.index)
    weights.iloc[0] = 0
    clusters = pd.Series(np.repeat(np.arange(60), 4), index=X.index, dtype=object)
    clusters.iloc[0] = None
    offset.iloc[0] = 1_000.0

    result = estimator.fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        freq_weights=weights,
        cov_type="cluster",
        clusters=clusters,
    )
    assert result.inference_valid
    assert result.n_clusters == 60
    assert np.isinf(result.fitted_values.iloc[0])

    clusters.iloc[1] = None
    with pytest.raises(ValueError, match="positive-weight rows"):
        estimator.fit(
            X,
            y,
            freq_weights=weights,
            cov_type="cluster",
            clusters=clusters,
        )


def test_nb2_reserves_its_dispersion_parameter_label():
    X, y, _, _ = _nb2_data(nobs=100)
    X = X.rename(columns={"x": "log_alpha"})
    with pytest.raises(ValueError, match="reserved for NB2 dispersion"):
        count.NegativeBinomial().fit(X, y)


@pytest.mark.parametrize("estimator,data_factory", [
    (count.PoissonRegressor(), _poisson_data),
    (count.NegativeBinomial(), _nb2_data),
])
def test_optimizer_status_cannot_bypass_stationarity_gate(estimator, data_factory):
    X, y, offset, exposure = data_factory(nobs=300)
    result = estimator.fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        maxiter=1,
        tolerance=1e-12,
    )
    assert not result.converged
    assert not result.inference_valid
    assert result.scaled_score_norm > 1e-5


@pytest.mark.parametrize("estimator,data_factory", [
    (count.PoissonRegressor(), _poisson_data),
    (count.NegativeBinomial(), _nb2_data),
])
def test_loose_tolerance_cannot_bypass_capped_stationarity_gate(estimator, data_factory):
    X, y, offset, exposure = data_factory(nobs=300)
    result = estimator.fit(
        X,
        y,
        offset=offset,
        exposure=exposure,
        tolerance=1e6,
    )

    assert result.scaled_score_norm > 1e-4
    assert not result.converged
    assert not result.inference_valid
