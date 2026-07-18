"""Stable API and inferential-contract tests for Firth Binary Logit."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

import limiteddepkit
import limiteddepkit.experimental as experimental
from limiteddepkit import small_sample


def separated_data() -> tuple[pd.DataFrame, np.ndarray]:
    design = pd.DataFrame(
        {"const": np.ones(9), "x": np.r_[np.zeros(4), np.ones(5)]},
        index=pd.Index(range(100, 109), name="case"),
    )
    outcomes = np.r_[np.zeros(4), np.ones(5)]
    return design, outcomes


def lbw_scale_synthetic_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Return a deterministic mixed-scale design resembling an LBW application."""
    rng = np.random.default_rng(0)
    nobs = 189
    design = pd.DataFrame(
        {
            "const": 1.0,
            "age10": rng.uniform(1.4, 4.6, nobs),
            "lwt100": rng.lognormal(np.log(1.3), 0.3, nobs),
            "smoke": rng.binomial(1, 0.4, nobs),
            "ht": rng.binomial(1, 0.12, nobs),
        }
    )
    linear_predictor = (
        1.6
        - 0.34 * design["age10"]
        - 1.59 * design["lwt100"]
        + 0.67 * design["smoke"]
        + 1.70 * design["ht"]
    )
    outcomes = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear_predictor)))
    return design, outcomes


def test_firth_is_stable_root_export_with_experimental_compatibility_alias() -> None:
    assert set(small_sample.__all__) == {"FirthBinaryLogit", "FirthBinaryLogitResult"}
    assert limiteddepkit.FirthBinaryLogit is small_sample.FirthBinaryLogit
    assert limiteddepkit.FirthBinaryLogitResult is small_sample.FirthBinaryLogitResult
    assert experimental.FirthBinaryLogit is small_sample.FirthBinaryLogit
    assert experimental.FirthBinaryLogitResult is small_sample.FirthBinaryLogitResult
    assert "RidgeBinaryLogit" not in small_sample.__all__
    assert not hasattr(small_sample, "RidgeBinaryLogit")


def test_profile_intervals_match_independent_separated_data_reference() -> None:
    design, outcomes = separated_data()
    result = small_sample.FirthBinaryLogit().fit(design, outcomes, tolerance=1e-10)

    intervals = result.conf_int(tolerance=1e-8)

    # firthmodels 0.7.2, NumPy backend, profile-likelihood intervals.
    expected = np.array(
        [
            [-7.08390716, 0.03971141],
            [1.28526156, 10.41467047],
        ]
    )
    assert intervals.to_numpy() == pytest.approx(expected, abs=2e-6)
    assert intervals.loc["x", "lower"] > 0.0
    assert not np.allclose(
        intervals.to_numpy(), result.conf_int(method="wald").to_numpy()
    )


def test_profile_bounds_invert_penalized_likelihood_ratio() -> None:
    design, outcomes = separated_data()
    result = small_sample.FirthBinaryLogit().fit(design, outcomes, tolerance=1e-10)
    intervals = result.conf_int(level=0.9, tolerance=1e-8)
    target = float(chi2.ppf(0.9, df=1))

    for parameter in result.params.index:
        profiles = result.profile_penalized_loglike(
            parameter,
            intervals.loc[parameter].to_numpy(),
            tolerance=1e-8,
        )
        deviances = 2.0 * (result.penalized_loglike - profiles.to_numpy())
        assert deviances == pytest.approx(np.repeat(target, 2), abs=2e-7)


def test_default_profiles_converge_on_mixed_scale_real_data_design() -> None:
    design, outcomes = lbw_scale_synthetic_data()
    result = small_sample.FirthBinaryLogit().fit(design, outcomes)

    intervals = result.conf_int()

    assert np.isfinite(intervals.to_numpy()).all()
    assert (intervals["lower"] < result.params).all()
    assert (result.params < intervals["upper"]).all()
    target = float(chi2.ppf(0.95, df=1))
    for parameter in result.params.index:
        profiles = result.profile_penalized_loglike(
            parameter,
            intervals.loc[parameter].to_numpy(),
        )
        deviances = 2.0 * (result.penalized_loglike - profiles.to_numpy())
        assert deviances == pytest.approx(np.repeat(target, 2), abs=2e-5)


def test_profile_api_validates_names_values_controls_and_methods() -> None:
    design, outcomes = separated_data()
    result = small_sample.FirthBinaryLogit().fit(design, outcomes)

    with pytest.raises(ValueError, match="Unknown parameter"):
        result.profile_penalized_loglike("missing", [0.0])
    with pytest.raises(IndexError, match="out of range"):
        result.profile_penalized_loglike(10, [0.0])
    with pytest.raises(ValueError, match="finite scalar"):
        result.profile_penalized_loglike("x", [np.nan])
    with pytest.raises(ValueError, match="method must"):
        result.conf_int(method="bootstrap")
    with pytest.raises(ValueError, match="positive integer"):
        result.conf_int(maxiter=0)
    with pytest.raises(ValueError, match="finite and positive"):
        result.conf_int(tolerance=0.0)


def test_loose_tolerance_cannot_certify_the_initial_firth_iterate() -> None:
    design, outcomes = separated_data()
    result = small_sample.FirthBinaryLogit().fit(design, outcomes, tolerance=1e6)

    assert result.converged
    assert result.inference_valid
    assert result.score_norm <= 1e-6
    assert not np.allclose(result.params.to_numpy(), 0.0)
