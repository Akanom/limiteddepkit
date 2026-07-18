"""Regression tests for the stable and experimental public namespaces."""

import importlib.util

import limiteddepkit
import limiteddepkit.experimental as experimental
import limiteddepkit.ml as ml

STABLE_BINARY_EXPORTS = {
    "BinaryLogit",
    "BinaryLogitResult",
    "BinaryProbit",
    "BinaryProbitResult",
}

STABLE_CENSORING_EXPORTS = {
    "IntervalRegression",
    "IntervalRegressionResult",
    "Tobit",
    "TobitResult",
    "TruncatedRegression",
    "TruncatedRegressionResult",
}

STABLE_COUNT_EXPORTS = {
    "NegativeBinomial",
    "NegativeBinomialNB2",
    "NegativeBinomialResult",
    "PoissonRegressor",
    "PoissonResult",
}

COUNT_COMPATIBILITY_EXPORTS = STABLE_COUNT_EXPORTS - {"NegativeBinomialNB2"}

STABLE_PANEL_ORDINAL_EXPORTS = {
    "RandomEffectsOrderedProbit",
    "RandomEffectsOrderedProbitResult",
    "RandomEffectsOrderedProbitSimulation",
    "simulate_random_effects_ordered_probit",
}

STABLE_SMALL_SAMPLE_EXPORTS = {
    "FirthBinaryLogit",
    "FirthBinaryLogitResult",
}

STABLE_DURATION_EXPORTS = {
    "DiscreteTimeDuration",
    "DiscreteTimeDurationResult",
    "ExponentialDuration",
    "ExponentialDurationResult",
    "GammaDuration",
    "GammaDurationResult",
    "GeometricDuration",
    "GeometricDurationResult",
    "WeibullDuration",
    "WeibullDurationResult",
}

DURATION_COMPATIBILITY_EXPORTS = STABLE_DURATION_EXPORTS - {
    "GeometricDuration",
    "GeometricDurationResult",
}

STABLE_FIXED_EFFECTS_ORDINAL_EXPORTS = {
    "FixedEffectsOrderedLogit",
    "FixedEffectsOrderedLogitResult",
}

PROVISIONAL_EXPORTS = {
    "CensoredQuantileRegression",
    "CensoredQuantileRegressionResult",
    "ConditionalLogit",
    "ConditionalLogitResult",
    "DynamicFixedEffectsOrderedLogit",
    "DynamicFixedEffectsOrderedLogitResult",
    "FixedEffectsOrderedProbit",
    "FixedEffectsOrderedProbitResult",
    "HurdlePoisson",
    "HurdlePoissonResult",
    "MultinomialLogit",
    "MultinomialLogitResult",
    "RidgeBinaryLogit",
    "RidgeBinaryLogitResult",
    "RidgeOrderedLogit",
    "RidgeOrderedLogitResult",
    "SampleSelection",
    "SampleSelectionResult",
    "SequentialLogit",
    "SequentialLogitResult",
    "ZeroInflatedPoisson",
    "ZeroInflatedPoissonResult",
}

EXTRACTED_EXPORTS = {
    "GaussianMixtureRegression",
    "GaussianMixtureRegressionResult",
    "SwitchingRegression",
    "SwitchingRegressionResult",
    "TreatmentEffect",
    "TreatmentEffectResult",
}

ML_WORKFLOW_EXPORTS = {
    "EntityHoldoutSplit",
    "ForwardPanelSplit",
    "GroupKFold",
    "KFold",
    "RepeatedStratifiedKFold",
    "ResidualBinaryMLP",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "binary_calibration_intercept_slope",
    "compare_models",
    "cross_validate",
    "fit_censoring_distribution",
    "nested_cross_validate",
    "paired_bootstrap_interval",
    "score_predictions",
    "statsmodels_bridge",
}


def test_certified_binary_estimators_are_stable_root_exports():
    assert set(limiteddepkit.__all__) >= STABLE_BINARY_EXPORTS
    assert all(hasattr(limiteddepkit, name) for name in STABLE_BINARY_EXPORTS)
    assert STABLE_BINARY_EXPORTS.isdisjoint(experimental.__all__)


def test_gaussian_censoring_estimators_are_stable_root_exports_and_compatibility_aliases():
    from limiteddepkit import censoring

    assert set(limiteddepkit.__all__) >= STABLE_CENSORING_EXPORTS
    assert set(censoring.__all__) == STABLE_CENSORING_EXPORTS
    assert all(hasattr(limiteddepkit, name) for name in STABLE_CENSORING_EXPORTS)
    assert all(
        getattr(limiteddepkit, name) is getattr(censoring, name)
        for name in STABLE_CENSORING_EXPORTS
    )
    assert all(
        getattr(experimental, name) is getattr(censoring, name)
        for name in STABLE_CENSORING_EXPORTS
    )


def test_foundational_count_estimators_are_stable_exports_and_compatibility_aliases():
    from limiteddepkit import count

    assert set(limiteddepkit.__all__) >= STABLE_COUNT_EXPORTS
    assert set(count.__all__) == STABLE_COUNT_EXPORTS
    assert all(hasattr(limiteddepkit, name) for name in STABLE_COUNT_EXPORTS)
    assert all(
        getattr(limiteddepkit, name) is getattr(count, name)
        for name in STABLE_COUNT_EXPORTS
    )
    assert all(
        getattr(experimental, name) is getattr(count, name)
        for name in COUNT_COMPATIBILITY_EXPORTS
    )


def test_random_effects_ordered_probit_is_a_stable_root_export():
    assert set(limiteddepkit.__all__) >= STABLE_PANEL_ORDINAL_EXPORTS
    assert all(
        hasattr(limiteddepkit, name) for name in STABLE_PANEL_ORDINAL_EXPORTS
    )
    assert STABLE_PANEL_ORDINAL_EXPORTS.isdisjoint(experimental.__all__)


def test_firth_binary_logit_is_stable_with_experimental_compatibility_aliases():
    from limiteddepkit import small_sample

    assert set(limiteddepkit.__all__) >= STABLE_SMALL_SAMPLE_EXPORTS
    assert set(small_sample.__all__) == STABLE_SMALL_SAMPLE_EXPORTS
    assert all(
        getattr(limiteddepkit, name) is getattr(small_sample, name)
        for name in STABLE_SMALL_SAMPLE_EXPORTS
    )
    assert all(
        getattr(experimental, name) is getattr(small_sample, name)
        for name in STABLE_SMALL_SAMPLE_EXPORTS
    )


def test_duration_family_is_stable_with_experimental_compatibility_aliases():
    from limiteddepkit import duration

    assert set(limiteddepkit.__all__) >= STABLE_DURATION_EXPORTS
    assert set(duration.__all__) == STABLE_DURATION_EXPORTS
    assert all(
        getattr(limiteddepkit, name) is getattr(duration, name)
        for name in STABLE_DURATION_EXPORTS
    )
    assert all(
        getattr(experimental, name) is getattr(duration, name)
        for name in DURATION_COMPATIBILITY_EXPORTS
    )


def test_buc_fixed_effects_ordered_logit_is_stable_and_probit_is_provisional():
    from limiteddepkit import panel

    assert set(limiteddepkit.__all__) >= STABLE_FIXED_EFFECTS_ORDINAL_EXPORTS
    assert all(
        getattr(limiteddepkit, name) is getattr(panel, name)
        for name in STABLE_FIXED_EFFECTS_ORDINAL_EXPORTS
    )
    assert STABLE_FIXED_EFFECTS_ORDINAL_EXPORTS.isdisjoint(experimental.__all__)


def test_provisional_estimators_are_quarantined_from_package_root():
    assert PROVISIONAL_EXPORTS.isdisjoint(limiteddepkit.__all__)
    assert all(not hasattr(limiteddepkit, name) for name in PROVISIONAL_EXPORTS)


def test_experimental_namespace_exports_every_provisional_estimator():
    assert set(experimental.__all__) == (
        PROVISIONAL_EXPORTS
        | STABLE_CENSORING_EXPORTS
        | COUNT_COMPATIBILITY_EXPORTS
        | STABLE_SMALL_SAMPLE_EXPORTS
        | DURATION_COMPATIBILITY_EXPORTS
    )
    assert all(hasattr(experimental, name) for name in PROVISIONAL_EXPORTS)


def test_ml_workflow_is_public_only_from_its_experimental_submodule():
    assert set(ml.__all__) >= ML_WORKFLOW_EXPORTS
    assert all(hasattr(ml, name) for name in ML_WORKFLOW_EXPORTS)
    assert ML_WORKFLOW_EXPORTS.isdisjoint(limiteddepkit.__all__)


def test_extracted_estimators_are_absent_from_the_distribution_namespaces():
    assert EXTRACTED_EXPORTS.isdisjoint(limiteddepkit.__all__)
    assert EXTRACTED_EXPORTS.isdisjoint(experimental.__all__)
    assert all(not hasattr(limiteddepkit, name) for name in EXTRACTED_EXPORTS)
    assert all(not hasattr(experimental, name) for name in EXTRACTED_EXPORTS)
    assert importlib.util.find_spec("limiteddepkit.treatment_effect") is None
    assert importlib.util.find_spec("limiteddepkit.switching_regression") is None
