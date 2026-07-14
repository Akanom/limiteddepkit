"""Regression tests for the stable and experimental public namespaces."""

import importlib.util

import limiteddepkit
import limiteddepkit.experimental as experimental

STABLE_BINARY_EXPORTS = {
    "BinaryLogit",
    "BinaryLogitResult",
    "BinaryProbit",
    "BinaryProbitResult",
}

PROVISIONAL_EXPORTS = {
    "CensoredQuantileRegression",
    "CensoredQuantileRegressionResult",
    "ConditionalLogit",
    "ConditionalLogitResult",
    "DiscreteTimeDuration",
    "DiscreteTimeDurationResult",
    "ExponentialDuration",
    "ExponentialDurationResult",
    "GammaDuration",
    "GammaDurationResult",
    "HurdlePoisson",
    "HurdlePoissonResult",
    "IntervalRegression",
    "IntervalRegressionResult",
    "MultinomialLogit",
    "MultinomialLogitResult",
    "NegativeBinomial",
    "NegativeBinomialResult",
    "PoissonRegressor",
    "PoissonResult",
    "SampleSelection",
    "SampleSelectionResult",
    "SequentialLogit",
    "SequentialLogitResult",
    "Tobit",
    "TobitResult",
    "TruncatedRegression",
    "TruncatedRegressionResult",
    "WeibullDuration",
    "WeibullDurationResult",
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


def test_certified_binary_estimators_are_stable_root_exports():
    assert set(limiteddepkit.__all__) >= STABLE_BINARY_EXPORTS
    assert all(hasattr(limiteddepkit, name) for name in STABLE_BINARY_EXPORTS)
    assert STABLE_BINARY_EXPORTS.isdisjoint(experimental.__all__)


def test_provisional_estimators_are_quarantined_from_package_root():
    assert PROVISIONAL_EXPORTS.isdisjoint(limiteddepkit.__all__)
    assert all(not hasattr(limiteddepkit, name) for name in PROVISIONAL_EXPORTS)


def test_experimental_namespace_exports_every_provisional_estimator():
    assert set(experimental.__all__) == PROVISIONAL_EXPORTS
    assert all(hasattr(experimental, name) for name in PROVISIONAL_EXPORTS)


def test_extracted_estimators_are_absent_from_the_distribution_namespaces():
    assert EXTRACTED_EXPORTS.isdisjoint(limiteddepkit.__all__)
    assert EXTRACTED_EXPORTS.isdisjoint(experimental.__all__)
    assert all(not hasattr(limiteddepkit, name) for name in EXTRACTED_EXPORTS)
    assert all(not hasattr(experimental, name) for name in EXTRACTED_EXPORTS)
    assert importlib.util.find_spec("limiteddepkit.treatment_effect") is None
    assert importlib.util.find_spec("limiteddepkit.switching_regression") is None
