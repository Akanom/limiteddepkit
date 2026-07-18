"""In-scope estimators whose stable contracts are not yet promoted.

These APIs may change while remaining covariance, workflow, documentation, and
compatibility promotion gates are completed.

The Gaussian censoring, foundational count, and Firth Binary Logit names remain
as temporary compatibility aliases after promotion to their stable family
namespaces and the package root.
"""

from ..censored_quantile import CensoredQuantileRegression, CensoredQuantileRegressionResult
from ..censoring import (
    IntervalRegression,
    IntervalRegressionResult,
    Tobit,
    TobitResult,
    TruncatedRegression,
    TruncatedRegressionResult,
)
from ..conditional_logit import ConditionalLogit, ConditionalLogitResult
from ..count import (
    NegativeBinomial,
    NegativeBinomialResult,
    PoissonRegressor,
    PoissonResult,
)
from ..duration import (
    DiscreteTimeDuration,
    DiscreteTimeDurationResult,
    ExponentialDuration,
    ExponentialDurationResult,
    GammaDuration,
    GammaDurationResult,
    WeibullDuration,
    WeibullDurationResult,
)
from ..dynamic_fixed_effects_ordinal import (
    DynamicFixedEffectsOrderedLogit,
    DynamicFixedEffectsOrderedLogitResult,
)
from ..fixed_effects_ordinal import (
    FixedEffectsOrderedProbit,
    FixedEffectsOrderedProbitResult,
)
from ..hurdle_poisson import HurdlePoisson, HurdlePoissonResult
from ..multinomial import MultinomialLogit, MultinomialLogitResult
from ..sample_selection import SampleSelection, SampleSelectionResult
from ..sequential_logit import SequentialLogit, SequentialLogitResult
from ..zero_inflated_poisson import ZeroInflatedPoisson, ZeroInflatedPoissonResult
from .small_sample import (
    FirthBinaryLogit,
    FirthBinaryLogitResult,
    RidgeBinaryLogit,
    RidgeBinaryLogitResult,
    RidgeOrderedLogit,
    RidgeOrderedLogitResult,
)

__all__ = [
    "CensoredQuantileRegression",
    "CensoredQuantileRegressionResult",
    "ConditionalLogit",
    "ConditionalLogitResult",
    "DiscreteTimeDuration",
    "DiscreteTimeDurationResult",
    "DynamicFixedEffectsOrderedLogit",
    "DynamicFixedEffectsOrderedLogitResult",
    "ExponentialDuration",
    "ExponentialDurationResult",
    "FirthBinaryLogit",
    "FirthBinaryLogitResult",
    "FixedEffectsOrderedProbit",
    "FixedEffectsOrderedProbitResult",
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
    "RidgeBinaryLogit",
    "RidgeBinaryLogitResult",
    "RidgeOrderedLogit",
    "RidgeOrderedLogitResult",
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
]
