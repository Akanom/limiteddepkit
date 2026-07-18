"""Validated limited-dependent-variable and duration estimators."""

from .binary import BinaryLogit, BinaryLogitResult
from .binary_probit import BinaryProbit, BinaryProbitResult
from .censoring import (
    IntervalRegression,
    IntervalRegressionResult,
    Tobit,
    TobitResult,
    TruncatedRegression,
    TruncatedRegressionResult,
)
from .count import (
    NegativeBinomial,
    NegativeBinomialNB2,
    NegativeBinomialResult,
    PoissonRegressor,
    PoissonResult,
)
from .duration import (
    DiscreteTimeDuration,
    DiscreteTimeDurationResult,
    ExponentialDuration,
    ExponentialDurationResult,
    GammaDuration,
    GammaDurationResult,
    GeometricDuration,
    GeometricDurationResult,
    WeibullDuration,
    WeibullDurationResult,
)
from .dynamic_ordinal import (
    DynamicRandomEffectsOrderedLogit,
    DynamicRandomEffectsOrderedLogitResult,
)
from .fixed_effects_ordinal import (
    FixedEffectsOrderedLogit,
    FixedEffectsOrderedLogitResult,
)
from .generalized_ordinal import (
    GeneralizedOrderedLogit,
    GeneralizedOrderedLogitResult,
    PartialProportionalOdds,
    PartialProportionalOddsResult,
)
from .integrations import add_to_outputhub, to_outputhub_model
from .model_comparison import LikelihoodRatioTestResult, likelihood_ratio_test
from .ordinal import (
    OrderedLogit,
    OrderedLogitResult,
    OrderedProbit,
    OrderedProbitResult,
    OrderedResult,
    ProportionalOddsTestResult,
)
from .panel_ordinal import (
    RandomEffectsOrderedLogit,
    RandomEffectsOrderedLogitResult,
    RandomEffectsOrderedProbit,
    RandomEffectsOrderedProbitResult,
    RandomEffectsOrderedResult,
)
from .plotting import plot_marginal_effects, plot_probabilities
from .postestimation import (
    confint,
    lincom,
    marginal_effects,
    margins,
    posterior_predict_proba,
    posterior_random_effects,
    predict,
    predict_proba,
    summary_frame,
    vcov,
    wald_test,
)
from .simulation import (
    DynamicOrderedLogitSimulation,
    GeneralizedOrdinalSimulation,
    RandomEffectsOrderedLogitSimulation,
    RandomEffectsOrderedProbitSimulation,
    simulate_dynamic_random_effects_ordered_logit,
    simulate_generalized_ordered_logit,
    simulate_random_effects_ordered_logit,
    simulate_random_effects_ordered_probit,
)
from .small_sample import FirthBinaryLogit, FirthBinaryLogitResult

__all__ = [
    "BinaryLogit",
    "BinaryLogitResult",
    "BinaryProbit",
    "BinaryProbitResult",
    "Tobit",
    "TobitResult",
    "TruncatedRegression",
    "TruncatedRegressionResult",
    "IntervalRegression",
    "IntervalRegressionResult",
    "PoissonRegressor",
    "PoissonResult",
    "NegativeBinomial",
    "NegativeBinomialNB2",
    "NegativeBinomialResult",
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
    "OrderedLogit",
    "OrderedLogitResult",
    "OrderedProbit",
    "OrderedProbitResult",
    "OrderedResult",
    "ProportionalOddsTestResult",
    "plot_marginal_effects",
    "plot_probabilities",
    "add_to_outputhub",
    "to_outputhub_model",
    "GeneralizedOrderedLogit",
    "GeneralizedOrderedLogitResult",
    "PartialProportionalOdds",
    "PartialProportionalOddsResult",
    "LikelihoodRatioTestResult",
    "likelihood_ratio_test",
    "confint",
    "lincom",
    "marginal_effects",
    "margins",
    "predict",
    "predict_proba",
    "posterior_predict_proba",
    "posterior_random_effects",
    "summary_frame",
    "vcov",
    "wald_test",
    "GeneralizedOrdinalSimulation",
    "simulate_generalized_ordered_logit",
    "RandomEffectsOrderedLogit",
    "RandomEffectsOrderedLogitResult",
    "RandomEffectsOrderedLogitSimulation",
    "simulate_random_effects_ordered_logit",
    "RandomEffectsOrderedProbit",
    "RandomEffectsOrderedProbitResult",
    "RandomEffectsOrderedProbitSimulation",
    "RandomEffectsOrderedResult",
    "simulate_random_effects_ordered_probit",
    "FixedEffectsOrderedLogit",
    "FixedEffectsOrderedLogitResult",
    "DynamicRandomEffectsOrderedLogit",
    "DynamicRandomEffectsOrderedLogitResult",
    "DynamicOrderedLogitSimulation",
    "simulate_dynamic_random_effects_ordered_logit",
    "FirthBinaryLogit",
    "FirthBinaryLogitResult",
]
__version__ = "0.1.0a1"
