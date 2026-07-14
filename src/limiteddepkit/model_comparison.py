"""Nested model comparisons for ordinal estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2

from .generalized_ordinal import (
    GeneralizedOrderedLogitResult,
    PartialProportionalOddsResult,
)
from .ordinal import OrderedResult


@dataclass(frozen=True)
class LikelihoodRatioTestResult:
    """Likelihood-ratio comparison of nested ordinal models."""

    statistic: float
    df: int
    p_value: float
    restricted_loglike: float
    unrestricted_loglike: float
    restricted_n_params: int
    unrestricted_n_params: int
    regular_chi2_reference: bool
    note: str


def _check_common_data_contract(restricted: Any, unrestricted: Any) -> None:
    if restricted.nobs != unrestricted.nobs:
        raise ValueError("Models must use the same number of observations.")
    if restricted.feature_names != unrestricted.feature_names:
        raise ValueError("Models must use the same features in the same order.")
    if not np.array_equal(restricted.categories, unrestricted.categories):
        raise ValueError("Models must use the same ordered outcome categories.")


def _check_nested(restricted: Any, unrestricted: Any) -> None:
    if isinstance(restricted, OrderedResult) and isinstance(
        unrestricted, (PartialProportionalOddsResult, GeneralizedOrderedLogitResult)
    ):
        if restricted.link != "logit":
            raise ValueError(
                "Generalized and Partial Proportional Odds models use a Logit link; "
                "the restricted model must also be Ordered Logit."
            )
        return
    if isinstance(restricted, PartialProportionalOddsResult):
        if isinstance(unrestricted, GeneralizedOrderedLogitResult):
            return
        if isinstance(unrestricted, PartialProportionalOddsResult) and set(
            restricted.varying_features
        ).issubset(unrestricted.varying_features):
            return
    raise ValueError("The supplied models are not a supported restricted/unrestricted pair.")


def likelihood_ratio_test(
    restricted: OrderedResult | PartialProportionalOddsResult,
    unrestricted: PartialProportionalOddsResult | GeneralizedOrderedLogitResult,
    *,
    active_constraint_tolerance: float = 1e-5,
) -> LikelihoodRatioTestResult:
    """Compare nested ordinal models by their maximized log likelihoods."""
    if active_constraint_tolerance <= 0:
        raise ValueError("active_constraint_tolerance must be positive.")
    _check_common_data_contract(restricted, unrestricted)
    _check_nested(restricted, unrestricted)

    degrees_of_freedom = unrestricted.n_params - restricted.n_params
    if degrees_of_freedom <= 0:
        raise ValueError("The unrestricted model must contain additional free parameters.")
    statistic = 2.0 * (unrestricted.loglike - restricted.loglike)
    if statistic < -1e-6:
        raise ValueError(
            "The unrestricted model has a lower log likelihood; check convergence and nesting."
        )
    statistic = max(float(statistic), 0.0)
    constraint_slack = getattr(unrestricted, "constraint_slack", np.inf)
    regular_reference = bool(constraint_slack > active_constraint_tolerance)
    if regular_reference:
        p_value = float(chi2.sf(statistic, degrees_of_freedom))
        note = "Standard chi-square reference used; no non-crossing constraint is active."
    else:
        p_value = np.nan
        note = (
            "A non-crossing inequality constraint is active; the standard chi-square "
            "reference is not valid. Use a constrained bootstrap for inference."
        )
    return LikelihoodRatioTestResult(
        statistic=statistic,
        df=degrees_of_freedom,
        p_value=p_value,
        restricted_loglike=float(restricted.loglike),
        unrestricted_loglike=float(unrestricted.loglike),
        restricted_n_params=restricted.n_params,
        unrestricted_n_params=unrestricted.n_params,
        regular_chi2_reference=regular_reference,
        note=note,
    )
