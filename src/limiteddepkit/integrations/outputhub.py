"""Universal Output Hub adapter for ordinal-model results."""

from __future__ import annotations

from typing import Any

from ..dynamic_ordinal import DynamicRandomEffectsOrderedLogitResult
from ..generalized_ordinal import GeneralizedOrderedLogitResult, PartialProportionalOddsResult
from ..ordinal import OrderedResult, ProportionalOddsTestResult
from ..panel_ordinal import RandomEffectsOrderedResult

OutputHubOrdinalResult = (
    OrderedResult
    | GeneralizedOrderedLogitResult
    | PartialProportionalOddsResult
    | RandomEffectsOrderedResult
    | DynamicRandomEffectsOrderedLogitResult
)


def _result_profile(result: OutputHubOrdinalResult) -> dict[str, Any]:
    if isinstance(result, OrderedResult):
        return {
            "default_name": f"Ordered {result.link.title()}",
            "estimator": "ordinal",
            "link": result.link,
            "threshold_parameterization": "ordered cuts",
            "metadata": {},
            "diagnostics": {},
        }
    if isinstance(result, GeneralizedOrderedLogitResult):
        return {
            "default_name": "Generalized Ordered Logit",
            "estimator": "generalized_ordered_logit",
            "link": "logit",
            "threshold_parameterization": "threshold-specific slopes",
            "metadata": {"inference_valid": result.inference_valid},
            "diagnostics": {
                "Minimum index gap": result.minimum_index_gap,
                "Constraint slack": result.constraint_slack,
            },
        }
    if isinstance(result, PartialProportionalOddsResult):
        return {
            "default_name": "Partial Proportional Odds",
            "estimator": "partial_proportional_odds",
            "link": "logit",
            "threshold_parameterization": "selected threshold-specific slopes",
            "metadata": {
                "inference_valid": result.inference_valid,
                "varying_features": list(result.varying_features),
            },
            "diagnostics": {
                "Minimum index gap": result.minimum_index_gap,
                "Constraint slack": result.constraint_slack,
            },
        }
    if isinstance(result, RandomEffectsOrderedResult):
        titled_link = result.link.title()
        return {
            "default_name": f"Random-effects Ordered {titled_link}",
            "estimator": f"random_effects_ordered_{result.link}",
            "link": result.link,
            "threshold_parameterization": "ordered cuts",
            "metadata": {
                "inference_valid": result.inference_valid,
                "n_entities": result.n_entities,
                "quadrature_points": result.quadrature_points,
                "backend": result.backend,
                "covariance_type": result.covariance_type,
            },
            "diagnostics": {"Random-effect SD": result.sigma_entity},
        }
    if isinstance(result, DynamicRandomEffectsOrderedLogitResult):
        return {
            "default_name": "Dynamic random-effects Ordered Logit",
            "estimator": "dynamic_random_effects_ordered_logit",
            "link": "logit",
            "threshold_parameterization": "ordered cuts",
            "metadata": {
                "inference_valid": result.inference_valid,
                "n_entities": result.n_entities,
                "backend": result.backend,
                "covariance_type": result.covariance_type,
                "conditioned_initial_observations": result.dropped_initial,
                "truncated_gap_observations": result.dropped_nonconsecutive,
            },
            "diagnostics": {"Random-effect SD": result.sigma_entity},
        }
    raise TypeError(
        "result must be an Ordered, Generalized Ordered Logit, or "
        "Partial Proportional Odds result."
    )


def _regression_model_class() -> Any:
    try:
        from universal_output_hub import RegressionModel
    except ImportError as error:
        raise ImportError(
            "Universal Output Hub is required for this integration. "
            "Install limiteddepkit with the 'outputhub' extra."
        ) from error
    return RegressionModel


def to_outputhub_model(
    result: OutputHubOrdinalResult,
    *,
    name: str | None = None,
    depvar: str | None = None,
    proportional_odds: ProportionalOddsTestResult | None = None,
) -> Any:
    """Convert an ordinal result to Output Hub's canonical model representation."""
    RegressionModel = _regression_model_class()
    profile = _result_profile(result)
    diagnostics = dict(profile["diagnostics"])
    if proportional_odds is not None:
        diagnostics.update(
            {
                "Proportional-odds chi2": proportional_odds.statistic,
                "Proportional-odds df": proportional_odds.df,
                "Proportional-odds p": proportional_odds.p_value,
            }
        )
    return RegressionModel(
        name=name or profile["default_name"],
        depvar=depvar,
        params=result.all_params.rename("coef"),
        std_errors=result.standard_errors.rename("se"),
        pvalues=result.pvalues.rename("pvalue"),
        statistics={
            "N": result.nobs,
            "Log Likelihood": result.loglike,
            "Converged": result.converged,
            "Categories": len(result.categories),
        },
        diagnostics=diagnostics,
        metadata={
            "estimator": profile["estimator"],
            "link": profile["link"],
            "categories": [str(category) for category in result.categories],
            "threshold_parameterization": profile["threshold_parameterization"],
            **profile["metadata"],
        },
        source="limiteddepkit",
    )


def add_to_outputhub(
    hub: Any,
    result: OutputHubOrdinalResult,
    *,
    name: str | None = None,
    depvar: str | None = None,
    X: Any | None = None,
    proportional_odds: ProportionalOddsTestResult | None = None,
) -> Any:
    """Add an ordinal model and optional AME inference table to an OutputHub."""
    if not hasattr(hub, "add_model") or not hasattr(hub, "add_table"):
        raise TypeError("hub must provide OutputHub-compatible add_model and add_table methods.")
    if X is not None and not hasattr(result, "average_marginal_effects_inference"):
        raise NotImplementedError(
            "Average marginal effects are not yet available for this ordinal result. "
            "Add the model without X to export its coefficient summary."
        )
    profile = _result_profile(result)
    model_name = name or profile["default_name"]
    model = to_outputhub_model(
        result,
        name=model_name,
        depvar=depvar,
        proportional_odds=proportional_odds,
    )
    hub.add_model(model)
    if X is not None:
        inference = result.average_marginal_effects_inference(X)
        inference_valid = bool(inference.attrs.get("inference_valid", True))
        marginal_effects = inference.reset_index()
        caption = "Average marginal effects with delta-method inference."
        if not inference_valid:
            caption = (
                "Average marginal effects; delta-method inference is unavailable because "
                "a non-crossing constraint is active."
            )
        hub.add_table(
            f"{model_name} average marginal effects",
            marginal_effects,
            caption=caption,
            metadata={
                "source": "limiteddepkit",
                "estimator": profile["estimator"],
                "link": profile["link"],
                "inference_valid": inference_valid,
            },
        )
    return model
