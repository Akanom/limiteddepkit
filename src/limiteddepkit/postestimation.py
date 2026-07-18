"""Ecosystem-aligned post-estimation functions for limiteddepkit results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import chi2, norm


def _all_params(result: Any) -> pd.Series:
    parameters = getattr(result, "all_params", None)
    if parameters is None:
        parameters = getattr(result, "params", None)
    if not isinstance(parameters, pd.Series):
        raise TypeError("Result does not expose a labeled parameter Series.")
    return parameters


def summary_frame(result: Any) -> pd.DataFrame:
    """Return the common coefficient-table representation used across the ecosystem."""
    parameters = _all_params(result)
    return pd.DataFrame(
        {
            "coef": parameters,
            "std_err": result.standard_errors.reindex(parameters.index),
            "z": result.zstats.reindex(parameters.index),
            "p_value": result.pvalues.reindex(parameters.index),
        }
    )


def vcov(result: Any) -> pd.DataFrame:
    """Return a defensive copy of the fitted covariance matrix."""
    covariance = getattr(result, "covariance", None)
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("Result does not expose a covariance DataFrame.")
    return covariance.copy()


def confint(result: Any, level: float = 0.95) -> pd.DataFrame:
    """Return the fitted estimator's default confidence intervals."""
    return result.conf_int(level=level)


def predict(result: Any, X: Any, **kwargs: Any) -> pd.Series:
    """Return predicted ordinal categories."""
    return result.predict(X, **kwargs)


def predict_proba(result: Any, X: Any, **kwargs: Any) -> pd.DataFrame:
    """Return predicted category probabilities."""
    return result.predict_proba(X, **kwargs)


def posterior_random_effects(
    result: Any, X: Any, y: Any, *, entity: Any
) -> pd.DataFrame:
    """Return entity posterior summaries for a fitted random-effects result."""
    if not hasattr(result, "posterior_random_effects"):
        raise TypeError("Result does not support posterior random effects.")
    return result.posterior_random_effects(X, y, entity=entity)


def posterior_predict_proba(
    result: Any,
    X: Any,
    *,
    entity: Any,
    posterior: pd.DataFrame,
) -> pd.DataFrame:
    """Return entity-specific posterior-predictive category probabilities."""
    if not hasattr(result, "posterior_predict_proba"):
        raise TypeError("Result does not support posterior-predictive probabilities.")
    return result.posterior_predict_proba(X, entity=entity, posterior=posterior)


def marginal_effects(result: Any, X: Any) -> pd.DataFrame:
    """Return category-specific probability derivatives."""
    return result.marginal_effects(X)


def margins(
    result: Any,
    X: Any,
    *,
    at: str | Mapping[str, float] = "overall",
    kind: str = "probability",
) -> pd.Series | pd.DataFrame:
    """Evaluate probabilities or marginal effects at requested covariate values."""
    return result.margins(X, at=at, kind=kind)


def lincom(
    result: Any,
    weights: Mapping[str, float],
    *,
    value: float = 0.0,
    level: float = 0.95,
) -> pd.Series:
    """Estimate and test a linear combination for any inferential ordinal result."""
    if not weights:
        raise ValueError("weights must contain at least one parameter.")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one.")
    parameters = _all_params(result)
    unknown = set(weights) - set(parameters.index)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}.")
    contrast = pd.Series(0.0, index=parameters.index)
    for name, weight in weights.items():
        contrast[name] = float(weight)
    vector = contrast.to_numpy(dtype=float)
    estimate = float(vector @ parameters.to_numpy(dtype=float))
    variance = float(vector @ vcov(result).to_numpy(dtype=float) @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    zstat = (estimate - float(value)) / standard_error if standard_error > 0 else np.nan
    critical = float(norm.ppf(0.5 + level / 2.0))
    return pd.Series(
        {
            "estimate": estimate,
            "standard_error": standard_error,
            "z_stat": zstat,
            "p_value": float(2.0 * ndtr(-abs(zstat))),
            "lower": estimate - critical * standard_error,
            "upper": estimate + critical * standard_error,
        },
        name="lincom",
    )


def wald_test(
    result: Any,
    restrictions: Mapping[str, float] | Sequence[Mapping[str, float]],
    *,
    values: float | Sequence[float] = 0.0,
) -> pd.Series:
    """Test one or more named linear restrictions."""
    rows = [restrictions] if isinstance(restrictions, Mapping) else list(restrictions)
    if not rows:
        raise ValueError("restrictions must contain at least one restriction.")
    parameters = _all_params(result)
    matrix = np.zeros((len(rows), len(parameters)), dtype=float)
    for row_index, row in enumerate(rows):
        unknown = set(row) - set(parameters.index)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}.")
        for name, weight in row.items():
            matrix[row_index, parameters.index.get_loc(name)] = float(weight)
    if np.isscalar(values):
        null_values = np.full(len(rows), float(values))
    else:
        null_values = np.asarray(values, dtype=float)
        if null_values.shape != (len(rows),):
            raise ValueError("values must provide one null value per restriction.")
    difference = matrix @ parameters.to_numpy(dtype=float) - null_values
    restricted_covariance = matrix @ vcov(result).to_numpy(dtype=float) @ matrix.T
    statistic = float(difference @ np.linalg.pinv(restricted_covariance) @ difference)
    degrees_of_freedom = int(np.linalg.matrix_rank(matrix))
    return pd.Series(
        {
            "statistic": statistic,
            "df": degrees_of_freedom,
            "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
        },
        name="wald_test",
    )
