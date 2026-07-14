"""Visualization helpers for fitted ordinal models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .generalized_ordinal import (
    GeneralizedOrderedLogitResult,
    PartialProportionalOddsResult,
)
from .ordinal import OrderedResult


def _evaluation_grid(
    result: OrderedResult | GeneralizedOrderedLogitResult | PartialProportionalOddsResult,
    X: Any,
    feature: str,
    values: Any | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Plotting requires X to be a DataFrame with fitted feature names.")
    if tuple(str(column) for column in X.columns) != result.feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    if feature not in result.feature_names:
        raise ValueError(f"Unknown feature: {feature!r}.")

    if values is None:
        lower, upper = np.quantile(X[feature].to_numpy(dtype=float), [0.05, 0.95])
        grid_values = np.linspace(lower, upper, 100)
    else:
        grid_values = np.asarray(values, dtype=float)
        if grid_values.ndim != 1 or grid_values.size < 2:
            raise ValueError("values must be a one-dimensional grid with at least two points.")
        if not np.isfinite(grid_values).all():
            raise ValueError("values must contain only finite numbers.")

    means = X.mean(axis=0, numeric_only=True)
    if len(means) != len(result.feature_names):
        raise ValueError("All fitted plotting features must be numeric.")
    grid = pd.DataFrame(
        np.tile(means.to_numpy(dtype=float), (grid_values.size, 1)),
        columns=result.feature_names,
    )
    grid[feature] = grid_values
    return grid, grid_values


def plot_probabilities(
    result: OrderedResult | GeneralizedOrderedLogitResult | PartialProportionalOddsResult,
    X: Any,
    *,
    feature: str,
    values: Any | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot predicted category probabilities for a fitted ordinal model."""
    import matplotlib.pyplot as plt

    grid, grid_values = _evaluation_grid(result, X, feature, values)
    probabilities = result.predict_proba(grid)
    if ax is None:
        _, ax = plt.subplots()
    for category in result.categories:
        ax.plot(grid_values, probabilities[category], label=str(category))
    ax.set_xlabel(feature)
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Category")
    return ax


def plot_marginal_effects(
    result: OrderedResult | GeneralizedOrderedLogitResult | PartialProportionalOddsResult,
    X: Any,
    *,
    feature: str,
    values: Any | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot category-specific marginal effects for a fitted ordinal model."""
    import matplotlib.pyplot as plt

    grid, grid_values = _evaluation_grid(result, X, feature, values)
    effects = result.marginal_effects(grid).xs(feature, axis=1, level="feature")
    if ax is None:
        _, ax = plt.subplots()
    for category in result.categories:
        ax.plot(grid_values, effects[category], label=str(category))
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel("Marginal effect")
    ax.legend(title="Category")
    return ax
