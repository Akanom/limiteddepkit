"""Dependency-light prediction metrics for limited dependent outcomes.

The functions in this module deliberately operate on arrays rather than fitted
model objects.  This keeps the scoring layer usable by both stable and
experimental estimators without requiring scikit-learn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

ArrayLike: TypeAlias = Sequence[Any] | np.ndarray | pd.Series | pd.Index | pd.DataFrame
OutcomeKind: TypeAlias = Literal[
    "binary",
    "choice",
    "multiclass",
    "ordinal",
    "continuous",
    "count",
    "quantile",
    "duration",
    "selection",
]

__all__ = [
    "binary_accuracy",
    "binary_balanced_accuracy",
    "binary_brier_score",
    "binary_log_loss",
    "binary_roc_auc",
    "choice_accuracy",
    "choice_brier_score",
    "choice_log_loss",
    "count_mean_absolute_error",
    "count_root_mean_squared_error",
    "continuous_bias",
    "continuous_mean_absolute_error",
    "continuous_root_mean_squared_error",
    "duration_brier_score",
    "duration_concordance_index",
    "multiclass_brier_score",
    "multiclass_log_loss",
    "observed_outcome_rmse",
    "ordinal_mean_absolute_error",
    "poisson_deviance",
    "quantile_check_loss",
    "ranked_probability_score",
    "score_predictions",
    "selection_scores",
    "zero_rate_calibration_error",
]


def _as_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    return array


def _as_numeric_1d(
    values: ArrayLike,
    name: str,
    *,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    array = _as_1d(values, name)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not allow_nonfinite and not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values.")
    return numeric


def _check_same_length(**arrays: np.ndarray) -> None:
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Inputs must contain the same number of observations ({detail}).")


def _binary_target(values: ArrayLike, name: str = "y_true") -> np.ndarray:
    target = _as_numeric_1d(values, name)
    if not np.all(np.isin(target, (0.0, 1.0))):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return target.astype(int)


def _probability_vector(values: ArrayLike, name: str) -> np.ndarray:
    probability = _as_numeric_1d(values, name)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{name} must contain probabilities between 0 and 1.")
    return probability


def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if not np.isfinite(eps) or not 0.0 < eps < 0.5:
        raise ValueError("eps must be finite and strictly between 0 and 0.5.")
    return eps


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be finite and strictly between 0 and 1.")
    return threshold


def _binary_inputs(y_true: ArrayLike, y_probability: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    target = _binary_target(y_true)
    probability_array = np.asarray(y_probability)
    if probability_array.ndim == 2:
        probability_matrix = _probability_matrix(y_probability, "y_probability")
        if probability_matrix.shape[1] != 2:
            raise ValueError(
                "Binary y_probability matrices must contain exactly two class columns."
            )
        event_column = 1
        if isinstance(y_probability, pd.DataFrame):
            resolved_labels = _resolve_labels(y_probability, 2, None)
            if 1 in resolved_labels:
                event_column = resolved_labels.index(1)
        probability = probability_matrix[:, event_column]
    else:
        probability = _probability_vector(y_probability, "y_probability")
    _check_same_length(y_true=target, y_probability=probability)
    return target, probability


def binary_log_loss(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    eps: float = 1e-15,
) -> float:
    """Return mean binary negative log likelihood.

    Exact zero and one probabilities are clipped only for evaluation; values
    outside the probability interval remain invalid.
    """

    target, probability = _binary_inputs(y_true, y_probability)
    eps = _validate_eps(eps)
    probability = np.clip(probability, eps, 1.0 - eps)
    return float(-np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability)))


def binary_brier_score(y_true: ArrayLike, y_probability: ArrayLike) -> float:
    """Return the mean squared error of binary event probabilities."""

    target, probability = _binary_inputs(y_true, y_probability)
    return float(np.mean((probability - target) ** 2))


def binary_accuracy(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    threshold: float = 0.5,
) -> float:
    """Return thresholded binary classification accuracy."""

    target, probability = _binary_inputs(y_true, y_probability)
    prediction = probability >= _validate_threshold(threshold)
    return float(np.mean(prediction == target))


def binary_balanced_accuracy(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    threshold: float = 0.5,
) -> float:
    """Return the unweighted mean of sensitivity and specificity.

    Both outcome classes must occur.  A one-class fold has no identified
    binary balanced accuracy and is rejected rather than assigned a misleading
    perfect score.
    """

    target, probability = _binary_inputs(y_true, y_probability)
    prediction = (probability >= _validate_threshold(threshold)).astype(int)
    class_counts = np.bincount(target, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("binary balanced accuracy requires both outcome classes.")
    sensitivity = np.mean(prediction[target == 1] == 1)
    specificity = np.mean(prediction[target == 0] == 0)
    return float((sensitivity + specificity) / 2.0)


def binary_roc_auc(y_true: ArrayLike, y_probability: ArrayLike) -> float:
    """Return the tie-aware area under the binary ROC curve."""

    target, probability = _binary_inputs(y_true, y_probability)
    class_counts = np.bincount(target, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("binary ROC AUC requires both outcome classes.")
    ranks = pd.Series(probability).rank(method="average").to_numpy(dtype=float)
    positive_rank_sum = float(np.sum(ranks[target == 1]))
    n_negative, n_positive = int(class_counts[0]), int(class_counts[1])
    mann_whitney = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(mann_whitney / (n_positive * n_negative))


def _grouped_choice_inputs(
    choice: ArrayLike,
    probability: ArrayLike,
    groups: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    selected = _binary_target(choice, "choice")
    predicted = _probability_vector(probability, "choice_probability")
    group_values = _as_1d(groups, "groups")
    _check_same_length(choice=selected, choice_probability=predicted, groups=group_values)
    if np.any(pd.isna(group_values)):
        raise ValueError("groups cannot contain missing values.")
    try:
        group_codes, group_labels = pd.factorize(group_values, sort=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("groups must contain hashable scalar labels.") from exc
    n_groups = len(group_labels)
    for group in range(n_groups):
        within = group_codes == group
        if int(np.sum(within)) < 2:
            raise ValueError("Every choice set must contain at least two alternatives.")
        if int(np.sum(selected[within])) != 1:
            raise ValueError("Every choice set must contain exactly one chosen alternative.")
        if not np.isclose(np.sum(predicted[within]), 1.0, rtol=1e-7, atol=1e-10):
            raise ValueError("Predicted probabilities must sum to 1 within each choice set.")
    return selected, predicted, np.asarray(group_codes, dtype=int), n_groups


def choice_log_loss(
    choice: ArrayLike,
    choice_probability: ArrayLike,
    groups: ArrayLike,
    *,
    eps: float = 1e-15,
) -> float:
    """Return mean negative log probability of the chosen alternative."""

    selected, probability, _, n_groups = _grouped_choice_inputs(
        choice, choice_probability, groups
    )
    eps = _validate_eps(eps)
    return float(-np.sum(np.log(np.clip(probability[selected == 1], eps, 1.0))) / n_groups)


def choice_brier_score(
    choice: ArrayLike,
    choice_probability: ArrayLike,
    groups: ArrayLike,
) -> float:
    """Return the mean choice-set sum of squared probability errors."""

    selected, probability, _, n_groups = _grouped_choice_inputs(
        choice, choice_probability, groups
    )
    return float(np.sum((probability - selected) ** 2) / n_groups)


def choice_accuracy(
    choice: ArrayLike,
    choice_probability: ArrayLike,
    groups: ArrayLike,
) -> float:
    """Return the share of choice sets whose highest-probability item was chosen."""

    selected, probability, group_codes, n_groups = _grouped_choice_inputs(
        choice, choice_probability, groups
    )
    correct = 0
    for group in range(n_groups):
        indices = np.flatnonzero(group_codes == group)
        predicted = indices[int(np.argmax(probability[indices]))]
        correct += int(selected[predicted] == 1)
    return float(correct / n_groups)


def _probability_matrix(values: ArrayLike, name: str) -> np.ndarray:
    try:
        probability = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if probability.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional probability matrix.")
    if probability.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if probability.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two outcome columns.")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{name} must contain probabilities between 0 and 1.")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError(f"Rows of {name} must sum to 1.")
    return probability


def _resolve_labels(
    probabilities: ArrayLike,
    n_classes: int,
    labels: Sequence[Any] | None,
) -> tuple[Any, ...]:
    frame_columns = tuple(probabilities.columns) if isinstance(probabilities, pd.DataFrame) else None
    if labels is None:
        resolved = frame_columns if frame_columns is not None else tuple(range(n_classes))
    else:
        resolved = tuple(labels)
        if frame_columns is not None and resolved != frame_columns:
            raise ValueError("labels must match probability DataFrame columns in the same order.")
    if len(resolved) != n_classes:
        raise ValueError("labels must contain one label for each probability column.")
    label_index = pd.Index(resolved)
    if not label_index.is_unique:
        raise ValueError("labels must be unique.")
    if label_index.hasnans:
        raise ValueError("labels cannot contain missing values.")
    return resolved


def _multiclass_inputs(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    labels: Sequence[Any] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    target = _as_1d(y_true, "y_true")
    if np.any(pd.isna(target)):
        raise ValueError("y_true cannot contain missing values.")
    probability = _probability_matrix(y_probability, "y_probability")
    _check_same_length(y_true=target, y_probability=probability)
    resolved_labels = _resolve_labels(y_probability, probability.shape[1], labels)
    codes = pd.Index(resolved_labels).get_indexer(target)
    if np.any(codes < 0):
        unknown = tuple(pd.unique(target[codes < 0]))
        raise ValueError(f"y_true contains labels absent from probability columns: {unknown!r}.")
    return codes, probability, resolved_labels


def multiclass_log_loss(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
    eps: float = 1e-15,
) -> float:
    """Return mean multiclass negative log likelihood.

    For a pandas probability DataFrame, its columns define class labels and
    order.  For an array, labels default to integer codes ``0, ..., K - 1``.
    """

    codes, probability, _ = _multiclass_inputs(y_true, y_probability, labels)
    eps = _validate_eps(eps)
    assigned = np.clip(probability[np.arange(len(codes)), codes], eps, 1.0)
    return float(-np.mean(np.log(assigned)))


def multiclass_brier_score(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
) -> float:
    """Return the mean sum of classwise squared probability errors."""

    codes, probability, _ = _multiclass_inputs(y_true, y_probability, labels)
    observed = np.zeros_like(probability)
    observed[np.arange(len(codes)), codes] = 1.0
    return float(np.mean(np.sum((probability - observed) ** 2, axis=1)))


def ranked_probability_score(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
) -> float:
    """Return the mean ranked probability score for ordered categories.

    ``labels`` (or probability DataFrame columns) define category order.  The
    score is the sum of squared cumulative-probability errors over the ``K-1``
    non-trivial category boundaries.
    """

    codes, probability, _ = _multiclass_inputs(y_true, y_probability, labels)
    forecast_cdf = np.cumsum(probability, axis=1)[:, :-1]
    boundaries = np.arange(probability.shape[1] - 1)
    observed_cdf = boundaries[None, :] >= codes[:, None]
    return float(np.mean(np.sum((forecast_cdf - observed_cdf) ** 2, axis=1)))


def ordinal_mean_absolute_error(
    y_true: ArrayLike,
    y_predicted: ArrayLike,
    *,
    labels: Sequence[Any],
) -> float:
    """Return MAE between the positions of observed and predicted categories."""

    target = _as_1d(y_true, "y_true")
    prediction = _as_1d(y_predicted, "y_predicted")
    _check_same_length(y_true=target, y_predicted=prediction)
    resolved = tuple(labels)
    if len(resolved) < 2:
        raise ValueError("labels must contain at least two ordered categories.")
    index = pd.Index(resolved)
    if not index.is_unique or index.hasnans:
        raise ValueError("labels must be unique and cannot contain missing values.")
    target_codes = index.get_indexer(target)
    prediction_codes = index.get_indexer(prediction)
    if np.any(target_codes < 0) or np.any(prediction_codes < 0):
        raise ValueError("Observed and predicted categories must all occur in labels.")
    return float(np.mean(np.abs(target_codes - prediction_codes)))


def _continuous_inputs(
    y_true: ArrayLike,
    y_prediction: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    target = _as_numeric_1d(y_true, "y_true")
    prediction = _as_numeric_1d(y_prediction, "y_prediction")
    _check_same_length(y_true=target, y_prediction=prediction)
    return target, prediction


def continuous_mean_absolute_error(
    y_true: ArrayLike,
    y_prediction: ArrayLike,
) -> float:
    """Return mean absolute error for an observed-scale continuous outcome."""

    target, prediction = _continuous_inputs(y_true, y_prediction)
    return float(np.mean(np.abs(prediction - target)))


def continuous_root_mean_squared_error(
    y_true: ArrayLike,
    y_prediction: ArrayLike,
) -> float:
    """Return root mean squared error for an observed-scale continuous outcome."""

    target, prediction = _continuous_inputs(y_true, y_prediction)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def continuous_bias(y_true: ArrayLike, y_prediction: ArrayLike) -> float:
    """Return mean signed prediction error (prediction minus observation)."""

    target, prediction = _continuous_inputs(y_true, y_prediction)
    return float(np.mean(prediction - target))


def _count_inputs(y_true: ArrayLike, y_mean: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    target = _as_numeric_1d(y_true, "y_true")
    prediction = _as_numeric_1d(y_mean, "y_mean")
    _check_same_length(y_true=target, y_mean=prediction)
    if np.any(target < 0.0) or not np.allclose(target, np.round(target), rtol=0.0, atol=1e-12):
        raise ValueError("y_true must contain non-negative integer counts.")
    if np.any(prediction < 0.0):
        raise ValueError("y_mean must contain non-negative predicted means.")
    return target, prediction


def count_mean_absolute_error(y_true: ArrayLike, y_mean: ArrayLike) -> float:
    """Return mean absolute error for predicted counts or count means."""

    target, prediction = _count_inputs(y_true, y_mean)
    return float(np.mean(np.abs(target - prediction)))


def count_root_mean_squared_error(y_true: ArrayLike, y_mean: ArrayLike) -> float:
    """Return root mean squared error for predicted counts or count means."""

    target, prediction = _count_inputs(y_true, y_mean)
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def poisson_deviance(y_true: ArrayLike, y_mean: ArrayLike) -> float:
    """Return mean Poisson deviance for strictly positive predicted means."""

    target, prediction = _count_inputs(y_true, y_mean)
    if np.any(prediction <= 0.0):
        raise ValueError("Poisson deviance requires strictly positive predicted means.")
    contribution = prediction.copy()
    positive = target > 0.0
    contribution[positive] = (
        target[positive] * np.log(target[positive] / prediction[positive])
        - target[positive]
        + prediction[positive]
    )
    return float(2.0 * np.mean(contribution))


def zero_rate_calibration_error(
    y_true: ArrayLike,
    zero_probability: ArrayLike,
) -> float:
    """Return absolute error between observed and mean predicted zero rates."""

    target = _as_numeric_1d(y_true, "y_true")
    probability = _probability_vector(zero_probability, "zero_probability")
    _check_same_length(y_true=target, zero_probability=probability)
    if np.any(target < 0.0) or not np.allclose(target, np.round(target), rtol=0.0, atol=1e-12):
        raise ValueError("y_true must contain non-negative integer counts.")
    return float(abs(np.mean(target == 0.0) - np.mean(probability)))


def quantile_check_loss(
    y_true: ArrayLike,
    y_quantile: ArrayLike,
    *,
    quantile: float,
) -> float:
    """Return mean pinball (check) loss at a requested quantile."""

    target = _as_numeric_1d(y_true, "y_true")
    prediction = _as_numeric_1d(y_quantile, "y_quantile")
    _check_same_length(y_true=target, y_quantile=prediction)
    quantile = float(quantile)
    if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be finite and strictly between 0 and 1.")
    residual = target - prediction
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def _duration_inputs(
    time: ArrayLike,
    event: ArrayLike,
    prediction: ArrayLike,
    prediction_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = _as_numeric_1d(time, "time")
    observed = _binary_target(event, "event")
    predicted = _as_numeric_1d(prediction, prediction_name)
    _check_same_length(time=duration, event=observed, prediction=predicted)
    if np.any(duration < 0.0):
        raise ValueError("time must contain non-negative durations.")
    return duration, observed, predicted


def duration_concordance_index(
    time: ArrayLike,
    event: ArrayLike,
    risk_score: ArrayLike,
    *,
    higher_risk: bool = True,
) -> float:
    """Return Harrell's concordance index for right-censored outcomes.

    By default, larger scores indicate an earlier event.  Set ``higher_risk``
    to ``False`` for predictions such as expected duration, where smaller
    values indicate an earlier event.  A pair is comparable only when the
    shorter observed time is an event.  At equal recorded times, an event and
    a censoring observation are comparable with the event ordered first;
    event/event and censor/censor time ties remain excluded.  Tied scores
    receive half credit.
    """

    duration, observed, risk = _duration_inputs(time, event, risk_score, "risk_score")
    if not isinstance(higher_risk, (bool, np.bool_)):
        raise ValueError("higher_risk must be a boolean.")
    concordance = 0.0
    comparable = 0
    for left in range(len(duration) - 1):
        for right in range(left + 1, len(duration)):
            if duration[left] < duration[right] and observed[left] == 1:
                earlier, later = left, right
            elif duration[right] < duration[left] and observed[right] == 1:
                earlier, later = right, left
            elif duration[left] == duration[right] and observed[left] != observed[right]:
                earlier, later = (left, right) if observed[left] == 1 else (right, left)
            else:
                continue
            comparable += 1
            score_difference = risk[earlier] - risk[later]
            if (higher_risk and score_difference > 0.0) or (
                not higher_risk and score_difference < 0.0
            ):
                concordance += 1.0
            elif score_difference == 0.0:
                concordance += 0.5
    if comparable == 0:
        raise ValueError("Concordance is undefined because there are no comparable pairs.")
    return float(concordance / comparable)


def duration_brier_score(
    time: ArrayLike,
    event: ArrayLike,
    survival_probability: ArrayLike,
    *,
    horizon: float,
) -> float:
    """Return a horizon-specific survival Brier score.

    This dependency-light score uses observations whose survival state at the
    horizon is known.  Events at or before the horizon have target zero,
    observations followed beyond the horizon have target one, and observations
    censored at or before the horizon are excluded.  It is not an IPCW score.
    """

    duration, observed, survival = _duration_inputs(
        time, event, survival_probability, "survival_probability"
    )
    if np.any((survival < 0.0) | (survival > 1.0)):
        raise ValueError("survival_probability must contain probabilities between 0 and 1.")
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon < 0.0:
        raise ValueError("horizon must be finite and non-negative.")
    event_by_horizon = (observed == 1) & (duration <= horizon)
    survived_horizon = duration > horizon
    known = event_by_horizon | survived_horizon
    if not np.any(known):
        raise ValueError("No observations have a known survival state at the horizon.")
    target_survival = survived_horizon[known].astype(float)
    return float(np.mean((survival[known] - target_survival) ** 2))


def observed_outcome_rmse(
    selected: ArrayLike,
    outcome_true: ArrayLike,
    outcome_predicted: ArrayLike,
) -> float:
    """Return outcome RMSE among observations with an observed outcome."""

    selected_array = _binary_target(selected, "selected")
    target = _as_numeric_1d(outcome_true, "outcome_true", allow_nonfinite=True)
    prediction = _as_numeric_1d(
        outcome_predicted, "outcome_predicted", allow_nonfinite=True
    )
    _check_same_length(
        selected=selected_array,
        outcome_true=target,
        outcome_predicted=prediction,
    )
    observed = selected_array == 1
    if not np.any(observed):
        raise ValueError("Observed-outcome RMSE requires at least one selected observation.")
    if not np.all(np.isfinite(target[observed])) or not np.all(np.isfinite(prediction[observed])):
        raise ValueError("Observed outcomes and predictions must be finite when selected=1.")
    return float(np.sqrt(np.mean((target[observed] - prediction[observed]) ** 2)))


def selection_scores(
    selected: ArrayLike,
    selection_probability: ArrayLike,
    outcome_true: ArrayLike,
    outcome_predicted: ArrayLike,
    *,
    eps: float = 1e-15,
) -> dict[str, float]:
    """Score sample selection and its observed conditional outcome jointly."""

    return {
        "selection_brier_score": binary_brier_score(selected, selection_probability),
        "selection_log_loss": binary_log_loss(selected, selection_probability, eps=eps),
        "observed_outcome_rmse": observed_outcome_rmse(
            selected, outcome_true, outcome_predicted
        ),
    }


def score_predictions(
    y_true: ArrayLike,
    y_prediction: ArrayLike | Mapping[str, ArrayLike],
    *,
    outcome_kind: OutcomeKind | None = None,
    outcome: OutcomeKind | None = None,
    labels: Sequence[Any] | None = None,
    threshold: float = 0.5,
    eps: float = 1e-15,
    quantile: float | None = None,
    zero_probability: ArrayLike | None = None,
    event: ArrayLike | None = None,
    horizon: float | None = None,
    survival_probability: ArrayLike | None = None,
    outcome_true: ArrayLike | None = None,
    outcome_predicted: ArrayLike | None = None,
    higher_risk: bool = True,
    score_context: ArrayLike | Mapping[str, ArrayLike] | None = None,
    selection: ArrayLike | None = None,
    selected: ArrayLike | None = None,
    selection_indicator: ArrayLike | None = None,
    groups: ArrayLike | None = None,
) -> dict[str, float]:
    """Dispatch prediction scoring according to the limited-outcome family.

    Parameters
    ----------
    y_true, y_prediction
        Their interpretation depends on ``outcome_kind``.  They are outcomes
        and event probabilities for binary scoring, class probabilities for
        multiclass/ordinal scoring, continuous outcomes and predictions for
        continuous scoring, count outcomes and predicted means for count
        scoring, outcomes and predicted quantiles for quantile scoring,
        durations and risk scores (or expected durations with
        ``higher_risk=False``) for duration scoring, long-format chosen-item
        indicators and within-choice-set probabilities for choice scoring, and
        selection indicators and selection probabilities for selection scoring.
    outcome_kind, outcome
        Equivalent names for the outcome family.  Supply exactly one.  Valid
        values are ``binary``, ``choice``, ``multiclass``, ``ordinal``,
        ``continuous``, ``count``, ``quantile``, ``duration``, or ``selection``.

    Notes
    -----
    Count predictions may be an array of means or a mapping containing ``mean``
    and, optionally, ``zero_probability``.  Duration predictions may be an
    array or a mapping containing exactly one of ``risk_score`` and
    ``expected_duration``, plus an optional ``survival_probability``.  Duration
    Brier scoring also requires ``horizon``.  Selection scoring accepts direct
    arrays or an adapter-friendly prediction mapping with
    ``selection_probability`` and ``outcome`` (or ``observed_outcome``).  In
    mapping form, ``y_true`` is the observed outcome and the selection indicator
    is accepted directly through ``selection``/``selected``/
    ``selection_indicator`` or through ``score_context``. Grouped-choice scoring
    requires one non-missing ``groups`` label per alternative row.
    """

    if outcome_kind is not None and outcome is not None and outcome_kind != outcome:
        raise ValueError("outcome and outcome_kind cannot specify different families.")
    kind = outcome_kind if outcome_kind is not None else outcome
    if kind is None:
        raise ValueError("outcome or outcome_kind is required.")

    if kind == "binary":
        binary_target = _binary_target(y_true)
        balanced_accuracy = (
            binary_balanced_accuracy(y_true, y_prediction, threshold=threshold)
            if np.unique(binary_target).size == 2
            else np.nan
        )
        return {
            "log_loss": binary_log_loss(y_true, y_prediction, eps=eps),
            "brier_score": binary_brier_score(y_true, y_prediction),
            "accuracy": binary_accuracy(y_true, y_prediction, threshold=threshold),
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": (
                binary_roc_auc(y_true, y_prediction)
                if np.unique(binary_target).size == 2
                else np.nan
            ),
        }
    if kind == "choice":
        if groups is None:
            raise ValueError("groups is required when outcome_kind='choice'.")
        return {
            "log_loss": choice_log_loss(y_true, y_prediction, groups, eps=eps),
            "brier_score": choice_brier_score(y_true, y_prediction, groups),
            "accuracy": choice_accuracy(y_true, y_prediction, groups),
        }
    if kind in {"multiclass", "ordinal"}:
        codes, probabilities, resolved_labels = _multiclass_inputs(
            y_true, y_prediction, labels
        )
        del codes  # validation is shared by the public metric functions below
        scores = {
            "log_loss": multiclass_log_loss(
                y_true, y_prediction, labels=resolved_labels, eps=eps
            ),
            "brier_score": multiclass_brier_score(
                y_true, y_prediction, labels=resolved_labels
            ),
        }
        if kind == "ordinal":
            predicted_labels = np.asarray(resolved_labels, dtype=object)[
                np.argmax(probabilities, axis=1)
            ]
            scores.update(
                {
                    "ranked_probability_score": ranked_probability_score(
                        y_true, y_prediction, labels=resolved_labels
                    ),
                    "ordinal_mae": ordinal_mean_absolute_error(
                        y_true, predicted_labels, labels=resolved_labels
                    ),
                }
            )
        return scores
    if kind == "continuous":
        return {
            "mae": continuous_mean_absolute_error(y_true, y_prediction),
            "rmse": continuous_root_mean_squared_error(y_true, y_prediction),
            "bias": continuous_bias(y_true, y_prediction),
        }
    if kind == "count":
        if isinstance(y_prediction, Mapping):
            if "mean" not in y_prediction:
                raise ValueError("Count prediction mappings require mean.")
            count_prediction = y_prediction["mean"]
            mapped_zero_probability = y_prediction.get("zero_probability")
            if zero_probability is None:
                zero_probability = mapped_zero_probability
        else:
            count_prediction = y_prediction
        scores = {
            "mae": count_mean_absolute_error(y_true, count_prediction),
            "rmse": count_root_mean_squared_error(y_true, count_prediction),
            "poisson_deviance": poisson_deviance(y_true, count_prediction),
        }
        if zero_probability is not None:
            scores["zero_rate_calibration_error"] = zero_rate_calibration_error(
                y_true, zero_probability
            )
        return scores
    if kind == "quantile":
        if quantile is None:
            raise ValueError("quantile is required when outcome_kind='quantile'.")
        return {
            "check_loss": quantile_check_loss(
                y_true, y_prediction, quantile=quantile
            )
        }
    if kind == "duration":
        if event is None:
            raise ValueError("event is required when outcome_kind='duration'.")
        if isinstance(y_prediction, Mapping):
            has_risk = "risk_score" in y_prediction
            has_duration = "expected_duration" in y_prediction
            if has_risk == has_duration:
                raise ValueError(
                    "Duration prediction mappings require exactly one of risk_score "
                    "or expected_duration."
                )
            if has_risk:
                duration_prediction = y_prediction["risk_score"]
                score_direction = higher_risk
            else:
                duration_prediction = y_prediction["expected_duration"]
                score_direction = False
            if survival_probability is None:
                survival_probability = y_prediction.get("survival_probability")
        else:
            duration_prediction = y_prediction
            score_direction = higher_risk
        scores = {
            "concordance_index": duration_concordance_index(
                y_true, event, duration_prediction, higher_risk=score_direction
            )
        }
        if (horizon is None) != (survival_probability is None):
            raise ValueError(
                "horizon and survival_probability must be supplied together."
            )
        if horizon is not None and survival_probability is not None:
            scores["brier_score_at_horizon"] = duration_brier_score(
                y_true,
                event,
                survival_probability,
                horizon=horizon,
            )
        return scores
    if kind == "selection":
        if isinstance(y_prediction, Mapping):
            if "selection_probability" not in y_prediction:
                raise ValueError(
                    "Selection prediction mappings require selection_probability."
                )
            prediction_key = (
                "observed_outcome"
                if "observed_outcome" in y_prediction
                else "outcome"
            )
            if prediction_key not in y_prediction:
                raise ValueError(
                    "Selection prediction mappings require outcome or observed_outcome."
                )
            explicit_selection = next(
                (
                    value
                    for value in (selection, selected, selection_indicator)
                    if value is not None
                ),
                None,
            )
            if explicit_selection is not None:
                selected_values = explicit_selection
            elif score_context is None:
                raise ValueError(
                    "score_context must supply the selection indicator for "
                    "selection prediction mappings."
                )
            elif isinstance(score_context, Mapping):
                selection_key = next(
                    (
                        key
                        for key in ("selection", "selected", "selection_indicator")
                        if key in score_context
                    ),
                    None,
                )
                if selection_key is None:
                    raise ValueError(
                        "score_context requires selection, selected, or "
                        "selection_indicator."
                    )
                selected_values = score_context[selection_key]
            else:
                selected_values = score_context
            return selection_scores(
                selected_values,
                y_prediction["selection_probability"],
                y_true,
                y_prediction[prediction_key],
                eps=eps,
            )
        if outcome_true is None or outcome_predicted is None:
            raise ValueError(
                "outcome_true and outcome_predicted are required when "
                "outcome_kind='selection'."
            )
        return selection_scores(
            y_true,
            y_prediction,
            outcome_true,
            outcome_predicted,
            eps=eps,
        )
    supported = (
        "binary, choice, multiclass, ordinal, continuous, count, quantile, "
        "duration, selection"
    )
    raise ValueError(f"Unsupported outcome {kind!r}; expected one of {supported}.")
