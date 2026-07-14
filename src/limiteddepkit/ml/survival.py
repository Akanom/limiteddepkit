"""Censoring-aware validation metrics for time-to-event predictions.

Every IPCW metric in this module requires an explicit censoring distribution
estimated from training data.  Estimating censoring weights on the evaluation
fold would leak test-fold information and is intentionally unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CensoringDistribution",
    "DynamicAUCResult",
    "TimeDependentBrierResult",
    "cumulative_dynamic_auc",
    "fit_censoring_distribution",
    "integrated_brier_score",
    "ipcw_concordance_index",
    "time_dependent_brier_score",
    "time_dependent_brier_scores",
]


@dataclass(frozen=True)
class CensoringDistribution:
    """Reverse Kaplan-Meier estimate of training-fold censoring survival."""

    times: np.ndarray
    survival: np.ndarray
    n_samples: int

    def __post_init__(self) -> None:
        try:
            times = np.asarray(self.times, dtype=float)
            survival = np.asarray(self.survival, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Censoring distribution arrays must be numeric.") from exc
        if times.ndim != 1 or survival.ndim != 1 or len(times) == 0:
            raise ValueError("Censoring distribution arrays must be non-empty and 1D.")
        if len(times) != len(survival):
            raise ValueError("Censoring distribution arrays must have the same length.")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(survival)):
            raise ValueError("Censoring distribution arrays must be finite.")
        if np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("Censoring distribution times must be non-negative and increasing.")
        if np.any((survival < 0.0) | (survival > 1.0)) or np.any(np.diff(survival) > 1e-14):
            raise ValueError(
                "Censoring survival must be between 0 and 1 and non-increasing."
            )
        if (
            isinstance(self.n_samples, (bool, np.bool_))
            or int(self.n_samples) != self.n_samples
            or int(self.n_samples) < 1
        ):
            raise ValueError("n_samples must be a positive integer.")
        times = times.copy()
        survival = survival.copy()
        times.setflags(write=False)
        survival.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "survival", survival)
        object.__setattr__(self, "n_samples", int(self.n_samples))

    @property
    def max_time(self) -> float:
        """Return the largest follow-up time represented by the estimate."""

        return float(self.times[-1])

    def survival_at(self, values: Any) -> float | np.ndarray:
        """Evaluate the right-continuous censoring survival ``G(t)``.

        Values beyond training follow-up are rejected instead of extrapolating
        censoring weights into an unidentified region.
        """

        raw = np.asarray(values)
        scalar = raw.ndim == 0
        try:
            query = np.atleast_1d(raw).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Censoring-distribution query times must be numeric.") from exc
        if query.ndim != 1 or not np.all(np.isfinite(query)):
            raise ValueError("Censoring-distribution query times must be finite and 1D.")
        if np.any(query < 0.0):
            raise ValueError("Censoring-distribution query times cannot be negative.")
        tolerance = np.finfo(float).eps * max(1.0, abs(self.max_time)) * 8.0
        if np.any(query > self.max_time + tolerance):
            raise ValueError(
                "Evaluation time exceeds the training censoring-distribution support."
            )
        positions = np.searchsorted(self.times, query, side="right") - 1
        output = np.ones(len(query), dtype=float)
        selected = positions >= 0
        output[selected] = self.survival[positions[selected]]
        return float(output[0]) if scalar else output


@dataclass(frozen=True)
class TimeDependentBrierResult:
    """Pointwise IPCW Brier scores on a requested evaluation grid."""

    times: np.ndarray
    scores: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return one row per evaluation time."""

        return pd.DataFrame({"time": self.times, "brier_score": self.scores})


@dataclass(frozen=True)
class DynamicAUCResult:
    """Pointwise cumulative/dynamic AUC estimates."""

    times: np.ndarray
    auc: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return one row per evaluation time."""

        return pd.DataFrame({"time": self.times, "auc": self.auc})


def _duration_event_inputs(duration: Any, event: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        observed_time = np.asarray(duration, dtype=float)
        status = np.asarray(event, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration and event must be numeric.") from exc
    if observed_time.ndim != 1 or status.ndim != 1:
        raise ValueError("duration and event must be one-dimensional.")
    if len(observed_time) == 0 or len(observed_time) != len(status):
        raise ValueError("duration and event must have the same non-zero length.")
    if not np.all(np.isfinite(observed_time)) or not np.all(np.isfinite(status)):
        raise ValueError("duration and event must contain only finite values.")
    if np.any(observed_time < 0.0):
        raise ValueError("duration cannot contain negative values.")
    if not np.all(np.isin(status, (0.0, 1.0))):
        raise ValueError("event must contain only binary values 0 and 1.")
    return observed_time, status.astype(bool)


def fit_censoring_distribution(
    train_duration: Any,
    train_event: Any,
) -> CensoringDistribution:
    """Fit reverse Kaplan-Meier censoring survival on training observations.

    When event and censoring times tie, observed events are removed before the
    censoring hazard is calculated.  This reverse-KM convention treats the
    event of interest as occurring before censoring at a common recorded time.
    """

    duration, event = _duration_event_inputs(train_duration, train_event)
    times = np.unique(duration)
    survival = np.empty(len(times), dtype=float)
    current = 1.0
    for index, time in enumerate(times):
        at_risk = int(np.sum(duration >= time))
        observed_events = int(np.sum((duration == time) & event))
        censorings = int(np.sum((duration == time) & ~event))
        if censorings:
            censoring_risk = at_risk - observed_events
            if censoring_risk < censorings or censoring_risk <= 0:
                raise RuntimeError("Reverse Kaplan-Meier censoring risk set is invalid.")
            current *= 1.0 - censorings / censoring_risk
        survival[index] = current
    return CensoringDistribution(
        times=times.astype(float, copy=True),
        survival=survival,
        n_samples=len(duration),
    )


def _require_censoring(value: Any) -> CensoringDistribution:
    if not isinstance(value, CensoringDistribution):
        raise TypeError(
            "censoring must be an explicit CensoringDistribution fitted on training data."
        )
    return value


def _numeric_vector(values: Any, name: str, *, length: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must be one-dimensional with one value per observation.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_censoring_probability(
    censoring: CensoringDistribution,
    values: Any,
) -> np.ndarray:
    probability = np.atleast_1d(censoring.survival_at(values)).astype(float)
    if np.any(probability <= 0.0):
        raise ValueError(
            "The training censoring survival is zero at a required evaluation time."
        )
    return probability


def ipcw_concordance_index(
    duration: Any,
    event: Any,
    risk_score: Any,
    *,
    censoring: CensoringDistribution,
    tau: float | None = None,
    higher_risk: bool = True,
    tied_tolerance: float = 1e-8,
) -> float:
    """Return Uno-style IPCW concordance using training-fold censoring weights.

    An observed event is compared with later observations and with observations
    censored at the same recorded time.  Event-event ties are not comparable.
    Earlier events receive weight ``1 / G(time)^2``.
    """

    observed_time, status = _duration_event_inputs(duration, event)
    risk = _numeric_vector(risk_score, "risk_score", length=len(observed_time))
    censoring = _require_censoring(censoring)
    tied_tolerance = float(tied_tolerance)
    if not np.isfinite(tied_tolerance) or tied_tolerance < 0.0:
        raise ValueError("tied_tolerance must be finite and non-negative.")
    if not isinstance(higher_risk, (bool, np.bool_)):
        raise TypeError("higher_risk must be Boolean.")
    if not higher_risk:
        risk = -risk

    eligible_event = status.copy()
    if tau is not None:
        tau = float(tau)
        if not np.isfinite(tau) or tau < 0.0:
            raise ValueError("tau must be finite and non-negative.")
        censoring.survival_at(tau)
        eligible_event &= observed_time < tau

    event_indices = np.flatnonzero(eligible_event)
    if not len(event_indices):
        raise ValueError("IPCW concordance requires at least one eligible observed event.")
    censoring_probability = _positive_censoring_probability(
        censoring, observed_time[event_indices]
    )
    event_weights = 1.0 / censoring_probability**2

    numerator = 0.0
    denominator = 0.0
    for case, weight in zip(event_indices, event_weights, strict=True):
        comparable = (observed_time > observed_time[case]) | (
            (observed_time == observed_time[case]) & ~status
        )
        comparable[case] = False
        controls = np.flatnonzero(comparable)
        if not len(controls):
            continue
        difference = risk[case] - risk[controls]
        concordance = (difference > tied_tolerance).astype(float)
        concordance[np.abs(difference) <= tied_tolerance] = 0.5
        numerator += float(weight * np.sum(concordance))
        denominator += float(weight * len(controls))
    if denominator <= 0.0:
        raise ValueError("IPCW concordance has no comparable observation pairs.")
    return numerator / denominator


def _horizon(value: Any, *, name: str = "horizon") -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def time_dependent_brier_score(
    duration: Any,
    event: Any,
    survival_probability: Any,
    *,
    horizon: float,
    censoring: CensoringDistribution,
) -> float:
    """Return the IPCW Brier score for survival probability at one horizon."""

    observed_time, status = _duration_event_inputs(duration, event)
    probability = _numeric_vector(
        survival_probability,
        "survival_probability",
        length=len(observed_time),
    )
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("survival_probability must be between 0 and 1.")
    censoring = _require_censoring(censoring)
    horizon = _horizon(horizon)
    contribution = np.zeros(len(observed_time), dtype=float)
    cases = status & (observed_time <= horizon)
    controls = observed_time > horizon
    if np.any(cases):
        censoring_at_event = _positive_censoring_probability(
            censoring,
            observed_time[cases],
        )
        contribution[cases] = probability[cases] ** 2 / censoring_at_event
    if np.any(controls):
        censoring_at_horizon = _positive_censoring_probability(censoring, horizon)[0]
        contribution[controls] = (
            (1.0 - probability[controls]) ** 2 / censoring_at_horizon
        )
    if not np.any(cases | controls):
        raise ValueError("No survival status is identifiable at the requested horizon.")
    return float(np.mean(contribution))


def _evaluation_times(values: Any, *, minimum: int = 1) -> np.ndarray:
    try:
        times = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be numeric.") from exc
    if times.ndim != 1 or len(times) < minimum:
        raise ValueError(f"times must be one-dimensional with at least {minimum} values.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times must contain finite, non-negative values.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    return times


def _prediction_matrix(
    values: Any,
    *,
    name: str,
    nobs: int,
    ntimes: int,
    broadcast_vector: bool,
) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if matrix.ndim == 1 and broadcast_vector:
        if len(matrix) != nobs:
            raise ValueError(f"{name} must contain one value per observation.")
        matrix = np.repeat(matrix[:, None], ntimes, axis=1)
    if matrix.ndim != 2 or matrix.shape != (nobs, ntimes):
        raise ValueError(f"{name} must have shape (n_observations, n_times).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def time_dependent_brier_scores(
    duration: Any,
    event: Any,
    survival_probabilities: Any,
    *,
    times: Any,
    censoring: CensoringDistribution,
) -> TimeDependentBrierResult:
    """Return IPCW Brier scores across a strictly increasing time grid."""

    observed_time, status = _duration_event_inputs(duration, event)
    grid = _evaluation_times(times)
    matrix = _prediction_matrix(
        survival_probabilities,
        name="survival_probabilities",
        nobs=len(observed_time),
        ntimes=len(grid),
        broadcast_vector=False,
    )
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("survival_probabilities must be between 0 and 1.")
    scores = np.array(
        [
            time_dependent_brier_score(
                observed_time,
                status,
                matrix[:, index],
                horizon=time,
                censoring=censoring,
            )
            for index, time in enumerate(grid)
        ],
        dtype=float,
    )
    return TimeDependentBrierResult(times=grid.copy(), scores=scores)


def integrated_brier_score(
    duration: Any,
    event: Any,
    survival_probabilities: Any,
    *,
    times: Any,
    censoring: CensoringDistribution,
) -> float:
    """Return the time-normalized trapezoidal integral of IPCW Brier scores."""

    grid = _evaluation_times(times, minimum=2)
    result = time_dependent_brier_scores(
        duration,
        event,
        survival_probabilities,
        times=grid,
        censoring=censoring,
    )
    widths = np.diff(grid)
    areas = widths * (result.scores[:-1] + result.scores[1:]) / 2.0
    return float(np.sum(areas) / (grid[-1] - grid[0]))


def cumulative_dynamic_auc(
    duration: Any,
    event: Any,
    risk_scores: Any,
    *,
    times: Any,
    censoring: CensoringDistribution,
    tied_tolerance: float = 1e-8,
) -> DynamicAUCResult:
    """Return cumulative-case/dynamic-control AUC at each requested time.

    Cases are observations with an event by ``t`` and receive ``1/G(T)``
    training-censoring weights.  Controls are observations known to remain
    event-free beyond ``t``.  A one-dimensional risk score is broadcast over
    the evaluation grid; a matrix may provide time-specific risks.
    """

    observed_time, status = _duration_event_inputs(duration, event)
    grid = _evaluation_times(times)
    censoring = _require_censoring(censoring)
    matrix = _prediction_matrix(
        risk_scores,
        name="risk_scores",
        nobs=len(observed_time),
        ntimes=len(grid),
        broadcast_vector=True,
    )
    tied_tolerance = float(tied_tolerance)
    if not np.isfinite(tied_tolerance) or tied_tolerance < 0.0:
        raise ValueError("tied_tolerance must be finite and non-negative.")

    auc = np.empty(len(grid), dtype=float)
    for column, time in enumerate(grid):
        censoring.survival_at(time)
        cases = status & (observed_time <= time)
        controls = observed_time > time
        if not np.any(cases) or not np.any(controls):
            raise ValueError(
                "Every dynamic-AUC time requires at least one cumulative case and control."
            )
        case_weights = 1.0 / _positive_censoring_probability(
            censoring,
            observed_time[cases],
        )
        difference = matrix[cases, column][:, None] - matrix[controls, column][None, :]
        concordance = (difference > tied_tolerance).astype(float)
        concordance[np.abs(difference) <= tied_tolerance] = 0.5
        weighted = case_weights[:, None] * concordance
        denominator = float(np.sum(case_weights) * np.sum(controls))
        auc[column] = float(np.sum(weighted) / denominator)
    return DynamicAUCResult(times=grid.copy(), auc=auc)
