"""Fixed-boundary censored quantile regression.

The estimator minimizes the Powell censored-quantile check-loss objective.
Because the objective is non-convex and non-smooth after censoring, fitting uses
multiple starts and derivative-free local searches. Inference is intentionally
absent unless the caller requests an experimental pairs bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog, minimize
from scipy.stats import norm

from .ordinal import _as_2d_array


def _check_loss(residual: np.ndarray, quantile: float) -> float:
    return float(
        np.sum(
            np.where(
                residual >= 0.0,
                quantile * residual,
                (quantile - 1.0) * residual,
            )
        )
    )


def _linear_quantile_start(
    design: np.ndarray, outcome: np.ndarray, quantile: float
) -> np.ndarray:
    """Solve ordinary linear quantile regression as a HiGHS linear program."""
    n_obs, n_features = design.shape
    objective = np.concatenate(
        [
            np.zeros(n_features),
            np.full(n_obs, quantile),
            np.full(n_obs, 1.0 - quantile),
        ]
    )
    equality = sparse.hstack(
        [sparse.csr_matrix(design), sparse.eye(n_obs), -sparse.eye(n_obs)],
        format="csr",
    )
    bounds = [(None, None)] * n_features + [(0.0, None)] * (2 * n_obs)
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=outcome,
        bounds=bounds,
        method="highs",
    )
    if not result.success or not np.isfinite(result.x[:n_features]).all():
        raise RuntimeError(f"Quantile-regression initialization failed: {result.message}")
    return np.asarray(result.x[:n_features], dtype=float)


def _prediction_design(
    X: Any, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, pd.Index]:
    design, names = _as_2d_array(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(
            f"X must contain {len(feature_names)} regressors; "
            f"received {design.shape[1]}."
        )
    if isinstance(X, pd.DataFrame) and tuple(names) != feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
    return design, index


@dataclass(frozen=True)
class CensoredQuantileRegressionResult:
    """Fitted fixed-boundary censored quantile regression result."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    covariance_type: str
    converged: bool
    objective_value: float
    quantile: float
    lower: float | None
    upper: float | None
    nobs: int
    n_censored_left: int
    n_censored_right: int
    n_starts: int
    feature_names: tuple[str, ...]
    optimizer_result: Any
    optimizer_results: tuple[Any, ...]
    bootstrap_estimates: pd.DataFrame | None

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def df_resid(self) -> int:
        return self.nobs - self.n_params

    @property
    def backend(self) -> str:
        return "experimental-powell-cqr"

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        if self.bootstrap_estimates is None:
            return pd.DataFrame(
                np.nan, index=self.feature_names, columns=["lower", "upper"]
            )
        tail = (1.0 - level) / 2.0
        return pd.DataFrame(
            {
                "lower": self.bootstrap_estimates.quantile(tail),
                "upper": self.bootstrap_estimates.quantile(1.0 - tail),
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def predict_latent(self, X: Any) -> pd.Series:
        """Predict the latent-outcome conditional quantile ``X beta``."""
        design, index = _prediction_design(X, self.feature_names)
        values = design @ self.params.to_numpy(dtype=float)
        return pd.Series(values, index=index, name="latent_quantile")

    def predict(self, X: Any) -> pd.Series:
        """Predict the observed conditional quantile after fixed censoring."""
        latent = self.predict_latent(X)
        values = latent.to_numpy(copy=True)
        if self.lower is not None:
            values = np.maximum(values, self.lower)
        if self.upper is not None:
            values = np.minimum(values, self.upper)
        return pd.Series(values, index=latent.index, name="predicted_quantile")


class CensoredQuantileRegression:
    """Powell-style quantile regression with fixed censoring boundaries.

    At least one of ``lower`` or ``upper`` must be supplied. For latent
    quantile ``X beta``, the observed quantile is clipped to those known fixed
    bounds. This is not a random-censoring survival quantile estimator.
    """

    def __init__(
        self,
        *,
        quantile: float = 0.5,
        lower: float | None = 0.0,
        upper: float | None = None,
    ) -> None:
        if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be strictly between zero and one.")
        if lower is None and upper is None:
            raise ValueError("At least one censoring boundary is required.")
        if lower is not None and not np.isfinite(lower):
            raise ValueError("lower must be finite when supplied.")
        if upper is not None and not np.isfinite(upper):
            raise ValueError("upper must be finite when supplied.")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower must be strictly less than upper.")
        self.quantile = float(quantile)
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        n_starts: int = 7,
        maxiter: int = 2_000,
        tolerance: float = 1e-8,
        random_state: int | None = 0,
        parameter_bounds: Sequence[tuple[float | None, float | None]] | None = None,
        n_bootstrap: int = 0,
    ) -> CensoredQuantileRegressionResult:
        """Fit the non-convex censored check-loss objective.

        ``n_bootstrap=0`` reports no inferential statistics. Values of at least
        20 request an experimental iid pairs-bootstrap covariance and percentile
        confidence intervals.
        """
        design, feature_names = _as_2d_array(X)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X feature names must be unique.")
        outcome_values = np.asarray(y)
        if outcome_values.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        try:
            outcome = outcome_values.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain numeric values.") from error
        if outcome.size != design.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if not np.isfinite(outcome).all():
            raise ValueError("y contains missing or non-finite values.")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("X must have full column rank.")
        if design.shape[0] <= design.shape[1]:
            raise ValueError("The number of observations must exceed the regressors.")
        if isinstance(n_starts, bool) or not isinstance(n_starts, (int, np.integer)):
            raise ValueError("n_starts must be a positive integer.")
        if n_starts < 1:
            raise ValueError("n_starts must be a positive integer.")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
            raise ValueError("maxiter must be a positive integer.")
        if maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if isinstance(n_bootstrap, bool) or not isinstance(
            n_bootstrap, (int, np.integer)
        ):
            raise ValueError("n_bootstrap must be zero or an integer of at least 20.")
        if n_bootstrap != 0 and n_bootstrap < 20:
            raise ValueError("n_bootstrap must be zero or an integer of at least 20.")

        scale = max(1.0, float(np.max(np.abs(outcome))))
        boundary_tolerance = 1e-10 * scale
        if self.lower is not None and np.any(outcome < self.lower - boundary_tolerance):
            raise ValueError("y contains values below the lower censoring boundary.")
        if self.upper is not None and np.any(outcome > self.upper + boundary_tolerance):
            raise ValueError("y contains values above the upper censoring boundary.")
        left_censored = (
            np.isclose(outcome, self.lower, rtol=0.0, atol=boundary_tolerance)
            if self.lower is not None
            else np.zeros(outcome.size, dtype=bool)
        )
        right_censored = (
            np.isclose(outcome, self.upper, rtol=0.0, atol=boundary_tolerance)
            if self.upper is not None
            else np.zeros(outcome.size, dtype=bool)
        )
        interior = ~(left_censored | right_censored)
        if interior.sum() <= design.shape[1]:
            raise ValueError("Too few uncensored observations identify the quantile model.")
        if np.linalg.matrix_rank(design[interior]) < design.shape[1]:
            raise ValueError("Uncensored X must have full column rank.")

        bounds: list[tuple[float | None, float | None]] | None = None
        if parameter_bounds is not None:
            bounds = list(parameter_bounds)
            if len(bounds) != design.shape[1]:
                raise ValueError("parameter_bounds must provide one pair per regressor.")
            for lower_bound, upper_bound in bounds:
                if lower_bound is not None and not np.isfinite(lower_bound):
                    raise ValueError("Finite or None coefficient bounds are required.")
                if upper_bound is not None and not np.isfinite(upper_bound):
                    raise ValueError("Finite or None coefficient bounds are required.")
                if (
                    lower_bound is not None
                    and upper_bound is not None
                    and lower_bound >= upper_bound
                ):
                    raise ValueError("Each lower coefficient bound must be below its upper.")

        def observed_prediction(
            parameters: np.ndarray, fit_design: np.ndarray = design
        ) -> np.ndarray:
            prediction = fit_design @ parameters
            if self.lower is not None:
                prediction = np.maximum(prediction, self.lower)
            if self.upper is not None:
                prediction = np.minimum(prediction, self.upper)
            return prediction

        def objective(parameters: np.ndarray) -> float:
            return _check_loss(
                outcome - observed_prediction(parameters), self.quantile
            )

        ordinary_start = _linear_quantile_start(design, outcome, self.quantile)
        interior_start = _linear_quantile_start(
            design[interior], outcome[interior], self.quantile
        )
        least_squares_start = np.linalg.lstsq(design, outcome, rcond=None)[0]
        candidate_starts = [ordinary_start, interior_start, least_squares_start]
        generator = np.random.default_rng(random_state)
        while len(candidate_starts) < n_starts:
            perturbation = generator.normal(
                scale=0.35 * (1.0 + np.abs(ordinary_start)),
                size=design.shape[1],
            )
            candidate_starts.append(ordinary_start + perturbation)
        candidate_starts = candidate_starts[:n_starts]

        def clip_to_bounds(parameters: np.ndarray) -> np.ndarray:
            if bounds is None:
                return np.asarray(parameters, dtype=float)
            clipped = np.asarray(parameters, dtype=float).copy()
            for index, (lower_bound, upper_bound) in enumerate(bounds):
                if lower_bound is not None:
                    clipped[index] = max(clipped[index], lower_bound)
                if upper_bound is not None:
                    clipped[index] = min(clipped[index], upper_bound)
            return clipped

        optimizer_results: list[OptimizeResult] = []
        for raw_start in candidate_starts:
            start = clip_to_bounds(raw_start)
            start_value = objective(start)
            result = minimize(
                objective,
                start,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": int(maxiter),
                    "xatol": float(tolerance),
                    "fatol": float(tolerance),
                },
            )
            if not np.isfinite(result.fun) or result.fun > start_value:
                result = OptimizeResult(
                    x=start,
                    fun=start_value,
                    success=True,
                    message="The quantile-regression initializer dominated local search.",
                )
            optimizer_results.append(result)

        optimizer_result = min(optimizer_results, key=lambda result: float(result.fun))
        parameters = np.asarray(optimizer_result.x, dtype=float)
        converged = bool(optimizer_result.success and np.isfinite(optimizer_result.fun))
        if not converged:
            raise RuntimeError(
                f"Censored quantile optimization failed: {optimizer_result.message}"
            )
        latent_prediction = design @ parameters
        active = np.ones(outcome.size, dtype=bool)
        if self.lower is not None:
            active &= latent_prediction > self.lower + boundary_tolerance
        if self.upper is not None:
            active &= latent_prediction < self.upper - boundary_tolerance
        if active.sum() <= design.shape[1] or np.linalg.matrix_rank(
            design[active]
        ) < design.shape[1]:
            raise RuntimeError(
                "The fitted latent quantile is censored on too much of the design; "
                "the coefficient vector is not identified."
            )

        bootstrap_frame: pd.DataFrame | None = None
        covariance_type = "not-estimated"
        covariance = np.full((design.shape[1], design.shape[1]), np.nan)
        standard_errors = np.full(design.shape[1], np.nan)
        zstats = np.full(design.shape[1], np.nan)
        pvalues = np.full(design.shape[1], np.nan)
        inference_valid = False
        if n_bootstrap:
            bootstrap_parameters: list[np.ndarray] = []
            for _ in range(int(n_bootstrap)):
                rows = generator.integers(0, outcome.size, size=outcome.size)
                bootstrap_design = design[rows]
                bootstrap_outcome = outcome[rows]
                if np.linalg.matrix_rank(bootstrap_design) < design.shape[1]:
                    continue

                def bootstrap_objective(
                    candidate: np.ndarray,
                    fit_design: np.ndarray = bootstrap_design,
                    fit_outcome: np.ndarray = bootstrap_outcome,
                ) -> float:
                    return _check_loss(
                        fit_outcome - observed_prediction(candidate, fit_design),
                        self.quantile,
                    )

                bootstrap_fit = minimize(
                    bootstrap_objective,
                    parameters,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={
                        "maxiter": int(maxiter),
                        "xatol": max(float(tolerance), 1e-7),
                        "fatol": max(float(tolerance), 1e-7),
                    },
                )
                if bootstrap_fit.success and np.isfinite(bootstrap_fit.x).all():
                    bootstrap_parameters.append(
                        np.asarray(bootstrap_fit.x, dtype=float)
                    )
            required = max(design.shape[1] + 2, int(np.ceil(0.8 * n_bootstrap)))
            if len(bootstrap_parameters) >= required:
                bootstrap_frame = pd.DataFrame(
                    bootstrap_parameters, columns=feature_names
                )
                covariance = np.cov(bootstrap_frame.to_numpy(), rowvar=False, ddof=1)
                covariance = np.atleast_2d(covariance)
                standard_errors = np.sqrt(np.diag(covariance))
                inference_valid = bool(
                    np.isfinite(covariance).all()
                    and np.all(standard_errors > 0.0)
                    and np.linalg.matrix_rank(covariance) == design.shape[1]
                )
                if inference_valid:
                    zstats = parameters / standard_errors
                    pvalues = 2.0 * norm.sf(np.abs(zstats))
                    covariance_type = "iid-pairs-bootstrap"

        labels = list(feature_names)
        return CensoredQuantileRegressionResult(
            params=pd.Series(parameters, index=labels, name="estimate"),
            covariance=pd.DataFrame(covariance, index=labels, columns=labels),
            standard_errors=pd.Series(standard_errors, index=labels, name="std_err"),
            zstats=pd.Series(zstats, index=labels, name="z"),
            pvalues=pd.Series(pvalues, index=labels, name="p_value"),
            inference_valid=inference_valid,
            covariance_type=covariance_type,
            converged=converged,
            objective_value=float(optimizer_result.fun),
            quantile=self.quantile,
            lower=self.lower,
            upper=self.upper,
            nobs=int(outcome.size),
            n_censored_left=int(left_censored.sum()),
            n_censored_right=int(right_censored.sum()),
            n_starts=len(optimizer_results),
            feature_names=tuple(feature_names),
            optimizer_result=optimizer_result,
            optimizer_results=tuple(optimizer_results),
            bootstrap_estimates=bootstrap_frame,
        )
