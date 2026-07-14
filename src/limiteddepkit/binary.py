"""Validated binary Logit maximum-likelihood estimator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.special import expit
from scipy.stats import norm

from .ordinal import _as_2d_array, _numerical_jacobian


def _validate_fit_data(X: Any, y: Any) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Validate a binary-response design and return numeric arrays."""
    design, feature_names = _as_2d_array(X)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("X feature names must be unique after conversion to strings.")

    outcomes = np.asarray(y)
    if outcomes.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if outcomes.size != design.shape[0]:
        raise ValueError("X and y must contain the same number of observations.")
    if pd.isna(outcomes).any():
        raise ValueError("y contains missing values.")
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("y must contain binary outcomes coded as 0/1.")
    outcomes = outcomes.astype(float)
    if np.unique(outcomes).size != 2:
        raise ValueError("y must contain observations from both binary outcome classes.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Binary-response inference requires more observations than regressors.")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("X is rank deficient; binary-response parameters are not identified.")
    if _has_separation(design, outcomes):
        raise ValueError(
            "The data exhibit complete or quasi-complete separation; a finite "
            "unpenalized maximum-likelihood estimate does not exist."
        )
    return design, feature_names, outcomes


def _has_separation(design: np.ndarray, outcomes: np.ndarray) -> bool:
    """Return whether a nonzero direction weakly separates the two classes.

    With a full-column-rank design, separation exists exactly when a nonzero
    vector ``b`` satisfies ``(2*y - 1) * X @ b >= 0``.  The homogeneous system
    is checked by scaling each coefficient to either +1 or -1 in turn.
    """
    signed_design = (2.0 * outcomes - 1.0)[:, None] * design
    constraints = -signed_design
    right_hand_side = np.zeros(design.shape[0], dtype=float)
    objective = np.zeros(design.shape[1], dtype=float)
    for column in range(design.shape[1]):
        for sign in (-1.0, 1.0):
            bounds: list[tuple[float | None, float | None]] = [
                (None, None) for _ in range(design.shape[1])
            ]
            bounds[column] = (1.0, None) if sign > 0 else (None, -1.0)
            feasibility = linprog(
                objective,
                A_ub=constraints,
                b_ub=right_hand_side,
                bounds=bounds,
                method="highs",
            )
            if feasibility.success:
                return True
    return False


def _validate_prediction_data(
    X: Any, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, pd.Index]:
    design, names = _as_2d_array(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(f"X has {design.shape[1]} columns; expected {len(feature_names)}.")
    if isinstance(X, pd.DataFrame) and tuple(names) != feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(design.shape[0])
    return design, index


def _invert_information(information: np.ndarray) -> np.ndarray:
    """Invert a well-identified observed-information matrix."""
    information = (np.asarray(information, dtype=float) + information.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(information)
    if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0.0:
        raise RuntimeError("The observed-information matrix is non-finite or non-positive.")
    if eigenvalues[0] <= max(1e-10 * eigenvalues[-1], 1e-12):
        raise RuntimeError(
            "The observed-information matrix is singular or ill-conditioned; "
            "coefficient inference is not reliable."
        )
    return np.linalg.inv(information)


def _binary_margins(
    result: Any,
    X: Any,
    *,
    at: str | Mapping[str, float] = "overall",
    kind: str = "probability",
) -> pd.Series:
    """Shared representative-value margins for binary response results."""
    values, _ = _validate_prediction_data(X, result.feature_names)
    if kind not in {"probability", "marginal_effect"}:
        raise ValueError("kind must be 'probability' or 'marginal_effect'.")

    evaluation: pd.DataFrame | np.ndarray
    if isinstance(at, str) and at == "overall":
        evaluation = X if isinstance(X, pd.DataFrame) else values
    elif isinstance(at, str) and at == "mean":
        evaluation = pd.DataFrame([values.mean(axis=0)], columns=result.feature_names)
    elif isinstance(at, Mapping):
        unknown = set(at) - set(result.feature_names)
        if unknown:
            raise ValueError(f"Unknown covariates in at: {sorted(unknown)}.")
        representative = {
            feature: values[:, index].mean()
            for index, feature in enumerate(result.feature_names)
        }
        for feature, value in at.items():
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError("User-specified margin values must be finite.")
            representative[feature] = numeric_value
        evaluation = pd.DataFrame([representative], columns=result.feature_names)
    else:
        raise ValueError("at must be 'overall', 'mean', or a covariate-value mapping.")

    if kind == "probability":
        probabilities = result.predict_proba(evaluation).mean(axis=0)
        probabilities.index.name = "category"
        return probabilities.rename("estimate")
    effects = result.marginal_effects(evaluation).mean(axis=0)
    effects.index.name = "feature"
    return effects.rename("estimate")


def _binary_ame_inference(
    result: Any,
    X: Any,
    *,
    scale_function: Any,
    level: float,
) -> pd.DataFrame:
    """Delta-method inference for average continuous-regressor effects."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one.")
    design, _ = _validate_prediction_data(X, result.feature_names)
    effect_features = [
        feature
        for feature in result.feature_names
        if feature not in result.constant_features
    ]
    effect_indices = [result.feature_names.index(feature) for feature in effect_features]
    if not effect_features:
        return pd.DataFrame(
            columns=[
                "estimate",
                "standard_error",
                "z_stat",
                "p_value",
                "lower",
                "upper",
            ],
            index=pd.Index([], name="feature"),
            dtype=float,
        )

    def average_effect(parameters: np.ndarray) -> np.ndarray:
        average_scale = float(np.mean(scale_function(design @ parameters)))
        return average_scale * parameters[effect_indices]

    parameters = result.params.to_numpy(dtype=float)
    estimates = average_effect(parameters)
    jacobian = _numerical_jacobian(average_effect, parameters)
    covariance = jacobian @ result.covariance.to_numpy(dtype=float) @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    zstats = np.divide(
        estimates,
        standard_errors,
        out=np.full_like(estimates, np.nan),
        where=standard_errors > 0.0,
    )
    pvalues = 2.0 * norm.sf(np.abs(zstats))
    critical = float(norm.ppf(0.5 + level / 2.0))
    return pd.DataFrame(
        {
            "estimate": estimates,
            "standard_error": standard_errors,
            "z_stat": zstats,
            "p_value": pvalues,
            "lower": estimates - critical * standard_errors,
            "upper": estimates + critical * standard_errors,
        },
        index=pd.Index(effect_features, name="feature"),
    )


@dataclass(frozen=True)
class BinaryLogitResult:
    """Fitted binary Logit result."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    constant_features: tuple[str, ...]
    score_norm: float
    optimizer_result: Any

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
    def aic(self) -> float:
        return -2.0 * self.loglike + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglike + np.log(self.nobs) * self.n_params

    @property
    def inference_valid(self) -> bool:
        return True

    @property
    def covariance_type(self) -> str:
        return "observed-information"

    @property
    def backend(self) -> str:
        return "native-mle"

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def predict_proba(self, X: Any) -> pd.DataFrame:
        design, index = _validate_prediction_data(X, self.feature_names)
        probabilities = expit(design @ self.params.to_numpy(dtype=float))
        return pd.DataFrame({0: 1.0 - probabilities, 1: probabilities}, index=index)

    def predict(self, X: Any, *, threshold: float = 0.5) -> pd.Series:
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be finite and strictly between zero and one.")
        probabilities = self.predict_proba(X)[1]
        return (probabilities >= threshold).astype(int).rename("prediction")

    def marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return derivatives of ``P(y=1)`` with respect to continuous regressors."""
        design, index = _validate_prediction_data(X, self.feature_names)
        probabilities = expit(design @ self.params.to_numpy(dtype=float))
        scale = probabilities * (1.0 - probabilities)
        effect_features = [
            feature for feature in self.feature_names if feature not in self.constant_features
        ]
        effect_indices = [self.feature_names.index(feature) for feature in effect_features]
        effects = scale[:, None] * self.params.to_numpy(dtype=float)[None, effect_indices]
        return pd.DataFrame(effects, index=index, columns=effect_features)

    def average_marginal_effects(self, X: Any) -> pd.Series:
        return self.marginal_effects(X).mean(axis=0).rename("estimate")

    def average_marginal_effects_inference(
        self, X: Any, *, level: float = 0.95
    ) -> pd.DataFrame:
        """Return delta-method inference for average marginal effects."""

        def logistic_scale(values: np.ndarray) -> np.ndarray:
            probability = expit(values)
            return probability * (1.0 - probability)

        return _binary_ame_inference(
            self, X, scale_function=logistic_scale, level=level
        )

    def margins(
        self,
        X: Any,
        *,
        at: str | Mapping[str, float] = "overall",
        kind: str = "probability",
    ) -> pd.Series:
        """Evaluate average or representative binary probabilities/effects."""
        return _binary_margins(self, X, at=at, kind=kind)

    def lincom(
        self,
        weights: Mapping[str, float],
        *,
        value: float = 0.0,
        level: float = 0.95,
    ) -> pd.Series:
        from .postestimation import lincom

        return lincom(self, weights, value=value, level=level)

    def wald_test(
        self,
        restrictions: Mapping[str, float] | Sequence[Mapping[str, float]],
        *,
        values: float | Sequence[float] = 0.0,
    ) -> pd.Series:
        from .postestimation import wald_test

        return wald_test(self, restrictions, values=values)


class BinaryLogit:
    """Binary Logit estimated by unpenalized maximum likelihood."""

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> BinaryLogitResult:
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        design, feature_names, outcomes = _validate_fit_data(X, y)

        def negative_loglike(beta: np.ndarray) -> float:
            linear = design @ beta
            return float(np.sum(np.logaddexp(0.0, linear) - outcomes * linear))

        def gradient(beta: np.ndarray) -> np.ndarray:
            return design.T @ (expit(design @ beta) - outcomes)

        optimizer_result = minimize(
            negative_loglike,
            np.zeros(design.shape[1], dtype=float),
            jac=gradient,
            method="BFGS",
            options={"maxiter": int(maxiter), "gtol": tolerance},
        )
        score_norm = float(np.max(np.abs(gradient(optimizer_result.x))))
        converged = bool(
            optimizer_result.success or score_norm <= max(10.0 * tolerance, 1e-7)
        )
        if (
            not converged
            or not np.isfinite(optimizer_result.fun)
            or not np.isfinite(optimizer_result.x).all()
        ):
            raise RuntimeError(
                "Binary Logit optimization failed: " + str(optimizer_result.message)
            )

        coefficients = np.asarray(optimizer_result.x, dtype=float)
        fitted_probabilities = expit(design @ coefficients)
        weights = fitted_probabilities * (1.0 - fitted_probabilities)
        information = design.T @ (weights[:, None] * design)
        covariance = _invert_information(information)
        standard_errors = np.sqrt(np.diag(covariance))
        zstats = coefficients / standard_errors
        pvalues = 2.0 * norm.sf(np.abs(zstats))

        params = pd.Series(coefficients, index=feature_names, name="estimate")
        covariance_frame = pd.DataFrame(covariance, index=feature_names, columns=feature_names)
        standard_errors_series = pd.Series(standard_errors, index=feature_names, name="std_err")
        zstats_series = pd.Series(zstats, index=feature_names, name="z")
        pvalues_series = pd.Series(pvalues, index=feature_names, name="p_value")

        return BinaryLogitResult(
            params=params,
            covariance=covariance_frame,
            standard_errors=standard_errors_series,
            zstats=zstats_series,
            pvalues=pvalues_series,
            converged=converged,
            loglike=-float(optimizer_result.fun),
            nobs=int(design.shape[0]),
            feature_names=tuple(feature_names),
            constant_features=tuple(
                feature_names[index]
                for index in range(design.shape[1])
                if np.ptp(design[:, index]) <= 1e-12
            ),
            score_norm=score_norm,
            optimizer_result=optimizer_result,
        )
