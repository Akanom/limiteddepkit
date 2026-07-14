"""Gaussian finite-mixture regression.

This module retains the historical ``SwitchingRegression`` name as a
deprecated alias. The model has independent latent classes; it is not a
Markov-switching or observed-regime regression.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp, softmax
from scipy.stats import norm

from .ordinal import _as_2d_array, _numerical_hessian


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
class GaussianMixtureRegressionResult:
    """Fitted independent-class Gaussian mixture regression."""

    params_regimes: tuple[pd.Series, ...]
    sigma_regimes: tuple[float, ...]
    mixture_probs: np.ndarray
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    converged: bool
    score_norm: float
    loglike: float
    nobs: int
    n_regimes: int
    feature_names: tuple[str, ...]
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        values: list[float] = []
        labels: list[str] = []
        for regime, params in enumerate(self.params_regimes, start=1):
            values.extend(params.to_numpy(dtype=float))
            labels.extend(f"regime{regime}:{name}" for name in self.feature_names)
        values.extend(np.log(self.sigma_regimes))
        labels.extend(
            f"regime{regime}:log_sigma" for regime in range(1, self.n_regimes + 1)
        )
        reference_probability = float(self.mixture_probs[-1])
        values.extend(np.log(self.mixture_probs[:-1] / reference_probability))
        labels.extend(
            f"mixture_logit:{regime}_vs_{self.n_regimes}"
            for regime in range(1, self.n_regimes)
        )
        return pd.Series(values, index=labels, name="estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

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
    def covariance_type(self) -> str:
        return "observed-information"

    @property
    def backend(self) -> str:
        return "experimental-native-mixture-mle"

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = float(norm.ppf(0.5 + level / 2.0))
        return pd.DataFrame(
            {
                "lower": self.all_params - critical * self.standard_errors,
                "upper": self.all_params + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def predict_component_means(self, X: Any) -> pd.DataFrame:
        """Return one conditional mean per latent component."""
        design, index = _prediction_design(X, self.feature_names)
        values = np.column_stack(
            [design @ params.to_numpy(dtype=float) for params in self.params_regimes]
        )
        return pd.DataFrame(
            values,
            index=index,
            columns=[f"regime_{index + 1}" for index in range(self.n_regimes)],
        )

    def predict(self, X: Any) -> pd.Series:
        """Return the mixture-averaged conditional outcome mean."""
        component_means = self.predict_component_means(X)
        values = component_means.to_numpy() @ self.mixture_probs
        return pd.Series(values, index=component_means.index, name="predicted")

    def predict_membership(self, X: Any, y: Any) -> pd.DataFrame:
        """Return posterior component probabilities for observed outcomes."""
        component_means = self.predict_component_means(X)
        outcomes = np.asarray(y)
        if outcomes.ndim != 1 or outcomes.size != len(component_means):
            raise ValueError("y must be one-dimensional and match X.")
        try:
            outcomes = outcomes.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain numeric values.") from error
        if not np.isfinite(outcomes).all():
            raise ValueError("y contains missing or non-finite values.")
        log_joint = np.log(self.mixture_probs)[None, :] + norm.logpdf(
            outcomes[:, None],
            loc=component_means.to_numpy(),
            scale=np.asarray(self.sigma_regimes)[None, :],
        )
        posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        return pd.DataFrame(
            posterior, index=component_means.index, columns=component_means.columns
        )


class GaussianMixtureRegression:
    """Independent latent-class Gaussian regression mixture.

    The component variances are constrained away from zero because an
    unrestricted Gaussian-mixture likelihood is unbounded at collapsed
    components. Multiple deterministic/randomized starts reduce, but cannot
    eliminate, the local-optimum risk.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        *,
        n_starts: int = 5,
        random_state: int | None = 0,
        min_sigma: float | None = None,
    ) -> None:
        if isinstance(n_regimes, bool) or not isinstance(n_regimes, (int, np.integer)):
            raise ValueError("n_regimes must be an integer of at least two.")
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least two.")
        if isinstance(n_starts, bool) or not isinstance(n_starts, (int, np.integer)):
            raise ValueError("n_starts must be a positive integer.")
        if n_starts < 1:
            raise ValueError("n_starts must be a positive integer.")
        if min_sigma is not None and (not np.isfinite(min_sigma) or min_sigma <= 0.0):
            raise ValueError("min_sigma must be finite and positive when supplied.")
        self.n_regimes = int(n_regimes)
        self.n_starts = int(n_starts)
        self.random_state = random_state
        self.min_sigma = None if min_sigma is None else float(min_sigma)

    def fit(
        self, X: Any, y: Any, *, maxiter: int = 500
    ) -> GaussianMixtureRegressionResult:
        design, feature_names = _as_2d_array(X)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X feature names must be unique.")
        outcomes = np.asarray(y)
        if outcomes.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        try:
            outcomes = outcomes.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain numeric values.") from error
        if outcomes.size != design.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if not np.isfinite(outcomes).all():
            raise ValueError("y contains missing or non-finite values.")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("X must have full column rank.")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
            raise ValueError("maxiter must be a positive integer.")
        if maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")

        n_obs, n_features = design.shape
        n_parameters = self.n_regimes * (n_features + 1) + self.n_regimes - 1
        if n_obs <= n_parameters:
            raise ValueError("The number of observations must exceed the parameters.")
        pooled_beta = np.linalg.lstsq(design, outcomes, rcond=None)[0]
        pooled_residual = outcomes - design @ pooled_beta
        pooled_sigma = float(np.sqrt(np.mean(pooled_residual**2)))
        if not np.isfinite(pooled_sigma) or pooled_sigma <= np.finfo(float).eps:
            raise ValueError("y has no residual variation after projection on X.")
        sigma_floor = (
            self.min_sigma
            if self.min_sigma is not None
            else max(1e-6, 0.01 * pooled_sigma)
        )
        sigma_ceiling = max(100.0 * pooled_sigma, 10.0 * sigma_floor)

        def unpack(
            parameters: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            beta_end = self.n_regimes * n_features
            betas = parameters[:beta_end].reshape(self.n_regimes, n_features)
            log_sigmas = parameters[beta_end : beta_end + self.n_regimes]
            sigmas = np.exp(log_sigmas)
            logits = np.append(parameters[beta_end + self.n_regimes :], 0.0)
            probabilities = softmax(logits)
            return betas, log_sigmas, sigmas, probabilities

        def log_components(parameters: np.ndarray) -> np.ndarray:
            betas, _, sigmas, probabilities = unpack(parameters)
            means = design @ betas.T
            return np.log(probabilities)[None, :] + norm.logpdf(
                outcomes[:, None], loc=means, scale=sigmas[None, :]
            )

        def negative_loglike(parameters: np.ndarray) -> float:
            value = -float(np.sum(logsumexp(log_components(parameters), axis=1)))
            return value if np.isfinite(value) else 1e300

        def gradient(parameters: np.ndarray) -> np.ndarray:
            betas, _, sigmas, probabilities = unpack(parameters)
            means = design @ betas.T
            residual = outcomes[:, None] - means
            components = log_components(parameters)
            responsibility = np.exp(
                components - logsumexp(components, axis=1, keepdims=True)
            )
            beta_gradient = np.vstack(
                [
                    design.T
                    @ (
                        responsibility[:, regime]
                        * (-residual[:, regime] / sigmas[regime] ** 2)
                    )
                    for regime in range(self.n_regimes)
                ]
            ).reshape(-1)
            sigma_gradient = np.sum(
                responsibility * (1.0 - residual**2 / sigmas[None, :] ** 2),
                axis=0,
            )
            probability_gradient = (
                n_obs * probabilities[:-1]
                - np.sum(responsibility[:, :-1], axis=0)
            )
            return np.concatenate(
                [beta_gradient, sigma_gradient, probability_gradient]
            )

        def pack_start(labels: np.ndarray) -> np.ndarray:
            betas: list[np.ndarray] = []
            sigmas: list[float] = []
            counts: list[int] = []
            for regime in range(self.n_regimes):
                mask = labels == regime
                counts.append(int(mask.sum()))
                if mask.sum() > n_features and np.linalg.matrix_rank(design[mask]) == n_features:
                    beta = np.linalg.lstsq(design[mask], outcomes[mask], rcond=None)[0]
                    residual = outcomes[mask] - design[mask] @ beta
                    sigma = float(np.sqrt(np.mean(residual**2)))
                else:
                    beta = pooled_beta.copy()
                    sigma = pooled_sigma
                betas.append(beta)
                sigmas.append(float(np.clip(sigma, sigma_floor * 1.5, sigma_ceiling)))
            probabilities = np.maximum(np.asarray(counts, dtype=float), 1.0)
            probabilities /= probabilities.sum()
            logits = np.log(probabilities[:-1] / probabilities[-1])
            return np.concatenate(
                [np.asarray(betas).reshape(-1), np.log(sigmas), logits]
            )

        order = np.argsort(pooled_residual, kind="stable")
        quantile_labels = np.empty(n_obs, dtype=int)
        for regime, rows in enumerate(np.array_split(order, self.n_regimes)):
            quantile_labels[rows] = regime
        starts = [pack_start(quantile_labels)]
        generator = np.random.default_rng(self.random_state)
        for _ in range(1, self.n_starts):
            randomized = generator.permutation(np.arange(n_obs) % self.n_regimes)
            starts.append(pack_start(randomized))

        beta_bounds = [(None, None)] * (self.n_regimes * n_features)
        sigma_bounds = [
            (float(np.log(sigma_floor)), float(np.log(sigma_ceiling)))
        ] * self.n_regimes
        probability_bounds = [(-10.0, 10.0)] * (self.n_regimes - 1)
        fits: list[OptimizeResult] = []
        for initial in starts:
            fits.append(
                minimize(
                    negative_loglike,
                    initial,
                    method="L-BFGS-B",
                    jac=gradient,
                    bounds=beta_bounds + sigma_bounds + probability_bounds,
                    options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7},
                )
            )
        optimizer_result = min(fits, key=lambda fit: float(fit.fun))
        parameters = np.asarray(optimizer_result.x, dtype=float)
        score_norm = float(np.max(np.abs(gradient(parameters))))
        converged = bool(optimizer_result.success or score_norm <= 1e-5)
        if not converged or not np.isfinite(optimizer_result.fun):
            raise RuntimeError(
                f"Gaussian mixture optimization failed: {optimizer_result.message}"
            )

        betas, _, sigmas, probabilities = unpack(parameters)
        regime_order = np.argsort(np.mean(design @ betas.T, axis=0), kind="stable")
        betas = betas[regime_order]
        sigmas = sigmas[regime_order]
        probabilities = probabilities[regime_order]
        canonical_logits = np.log(probabilities[:-1] / probabilities[-1])
        parameters = np.concatenate(
            [betas.reshape(-1), np.log(sigmas), canonical_logits]
        )
        optimizer_result.x = parameters
        optimizer_result.fun = negative_loglike(parameters)
        optimizer_result.jac = gradient(parameters)
        score_norm = float(np.max(np.abs(optimizer_result.jac)))

        hessian = _numerical_hessian(negative_loglike, parameters)
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues = (
            np.linalg.eigvalsh(hessian)
            if np.isfinite(hessian).all()
            else np.array([-np.inf])
        )
        at_scale_boundary = bool(
            np.any(sigmas <= sigma_floor * (1.0 + 1e-6))
            or np.any(sigmas >= sigma_ceiling * (1.0 - 1e-6))
        )
        inference_valid = bool(
            not at_scale_boundary
            and eigenvalues[0] > max(1e-9, 1e-10 * eigenvalues[-1])
        )
        if inference_valid:
            covariance = np.linalg.inv(hessian)
            standard_errors = np.sqrt(np.diag(covariance))
            zstats = parameters / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full_like(hessian, np.nan)
            standard_errors = np.full(parameters.shape, np.nan)
            zstats = np.full(parameters.shape, np.nan)
            pvalues = np.full(parameters.shape, np.nan)

        labels: list[str] = []
        for regime in range(1, self.n_regimes + 1):
            labels.extend(f"regime{regime}:{name}" for name in feature_names)
        labels.extend(
            f"regime{regime}:log_sigma" for regime in range(1, self.n_regimes + 1)
        )
        labels.extend(
            f"mixture_logit:{regime}_vs_{self.n_regimes}"
            for regime in range(1, self.n_regimes)
        )
        params_regimes = tuple(
            pd.Series(betas[index], index=feature_names, name=f"regime_{index + 1}")
            for index in range(self.n_regimes)
        )
        return GaussianMixtureRegressionResult(
            params_regimes=params_regimes,
            sigma_regimes=tuple(float(value) for value in sigmas),
            mixture_probs=np.asarray(probabilities, dtype=float),
            covariance=pd.DataFrame(covariance, index=labels, columns=labels),
            standard_errors=pd.Series(standard_errors, index=labels, name="std_err"),
            zstats=pd.Series(zstats, index=labels, name="z"),
            pvalues=pd.Series(pvalues, index=labels, name="p_value"),
            inference_valid=inference_valid,
            converged=converged,
            score_norm=score_norm,
            loglike=-float(optimizer_result.fun),
            nobs=n_obs,
            n_regimes=self.n_regimes,
            feature_names=tuple(feature_names),
            optimizer_result=optimizer_result,
        )


SwitchingRegressionResult = GaussianMixtureRegressionResult


class SwitchingRegression(GaussianMixtureRegression):
    """Deprecated alias for :class:`GaussianMixtureRegression`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "SwitchingRegression is an iid Gaussian mixture, not a switching model; "
            "use GaussianMixtureRegression instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
