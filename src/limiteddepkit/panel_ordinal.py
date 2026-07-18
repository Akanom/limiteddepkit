"""Panel ordinal-response estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import log_expit, log_ndtr, logsumexp, ndtr
from scipy.stats import norm

from .ordinal import (
    OrderedLogit,
    OrderedProbit,
    _as_2d_array,
    _link_cdf,
    _numerical_hessian,
    _ordered_categories,
    _threshold_jacobian,
    _unpack_thresholds,
)


def _selected_log_probabilities(
    values: np.ndarray,
    encoded: np.ndarray,
    beta: np.ndarray,
    thresholds: np.ndarray,
    random_intercept: float,
    link: str,
) -> np.ndarray:
    """Stable log probabilities for observed ordinal categories."""
    n_thresholds = thresholds.size
    indices = thresholds[None, :] - values @ beta[:, None] - random_intercept
    if link == "logit":
        log_cumulative = log_expit(indices)
        log_survival = log_expit(-indices[:, -1])
    elif link == "probit":
        log_cumulative = log_ndtr(indices)
        log_survival = log_ndtr(-indices[:, -1])
    else:
        raise ValueError(f"Unsupported panel ordinal link: {link!r}.")
    selected_log_probabilities = np.empty(encoded.size, dtype=float)
    first = encoded == 0
    last = encoded == n_thresholds
    selected_log_probabilities[first] = log_cumulative[first, 0]
    selected_log_probabilities[last] = log_survival[last]
    for category in range(1, n_thresholds):
        selected = encoded == category
        if link == "probit":
            lower_index = indices[selected, category - 1]
            upper_index = indices[selected, category]
            use_survival = lower_index > 0.0
            log_larger = log_ndtr(upper_index)
            log_smaller = log_ndtr(lower_index)
            log_larger[use_survival] = log_ndtr(-lower_index[use_survival])
            log_smaller[use_survival] = log_ndtr(-upper_index[use_survival])
        else:
            log_larger = log_cumulative[selected, category]
            log_smaller = log_cumulative[selected, category - 1]
        log_ratio = log_smaller - log_larger
        selected_log_probabilities[selected] = log_larger + np.log(
            -np.expm1(np.minimum(log_ratio, -1e-15))
        )
    return selected_log_probabilities


def _encode_result_categories(y: Any, categories: np.ndarray) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if pd.isna(values).any():
        raise ValueError("y contains missing values.")
    mapping = {category: index for index, category in enumerate(categories)}
    try:
        encoded = np.array([mapping[value] for value in values], dtype=int)
    except (KeyError, TypeError) as error:
        raise ValueError("y contains categories not present in the fitted result.") from error
    return encoded


def _conditional_probabilities(
    values: np.ndarray,
    beta: np.ndarray,
    thresholds: np.ndarray,
    random_effects: np.ndarray,
    link: str,
) -> np.ndarray:
    linear_predictor = values @ beta + random_effects
    cumulative = _link_cdf(
        thresholds[None, :] - linear_predictor[:, None], link
    )
    bounds = np.column_stack(
        [np.zeros(values.shape[0]), cumulative, np.ones(values.shape[0])]
    )
    return np.diff(bounds, axis=1)


def _resolved_random_effects(
    random_effects: Any,
    *,
    nobs: int,
    entity: Any | None,
) -> np.ndarray:
    if np.isscalar(random_effects):
        effects = np.full(nobs, float(random_effects), dtype=float)
    elif isinstance(random_effects, (Mapping, pd.Series)):
        if entity is None:
            raise ValueError("entity is required when random_effects is keyed by entity.")
        entities = np.asarray(entity)
        if entities.ndim != 1 or entities.size != nobs:
            raise ValueError("entity must contain one label per prediction row.")
        keyed = dict(random_effects)
        missing = [label for label in pd.unique(entities) if label not in keyed]
        if missing:
            raise ValueError(f"Missing random effects for entities: {missing}.")
        effects = np.array([keyed[label] for label in entities], dtype=float)
    else:
        effects = np.asarray(random_effects, dtype=float)
        if effects.shape != (nobs,):
            raise ValueError("random_effects must be scalar or contain one value per row.")
    if not np.isfinite(effects).all():
        raise ValueError("random_effects must contain only finite values.")
    return effects


@dataclass(frozen=True)
class RandomEffectsOrderedResult:
    """Shared result contract for random-intercept ordered-response models."""

    params: pd.Series
    thresholds: pd.Series
    sigma_entity: float
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    categories: np.ndarray
    converged: bool
    score_norm: float
    scaled_score_norm: float
    loglike: float
    nobs: int
    n_groups: int
    feature_names: tuple[str, ...]
    quadrature_points: int
    optimizer_result: Any
    link: str = "logit"

    @property
    def all_params(self) -> pd.Series:
        cuts = self.thresholds.copy()
        cuts.index = [f"threshold: {name}" for name in cuts.index]
        sigma = pd.Series({"sigma_entity": self.sigma_entity})
        return pd.concat([self.params, cuts, sigma]).rename("estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def n_entities(self) -> int:
        """Alias matching the ecosystem's panel terminology."""
        return self.n_groups

    @property
    def random_effect_sd(self) -> float:
        return self.sigma_entity

    @property
    def n_quadrature_points(self) -> int:
        return self.quadrature_points

    @property
    def backend(self) -> str:
        return "native-ghq"

    @property
    def covariance_type(self) -> str:
        return "observed-information"

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.DataFrame(
            {
                "lower": self.all_params - critical * self.standard_errors,
                "upper": self.all_params + critical * self.standard_errors,
            }
        )

    def sigma_conf_int(self, level: float = 0.95) -> pd.Series:
        """Return a positive log-scale interval for the random-effect SD."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        standard_error = float(self.standard_errors.get("sigma_entity", np.nan))
        if not np.isfinite(standard_error) or self.sigma_entity <= 0:
            return pd.Series({"lower": np.nan, "upper": np.nan}, name="sigma_entity")
        log_scale_se = standard_error / self.sigma_entity
        critical = norm.ppf(0.5 + level / 2.0)
        return pd.Series(
            {
                "lower": self.sigma_entity * np.exp(-critical * log_scale_se),
                "upper": self.sigma_entity * np.exp(critical * log_scale_se),
            },
            name="sigma_entity",
        )

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def predict_proba(
        self,
        X: Any,
        *,
        random_effects: Any | None = None,
        entity: Any | None = None,
    ) -> pd.DataFrame:
        """Return population-averaged or random-effect-conditional probabilities."""
        values, names = _as_2d_array(X)
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        beta = self.params.to_numpy(dtype=float)
        thresholds = self.thresholds.to_numpy(dtype=float)
        if random_effects is None:
            nodes, weights = hermgauss(self.quadrature_points)
            probabilities = np.zeros((values.shape[0], len(self.categories)), dtype=float)
            for node, weight in zip(nodes, weights, strict=True):
                random_intercept = np.sqrt(2.0) * self.sigma_entity * node
                probabilities += (weight / np.sqrt(np.pi)) * _conditional_probabilities(
                    values,
                    beta,
                    thresholds,
                    np.full(values.shape[0], random_intercept),
                    self.link,
                )
        else:
            effects = _resolved_random_effects(
                random_effects, nobs=values.shape[0], entity=entity
            )
            probabilities = _conditional_probabilities(
                values, beta, thresholds, effects, self.link
            )
        return pd.DataFrame(probabilities, columns=self.categories)

    def predict(
        self,
        X: Any,
        *,
        random_effects: Any | None = None,
        entity: Any | None = None,
    ) -> pd.Series:
        probabilities = self.predict_proba(
            X, random_effects=random_effects, entity=entity
        ).to_numpy()
        return pd.Series(self.categories[np.argmax(probabilities, axis=1)], name="prediction")

    def posterior_random_effects(self, X: Any, y: Any, *, entity: Any) -> pd.DataFrame:
        """Return quadrature posterior summaries for each observed entity."""
        values, names = _as_2d_array(X)
        encoded = _encode_result_categories(y, self.categories)
        entities = np.asarray(entity)
        if values.shape[0] != encoded.size or entities.ndim != 1 or entities.size != encoded.size:
            raise ValueError("X, y, and entity must contain the same number of observations.")
        if values.shape[0] == 0:
            raise ValueError("Posterior conditioning data must not be empty.")
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        if pd.isna(entities).any():
            raise ValueError("entity contains missing values.")

        group_codes, entity_labels = pd.factorize(entities, sort=False)
        nodes, weights = hermgauss(self.quadrature_points)
        node_effects = np.sqrt(2.0) * self.sigma_entity * nodes
        log_weights = np.log(weights) - 0.5 * np.log(np.pi)
        beta = self.params.to_numpy(dtype=float)
        thresholds = self.thresholds.to_numpy(dtype=float)
        rows = []
        for group, label in enumerate(entity_labels):
            selected_rows = np.flatnonzero(group_codes == group)
            log_posterior = np.empty(len(nodes), dtype=float)
            for node_index, random_intercept in enumerate(node_effects):
                log_posterior[node_index] = log_weights[node_index] + np.sum(
                    _selected_log_probabilities(
                        values[selected_rows],
                        encoded[selected_rows],
                        beta,
                        thresholds,
                        float(random_intercept),
                        self.link,
                    )
                )
            log_marginal = float(logsumexp(log_posterior))
            posterior_weights = np.exp(log_posterior - log_marginal)
            posterior_mean = float(posterior_weights @ node_effects)
            posterior_variance = float(
                posterior_weights @ (node_effects - posterior_mean) ** 2
            )
            rows.append(
                {
                    "entity": label,
                    "posterior_mean": posterior_mean,
                    "posterior_sd": np.sqrt(max(posterior_variance, 0.0)),
                    "highest_posterior_mass_node": float(
                        node_effects[np.argmax(posterior_weights)]
                    ),
                    "log_marginal_likelihood": log_marginal,
                    "nobs": len(selected_rows),
                    "posterior_weights": posterior_weights.copy(),
                }
            )
        output = pd.DataFrame(rows).set_index("entity")
        output.attrs["quadrature_effects"] = node_effects.copy()
        return output

    def posterior_predict_proba(
        self,
        X: Any,
        *,
        entity: Any,
        posterior: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return exact quadrature posterior-predictive probabilities."""
        values, names = _as_2d_array(X)
        entities = np.asarray(entity)
        if values.shape[0] != entities.size or entities.ndim != 1:
            raise ValueError("entity must contain one label per prediction row.")
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        if "posterior_weights" not in posterior.columns:
            raise ValueError("posterior must come from posterior_random_effects().")
        if not posterior.index.is_unique:
            raise ValueError("posterior entity labels must be unique.")
        node_effects = np.asarray(posterior.attrs.get("quadrature_effects"), dtype=float)
        if node_effects.shape != (self.quadrature_points,):
            raise ValueError("posterior quadrature metadata is missing or incompatible.")
        missing = [label for label in pd.unique(entities) if label not in posterior.index]
        if missing:
            raise ValueError(f"Posterior summaries are missing entities: {missing}.")

        probabilities = np.empty((values.shape[0], len(self.categories)), dtype=float)
        beta = self.params.to_numpy(dtype=float)
        thresholds = self.thresholds.to_numpy(dtype=float)
        for label in pd.unique(entities):
            selected = np.flatnonzero(entities == label)
            posterior_weights = np.asarray(
                posterior.at[label, "posterior_weights"], dtype=float
            )
            if posterior_weights.shape != node_effects.shape:
                raise ValueError("posterior node weights are incompatible with this result.")
            group_probabilities = np.zeros(
                (len(selected), len(self.categories)), dtype=float
            )
            for weight, random_intercept in zip(
                posterior_weights, node_effects, strict=True
            ):
                group_probabilities += weight * _conditional_probabilities(
                    values[selected],
                    beta,
                    thresholds,
                    np.full(len(selected), random_intercept),
                    self.link,
                )
            probabilities[selected] = group_probabilities
        return pd.DataFrame(probabilities, columns=self.categories)


@dataclass(frozen=True)
class RandomEffectsOrderedLogitResult(RandomEffectsOrderedResult):
    """Random-intercept Ordered Logit result."""

    link: str = "logit"


@dataclass(frozen=True)
class RandomEffectsOrderedProbitResult(RandomEffectsOrderedResult):
    """Random-intercept Ordered Probit result."""

    link: str = "probit"


class _RandomEffectsOrdered:
    """Shared non-adaptive GHQ estimator for random-intercept ordered models."""

    _link: str
    _model_label: str
    _pooled_estimator: type[OrderedLogit] | type[OrderedProbit]
    _result_type: type[RandomEffectsOrderedResult]

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any,
        category_order: Sequence[Any] | None = None,
        quadrature_points: int = 12,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> RandomEffectsOrderedResult:
        values, feature_names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=category_order)
        entities = np.asarray(entity)
        if values.shape[0] != encoded.size or entities.ndim != 1 or entities.size != encoded.size:
            raise ValueError("X, y, and entity must contain the same number of observations.")
        if pd.isna(entities).any():
            raise ValueError("entity contains missing values.")
        if (
            isinstance(quadrature_points, bool)
            or not isinstance(quadrature_points, (int, np.integer))
            or quadrature_points < 3
        ):
            raise ValueError("quadrature_points must be at least three.")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        constant_features = [
            feature_names[index]
            for index in range(values.shape[1])
            if np.ptp(values[:, index]) <= 1e-12
        ]
        if constant_features:
            raise ValueError(
                "Ordered models identify location through thresholds; constant regressors "
                f"are not permitted: {constant_features}."
            )
        unique_entities, group_codes = np.unique(entities, return_inverse=True)
        if unique_entities.size < 2:
            raise ValueError("Random-effects estimation requires at least two entities.")
        group_rows = [np.flatnonzero(group_codes == group) for group in range(unique_entities.size)]
        if max(len(rows) for rows in group_rows) < 2:
            raise ValueError(
                "Random-effects estimation requires repeated observations within entities."
            )

        pooled = self._pooled_estimator().fit(X, y, category_order=categories)
        n_features = values.shape[1]
        n_thresholds = categories.size - 1
        pooled_thresholds = pooled.thresholds.to_numpy(dtype=float)
        raw_thresholds = np.r_[
            pooled_thresholds[0], np.log(np.diff(pooled_thresholds))
        ]
        initial = np.r_[pooled.params.to_numpy(dtype=float), raw_thresholds, np.log(0.5)]
        nodes, weights = hermgauss(quadrature_points)
        log_weights = np.log(weights) - 0.5 * np.log(np.pi)

        def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            beta = parameters[:n_features]
            raw_cuts = parameters[n_features : n_features + n_thresholds]
            thresholds = _unpack_thresholds(raw_cuts)
            sigma = float(np.exp(parameters[-1]))
            return beta, thresholds, sigma

        def negative_loglike(parameters: np.ndarray) -> float:
            beta, thresholds, sigma = unpack(parameters)
            node_log_probabilities = np.empty((len(nodes), values.shape[0]), dtype=float)
            for node_index, node in enumerate(nodes):
                random_intercept = np.sqrt(2.0) * sigma * node
                node_log_probabilities[node_index] = _selected_log_probabilities(
                    values,
                    encoded,
                    beta,
                    thresholds,
                    float(random_intercept),
                    self._link,
                )
            loglike = 0.0
            for rows in group_rows:
                conditional_group_loglike = node_log_probabilities[:, rows].sum(axis=1)
                loglike += logsumexp(log_weights + conditional_group_loglike)
            return float(-loglike)

        fitted = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * (initial.size - 1) + [(-10.0, 4.0)],
            options={
                "maxiter": int(maxiter),
                "ftol": min(float(tolerance), 1e-12),
                "gtol": min(float(tolerance), 1e-5),
                "maxls": 50,
            },
        )
        if not np.isfinite(fitted.fun):
            raise RuntimeError(
                f"Random-effects {self._model_label} produced a non-finite likelihood."
            )

        beta, thresholds, sigma = unpack(fitted.x)
        score_norm = float(np.linalg.norm(np.asarray(fitted.jac), ord=np.inf))
        scaled_score_norm = score_norm / max(1, values.shape[0])
        stationarity_limit = max(min(100.0 * float(tolerance), 1e-4), 1e-5)
        converged = bool(
            np.isfinite(fitted.fun)
            and np.isfinite(fitted.x).all()
            and np.isfinite(scaled_score_norm)
            and scaled_score_norm <= stationarity_limit
        )
        threshold_names = [
            f"{categories[index]} | {categories[index + 1]}"
            for index in range(n_thresholds)
        ]
        parameter_names = (
            feature_names
            + [f"threshold: {name}" for name in threshold_names]
            + ["sigma_entity"]
        )
        information = _numerical_hessian(negative_loglike, fitted.x)
        information = (information + information.T) / 2.0
        information_eigenvalues = (
            np.linalg.eigvalsh(information)
            if np.isfinite(information).all()
            else np.array([np.nan])
        )
        maximum_eigenvalue = float(np.max(information_eigenvalues))
        minimum_eigenvalue = float(np.min(information_eigenvalues))
        inference_valid = bool(
            converged
            and fitted.x[-1] > -9.5
            and np.isfinite(information_eigenvalues).all()
            and maximum_eigenvalue > 0.0
            and minimum_eigenvalue > 0.0
            and minimum_eigenvalue / maximum_eigenvalue > 1e-12
        )
        raw_covariance = (
            np.linalg.pinv(information)
            if np.isfinite(information).all()
            else np.full_like(information, np.nan)
        )
        transformation = np.eye(fitted.x.size)
        transformation[
            n_features : n_features + n_thresholds,
            n_features : n_features + n_thresholds,
        ] = _threshold_jacobian(fitted.x[n_features : n_features + n_thresholds])
        transformation[-1, -1] = sigma
        covariance_values = transformation @ raw_covariance @ transformation.T
        covariance_values = (covariance_values + covariance_values.T) / 2.0
        inference_valid = bool(inference_valid and np.isfinite(covariance_values).all())
        reported_parameters = np.r_[beta, thresholds, sigma]
        standard_errors = np.sqrt(np.clip(np.diag(covariance_values), 0.0, None))
        zstats = np.divide(
            reported_parameters,
            standard_errors,
            out=np.full_like(reported_parameters, np.nan),
            where=standard_errors > 0,
        )
        pvalues = 2.0 * ndtr(-np.abs(zstats))
        zstats[-1] = np.nan
        pvalues[-1] = np.nan
        if not inference_valid:
            covariance_values[:] = np.nan
            standard_errors[:] = np.nan
            zstats[:] = np.nan
            pvalues[:] = np.nan
        return self._result_type(
            params=pd.Series(beta, index=feature_names, name="coefficient"),
            thresholds=pd.Series(thresholds, index=threshold_names, name="threshold"),
            sigma_entity=sigma,
            covariance=pd.DataFrame(
                covariance_values, index=parameter_names, columns=parameter_names
            ),
            standard_errors=pd.Series(
                standard_errors, index=parameter_names, name="standard_error"
            ),
            zstats=pd.Series(zstats, index=parameter_names, name="z_stat"),
            pvalues=pd.Series(pvalues, index=parameter_names, name="p_value"),
            inference_valid=inference_valid,
            categories=categories,
            converged=converged,
            score_norm=score_norm,
            scaled_score_norm=scaled_score_norm,
            loglike=float(-fitted.fun),
            nobs=values.shape[0],
            n_groups=unique_entities.size,
            feature_names=tuple(feature_names),
            quadrature_points=quadrature_points,
            optimizer_result=fitted,
            link=self._link,
        )


class RandomEffectsOrderedLogit(_RandomEffectsOrdered):
    """Random-intercept Ordered Logit estimated by quadrature likelihood."""

    _link = "logit"
    _model_label = "Ordered Logit"
    _pooled_estimator = OrderedLogit
    _result_type = RandomEffectsOrderedLogitResult


class RandomEffectsOrderedProbit(_RandomEffectsOrdered):
    """Random-intercept Ordered Probit estimated by quadrature likelihood."""

    _link = "probit"
    _model_label = "Ordered Probit"
    _pooled_estimator = OrderedProbit
    _result_type = RandomEffectsOrderedProbitResult
