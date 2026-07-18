"""Fixed-effects estimators for static ordinal panel outcomes.

Ordered Logit uses the blow-up-and-cluster conditional composite likelihood of
Baetschmann, Staub, and Winkelmann (2015). Ordered Probit uses an explicitly
experimental unconditional entity-effects likelihood with split-panel-
jackknife bias correction for balanced large panels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

from .ordinal import (
    OrderedProbit,
    _as_2d_array,
    _category_probabilities,
    _numerical_hessian,
    _ordered_categories,
    _threshold_jacobian,
    _unpack_thresholds,
)

__all__ = [
    "FixedEffectsOrderedLogit",
    "FixedEffectsOrderedLogitResult",
    "FixedEffectsOrderedProbit",
    "FixedEffectsOrderedProbitResult",
]


def _conditional_logit_clone(
    beta: np.ndarray,
    design: np.ndarray,
    outcome: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return one clone's conditional log likelihood and slope score."""
    nobs = len(outcome)
    successes = int(np.sum(outcome))
    if successes <= 0 or successes >= nobs:
        return 0.0, np.zeros(design.shape[1], dtype=float)

    eta = design @ beta
    prefix = np.full((nobs + 1, successes + 1), -np.inf, dtype=float)
    suffix = np.full((nobs + 1, successes + 1), -np.inf, dtype=float)
    prefix[0, 0] = 0.0
    for row in range(nobs):
        prefix[row + 1] = prefix[row]
        maximum = min(successes, row + 1)
        for count in range(1, maximum + 1):
            prefix[row + 1, count] = np.logaddexp(
                prefix[row, count],
                prefix[row, count - 1] + eta[row],
            )

    suffix[nobs, 0] = 0.0
    for row in range(nobs - 1, -1, -1):
        suffix[row] = suffix[row + 1]
        maximum = min(successes, nobs - row)
        for count in range(1, maximum + 1):
            suffix[row, count] = np.logaddexp(
                suffix[row + 1, count],
                suffix[row + 1, count - 1] + eta[row],
            )

    log_denominator = float(prefix[nobs, successes])
    inclusion = np.empty(nobs, dtype=float)
    for row in range(nobs):
        terms = []
        for left_count in range(successes):
            right_count = successes - 1 - left_count
            if left_count <= row and right_count <= nobs - row - 1:
                terms.append(
                    prefix[row, left_count] + suffix[row + 1, right_count]
                )
        log_excluding = float(logsumexp(terms))
        inclusion[row] = np.exp(eta[row] + log_excluding - log_denominator)

    loglike = float(outcome @ eta - log_denominator)
    score = design.T @ (outcome - inclusion)
    return loglike, np.asarray(score, dtype=float)


@dataclass(frozen=True)
class FixedEffectsOrderedLogitResult:
    """BUC conditional fixed-effects Ordered-Logit slope result."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    inference_valid: bool
    score_norm: float
    scaled_score_norm: float
    composite_loglike: float
    nobs: int
    n_entities: int
    n_contributing_entities: int
    n_cutoff_clones: int
    n_pseudo_observations: int
    categories: np.ndarray
    feature_names: tuple[str, ...]
    optimizer_result: Any
    _entity_scores: pd.DataFrame = field(repr=False, compare=False)

    @property
    def all_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def loglike(self) -> float:
        """Return the BUC composite conditional log likelihood."""
        return self.composite_loglike

    @property
    def covariance_type(self) -> str:
        return "entity-cluster-sandwich"

    @property
    def backend(self) -> str:
        return "conditional-buc"

    @property
    def thresholds_identified(self) -> bool:
        return False

    @property
    def entity_effects_identified(self) -> bool:
        return False

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = float(norm.ppf(0.5 + level / 2.0))
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        from .postestimation import summary_frame

        return summary_frame(self)

    def odds_ratios(self) -> pd.Series:
        """Return continuation-odds ratios for the common slopes."""
        return np.exp(self.params).rename("odds_ratio")

    def linear_index(self, X: Any) -> pd.Series:
        """Return ``X beta`` without pretending fixed effects are available."""
        design, names = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {design.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
        return pd.Series(design @ self.params.to_numpy(), index=index, name="linear_index")

    def entity_score_frame(self) -> pd.DataFrame:
        """Return entity-cluster slope scores used by the sandwich covariance."""
        return self._entity_scores.copy()


class FixedEffectsOrderedLogit:
    """Estimate common Ordered-Logit slopes by BUC conditional likelihood.

    Entity effects may be arbitrarily correlated with the regressors.  The
    estimator blows each entity into one binary conditional-logit clone per
    ordered cutoff, omits clones without within-entity binary variation, and
    clusters the composite-likelihood covariance by original entity.

    Thresholds and entity effects are conditioned out.  Category probabilities,
    marginal effects, and ordinary ordered-model information criteria are
    therefore intentionally unavailable.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any,
        category_order: Any | None = None,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> FixedEffectsOrderedLogitResult:
        design, feature_names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=category_order)
        entities = np.asarray(entity)
        nobs, nfeatures = design.shape
        if entities.ndim != 1 or len(entities) != nobs or len(encoded) != nobs:
            raise ValueError("X, y, and entity must contain the same observations.")
        if pd.isna(entities).any():
            raise ValueError("entity must not contain missing values.")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X feature names must be unique after conversion to strings.")
        if categories.size < 3:
            raise ValueError("Fixed-effects Ordered Logit requires at least three categories.")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")

        try:
            entity_codes, entity_levels = pd.factorize(entities, sort=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("entity must contain hashable scalar labels.") from exc
        entity_codes = np.asarray(entity_codes, dtype=np.int64)
        if len(entity_levels) < 2:
            raise ValueError("Fixed-effects estimation requires at least two entities.")

        group_rows = [
            np.flatnonzero(entity_codes == code)
            for code in range(len(entity_levels))
        ]
        clones: list[tuple[int, np.ndarray, np.ndarray]] = []
        contributing_codes: set[int] = set()
        pseudo_observations = 0
        for code, rows in enumerate(group_rows):
            for cutoff in range(1, len(categories)):
                binary = (encoded[rows] >= cutoff).astype(float)
                successes = int(binary.sum())
                if successes == 0 or successes == len(binary):
                    continue
                clones.append((code, rows, binary))
                contributing_codes.add(code)
                pseudo_observations += len(rows)

        if not clones:
            raise ValueError(
                "No entity crosses any ordered cutoff; conditional slopes are unidentified."
            )
        contributing = np.array(sorted(contributing_codes), dtype=np.int64)
        if len(contributing) <= nfeatures:
            raise ValueError(
                "The number of outcome-varying entities must exceed the regressors."
            )
        contributing_rows = np.concatenate([group_rows[code] for code in contributing])
        within = design[contributing_rows].copy()
        contributing_entity = entity_codes[contributing_rows]
        for code in contributing:
            selected = contributing_entity == code
            within[selected] -= within[selected].mean(axis=0)
        if np.linalg.matrix_rank(within) < nfeatures:
            raise ValueError(
                "The within-entity design is rank deficient; constants and other "
                "time-invariant or within-collinear regressors are not identified."
            )

        def entity_loglikes_and_scores(
            beta: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            loglikes = np.zeros(len(entity_levels), dtype=float)
            scores = np.zeros((len(entity_levels), nfeatures), dtype=float)
            for code, rows, binary in clones:
                loglike, score = _conditional_logit_clone(beta, design[rows], binary)
                loglikes[code] += loglike
                scores[code] += score
            return loglikes, scores

        def objective(beta: np.ndarray) -> float:
            loglikes, _ = entity_loglikes_and_scores(beta)
            return -float(np.sum(loglikes))

        def gradient(beta: np.ndarray) -> np.ndarray:
            _, scores = entity_loglikes_and_scores(beta)
            return -np.sum(scores, axis=0)

        fitted = minimize(
            objective,
            np.zeros(nfeatures, dtype=float),
            jac=gradient,
            method="BFGS",
            options={"maxiter": int(maxiter), "gtol": float(min(tolerance, 1e-6))},
        )
        beta = np.asarray(fitted.x, dtype=float)
        score_norm = float(np.linalg.norm(gradient(beta), ord=np.inf))
        scaled_score_norm = score_norm / max(1, pseudo_observations)
        stationarity_limit = max(min(10.0 * tolerance, 1e-6), 1e-8)
        converged = bool(
            np.isfinite(fitted.fun)
            and np.isfinite(beta).all()
            and np.isfinite(scaled_score_norm)
            and scaled_score_norm <= stationarity_limit
        )
        information = _numerical_hessian(objective, beta)
        information = (information + information.T) / 2.0
        _, all_entity_scores = entity_loglikes_and_scores(beta)
        entity_scores = all_entity_scores[contributing]
        eigenvalues = (
            np.linalg.eigvalsh(information)
            if np.isfinite(information).all()
            else np.array([np.nan])
        )
        maximum_eigenvalue = float(np.max(eigenvalues))
        minimum_eigenvalue = float(np.min(eigenvalues))
        inference_valid = bool(
            converged
            and np.isfinite(eigenvalues).all()
            and maximum_eigenvalue > 0.0
            and minimum_eigenvalue > 0.0
            and minimum_eigenvalue / maximum_eigenvalue > 1e-12
            and len(contributing) > nfeatures
        )
        if inference_valid:
            bread = np.linalg.inv(information)
            meat = entity_scores.T @ entity_scores
            correction = len(contributing) / (len(contributing) - 1.0)
            if pseudo_observations > nfeatures:
                correction *= (pseudo_observations - 1.0) / (
                    pseudo_observations - nfeatures
                )
            covariance = correction * bread @ meat @ bread
            covariance = (covariance + covariance.T) / 2.0
            inference_valid = bool(np.isfinite(covariance).all())
        if inference_valid:
            standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
            zstats = beta / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full((nfeatures, nfeatures), np.nan)
            standard_errors = np.full(nfeatures, np.nan)
            zstats = np.full(nfeatures, np.nan)
            pvalues = np.full(nfeatures, np.nan)

        names = tuple(feature_names)
        params = pd.Series(beta, index=names, name="estimate")
        covariance_frame = pd.DataFrame(covariance, index=names, columns=names)
        standard_error_series = pd.Series(standard_errors, index=names, name="std_err")
        zstat_series = pd.Series(zstats, index=names, name="z")
        pvalue_series = pd.Series(pvalues, index=names, name="p_value")
        score_frame = pd.DataFrame(
            entity_scores,
            index=pd.Index(entity_levels[contributing], name="entity"),
            columns=names,
        )
        return FixedEffectsOrderedLogitResult(
            params=params,
            covariance=covariance_frame,
            standard_errors=standard_error_series,
            zstats=zstat_series,
            pvalues=pvalue_series,
            converged=converged,
            inference_valid=inference_valid,
            score_norm=score_norm,
            scaled_score_norm=scaled_score_norm,
            composite_loglike=-float(fitted.fun),
            nobs=nobs,
            n_entities=len(entity_levels),
            n_contributing_entities=len(contributing),
            n_cutoff_clones=len(clones),
            n_pseudo_observations=pseudo_observations,
            categories=categories.copy(),
            feature_names=names,
            optimizer_result=fitted,
            _entity_scores=score_frame,
        )


def _raw_thresholds(thresholds: np.ndarray) -> np.ndarray:
    values = np.asarray(thresholds, dtype=float)
    return np.r_[values[0], np.log(np.diff(values))]


@dataclass(frozen=True)
class _UnconditionalFEProbitFit:
    beta: np.ndarray
    thresholds: np.ndarray
    raw_thresholds: np.ndarray
    entity_effects: np.ndarray
    loglike: float
    converged: bool
    score_norm: float
    scaled_score_norm: float
    optimizer_result: Any


def _fit_unconditional_fe_probit(
    panel_X: np.ndarray,
    panel_y: np.ndarray,
    *,
    n_categories: int,
    initial_common: np.ndarray,
    maxiter: int,
    tolerance: float,
) -> _UnconditionalFEProbitFit:
    """Fit one balanced-panel unconditional FE Ordered Probit."""
    n_entities, n_periods, n_features = panel_X.shape
    design = panel_X.reshape(-1, n_features)
    encoded = panel_y.reshape(-1)
    entity_codes = np.repeat(np.arange(n_entities), n_periods)
    n_thresholds = n_categories - 1
    # Keep one raw effect per entity and center the vector inside the
    # likelihood.  The redundant common-shift direction has exactly zero
    # gradient, while this symmetric representation avoids making estimates
    # depend on which entity happens to be chosen as a reference category.
    initial = np.r_[initial_common, np.zeros(n_entities, dtype=float)]

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        beta = parameters[:n_features]
        raw = parameters[n_features : n_features + n_thresholds]
        thresholds = _unpack_thresholds(raw)
        raw_effects = parameters[n_features + n_thresholds :]
        effects = raw_effects - raw_effects.mean()
        return beta, thresholds, effects

    def value_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        beta, thresholds, effects = unpack(parameters)
        eta = design @ beta + effects[entity_codes]
        lower = np.full(len(encoded), -np.inf, dtype=float)
        upper = np.full(len(encoded), np.inf, dtype=float)
        has_lower = encoded > 0
        has_upper = encoded < n_thresholds
        lower[has_lower] = thresholds[encoded[has_lower] - 1] - eta[has_lower]
        upper[has_upper] = thresholds[encoded[has_upper]] - eta[has_upper]
        probability = ndtr(upper) - ndtr(lower)
        probability = np.clip(probability, 1e-15, 1.0)
        lower_density = np.zeros(len(encoded), dtype=float)
        upper_density = np.zeros(len(encoded), dtype=float)
        lower_density[has_lower] = norm.pdf(lower[has_lower])
        upper_density[has_upper] = norm.pdf(upper[has_upper])
        eta_score = (lower_density - upper_density) / probability

        beta_score = design.T @ eta_score
        threshold_score = np.zeros(n_thresholds, dtype=float)
        for threshold in range(n_thresholds):
            as_upper = encoded == threshold
            as_lower = encoded == threshold + 1
            threshold_score[threshold] += np.sum(
                upper_density[as_upper] / probability[as_upper]
            )
            threshold_score[threshold] -= np.sum(
                lower_density[as_lower] / probability[as_lower]
            )
        raw = parameters[n_features : n_features + n_thresholds]
        raw_threshold_score = _threshold_jacobian(raw).T @ threshold_score

        effect_score = np.bincount(
            entity_codes,
            weights=eta_score,
            minlength=n_entities,
        )
        centered_effect_score = effect_score - np.mean(effect_score)
        score = np.r_[beta_score, raw_threshold_score, centered_effect_score]
        return -float(np.sum(np.log(probability))), -score

    def objective(parameters: np.ndarray) -> float:
        return value_and_gradient(parameters)[0]

    def gradient(parameters: np.ndarray) -> np.ndarray:
        return value_and_gradient(parameters)[1]

    bounds = (
        [(None, None)] * (n_features + 1)
        + [(-8.0, 8.0)] * (n_thresholds - 1)
        + [(-12.0, 12.0)] * n_entities
    )
    fitted = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(maxiter),
            # Relative objective stopping can occur while the high-dimensional
            # nuisance-effect score is still material.  Tightening ``ftol``
            # lets the score criterion below, rather than a flat likelihood,
            # certify stationarity.
            "ftol": float(min(tolerance**2, 1e-14)),
            "gtol": float(min(tolerance, 1e-6)),
            "maxls": 50,
        },
    )
    beta, thresholds, effects = unpack(np.asarray(fitted.x, dtype=float))
    score_norm = float(np.linalg.norm(gradient(fitted.x), ord=np.inf))
    scaled_score_norm = score_norm / np.sqrt(len(encoded))
    effect_parameters = np.asarray(fitted.x)[n_features + n_thresholds :]
    converged = bool(
        scaled_score_norm <= max(min(100.0 * tolerance, 1e-5), 1e-6)
        and np.max(np.abs(effect_parameters)) < 11.9
    )
    return _UnconditionalFEProbitFit(
        beta=np.asarray(beta, dtype=float),
        thresholds=np.asarray(thresholds, dtype=float),
        raw_thresholds=np.asarray(
            fitted.x[n_features : n_features + n_thresholds], dtype=float
        ),
        entity_effects=np.asarray(effects, dtype=float),
        loglike=-float(fitted.fun),
        converged=converged,
        score_norm=score_norm,
        scaled_score_norm=scaled_score_norm,
        optimizer_result=fitted,
    )


def _split_panel_jackknife_probit(
    panel_X: np.ndarray,
    panel_y: np.ndarray,
    *,
    n_categories: int,
    initial_common: np.ndarray,
    maxiter: int,
    tolerance: float,
) -> tuple[
    _UnconditionalFEProbitFit,
    _UnconditionalFEProbitFit,
    _UnconditionalFEProbitFit,
    np.ndarray,
]:
    full = _fit_unconditional_fe_probit(
        panel_X,
        panel_y,
        n_categories=n_categories,
        initial_common=initial_common,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    fitted_common = np.r_[full.beta, full.raw_thresholds]
    midpoint = panel_X.shape[1] // 2
    first = _fit_unconditional_fe_probit(
        panel_X[:, :midpoint],
        panel_y[:, :midpoint],
        n_categories=n_categories,
        initial_common=fitted_common,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    second = _fit_unconditional_fe_probit(
        panel_X[:, midpoint:],
        panel_y[:, midpoint:],
        n_categories=n_categories,
        initial_common=fitted_common,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    full_reported = np.r_[full.beta, full.thresholds]
    half_average = 0.5 * (
        np.r_[first.beta, first.thresholds] + np.r_[second.beta, second.thresholds]
    )
    corrected = 2.0 * full_reported - half_average
    return full, first, second, corrected


@dataclass(frozen=True)
class FixedEffectsOrderedProbitResult:
    """Split-panel-jackknife entity-FE Ordered-Probit result."""

    params: pd.Series
    thresholds: pd.Series
    uncorrected_params: pd.Series
    uncorrected_thresholds: pd.Series
    half_panel_common_parameters: pd.DataFrame
    entity_effects: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    inference_valid: bool
    nobs: int
    n_entities: int
    n_periods: int
    categories: np.ndarray
    feature_names: tuple[str, ...]
    bootstrap_repetitions: int
    bootstrap_successes: int
    full_loglike: float
    score_norms: pd.Series
    optimizer_results: tuple[Any, Any, Any] = field(repr=False, compare=False)

    @property
    def all_params(self) -> pd.Series:
        cuts = self.thresholds.copy()
        cuts.index = [f"threshold: {label}" for label in cuts.index]
        return pd.concat([self.params, cuts]).rename("estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def covariance_type(self) -> str:
        return "entity-bootstrap" if self.inference_valid else "none"

    @property
    def backend(self) -> str:
        return "experimental-fe-probit-spj"

    @property
    def bias_correction(self) -> str:
        return "split-panel-jackknife"

    def vcov(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        if not self.inference_valid:
            raise RuntimeError(
                "Confidence intervals require a successful entity bootstrap."
            )
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

    def linear_index(self, X: Any) -> pd.Series:
        design, names = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {design.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
        return pd.Series(design @ self.params.to_numpy(), index=index, name="linear_index")

    def predict_proba(self, X: Any, *, entity: Any) -> pd.DataFrame:
        """Predict conditionally using full-sample nuisance entity effects.

        Common slopes and thresholds use the split-panel correction, whereas
        entity effects are uncorrected nuisance estimates.  These probabilities
        are therefore diagnostic plug-ins, not bias-corrected predictive
        distributions or evidence for new entities.
        """
        design, names = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {design.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        labels = np.asarray(entity)
        if labels.ndim != 1 or len(labels) != len(design):
            raise ValueError("entity must contain one label per prediction row.")
        missing = [label for label in pd.unique(labels) if label not in self.entity_effects.index]
        if missing:
            raise ValueError(f"No fitted fixed effect is available for entities: {missing!r}.")
        effects = np.array([self.entity_effects.loc[label] for label in labels], dtype=float)
        probabilities = _category_probabilities(
            np.column_stack([design, effects]),
            np.r_[self.params.to_numpy(dtype=float), 1.0],
            self.thresholds.to_numpy(dtype=float),
            "probit",
        )
        index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
        return pd.DataFrame(probabilities, index=index, columns=self.categories)


class FixedEffectsOrderedProbit:
    """Bias-corrected unconditional entity-FE Ordered Probit for balanced panels.

    The estimator reports the split-panel-jackknife correction
    ``2 * full - (first_half + second_half) / 2`` for common slopes and
    thresholds.  It requires a balanced panel with the same even time grid for
    every entity and at least six periods.  This is a large-``N``, large-``T``
    estimator, not a short-panel conditional-likelihood analogue of BUC.

    Inference is deliberately unavailable unless an entity bootstrap is
    requested and at least 80 percent of its replications converge.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any,
        time: Any,
        category_order: Any | None = None,
        bootstrap_repetitions: int = 0,
        random_state: int | None = None,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
    ) -> FixedEffectsOrderedProbitResult:
        design, feature_names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=category_order)
        entities = np.asarray(entity)
        times = np.asarray(time)
        nobs, nfeatures = design.shape
        if entities.shape != (nobs,) or times.shape != (nobs,) or len(encoded) != nobs:
            raise ValueError("X, y, entity, and time must contain the same observations.")
        if pd.isna(entities).any() or pd.isna(times).any():
            raise ValueError("entity and time must not contain missing values.")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X feature names must be unique after conversion to strings.")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if (
            isinstance(bootstrap_repetitions, bool)
            or not isinstance(bootstrap_repetitions, (int, np.integer))
            or bootstrap_repetitions < 0
        ):
            raise ValueError("bootstrap_repetitions must be a non-negative integer.")
        if 0 < bootstrap_repetitions < 20:
            raise ValueError("bootstrap_repetitions must be zero or at least 20.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("random_state must be an integer or None.")

        try:
            entity_codes, entity_levels = pd.factorize(entities, sort=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("entity must contain hashable scalar labels.") from exc
        entity_codes = np.asarray(entity_codes, dtype=np.int64)
        if len(entity_levels) <= nfeatures:
            raise ValueError("The number of entities must exceed the regressors.")
        records: list[tuple[Any, np.ndarray]] = []
        reference_times: np.ndarray | None = None
        for code, label in enumerate(entity_levels):
            rows = np.flatnonzero(entity_codes == code)
            order = np.argsort(times[rows], kind="stable")
            ordered_rows = rows[order]
            ordered_times = times[ordered_rows]
            if len(pd.unique(ordered_times)) != len(ordered_times):
                raise ValueError(f"Entity {label!r} has duplicate time observations.")
            if reference_times is None:
                reference_times = ordered_times.copy()
            elif not np.array_equal(ordered_times, reference_times):
                raise ValueError(
                    "FixedEffectsOrderedProbit requires a balanced common time grid."
                )
            records.append((label, ordered_rows))
        if reference_times is None:  # pragma: no cover - non-empty data guard
            raise ValueError("No panel observations are available.")
        n_periods = len(reference_times)
        if n_periods < 6 or n_periods % 2:
            raise ValueError(
                "Split-panel correction requires an even common panel length of at least six."
            )

        panel_X = np.stack([design[rows] for _, rows in records])
        panel_y = np.stack([encoded[rows] for _, rows in records])
        within = panel_X - panel_X.mean(axis=1, keepdims=True)
        if np.linalg.matrix_rank(within.reshape(-1, nfeatures)) < nfeatures:
            raise ValueError(
                "The within-entity design is rank deficient; constants and time-invariant "
                "regressors are not identified."
            )
        midpoint = n_periods // 2
        for portion in (panel_y, panel_y[:, :midpoint], panel_y[:, midpoint:]):
            if len(np.unique(portion)) != len(categories):
                raise ValueError(
                    "Every outcome category must occur in the full panel and both time halves."
                )
        extreme_only_entities: list[Any] = []
        for index, (label, _) in enumerate(records):
            portions = (
                panel_y[index],
                panel_y[index, :midpoint],
                panel_y[index, midpoint:],
            )
            if any(
                np.all(portion == 0) or np.all(portion == len(categories) - 1)
                for portion in portions
            ):
                extreme_only_entities.append(label)
        if extreme_only_entities:
            raise ValueError(
                "A finite entity-effect MLE is required in the full panel and both "
                "time halves. Entities observed only in the lowest or highest category "
                f"within one of those samples include {extreme_only_entities[:5]}."
            )

        pooled = OrderedProbit().fit(
            panel_X.reshape(-1, nfeatures),
            panel_y.reshape(-1),
            category_order=np.arange(len(categories)),
            maxiter=maxiter,
            tolerance=tolerance,
        )
        initial_common = np.r_[
            pooled.params.to_numpy(dtype=float),
            _raw_thresholds(pooled.thresholds.to_numpy(dtype=float)),
        ]
        full, first, second, corrected = _split_panel_jackknife_probit(
            panel_X,
            panel_y,
            n_categories=len(categories),
            initial_common=initial_common,
            maxiter=maxiter,
            tolerance=tolerance,
        )
        corrected_beta = corrected[:nfeatures]
        corrected_thresholds = corrected[nfeatures:]
        if np.any(np.diff(corrected_thresholds) <= 1e-8):
            raise RuntimeError(
                "Split-panel bias correction produced crossing thresholds; "
                "the panel is not adequate for this estimator."
            )
        converged = bool(full.converged and first.converged and second.converged)

        parameter_names = [*feature_names]
        threshold_names = [
            f"{categories[index]} | {categories[index + 1]}"
            for index in range(len(categories) - 1)
        ]
        reported_names = [
            *parameter_names,
            *[f"threshold: {name}" for name in threshold_names],
        ]
        bootstrap_draws: list[np.ndarray] = []
        if bootstrap_repetitions:
            rng = np.random.default_rng(random_state)
            for _ in range(int(bootstrap_repetitions)):
                sampled = rng.integers(0, len(entity_levels), size=len(entity_levels))
                try:
                    boot_full, boot_first, boot_second, boot_corrected = (
                        _split_panel_jackknife_probit(
                            panel_X[sampled],
                            panel_y[sampled],
                            n_categories=len(categories),
                            initial_common=np.r_[full.beta, full.raw_thresholds],
                            maxiter=maxiter,
                            tolerance=tolerance,
                        )
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                if (
                    boot_full.converged
                    and boot_first.converged
                    and boot_second.converged
                    and np.all(np.diff(boot_corrected[nfeatures:]) > 1e-8)
                ):
                    bootstrap_draws.append(boot_corrected)

        successes = len(bootstrap_draws)
        inference_valid = bool(
            converged
            and bootstrap_repetitions >= 20
            and successes >= max(20, int(np.ceil(0.8 * bootstrap_repetitions)))
        )
        if inference_valid:
            covariance = np.cov(np.vstack(bootstrap_draws), rowvar=False, ddof=1)
            standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
            zstats = corrected / standard_errors
            pvalues = 2.0 * norm.sf(np.abs(zstats))
        else:
            covariance = np.full((len(corrected), len(corrected)), np.nan)
            standard_errors = np.full(len(corrected), np.nan)
            zstats = np.full(len(corrected), np.nan)
            pvalues = np.full(len(corrected), np.nan)

        params = pd.Series(corrected_beta, index=parameter_names, name="estimate")
        thresholds = pd.Series(
            corrected_thresholds,
            index=threshold_names,
            name="threshold",
        )
        uncorrected_params = pd.Series(
            full.beta,
            index=parameter_names,
            name="uncorrected_estimate",
        )
        uncorrected_thresholds = pd.Series(
            full.thresholds,
            index=threshold_names,
            name="uncorrected_threshold",
        )
        half_panel = pd.DataFrame(
            [np.r_[first.beta, first.thresholds], np.r_[second.beta, second.thresholds]],
            index=pd.Index(["first", "second"], name="time_half"),
            columns=reported_names,
        )
        entity_effects = pd.Series(
            full.entity_effects,
            index=pd.Index(entity_levels, name="entity"),
            name="fixed_effect",
        )
        covariance_frame = pd.DataFrame(
            covariance,
            index=reported_names,
            columns=reported_names,
        )
        standard_error_series = pd.Series(
            standard_errors,
            index=reported_names,
            name="std_err",
        )
        zstat_series = pd.Series(zstats, index=reported_names, name="z")
        pvalue_series = pd.Series(pvalues, index=reported_names, name="p_value")
        return FixedEffectsOrderedProbitResult(
            params=params,
            thresholds=thresholds,
            uncorrected_params=uncorrected_params,
            uncorrected_thresholds=uncorrected_thresholds,
            half_panel_common_parameters=half_panel,
            entity_effects=entity_effects,
            covariance=covariance_frame,
            standard_errors=standard_error_series,
            zstats=zstat_series,
            pvalues=pvalue_series,
            converged=converged,
            inference_valid=inference_valid,
            nobs=nobs,
            n_entities=len(entity_levels),
            n_periods=n_periods,
            categories=categories.copy(),
            feature_names=tuple(feature_names),
            bootstrap_repetitions=int(bootstrap_repetitions),
            bootstrap_successes=successes,
            full_loglike=full.loglike,
            score_norms=pd.Series(
                [
                    full.scaled_score_norm,
                    first.scaled_score_norm,
                    second.scaled_score_norm,
                ],
                index=["full", "first", "second"],
                name="scaled_score_norm",
            ),
            optimizer_results=(
                full.optimizer_result,
                first.optimizer_result,
                second.optimizer_result,
            ),
        )
