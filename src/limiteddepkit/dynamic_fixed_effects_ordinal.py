"""Fixed-T dynamic Ordered-Logit estimation with unrestricted entity effects.

This module implements the composite conditional maximum-likelihood estimator
in Muris, Raposo, and Vandoros (2025).  The deliberately narrow estimator uses
exactly four consecutive outcomes, discrete regressors with positive mass on
``X_2 == X_3``, and state dependence through one known binary cutoff.  Under
those restrictions the conditional likelihood removes the entity effects.

It is not a dummy-variable fixed-effects likelihood and it is not the static
blow-up-and-cluster likelihood with a lagged outcome appended as a regressor.
Both of those shortcuts suffer from incidental-parameter/endogeneity problems
in a short dynamic panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize, nnls
from scipy.special import expit, ndtr
from scipy.stats import norm

from .ordinal import _as_2d_array, _ordered_categories

__all__ = [
    "DynamicFixedEffectsOrderedLogit",
    "DynamicFixedEffectsOrderedLogitResult",
]


_STATE_PARAMETER = "state_dependence"
_META_COLUMNS = (
    "_entity",
    "_lower_cutoff",
    "_upper_cutoff",
    "_response",
    "_initial_state",
    "_terminal_state",
)


def _threshold_name(categories: np.ndarray, cutoff: int) -> str:
    """Return the label for the threshold at a zero-based category cutoff."""
    return f"{categories[cutoff - 1]} | {categories[cutoff]}"


@dataclass(frozen=True)
class DynamicFixedEffectsOrderedLogitResult:
    """Muris--Raposo--Vandoros composite conditional likelihood result."""

    params: pd.Series
    state_dependence: float
    thresholds: pd.Series
    normalized_threshold: str
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    converged: bool
    inference_valid: bool
    scaled_kkt_residual: float
    composite_loglike: float
    nobs: int
    n_entities: int
    n_stayer_entities: int
    n_contributing_entities: int
    n_conditional_contributions: int
    categories: np.ndarray
    state_cutoff: Any
    feature_names: tuple[str, ...]
    optimizer_result: Any
    _conditional_sample: pd.DataFrame = field(repr=False, compare=False)
    _entity_scores: pd.DataFrame = field(repr=False, compare=False)

    @property
    def estimated_thresholds(self) -> pd.Series:
        """Return thresholds other than the zero-normalized state threshold."""
        return self.thresholds.drop(index=self.normalized_threshold)

    @property
    def all_params(self) -> pd.Series:
        """Return all freely estimated common parameters in covariance order."""
        state = pd.Series(
            [self.state_dependence], index=[_STATE_PARAMETER], name="coefficient"
        )
        cuts = self.estimated_thresholds.copy()
        cuts.index = [f"threshold: {name}" for name in cuts.index]
        return pd.concat([self.params, state, cuts]).rename("estimate")

    @property
    def n_params(self) -> int:
        return len(self.all_params)

    @property
    def n_groups(self) -> int:
        return self.n_entities

    @property
    def structural_params(self) -> pd.Series:
        return self.params.copy()

    @property
    def state_dependence_params(self) -> pd.Series:
        return pd.Series(
            [self.state_dependence], index=[_STATE_PARAMETER], name="coefficient"
        )

    @property
    def loglike(self) -> float:
        """Return the composite conditional log likelihood."""
        return self.composite_loglike

    @property
    def backend(self) -> str:
        return "mrv-fixed-t-ccmle"

    @property
    def covariance_type(self) -> str:
        return "entity-cluster-godambe"

    @property
    def n_inference_groups(self) -> int:
        """Return entities contributing clusters to the Godambe covariance."""
        return self.n_contributing_entities

    @property
    def entity_effects_identified(self) -> bool:
        return False

    @property
    def thresholds_identified(self) -> bool:
        """Thresholds are identified relative to the zero normalization."""
        return True

    @property
    def probability_prediction_available(self) -> bool:
        return False

    @property
    def state_odds_ratio(self) -> float:
        """Conditional proportional-odds multiplier for the lagged high state."""
        return float(np.exp(self.state_dependence))

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

    def conditional_sample_frame(self) -> pd.DataFrame:
        """Return the auditable binary contributions in MRV equation (8)."""
        return self._conditional_sample.copy()

    def entity_score_frame(self) -> pd.DataFrame:
        """Return entity-level composite scores used in the sandwich meat."""
        return self._entity_scores.copy()

    def common_index(self, X: Any, *, lagged_y: Any) -> pd.Series:
        """Return ``X beta + rho 1(lagged_y >= cutoff)``, excluding entity effects.

        The common index cannot be mapped to category probabilities because the
        unrestricted entity effects were conditioned out and are not estimated.
        """
        design, names = _as_2d_array(X)
        if design.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {design.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        lagged = np.asarray(lagged_y, dtype=object)
        if lagged.shape != (len(design),):
            raise ValueError("lagged_y must contain one category per row.")
        positions = {category: index for index, category in enumerate(self.categories)}
        outside = [value for value in pd.unique(lagged) if value not in positions]
        if outside:
            raise ValueError(f"lagged_y contains categories outside the fitted order: {outside}.")
        encoded = np.array([positions[value] for value in lagged], dtype=int)
        cutoff = positions[self.state_cutoff]
        values = design @ self.params.to_numpy() + self.state_dependence * (
            encoded >= cutoff
        )
        index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
        return pd.Series(values, index=index, name="common_index")

    def predict_proba(self, X: Any, **kwargs: Any) -> pd.DataFrame:
        """Reject probability prediction because entity effects are unidentified."""
        raise NotImplementedError(
            "Dynamic fixed-effects conditional likelihood does not estimate entity effects; "
            "category probabilities are unavailable. Use common_index() for the common "
            "component only."
        )

    def predict(self, X: Any, **kwargs: Any) -> pd.Series:
        """Reject category prediction because probabilities are unavailable."""
        raise NotImplementedError(
            "Dynamic fixed-effects conditional likelihood cannot predict categories "
            "without the conditioned-out entity effects."
        )


class DynamicFixedEffectsOrderedLogit:
    """Estimate the restricted fixed-T dynamic Ordered Logit of MRV (2025).

    The lag enters only as ``rho * 1(y[t-1] >= state_cutoff)``.  Each entity
    must have exactly four consecutive observations.  Estimation uses only
    entities whose entire discrete regressor vector is exactly unchanged from
    period 2 to period 3, as required by the paper's stayer assumption.

    The entity effects are unrestricted and conditioned out.  Consequently,
    entity effects and fitted category probabilities are not available.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any,
        time: Any,
        state_cutoff: Any,
        category_order: Any | None = None,
        time_step: float = 1.0,
        maxiter: int = 2_000,
        tolerance: float = 1e-9,
    ) -> DynamicFixedEffectsOrderedLogitResult:
        design, feature_names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=category_order)
        entities = np.asarray(entity, dtype=object)
        times = np.asarray(time)
        nobs, nfeatures = design.shape

        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X feature names must be unique after conversion to strings.")
        reserved = set(feature_names) & ({_STATE_PARAMETER} | set(_META_COLUMNS))
        if reserved:
            raise ValueError(f"X uses reserved feature names: {sorted(reserved)}.")
        if entities.shape != (nobs,) or times.shape != (nobs,) or encoded.shape != (nobs,):
            raise ValueError("X, y, entity, and time must contain the same observations.")
        if pd.isna(entities).any() or pd.isna(times).any():
            raise ValueError("entity and time must not contain missing values.")
        if not np.issubdtype(times.dtype, np.number):
            raise ValueError("time must be numeric for consecutive-period validation.")
        numeric_times = times.astype(float)
        if not np.isfinite(numeric_times).all():
            raise ValueError("time must contain only finite values.")
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")

        cutoff_matches = [
            index for index, category in enumerate(categories) if category == state_cutoff
        ]
        if len(cutoff_matches) != 1 or cutoff_matches[0] == 0:
            raise ValueError(
                "state_cutoff must be one observed category above the lowest category."
            )
        cutoff = cutoff_matches[0]
        ncategories = len(categories)
        if ncategories < 3:
            raise ValueError(
                "Dynamic fixed-effects Ordered Logit requires at least three categories."
            )
        all_threshold_names = [
            f"threshold: {_threshold_name(categories, value)}"
            for value in range(1, ncategories)
        ]
        if len(set(all_threshold_names)) != len(all_threshold_names):
            raise ValueError(
                "Category labels produce duplicate threshold names after string conversion."
            )
        threshold_collisions = set(feature_names) & set(all_threshold_names)
        if threshold_collisions:
            raise ValueError(
                "X feature names collide with reported threshold parameters: "
                f"{sorted(threshold_collisions)}."
            )

        try:
            entity_codes, entity_levels = pd.factorize(entities, sort=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("entity must contain hashable scalar labels.") from exc
        entity_codes = np.asarray(entity_codes, dtype=np.int64)
        nentities = len(entity_levels)
        if nentities < 2:
            raise ValueError("Fixed-effects estimation requires at least two entities.")

        order = np.lexsort((numeric_times, entity_codes))
        sorted_codes = entity_codes[order]
        sorted_times = numeric_times[order]
        sorted_design = design[order]
        sorted_outcome = encoded[order]
        if np.any((sorted_codes[1:] == sorted_codes[:-1]) & (sorted_times[1:] == sorted_times[:-1])):
            raise ValueError("Duplicate entity-time observations are not permitted.")
        counts = np.bincount(sorted_codes, minlength=nentities)
        if not np.all(counts == 4):
            bad = [entity_levels[index] for index in np.flatnonzero(counts != 4)[:5]]
            raise ValueError(
                "The MRV fixed-T estimator requires exactly four observations per entity; "
                f"invalid entities include {bad}."
            )

        panel_design = sorted_design.reshape(nentities, 4, nfeatures)
        panel_outcome = sorted_outcome.reshape(nentities, 4)
        panel_times = sorted_times.reshape(nentities, 4)
        if not np.allclose(
            np.diff(panel_times, axis=1),
            float(time_step),
            rtol=1e-10,
            atol=max(1e-12, abs(float(time_step)) * 1e-10),
        ):
            raise ValueError(
                "Every entity must have four consecutive observations separated by time_step."
            )

        # MRV's stayer condition: positive mass on exact discrete-regressor stayers.
        stayer_mask = np.all(panel_design[:, 2, :] == panel_design[:, 3, :], axis=1)
        stayer_codes = np.flatnonzero(stayer_mask)
        if stayer_codes.size == 0:
            raise ValueError(
                "No exact X[2] == X[3] regressor stayers were found. The current estimator "
                "implements the paper's discrete-regressor case, not continuous matching."
            )

        free_cutoffs = [value for value in range(1, ncategories) if value != cutoff]
        threshold_names = {
            value: f"threshold: {_threshold_name(categories, value)}"
            for value in free_cutoffs
        }
        parameter_names = feature_names + [_STATE_PARAMETER] + [
            threshold_names[value] for value in free_cutoffs
        ]
        threshold_columns = {
            value: nfeatures + 1 + index for index, value in enumerate(free_cutoffs)
        }
        nparameters = len(parameter_names)

        rows: list[np.ndarray] = []
        responses: list[float] = []
        contribution_codes: list[int] = []
        lower_cutoffs: list[Any] = []
        upper_cutoffs: list[Any] = []
        initial_states: list[float] = []
        terminal_states: list[float] = []

        for code in stayer_codes:
            outcome = panel_outcome[code]
            # For the latent-index convention Y* = alpha + X beta + ... - U,
            # the log odds of the downward (B) history relative to the upward
            # (A) history contain (X_1 - X_2) beta.  This is the structural-sign
            # form of the conditional likelihood and makes the reported slope
            # directly comparable with the coefficient in the outcome equation.
            delta_x = panel_design[code, 1] - panel_design[code, 2]
            d0 = float(outcome[0] >= cutoff)
            d1 = float(outcome[1] >= cutoff)
            for lower in range(1, cutoff + 1):
                for upper in range(cutoff, ncategories):
                    is_event = (d1 == 0.0 and outcome[2] >= upper) or (
                        d1 == 1.0 and outcome[2] < lower
                    )
                    if not is_event:
                        continue
                    d3 = float(
                        outcome[3] >= (lower if d1 == 0.0 else upper)
                    )
                    row = np.zeros(nparameters, dtype=float)
                    row[:nfeatures] = delta_x
                    row[nfeatures] = d0 - d3
                    selected_threshold = lower if d3 == 1.0 else upper
                    if selected_threshold != cutoff:
                        row[threshold_columns[selected_threshold]] = 1.0
                    rows.append(row)
                    responses.append(d1)
                    contribution_codes.append(int(code))
                    lower_cutoffs.append(categories[lower])
                    upper_cutoffs.append(categories[upper])
                    initial_states.append(d0)
                    terminal_states.append(d3)

        if not rows:
            raise ValueError(
                "No stayer entity follows an MRV conditioning event; common parameters "
                "are unidentified in this sample."
            )
        pseudo_design = np.vstack(rows)
        pseudo_outcome = np.asarray(responses, dtype=float)
        pseudo_codes = np.asarray(contribution_codes, dtype=np.int64)
        contributing_codes = np.unique(pseudo_codes)
        if np.unique(pseudo_outcome).size < 2:
            raise ValueError(
                "The conditional sample contains only one binary outcome; the composite "
                "likelihood is separated and common parameters are unidentified."
            )
        rank = np.linalg.matrix_rank(pseudo_design)
        if rank < nparameters:
            raise ValueError(
                "The MRV conditional design is rank deficient. Identification requires "
                "variation in X[1]-X[2], initial/terminal states, and every free threshold. "
                f"rank={rank}, parameters={nparameters}."
            )
        if contributing_codes.size <= nparameters:
            raise ValueError(
                "The number of contributing entities must exceed the freely estimated "
                "common parameters."
            )

        def objective(parameters: np.ndarray) -> float:
            linear = pseudo_design @ parameters
            return float(np.sum(np.logaddexp(0.0, linear) - pseudo_outcome * linear))

        def gradient(parameters: np.ndarray) -> np.ndarray:
            linear = pseudo_design @ parameters
            return pseudo_design.T @ (expit(linear) - pseudo_outcome)

        # Enforce the ordered threshold parameter space while retaining the
        # paper's zero normalization gamma_k = 0.
        constraint_matrix = np.zeros((ncategories - 2, nparameters), dtype=float)
        for row_index, left in enumerate(range(1, ncategories - 1)):
            right = left + 1
            if right != cutoff:
                constraint_matrix[row_index, threshold_columns[right]] += 1.0
            if left != cutoff:
                constraint_matrix[row_index, threshold_columns[left]] -= 1.0

        # A finite conditional Logit MLE does not exist under complete or
        # quasi-complete separation. Search for a feasible direction whose
        # signed margins are all nonnegative and at least one is positive,
        # while preserving threshold order along the direction. In that case
        # the homogeneous LP objective is unbounded below.
        signed_design = (2.0 * pseudo_outcome - 1.0)[:, None] * pseudo_design
        separation_constraints = np.vstack((-signed_design, -constraint_matrix))
        separation_check = linprog(
            -signed_design.sum(axis=0),
            A_ub=separation_constraints,
            b_ub=np.zeros(len(separation_constraints), dtype=float),
            bounds=[(None, None)] * nparameters,
            method="highs",
        )
        if separation_check.status == 3:
            raise ValueError(
                "The MRV conditional sample is completely or quasi-completely "
                "separated; a finite conditional MLE does not exist."
            )
        minimum_gap = 1e-8
        constraint = LinearConstraint(
            constraint_matrix,
            lb=np.full(ncategories - 2, minimum_gap),
            ub=np.full(ncategories - 2, np.inf),
        )
        initial = np.zeros(nparameters, dtype=float)
        for value in free_cutoffs:
            initial[threshold_columns[value]] = float(value - cutoff)
        fitted = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            constraints=[constraint],
            options={"maxiter": int(maxiter), "ftol": float(tolerance)},
        )
        estimates = np.asarray(fitted.x, dtype=float)

        constraint_slack = constraint_matrix @ estimates - minimum_gap
        active_constraints = constraint_slack <= max(10.0 * minimum_gap, 1e-7)
        fitted_gradient = gradient(estimates)
        if np.any(active_constraints):
            active_matrix = constraint_matrix[active_constraints]
            multipliers, _ = nnls(active_matrix.T, fitted_gradient)
            kkt_gradient = fitted_gradient - active_matrix.T @ multipliers
        else:
            kkt_gradient = fitted_gradient
        scaled_kkt_residual = max(
            float(np.linalg.norm(kkt_gradient, ord=np.inf)) / len(pseudo_outcome),
            float(np.maximum(-constraint_slack, 0.0).max(initial=0.0)),
        )
        stationarity_tolerance = max(min(100.0 * float(tolerance), 1e-5), 1e-7)
        converged = bool(
            fitted.success
            and np.isfinite(fitted.fun)
            and np.isfinite(scaled_kkt_residual)
            and scaled_kkt_residual <= stationarity_tolerance
        )

        linear = pseudo_design @ estimates
        probabilities = expit(linear)
        weights = probabilities * (1.0 - probabilities)
        information = pseudo_design.T @ (weights[:, None] * pseudo_design)
        information = (information + information.T) / 2.0
        entity_scores = np.zeros((contributing_codes.size, nparameters), dtype=float)
        score_lookup = {code: index for index, code in enumerate(contributing_codes)}
        row_scores = pseudo_design * (pseudo_outcome - probabilities)[:, None]
        for row_index, code in enumerate(pseudo_codes):
            entity_scores[score_lookup[code]] += row_scores[row_index]

        threshold_values = np.zeros(ncategories - 1, dtype=float)
        for value in free_cutoffs:
            threshold_values[value - 1] = estimates[threshold_columns[value]]
        fitted_gaps = np.diff(threshold_values)
        eigenvalues = (
            np.linalg.eigvalsh(information)
            if np.isfinite(information).all()
            else np.array([np.nan])
        )
        maximum_eigenvalue = float(np.max(eigenvalues))
        minimum_eigenvalue = float(np.min(eigenvalues))
        information_well_conditioned = bool(
            np.isfinite(eigenvalues).all()
            and maximum_eigenvalue > 0.0
            and minimum_eigenvalue > 0.0
            and minimum_eigenvalue / maximum_eigenvalue > 1e-12
        )
        inference_valid = bool(
            converged
            and information_well_conditioned
            and (fitted_gaps.size == 0 or fitted_gaps.min() > 100.0 * minimum_gap)
            and contributing_codes.size > nparameters
            and len(pseudo_outcome) > nparameters
        )
        if inference_valid:
            bread = np.linalg.inv(information)
            meat = entity_scores.T @ entity_scores
            correction = contributing_codes.size / (contributing_codes.size - 1.0)
            correction *= (len(pseudo_outcome) - 1.0) / (
                len(pseudo_outcome) - nparameters
            )
            covariance_values = correction * bread @ meat @ bread
            covariance_values = (covariance_values + covariance_values.T) / 2.0
            covariance_scale = max(
                1.0, float(np.nanmax(np.abs(np.diag(covariance_values))))
            )
            inference_valid = bool(
                np.isfinite(covariance_values).all()
                and np.diag(covariance_values).min() >= -1e-12 * covariance_scale
            )
        if inference_valid:
            standard_error_values = np.sqrt(
                np.clip(np.diag(covariance_values), 0.0, None)
            )
            zstat_values = np.divide(
                estimates,
                standard_error_values,
                out=np.full_like(estimates, np.nan),
                where=standard_error_values > 0.0,
            )
            pvalue_values = 2.0 * ndtr(-np.abs(zstat_values))
        else:
            covariance_values = np.full((nparameters, nparameters), np.nan)
            standard_error_values = np.full(nparameters, np.nan)
            zstat_values = np.full(nparameters, np.nan)
            pvalue_values = np.full(nparameters, np.nan)

        threshold_index = [
            _threshold_name(categories, value) for value in range(1, ncategories)
        ]
        normalized_name = _threshold_name(categories, cutoff)
        conditional_sample = pd.DataFrame(pseudo_design, columns=parameter_names)
        conditional_sample.insert(0, "_terminal_state", terminal_states)
        conditional_sample.insert(0, "_initial_state", initial_states)
        conditional_sample.insert(0, "_response", pseudo_outcome)
        conditional_sample.insert(0, "_upper_cutoff", upper_cutoffs)
        conditional_sample.insert(0, "_lower_cutoff", lower_cutoffs)
        conditional_sample.insert(
            0, "_entity", [entity_levels[code] for code in pseudo_codes]
        )
        score_frame = pd.DataFrame(
            entity_scores,
            index=pd.Index(
                [entity_levels[code] for code in contributing_codes], name="entity"
            ),
            columns=parameter_names,
        )

        return DynamicFixedEffectsOrderedLogitResult(
            params=pd.Series(
                estimates[:nfeatures], index=feature_names, name="coefficient"
            ),
            state_dependence=float(estimates[nfeatures]),
            thresholds=pd.Series(
                threshold_values, index=threshold_index, name="threshold"
            ),
            normalized_threshold=normalized_name,
            covariance=pd.DataFrame(
                covariance_values, index=parameter_names, columns=parameter_names
            ),
            standard_errors=pd.Series(
                standard_error_values,
                index=parameter_names,
                name="standard_error",
            ),
            zstats=pd.Series(zstat_values, index=parameter_names, name="z_stat"),
            pvalues=pd.Series(pvalue_values, index=parameter_names, name="p_value"),
            converged=converged,
            inference_valid=inference_valid,
            scaled_kkt_residual=scaled_kkt_residual,
            composite_loglike=float(-fitted.fun),
            nobs=nobs,
            n_entities=nentities,
            n_stayer_entities=int(stayer_codes.size),
            n_contributing_entities=int(contributing_codes.size),
            n_conditional_contributions=len(pseudo_outcome),
            categories=categories,
            state_cutoff=state_cutoff,
            feature_names=tuple(feature_names),
            optimizer_result=fitted,
            _conditional_sample=conditional_sample,
            _entity_scores=score_frame,
        )
