"""Ordinal-response estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr
from scipy.stats import chi2, norm


def _as_2d_array(X: Any) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        names = [str(column) for column in X.columns]
        values = X.to_numpy(dtype=float)
    else:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        names = [f"x{index}" for index in range(values.shape[1])]

    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional array or DataFrame.")
    if values.shape[1] == 0:
        raise ValueError("X must contain at least one regressor.")
    if not np.isfinite(values).all():
        raise ValueError("X contains missing or non-finite values.")
    return values, names


def _ordered_categories(
    y: Any, category_order: Sequence[Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(y, dtype=object)
    if values.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if pd.isna(values).any():
        raise ValueError("y contains missing values.")

    if category_order is None:
        if isinstance(y, pd.Series) and isinstance(y.dtype, pd.CategoricalDtype):
            if not y.dtype.ordered:
                raise ValueError(
                    "Categorical y must be ordered or category_order must be supplied."
                )
            ordered_values = list(y.cat.categories)
        elif isinstance(y, (pd.Categorical, pd.CategoricalIndex)):
            if not y.ordered:
                raise ValueError(
                    "Categorical y must be ordered or category_order must be supplied."
                )
            ordered_values = list(y.categories)
        else:
            try:
                ordered_values = list(np.unique(values))
            except TypeError as error:
                raise ValueError(
                    "Outcome labels cannot be sorted safely; supply category_order."
                ) from error
    else:
        if isinstance(category_order, (str, bytes)):
            raise ValueError("category_order must be a sequence of category labels.")
        ordered_values = list(category_order)
    if any(
        isinstance(category, (dict, list, set, tuple, np.ndarray))
        for category in ordered_values
    ):
        raise ValueError("Ordinal category labels must be scalar values.")
    if any(pd.isna(category) for category in ordered_values):
        raise ValueError("category_order must not contain missing labels.")
    if len(pd.unique(pd.Series(ordered_values, dtype="object"))) != len(ordered_values):
        raise ValueError("category_order must contain unique labels.")
    unobserved = [
        category for category in ordered_values if not np.any(values == category)
    ]
    outside_order = [
        value
        for value in pd.unique(values)
        if not any(value == category for category in ordered_values)
    ]
    if unobserved or outside_order:
        raise ValueError(
            "category_order must contain each observed category exactly once; "
            f"unobserved={unobserved}, missing={outside_order}."
        )
    categories = np.asarray(ordered_values, dtype=object)
    if categories.size < 3:
        raise ValueError("Ordered Logit requires at least three observed categories.")
    category_positions = {category: index for index, category in enumerate(categories)}
    encoded = np.array([category_positions[value] for value in values], dtype=int)
    return encoded.astype(int), categories


def _unpack_thresholds(raw: np.ndarray) -> np.ndarray:
    thresholds = np.empty_like(raw, dtype=float)
    thresholds[0] = raw[0]
    if raw.size > 1:
        thresholds[1:] = raw[0] + np.cumsum(np.exp(raw[1:]))
    return thresholds


def _link_cdf(values: np.ndarray, link: str) -> np.ndarray:
    if link == "logit":
        return expit(values)
    if link == "probit":
        return ndtr(values)
    raise ValueError(f"Unsupported ordinal link: {link!r}.")


def _link_pdf(values: np.ndarray, link: str) -> np.ndarray:
    if link == "logit":
        probabilities = expit(values)
        return probabilities * (1.0 - probabilities)
    if link == "probit":
        return np.exp(-0.5 * values**2) / np.sqrt(2.0 * np.pi)
    raise ValueError(f"Unsupported ordinal link: {link!r}.")


def _category_probabilities(
    X: np.ndarray, beta: np.ndarray, thresholds: np.ndarray, link: str
) -> np.ndarray:
    linear_predictor = X @ beta
    cumulative = _link_cdf(thresholds[None, :] - linear_predictor[:, None], link)
    bounds = np.column_stack(
        [np.zeros(X.shape[0]), cumulative, np.ones(X.shape[0])]
    )
    return np.diff(bounds, axis=1)


def _numerical_hessian(function: Any, point: np.ndarray) -> np.ndarray:
    """Compute a central-difference Hessian at an optimum."""
    point = np.asarray(point, dtype=float)
    steps = 1e-4 * (1.0 + np.abs(point))
    hessian = np.empty((point.size, point.size), dtype=float)
    center = function(point)

    for row in range(point.size):
        row_step = np.zeros_like(point)
        row_step[row] = steps[row]
        hessian[row, row] = (
            function(point + row_step) - 2.0 * center + function(point - row_step)
        ) / steps[row] ** 2

        for column in range(row):
            column_step = np.zeros_like(point)
            column_step[column] = steps[column]
            value = (
                function(point + row_step + column_step)
                - function(point + row_step - column_step)
                - function(point - row_step + column_step)
                + function(point - row_step - column_step)
            ) / (4.0 * steps[row] * steps[column])
            hessian[row, column] = value
            hessian[column, row] = value

    return hessian


def _threshold_jacobian(raw: np.ndarray) -> np.ndarray:
    """Jacobian mapping unconstrained threshold parameters to ordered cuts."""
    jacobian = np.zeros((raw.size, raw.size), dtype=float)
    jacobian[:, 0] = 1.0
    for column in range(1, raw.size):
        jacobian[column:, column] = np.exp(raw[column])
    return jacobian


def _numerical_jacobian(function: Any, point: np.ndarray) -> np.ndarray:
    """Compute a central-difference Jacobian for a vector-valued function."""
    point = np.asarray(point, dtype=float)
    baseline = np.asarray(function(point), dtype=float)
    jacobian = np.empty((baseline.size, point.size), dtype=float)
    steps = 1e-5 * (1.0 + np.abs(point))
    for column, step in enumerate(steps):
        shift = np.zeros_like(point)
        shift[column] = step
        jacobian[:, column] = (function(point + shift) - function(point - shift)) / (
            2.0 * step
        )
    return jacobian


@dataclass(frozen=True)
class OrderedResult:
    """Fitted ordinal-response result."""

    params: pd.Series
    thresholds: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    zstats: pd.Series
    pvalues: pd.Series
    inference_valid: bool
    categories: np.ndarray
    converged: bool
    loglike: float
    nobs: int
    feature_names: tuple[str, ...]
    link: str
    optimizer_result: Any

    @property
    def all_params(self) -> pd.Series:
        """Return coefficients and thresholds in covariance-matrix order."""
        cuts = self.thresholds.copy()
        cuts.index = [f"threshold: {name}" for name in cuts.index]
        return pd.concat([self.params, cuts]).rename("estimate")

    @property
    def n_params(self) -> int:
        """Number of freely estimated coefficients and thresholds."""
        return len(self.all_params)

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        """Return normal-approximation confidence intervals."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        estimates = self.all_params
        return pd.DataFrame(
            {
                "lower": estimates - critical * self.standard_errors,
                "upper": estimates + critical * self.standard_errors,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return the ecosystem-standard coefficient table."""
        from .postestimation import summary_frame

        return summary_frame(self)

    def vcov(self) -> pd.DataFrame:
        """Return a defensive copy of the covariance matrix."""
        return self.covariance.copy()

    def lincom(
        self,
        weights: Mapping[str, float],
        *,
        value: float = 0.0,
        level: float = 0.95,
    ) -> pd.Series:
        """Estimate and test a linear combination of fitted parameters."""
        if not weights:
            raise ValueError("weights must contain at least one parameter.")
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        parameter_names = list(self.all_params.index)
        unknown = set(weights) - set(parameter_names)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}.")

        contrast = np.zeros(len(parameter_names), dtype=float)
        for name, weight in weights.items():
            numeric_weight = float(weight)
            if not np.isfinite(numeric_weight):
                raise ValueError("Linear-combination weights must be finite.")
            contrast[parameter_names.index(name)] = numeric_weight

        estimate = float(contrast @ self.all_params.to_numpy(dtype=float))
        variance = float(contrast @ self.covariance.to_numpy(dtype=float) @ contrast)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        zstat = (estimate - float(value)) / standard_error if standard_error > 0 else np.nan
        pvalue = float(2.0 * ndtr(-abs(zstat)))
        critical = float(norm.ppf(0.5 + level / 2.0))
        return pd.Series(
            {
                "estimate": estimate,
                "standard_error": standard_error,
                "z_stat": zstat,
                "p_value": pvalue,
                "lower": estimate - critical * standard_error,
                "upper": estimate + critical * standard_error,
            },
            name="lincom",
        )

    def wald_test(
        self,
        restrictions: Mapping[str, float] | Sequence[Mapping[str, float]],
        *,
        values: float | Sequence[float] = 0.0,
    ) -> pd.Series:
        """Test one or more linear restrictions using a Wald chi-square test."""
        rows = [restrictions] if isinstance(restrictions, Mapping) else list(restrictions)
        if not rows:
            raise ValueError("restrictions must contain at least one restriction.")
        if any(not isinstance(row, Mapping) or not row for row in rows):
            raise ValueError("Each restriction must be a non-empty parameter-weight mapping.")

        parameter_names = list(self.all_params.index)
        restriction_matrix = np.zeros((len(rows), len(parameter_names)), dtype=float)
        for row_index, row in enumerate(rows):
            unknown = set(row) - set(parameter_names)
            if unknown:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}.")
            for name, weight in row.items():
                numeric_weight = float(weight)
                if not np.isfinite(numeric_weight):
                    raise ValueError("Restriction weights must be finite.")
                restriction_matrix[row_index, parameter_names.index(name)] = numeric_weight

        if np.isscalar(values):
            null_values = np.full(len(rows), float(values))
        else:
            null_values = np.asarray(values, dtype=float)
            if null_values.shape != (len(rows),):
                raise ValueError("values must provide one null value per restriction.")
        if not np.isfinite(null_values).all():
            raise ValueError("Restriction null values must be finite.")

        difference = restriction_matrix @ self.all_params.to_numpy(dtype=float) - null_values
        restricted_covariance = (
            restriction_matrix
            @ self.covariance.to_numpy(dtype=float)
            @ restriction_matrix.T
        )
        statistic = float(difference @ np.linalg.pinv(restricted_covariance) @ difference)
        degrees_of_freedom = int(np.linalg.matrix_rank(restriction_matrix))
        return pd.Series(
            {
                "statistic": statistic,
                "df": degrees_of_freedom,
                "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
            },
            name="wald_test",
        )

    def predict_proba(self, X: Any) -> pd.DataFrame:
        """Return one predicted probability per ordered category."""
        values, names = _as_2d_array(X)
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")

        probabilities = _category_probabilities(
            values,
            self.params.to_numpy(dtype=float),
            self.thresholds.to_numpy(dtype=float),
            self.link,
        )
        return pd.DataFrame(probabilities, columns=self.categories)

    def predict(self, X: Any) -> pd.Series:
        """Return the category with the highest predicted probability."""
        probabilities = self.predict_proba(X).to_numpy()
        return pd.Series(self.categories[np.argmax(probabilities, axis=1)], name="prediction")

    def marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return observation-level effects on each category probability.

        Effects are analytical derivatives with respect to continuous
        regressors. Columns use a ``(category, feature)`` MultiIndex.
        """
        values, names = _as_2d_array(X)
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")

        linear_predictor = values @ self.params.to_numpy(dtype=float)
        threshold_values = self.thresholds.to_numpy(dtype=float)
        density_at_cuts = _link_pdf(
            threshold_values[None, :] - linear_predictor[:, None], self.link
        )
        density_bounds = np.column_stack(
            [np.zeros(values.shape[0]), density_at_cuts, np.zeros(values.shape[0])]
        )
        category_slopes = density_bounds[:, :-1] - density_bounds[:, 1:]
        effects = category_slopes[:, :, None] * self.params.to_numpy()[None, None, :]
        columns = pd.MultiIndex.from_product(
            [self.categories, self.feature_names], names=["category", "feature"]
        )
        return pd.DataFrame(effects.reshape(values.shape[0], -1), columns=columns)

    def average_marginal_effects(self, X: Any) -> pd.DataFrame:
        """Return sample-average category-specific marginal effects."""
        effects = self.marginal_effects(X)
        average = effects.mean(axis=0).unstack("feature")
        average.index.name = "category"
        average.columns.name = "feature"
        return average

    def average_marginal_effects_inference(
        self, X: Any, *, level: float = 0.95
    ) -> pd.DataFrame:
        """Return delta-method inference for average marginal effects."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        values, names = _as_2d_array(X)
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")

        n_features = len(self.feature_names)

        def flattened_ame(parameters: np.ndarray) -> np.ndarray:
            beta = parameters[:n_features]
            thresholds = parameters[n_features:]
            linear_predictor = values @ beta
            densities = _link_pdf(
                thresholds[None, :] - linear_predictor[:, None], self.link
            )
            density_bounds = np.column_stack(
                [np.zeros(values.shape[0]), densities, np.zeros(values.shape[0])]
            )
            category_slopes = (density_bounds[:, :-1] - density_bounds[:, 1:]).mean(
                axis=0
            )
            return (category_slopes[:, None] * beta[None, :]).reshape(-1)

        reported_parameters = self.all_params.to_numpy(dtype=float)
        estimates = flattened_ame(reported_parameters)
        jacobian = _numerical_jacobian(flattened_ame, reported_parameters)
        ame_covariance = jacobian @ self.covariance.to_numpy(dtype=float) @ jacobian.T
        ame_covariance = (ame_covariance + ame_covariance.T) / 2.0
        standard_errors = np.sqrt(np.clip(np.diag(ame_covariance), 0.0, None))
        zstats = np.divide(
            estimates,
            standard_errors,
            out=np.full_like(estimates, np.nan),
            where=standard_errors > 0,
        )
        pvalues = 2.0 * ndtr(-np.abs(zstats))
        critical = norm.ppf(0.5 + level / 2.0)
        index = pd.MultiIndex.from_product(
            [self.categories, self.feature_names], names=["category", "feature"]
        )
        return pd.DataFrame(
            {
                "estimate": estimates,
                "standard_error": standard_errors,
                "z_stat": zstats,
                "p_value": pvalues,
                "lower": estimates - critical * standard_errors,
                "upper": estimates + critical * standard_errors,
            },
            index=index,
        )

    def margins(
        self,
        X: Any,
        *,
        at: str | Mapping[str, float] = "overall",
        kind: str = "probability",
    ) -> pd.Series | pd.DataFrame:
        """Evaluate ordinal margins over observed or representative covariates.

        Parameters
        ----------
        X
            Evaluation sample.
        at
            ``"overall"`` averages over observations, ``"mean"`` evaluates at
            covariate means, and a mapping overrides selected means with
            user-specified values.
        kind
            ``"probability"`` for category probabilities or
            ``"marginal_effect"`` for continuous-regressor derivatives.
        """
        values, names = _as_2d_array(X)
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} columns; expected {len(self.feature_names)}."
            )
        if isinstance(X, pd.DataFrame) and tuple(names) != self.feature_names:
            raise ValueError("DataFrame columns must match the fitted feature names and order.")
        if kind not in {"probability", "marginal_effect"}:
            raise ValueError("kind must be 'probability' or 'marginal_effect'.")

        evaluation: pd.DataFrame | np.ndarray
        if at == "overall":
            evaluation = X if isinstance(X, pd.DataFrame) else values
        elif at == "mean":
            evaluation = pd.DataFrame(
                [values.mean(axis=0)], columns=self.feature_names
            )
        elif isinstance(at, Mapping):
            unknown = set(at) - set(self.feature_names)
            if unknown:
                raise ValueError(f"Unknown covariates in at: {sorted(unknown)}.")
            means = values.mean(axis=0)
            representative = {
                feature: means[index] for index, feature in enumerate(self.feature_names)
            }
            for feature, value in at.items():
                numeric_value = float(value)
                if not np.isfinite(numeric_value):
                    raise ValueError("User-specified margin values must be finite.")
                representative[feature] = numeric_value
            evaluation = pd.DataFrame([representative], columns=self.feature_names)
        else:
            raise ValueError("at must be 'overall', 'mean', or a covariate-value mapping.")

        if kind == "probability":
            probabilities = self.predict_proba(evaluation).mean(axis=0)
            probabilities.index.name = "category"
            return probabilities.rename("estimate")

        effects = self.marginal_effects(evaluation).mean(axis=0).unstack("feature")
        effects.index.name = "category"
        effects.columns.name = "feature"
        return effects

    def proportional_odds_test(self, X: Any, y: Any) -> ProportionalOddsTestResult:
        """Run a Brant-style joint slope-stability diagnostic.

        The diagnostic fits cumulative binary logits for ``y > category`` and
        uses their stacked score contributions to retain cross-equation
        covariance in the joint Wald test.
        """
        if self.link != "logit":
            raise ValueError("The proportional-odds test is currently available for Logit only.")
        values, names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=self.categories)
        if values.shape[0] != encoded.size:
            raise ValueError("X and y must contain the same number of observations.")
        if tuple(names) != self.feature_names:
            raise ValueError("X columns must match the fitted feature names and order.")
        if not np.array_equal(categories, self.categories):
            raise ValueError("y categories must match the fitted result categories.")

        design = np.column_stack([np.ones(values.shape[0]), values])
        equation_parameters = []
        equation_breads = []
        equation_scores = []

        for threshold_index in range(categories.size - 1):
            binary = (encoded > threshold_index).astype(float)

            def objective(parameters: np.ndarray, target: np.ndarray = binary) -> float:
                probabilities = expit(design @ parameters)
                return float(
                    -np.sum(
                        target * np.log(np.clip(probabilities, 1e-15, 1.0))
                        + (1.0 - target)
                        * np.log(np.clip(1.0 - probabilities, 1e-15, 1.0))
                    )
                )

            fitted = minimize(objective, np.zeros(design.shape[1]), method="BFGS")
            if not fitted.success and not np.isfinite(fitted.fun):
                raise RuntimeError("A threshold-specific binary Logit fit failed.")
            probabilities = expit(design @ fitted.x)
            weights = probabilities * (1.0 - probabilities)
            bread = np.linalg.pinv(design.T @ (weights[:, None] * design))
            scores = design * (binary - probabilities)[:, None]
            equation_parameters.append(fitted.x)
            equation_breads.append(bread)
            equation_scores.append(scores)

        n_equations = len(equation_parameters)
        equation_size = design.shape[1]
        stacked_covariance = np.empty(
            (n_equations * equation_size, n_equations * equation_size), dtype=float
        )
        for row in range(n_equations):
            for column in range(n_equations):
                score_cross_product = equation_scores[row].T @ equation_scores[column]
                block = equation_breads[row] @ score_cross_product @ equation_breads[column]
                row_slice = slice(row * equation_size, (row + 1) * equation_size)
                column_slice = slice(column * equation_size, (column + 1) * equation_size)
                stacked_covariance[row_slice, column_slice] = block

        stacked_parameters = np.concatenate(equation_parameters)
        n_restrictions = (n_equations - 1) * len(self.feature_names)
        restriction_matrix = np.zeros(
            (n_restrictions, stacked_parameters.size), dtype=float
        )
        restriction_row = 0
        for equation in range(1, n_equations):
            for feature_index in range(len(self.feature_names)):
                parameter_offset = feature_index + 1
                restriction_matrix[restriction_row, parameter_offset] = -1.0
                restriction_matrix[
                    restriction_row, equation * equation_size + parameter_offset
                ] = 1.0
                restriction_row += 1

        differences = restriction_matrix @ stacked_parameters
        restriction_covariance = (
            restriction_matrix @ stacked_covariance @ restriction_matrix.T
        )
        statistic = float(
            differences @ np.linalg.pinv(restriction_covariance) @ differences
        )
        degrees_of_freedom = int(np.linalg.matrix_rank(restriction_matrix))
        coefficient_table = pd.DataFrame(
            [parameters[1:] for parameters in equation_parameters],
            index=[
                f"{categories[index]} | higher" for index in range(n_equations)
            ],
            columns=self.feature_names,
        )
        coefficient_table.index.name = "cumulative_split"
        return ProportionalOddsTestResult(
            statistic=statistic,
            df=degrees_of_freedom,
            p_value=float(chi2.sf(statistic, degrees_of_freedom)),
            threshold_coefficients=coefficient_table,
        )


@dataclass(frozen=True)
class ProportionalOddsTestResult:
    """Brant-style joint diagnostic for proportional-odds slope stability."""

    statistic: float
    df: int
    p_value: float
    threshold_coefficients: pd.DataFrame


class _OrderedModel:
    link: str

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        category_order: Sequence[Any] | None = None,
        maxiter: int = 1_000,
        tolerance: float = 1e-9,
    ) -> OrderedResult:
        values, feature_names = _as_2d_array(X)
        encoded, categories = _ordered_categories(y, category_order=category_order)
        if values.shape[0] != encoded.size:
            raise ValueError("X and y must contain the same number of observations.")
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

        n_features = values.shape[1]
        n_thresholds = categories.size - 1
        cumulative_shares = np.array(
            [np.mean(encoded <= index) for index in range(n_thresholds)], dtype=float
        )
        initial_thresholds = np.log(
            np.clip(cumulative_shares, 1e-6, 1 - 1e-6)
            / np.clip(1 - cumulative_shares, 1e-6, 1 - 1e-6)
        )
        raw_thresholds = np.r_[
            initial_thresholds[0], np.log(np.diff(initial_thresholds))
        ]
        initial = np.r_[np.zeros(n_features), raw_thresholds]

        def negative_loglike(parameters: np.ndarray) -> float:
            beta = parameters[:n_features]
            thresholds = _unpack_thresholds(parameters[n_features:])
            probabilities = _category_probabilities(values, beta, thresholds, self.link)
            selected = probabilities[np.arange(encoded.size), encoded]
            return float(-np.log(np.clip(selected, 1e-15, 1.0)).sum())

        fitted = minimize(
            negative_loglike,
            initial,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": tolerance},
        )
        if not np.isfinite(fitted.fun):
            raise RuntimeError(
                f"Ordered {self.link.title()} optimization produced a non-finite likelihood."
            )

        beta = fitted.x[:n_features]
        fitted_raw_thresholds = fitted.x[n_features:]
        thresholds = _unpack_thresholds(fitted_raw_thresholds)
        threshold_names = [
            f"{categories[index]} | {categories[index + 1]}"
            for index in range(n_thresholds)
        ]
        parameter_names = feature_names + [f"threshold: {name}" for name in threshold_names]

        information = _numerical_hessian(negative_loglike, fitted.x)
        information = (information + information.T) / 2.0
        information_eigenvalues = np.linalg.eigvalsh(information)
        inference_valid = bool(
            fitted.success
            and np.isfinite(information_eigenvalues).all()
            and np.min(information_eigenvalues) > 1e-8
        )
        raw_covariance = np.linalg.pinv(information)
        transformation = np.eye(fitted.x.size)
        transformation[n_features:, n_features:] = _threshold_jacobian(
            fitted_raw_thresholds
        )
        covariance_values = transformation @ raw_covariance @ transformation.T
        covariance_values = (covariance_values + covariance_values.T) / 2.0
        standard_error_values = np.sqrt(np.clip(np.diag(covariance_values), 0.0, None))
        reported_values = np.r_[beta, thresholds]
        zstat_values = np.divide(
            reported_values,
            standard_error_values,
            out=np.full_like(reported_values, np.nan),
            where=standard_error_values > 0,
        )
        pvalue_values = 2.0 * ndtr(-np.abs(zstat_values))
        if not inference_valid:
            covariance_values[:] = np.nan
            standard_error_values[:] = np.nan
            zstat_values[:] = np.nan
            pvalue_values[:] = np.nan

        return OrderedResult(
            params=pd.Series(beta, index=feature_names, name="coefficient"),
            thresholds=pd.Series(thresholds, index=threshold_names, name="threshold"),
            covariance=pd.DataFrame(
                covariance_values, index=parameter_names, columns=parameter_names
            ),
            standard_errors=pd.Series(
                standard_error_values, index=parameter_names, name="standard_error"
            ),
            zstats=pd.Series(zstat_values, index=parameter_names, name="z_stat"),
            pvalues=pd.Series(pvalue_values, index=parameter_names, name="p_value"),
            inference_valid=inference_valid,
            categories=categories,
            converged=bool(fitted.success),
            loglike=float(-fitted.fun),
            nobs=values.shape[0],
            feature_names=tuple(feature_names),
            link=self.link,
            optimizer_result=fitted,
        )


class OrderedLogit(_OrderedModel):
    """Proportional-odds Ordered Logit estimated by maximum likelihood."""

    link = "logit"


class OrderedProbit(_OrderedModel):
    """Ordered Probit estimated by maximum likelihood."""

    link = "probit"


# Compatibility aliases retained while the public result API is still young.
OrderedLogitResult = OrderedResult
OrderedProbitResult = OrderedResult
