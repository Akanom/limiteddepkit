"""Persisted Stata-style factor-variable design compilation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, combinations, product
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from ._varlist import varlist

__all__ = ["FactorVariableCompiler"]


@dataclass(frozen=True)
class _Component:
    kind: str
    source: str


@dataclass(frozen=True)
class _Atom:
    kind: str
    source: str
    level: Any | None
    main_name: str
    interaction_name: str

    @property
    def semantic_key(self) -> tuple[str, str, str, str]:
        level_type = "" if self.level is None else type(self.level).__qualname__
        level_value = "" if self.level is None else repr(self.level)
        return self.kind, self.source, level_type, level_value


@dataclass(frozen=True)
class _OutputColumn:
    name: str
    atoms: tuple[_Atom, ...]
    semantic_key: tuple[tuple[str, str, str, str], ...]


def _normalize_terms(variables: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(variables, str):
        terms = variables.split()
    else:
        if isinstance(variables, (bytes, bytearray, Mapping, set, frozenset)):
            raise TypeError("variables must be a string or an ordered iterable of strings.")
        try:
            terms = list(variables)
        except TypeError as error:
            raise TypeError(
                "variables must be a string or an ordered iterable of strings."
            ) from error
        if not all(isinstance(term, str) for term in terms):
            raise TypeError("Every variables entry must be a string.")
        terms = [term.strip() for term in terms]

    if not terms:
        raise ValueError("variables must contain at least one factor-variable term.")
    if any(not term for term in terms):
        raise ValueError("variables entries must not be empty strings.")
    return tuple(terms)


def _copy_mapping(value: Mapping[str, Any] | None, *, argument: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{argument} must be a mapping keyed by source column name.")
    copied = dict(value)
    if not all(isinstance(key, str) for key in copied):
        raise TypeError(f"Every {argument} key must be a string source column name.")
    return copied


def _parse_term(term: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    if "##" in term:
        if "#" in term.replace("##", ""):
            raise ValueError(
                f"Factor-variable term {term!r} mixes # and ##; split it into explicit terms."
            )
        operator = "full"
        raw_components = term.split("##")
    elif "#" in term:
        operator = "interaction"
        raw_components = term.split("#")
    else:
        operator = "main"
        raw_components = [term]

    if any(not component for component in raw_components):
        raise ValueError(f"Factor-variable term {term!r} contains an empty component.")

    parsed: list[tuple[str, str]] = []
    for component in raw_components:
        if component.startswith("i."):
            kind = "categorical"
            selector = component[2:]
        elif component.startswith("c."):
            kind = "continuous"
            selector = component[2:]
        else:
            kind = "continuous"
            selector = component
            if component.startswith(("b.", "bn.", "o.")) or (
                component.startswith("ib")
                and len(component) > 2
                and (component[2].isdigit() or component[2] in ".(")
            ):
                raise ValueError(
                    f"Factor-variable component {component!r} uses unsupported inline "
                    "base/omission syntax; pass base_categories instead."
                )
        if not selector:
            raise ValueError(f"Factor-variable component {component!r} has no variable name.")
        parsed.append((kind, selector))
    return operator, tuple(parsed)


def _ordered_levels(value: Any, *, argument: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise TypeError(f"{argument} must be an ordered sequence of category levels.")
    if not isinstance(value, Sequence):
        try:
            levels = list(value)
        except TypeError as error:
            raise TypeError(
                f"{argument} must be an ordered sequence of category levels."
            ) from error
    else:
        levels = list(value)
    if not levels:
        raise ValueError(f"{argument} must contain at least two category levels.")
    return levels


def _validate_levels(levels: list[Any], *, source: str) -> tuple[Any, ...]:
    if len(levels) < 2:
        raise ValueError(f"Categorical variable {source!r} must contain at least two levels.")
    for level in levels:
        try:
            hash(level)
        except TypeError as error:
            raise TypeError(
                f"Categorical variable {source!r} contains an unhashable level {level!r}."
            ) from error
        missing = pd.isna(level)
        if not isinstance(missing, (bool, np.bool_)) or bool(missing):
            raise ValueError(f"Categorical variable {source!r} contains a missing level.")
    if len(set(levels)) != len(levels):
        raise ValueError(f"Category order for {source!r} contains duplicate levels.")
    return tuple(levels)


def _fit_category_levels(
    series: pd.Series,
    *,
    source: str,
    configured_order: Any | None,
) -> tuple[Any, ...]:
    if series.isna().any():
        raise ValueError(f"Categorical variable {source!r} contains missing values.")
    observed = list(pd.unique(series))
    if configured_order is not None:
        levels = _ordered_levels(
            configured_order,
            argument=f"category_orders[{source!r}]",
        )
    elif isinstance(series.dtype, pd.CategoricalDtype):
        levels = list(series.cat.categories)
    else:
        try:
            levels = sorted(observed)
        except TypeError as error:
            raise ValueError(
                f"Categorical variable {source!r} has levels without a deterministic sort "
                "order; pass category_orders for that variable."
            ) from error

    validated = _validate_levels(levels, source=source)
    observed_set = set(observed)
    level_set = set(validated)
    unlisted = [level for level in observed if level not in level_set]
    unobserved = [level for level in validated if level not in observed_set]
    if unlisted or unobserved:
        raise ValueError(
            f"Category order for {source!r} must contain each observed level exactly once; "
            f"unlisted={unlisted!r}, unobserved={unobserved!r}."
        )
    return validated


def _validate_continuous(series: pd.Series, *, source: str) -> np.ndarray:
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise TypeError(f"Continuous variable {source!r} must have a numeric, non-boolean dtype.")
    if is_complex_dtype(series.dtype):
        raise TypeError(f"Continuous variable {source!r} must be real-valued.")
    if series.isna().any():
        raise ValueError(f"Continuous variable {source!r} contains missing values.")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Continuous variable {source!r} must contain only finite values.")
    return values


def _format_level(level: Any) -> str:
    return level if isinstance(level, str) else str(level)


class FactorVariableCompiler:
    """Compile a persisted numeric design from Stata-style factor variables.

    Supported components are ``c.name`` (continuous), ``i.name``
    (categorical), and unprefixed continuous names. ``#`` emits only an
    interaction; ``##`` emits main effects and every lower-order interaction.
    ``*`` and ``?`` selectors are expanded by :func:`limiteddepkit.varlist`
    during fitting and are persisted as exact source columns for prediction.

    The compiler never adds an intercept unless ``add_constant=True``. Category
    levels, base categories, source columns, and output feature order are fitted
    once and reused by :meth:`transform`; unknown prediction levels raise.
    """

    def __init__(
        self,
        variables: str | Iterable[str],
        *,
        category_orders: Mapping[str, Sequence[Any]] | None = None,
        base_categories: Mapping[str, Any] | None = None,
        add_constant: bool = False,
        constant_name: str = "const",
        max_columns: int = 10_000,
    ) -> None:
        self.variables = _normalize_terms(variables)
        self.category_orders = _copy_mapping(category_orders, argument="category_orders")
        self.base_categories = _copy_mapping(base_categories, argument="base_categories")
        if not isinstance(add_constant, bool):
            raise TypeError("add_constant must be a boolean.")
        if not isinstance(constant_name, str) or not constant_name:
            raise ValueError("constant_name must be a non-empty string.")
        if isinstance(max_columns, bool) or not isinstance(max_columns, int) or max_columns < 1:
            raise ValueError("max_columns must be a positive integer.")
        self.add_constant = add_constant
        self.constant_name = constant_name
        self.max_columns = max_columns

    def _check_fitted(self) -> None:
        if not hasattr(self, "_output_columns"):
            raise RuntimeError("FactorVariableCompiler is not fitted; call fit() first.")

    @property
    def input_columns_(self) -> tuple[str, ...]:
        """Exact source columns fitted from the specification."""
        self._check_fitted()
        return self._input_columns

    @property
    def category_levels_(self) -> dict[str, tuple[Any, ...]]:
        """Copy of the fitted category order for every categorical source."""
        self._check_fitted()
        return dict(self._category_levels)

    @property
    def base_categories_(self) -> dict[str, Any]:
        """Copy of the fitted omitted category for every categorical source."""
        self._check_fitted()
        return dict(self._fitted_base_categories)

    @property
    def feature_names_(self) -> tuple[str, ...]:
        """Compiled output names in exact estimator column order."""
        self._check_fitted()
        return self._feature_names

    def fit(self, data: pd.DataFrame) -> FactorVariableCompiler:
        """Fit wildcard, category, base-level, and output-schema metadata."""
        expanded_terms: list[tuple[str, tuple[_Component, ...]]] = []
        used_sources: list[str] = []
        source_kinds: dict[str, str] = {}

        for term in self.variables:
            operator, parsed_components = _parse_term(term)
            component_options: list[list[_Component]] = []
            for kind, selector in parsed_components:
                sources = varlist(data, [selector])
                options = [_Component(kind=kind, source=source) for source in sources]
                component_options.append(options)
                for source in sources:
                    previous = source_kinds.get(source)
                    if previous is not None and previous != kind:
                        raise ValueError(
                            f"Variable {source!r} is declared as both continuous and categorical."
                        )
                    source_kinds[source] = kind
                    if source not in used_sources:
                        used_sources.append(source)
            for components in product(*component_options):
                if len(expanded_terms) >= self.max_columns:
                    raise ValueError(
                        f"Factor-variable source expansion exceeds max_columns="
                        f"{self.max_columns}; narrow wildcard interactions or raise the "
                        "limit explicitly."
                    )
                categorical_sources = [
                    component.source for component in components if component.kind == "categorical"
                ]
                if len(set(categorical_sources)) != len(categorical_sources):
                    raise ValueError(
                        "A categorical variable cannot interact with itself in one term; "
                        "remove the repeated i. component."
                    )
                expanded_terms.append((operator, tuple(components)))

        categorical_sources = {
            source for source, kind in source_kinds.items() if kind == "categorical"
        }
        unused_orders = set(self.category_orders) - categorical_sources
        unused_bases = set(self.base_categories) - categorical_sources
        if unused_orders:
            raise ValueError(
                f"category_orders contains variables not used as categorical factors: "
                f"{sorted(unused_orders)!r}."
            )
        if unused_bases:
            raise ValueError(
                f"base_categories contains variables not used as categorical factors: "
                f"{sorted(unused_bases)!r}."
            )

        category_levels: dict[str, tuple[Any, ...]] = {}
        fitted_bases: dict[str, Any] = {}
        for source in used_sources:
            kind = source_kinds[source]
            if kind == "continuous":
                _validate_continuous(data[source], source=source)
                continue
            levels = _fit_category_levels(
                data[source],
                source=source,
                configured_order=self.category_orders.get(source),
            )
            base = self.base_categories.get(source, levels[0])
            if base not in levels:
                raise ValueError(
                    f"Base category {base!r} is not a fitted level of {source!r}: {levels!r}."
                )
            category_levels[source] = levels
            fitted_bases[source] = base

        def component_basis(component: _Component) -> tuple[_Atom, ...]:
            if component.kind == "continuous":
                return (
                    _Atom(
                        kind="continuous",
                        source=component.source,
                        level=None,
                        main_name=component.source,
                        interaction_name=f"c.{component.source}",
                    ),
                )
            levels = category_levels[component.source]
            if len(levels) - 1 > self.max_columns:
                raise ValueError(
                    f"Categorical variable {component.source!r} alone exceeds "
                    f"max_columns={self.max_columns}; reduce its fitted levels or raise "
                    "the limit explicitly."
                )
            return tuple(
                _Atom(
                    kind="categorical",
                    source=component.source,
                    level=level,
                    main_name=f"{_format_level(level)}.{component.source}",
                    interaction_name=f"{_format_level(level)}.{component.source}",
                )
                for level in levels
                if level != fitted_bases[component.source]
            )

        output_columns: list[_OutputColumn] = []
        semantic_keys: set[tuple[tuple[str, str, str, str], ...]] = set()
        names: dict[str, tuple[tuple[str, str, str, str], ...]] = {}

        def add_output(name: str, atoms: tuple[_Atom, ...]) -> None:
            if atoms:
                semantic_key = tuple(sorted(atom.semantic_key for atom in atoms))
            else:
                semantic_key = (("constant", self.constant_name, "", ""),)
            if semantic_key in semantic_keys:
                return
            previous = names.get(name)
            if previous is not None and previous != semantic_key:
                raise ValueError(
                    f"Compiled feature name {name!r} is ambiguous; rename a source column "
                    "or category level."
                )
            if len(output_columns) >= self.max_columns:
                raise ValueError(
                    f"Factor-variable design exceeds max_columns={self.max_columns}; "
                    "reduce factor levels/interactions or raise the limit explicitly."
                )
            output_columns.append(_OutputColumn(name=name, atoms=atoms, semantic_key=semantic_key))
            semantic_keys.add(semantic_key)
            names[name] = semantic_key

        if self.add_constant:
            add_output(self.constant_name, ())

        for operator, components in expanded_terms:
            if operator == "main":
                subsets = [(0,)]
            elif operator == "interaction":
                subsets = (tuple(range(len(components))),)
            else:
                subsets = chain.from_iterable(
                    combinations(range(len(components)), size)
                    for size in range(1, len(components) + 1)
                )
            for subset in subsets:
                bases = [component_basis(components[index]) for index in subset]
                for atoms in product(*bases):
                    atom_tuple = tuple(atoms)
                    if len(atom_tuple) == 1:
                        name = atom_tuple[0].main_name
                    else:
                        name = "#".join(atom.interaction_name for atom in atom_tuple)
                    add_output(name, atom_tuple)

        if not output_columns:
            raise ValueError("Factor-variable specification compiled to no design columns.")

        # Validate the complete fitted design before publishing any fitted state.
        self._build_matrix(
            data,
            input_columns=tuple(used_sources),
            category_levels=category_levels,
            output_columns=tuple(output_columns),
        )
        self._input_columns = tuple(used_sources)
        self._category_levels = dict(category_levels)
        self._fitted_base_categories = dict(fitted_bases)
        self._feature_names = tuple(column.name for column in output_columns)
        self._output_columns = tuple(output_columns)
        return self

    @staticmethod
    def _build_matrix(
        data: pd.DataFrame,
        *,
        input_columns: tuple[str, ...],
        category_levels: Mapping[str, tuple[Any, ...]],
        output_columns: tuple[_OutputColumn, ...],
    ) -> pd.DataFrame:
        varlist(data, list(input_columns))
        continuous: dict[str, np.ndarray] = {}
        indicators: dict[tuple[str, str, str], np.ndarray] = {}

        categorical_sources = set(category_levels)
        for source in input_columns:
            series = data[source]
            if source not in categorical_sources:
                continuous[source] = _validate_continuous(series, source=source)
                continue
            if series.isna().any():
                raise ValueError(f"Categorical variable {source!r} contains missing values.")
            levels = category_levels[source]
            unknown = [level for level in pd.unique(series) if level not in set(levels)]
            if unknown:
                raise ValueError(
                    f"Categorical variable {source!r} contains unknown levels {unknown!r}; "
                    f"fitted levels are {levels!r}."
                )
            for level in levels:
                key = (source, type(level).__qualname__, repr(level))
                indicators[key] = (series == level).to_numpy(dtype=float)

        matrix: dict[str, np.ndarray] = {}
        nobs = len(data)
        for column in output_columns:
            if not column.atoms:
                values = np.ones(nobs, dtype=float)
            else:
                values = np.ones(nobs, dtype=float)
                for atom in column.atoms:
                    if atom.kind == "continuous":
                        values = values * continuous[atom.source]
                    else:
                        key = (atom.source, type(atom.level).__qualname__, repr(atom.level))
                        values = values * indicators[key]
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Compiled feature {column.name!r} contains non-finite values; "
                        "rescale the source variables before forming this interaction."
                    )
            matrix[column.name] = values
        return pd.DataFrame(
            matrix, index=data.index, columns=[column.name for column in output_columns]
        )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted factor-variable schema to a new DataFrame."""
        self._check_fitted()
        return self._build_matrix(
            data,
            input_columns=self._input_columns,
            category_levels=self._category_levels,
            output_columns=self._output_columns,
        )

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the compiler and return the numeric training design."""
        return self.fit(data).transform(data)

    def get_feature_names_out(self) -> np.ndarray:
        """Return fitted output names in exact estimator column order."""
        self._check_fitted()
        return np.asarray(self._feature_names, dtype=object)
