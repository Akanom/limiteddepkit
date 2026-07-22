"""Stata-style wildcard expansion for pandas column names."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def _normalize_patterns(
    value: str | Iterable[str] | None,
    *,
    argument: str,
    allow_empty: bool,
) -> list[str]:
    if value is None:
        patterns: list[str] = []
    elif isinstance(value, str):
        patterns = value.split()
    else:
        if isinstance(value, (bytes, bytearray, Mapping, set, frozenset)):
            raise TypeError(f"{argument} must be a string or an ordered iterable of strings.")
        try:
            patterns = list(value)
        except TypeError as error:
            raise TypeError(
                f"{argument} must be a string or an ordered iterable of strings."
            ) from error
        if not all(isinstance(pattern, str) for pattern in patterns):
            raise TypeError(f"Every {argument} entry must be a string.")

    if not patterns and not allow_empty:
        raise ValueError(f"{argument} must contain at least one variable or pattern.")
    if any(pattern == "" for pattern in patterns):
        raise ValueError(f"{argument} entries must not be empty strings.")
    return patterns


def _wildcard_matches(pattern: str, value: str) -> bool:
    """Match only ``*`` and ``?`` without evaluating a regular expression."""
    pattern_index = 0
    value_index = 0
    last_star = -1
    star_value_index = 0
    while value_index < len(value):
        if pattern_index < len(pattern) and (
            pattern[pattern_index] == "?"
            or (pattern[pattern_index] != "*" and pattern[pattern_index] == value[value_index])
        ):
            pattern_index += 1
            value_index += 1
        elif pattern_index < len(pattern) and pattern[pattern_index] == "*":
            last_star = pattern_index
            pattern_index += 1
            star_value_index = value_index
        elif last_star >= 0:
            pattern_index = last_star + 1
            star_value_index += 1
            value_index = star_value_index
        else:
            return False
    while pattern_index < len(pattern) and pattern[pattern_index] == "*":
        pattern_index += 1
    return pattern_index == len(pattern)


def _expand_patterns(columns: list[str], patterns: list[str], *, argument: str) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            matches = [column for column in columns if _wildcard_matches(pattern, column)]
        else:
            matches = [pattern] if pattern in columns else []
        if not matches:
            raise ValueError(f"{argument} pattern {pattern!r} matched no columns.")
        for column in matches:
            if column not in seen:
                expanded.append(column)
                seen.add(column)
    return expanded


def varlist(
    data: pd.DataFrame,
    variables: str | Iterable[str],
    *,
    exclude: str | Iterable[str] | None = None,
) -> list[str]:
    """Expand exact names and Stata-style ``*``/``?`` column wildcards.

    A string is split on whitespace into variable tokens. An ordered iterable
    treats each entry as one token, which also permits exact pandas column names
    containing spaces. Tokens are expanded in user order; wildcard matches retain
    the DataFrame's column order. Repeated matches are returned only once.

    Parameters
    ----------
    data:
        DataFrame whose columns define the available variables and their order.
    variables:
        Exact column names or wildcard patterns. ``*`` matches zero or more
        characters and ``?`` matches exactly one character.
    exclude:
        Optional exact names or wildcard patterns to remove. Exclusions are
        strict: every exclusion token must match at least one column.

    Returns
    -------
    list[str]
        Expanded column names suitable for ``data[columns]``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    columns = list(data.columns)
    non_string = [column for column in columns if not isinstance(column, str)]
    if non_string:
        raise TypeError(
            "varlist requires string DataFrame column names; "
            f"found non-string columns {non_string!r}."
        )
    duplicated = list(pd.Index(columns)[pd.Index(columns).duplicated()].unique())
    if duplicated:
        raise ValueError(
            f"varlist requires unique DataFrame column names; found duplicates {duplicated!r}."
        )

    include_patterns = _normalize_patterns(variables, argument="variables", allow_empty=False)
    exclude_patterns = _normalize_patterns(exclude, argument="exclude", allow_empty=True)
    selected = _expand_patterns(columns, include_patterns, argument="variables")
    excluded = (
        set(_expand_patterns(columns, exclude_patterns, argument="exclude"))
        if exclude_patterns
        else set()
    )
    selected = [column for column in selected if column not in excluded]
    if not selected:
        raise ValueError("No columns remain after applying exclusions.")
    return selected
