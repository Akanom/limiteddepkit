# Stata-style variable lists

`limiteddepkit.varlist` expands exact pandas column names and Stata-style `*`
and `?` wildcards into a deterministic Python list. It is an explicit bridge
between a single analysis DataFrame and the package's existing matrix-based
estimator contracts; it does not introduce hidden data or formula state.

```python
import limiteddepkit as ldk

features = ldk.varlist(
    data,
    "const income_* controls_?",
    exclude="outcome entity_id time_id",
)

result = ldk.BinaryProbit().fit(data[features], data["outcome"])
probabilities = result.predict_proba(new_data[features])
```

## Expansion rules

- A string is split on whitespace into tokens.
- An ordered sequence treats each entry as one token. This form can select an
  exact pandas column name containing spaces.
- Exact names remain in user-supplied token order.
- Each wildcard retains the DataFrame's column order.
- `*` matches zero or more characters and `?` matches exactly one character.
- Repeated exact or wildcard matches are returned only once.
- Exclusions use the same exact-name and wildcard rules.

The contract is strict. An unmatched inclusion or exclusion raises instead of
silently changing a specification. Duplicate or non-string DataFrame columns
also raise because their parameter labels would be ambiguous. If exclusions
remove every selected variable, expansion fails.

## Current boundary

`varlist` returns names; it does not mutate the DataFrame or automatically add
an intercept. Store the returned list and use it for both fitting and
prediction. Fitted estimators continue to enforce exact feature names and
order.

This first increment supports only `*` and `?`. Stata factor-variable tokens
such as `i.`, `c.`, `#`, and `##`, base-category declarations, and hyphenated
variable ranges are not interpreted yet. They require a separate design-matrix
parser with persisted category and interaction metadata.
