# Factor-variable design compiler

`limiteddepkit.FactorVariableCompiler` turns Stata-style factor-variable terms
into a finite numeric pandas design matrix. It fits wildcard expansion,
category order, base levels, interaction structure, and output names once, then
reuses that exact schema for prediction.

```python
import limiteddepkit as ldk

compiler = ldk.FactorVariableCompiler(
    "c.age c.income_* i.education i.education##c.age c.age##c.age",
    category_orders={
        "education": ["secondary", "college", "graduate"],
    },
    base_categories={"education": "secondary"},
    add_constant=True,
)

X = compiler.fit_transform(data)
result = ldk.BinaryProbit().fit(X, data["outcome"])

X_new = compiler.transform(new_data)
probabilities = result.predict_proba(X_new)
```

The compiler is explicit preprocessing state. It does not retain the training
DataFrame, register an active model, alter estimator signatures, or place
calculation state in Universal Output Hub.

## Supported syntax

| Syntax | Meaning |
| --- | --- |
| `age` or `c.age` | Continuous main effect |
| `i.education` | Treatment-coded categorical main effects |
| `c.income_*` | Continuous wildcard expansion through `varlist` |
| `i.region#c.age` | Interaction only |
| `i.region##c.age` | Both main effects and their interaction |
| `c.age##c.age` | Continuous main effect and square |
| `a##b##c` | Main effects plus every two- and three-way interaction |

`#` and `##` may connect more than two components. A term must use one
operator consistently; mixed forms should be written as separate explicit
terms. Repeated or reversed equivalent terms are emitted once.

String specifications are whitespace-delimited. An ordered sequence treats
each entry as a complete term, allowing exact source names containing spaces:

```python
compiler = ldk.FactorVariableCompiler(["c.household income", "i.region"])
```

Inline Stata base declarations such as `ib2.education` are intentionally not
accepted. Use `base_categories={"education": 2}` so the persisted choice is
plain Python data and can be serialized with the rest of the analysis
configuration.

## Category contract

Category order determines both the omitted base and output column order:

1. `category_orders` wins when supplied.
2. A pandas categorical uses its declared category order.
3. Other values use deterministic sorted order.

The default base is the first fitted category. `base_categories` overrides it.
Every configured or pandas-declared level must occur in the fitting data; this
prevents unidentified all-zero columns. Missing values, duplicate levels,
unorderable mixed labels, and factors with fewer than two levels raise.

Prediction data may omit fitted levels, but it may not introduce an unknown
level. The compiler always emits the complete fitted output schema, filling
the indicators for absent fitted levels with zero.

## Identification and safety

- No constant is added by default. Set `add_constant=True` only for estimators
  whose identification permits an intercept; ordinal designs generally must
  remain constant-free.
- Continuous sources must be numeric, real, finite, and non-boolean.
- Source columns must follow the `varlist` unique-string-name contract.
- A source cannot be declared both `c.` and `i.` in one compiler.
- Categorical self-interactions are rejected; continuous self-interactions are
  supported for polynomial terms.
- Compiled feature-name collisions raise instead of overwriting columns.
- `max_columns=10000` bounds accidental high-cardinality interaction growth;
  raise it explicitly only after checking identification and memory cost.

Store the fitted compiler alongside the fitted estimator. The public metadata
`input_columns_`, `category_levels_`, `base_categories_`, and
`feature_names_` records the exact transformation contract.

## Current boundary

This is a design compiler, not an outcome formula language. It does not parse
`y ~ x`, add lag/lead/time-series operators, implement hyphenated Stata ranges,
silently drop missing rows, choose a statistically valid intercept, or modify
an estimator's covariance and identification options.
