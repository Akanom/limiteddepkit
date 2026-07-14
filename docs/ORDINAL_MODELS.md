# Ordinal model guide

The supported alpha API contains six estimators:

| Estimator | Primary use | Important boundary |
| --- | --- | --- |
| `OrderedLogit` | Pooled proportional-odds Logit | Common slopes at every cumulative split |
| `OrderedProbit` | Pooled ordered Probit | Common slopes at every cumulative split |
| `GeneralizedOrderedLogit` | Every slope may vary by split | Non-crossing is enforced on the estimation support, not globally |
| `PartialProportionalOdds` | Selected slopes vary by split | Pass varying DataFrame column names to `varying=` |
| `RandomEffectsOrderedLogit` | Static panel with an entity intercept | Uses non-adaptive Gaussian-Hermite quadrature |
| `DynamicRandomEffectsOrderedLogit` | Panel state dependence with initial-conditions controls | Uses only the initial contiguous spell for each entity |

## Common data contract

Pass a two-dimensional numeric NumPy array or pandas DataFrame as `X` and a
one-dimensional outcome as `y`. DataFrames are recommended because fitted
feature names are then retained. Do not include a constant column: ordinal
thresholds identify the model location. Missing or non-finite design values are
rejected.

Supply `category_order=` for substantive labels unless `y` is an ordered pandas
categorical. Without either mechanism, sortable numeric or string labels use
their sorted order. See [Category ordering](CATEGORY_ORDER.md) for the complete
label contract.

```python
from limiteddepkit import OrderedLogit

result = OrderedLogit().fit(
    X,
    y,
    category_order=["low", "medium", "high"],
)

probabilities = result.predict_proba(X_new)
categories = result.predict(X_new)
coefficient_table = result.summary_frame()
```

Formula parsing is not part of the alpha API. Prepare categorical encodings and
other design transformations before fitting, and reproduce the same columns and
column order at prediction time.

## Flexible slopes and nesting

`PartialProportionalOdds` takes the names of columns whose slopes may differ
across cumulative splits:

```python
from limiteddepkit import OrderedLogit, PartialProportionalOdds, likelihood_ratio_test

restricted = OrderedLogit().fit(X, y, category_order=category_order)
unrestricted = PartialProportionalOdds(varying=["income"]).fit(
    X,
    y,
    category_order=category_order,
)
comparison = likelihood_ratio_test(restricted, unrestricted)
```

For generalized and partial-proportional-odds fits, prediction raises an error
when new covariates make cumulative indices cross. It does not clip invalid
probabilities. If a non-crossing constraint is active at the solution, ordinary
Hessian inference is unavailable and a likelihood-ratio comparison suppresses
the usual chi-square p-value.

## Post-estimation availability

Every supported result provides labeled parameters, covariance information,
confidence intervals, a coefficient summary, category prediction, and category
probabilities. Package-level `lincom()` and `wald_test()` work with these common
inferential fields.

Pooled and flexible-slope results also provide `marginal_effects()`,
`average_marginal_effects()`, and `margins()`. Probability and marginal-effect
plots support those result families when `limiteddepkit[plots]` is installed.

Static and dynamic random-effects models use panel-specific prediction instead.
See [Panel ordinal models](PANEL_ORDINAL.md) for population-averaged,
conditional, and posterior probabilities, and [Dynamic ordinal models](DYNAMIC_ORDINAL.md)
for the additional lagged-state and initial-condition inputs.

Always inspect `converged` and `inference_valid` before interpreting estimates or
normal-approximation inference.
