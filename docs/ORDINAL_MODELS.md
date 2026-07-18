# Ordinal model guide

The stable alpha API contains eight estimators, with additional research-grade
fixed-effects estimators kept experimental:

| Estimator | Primary use | Important boundary |
| --- | --- | --- |
| `OrderedLogit` | Pooled proportional-odds Logit | Common slopes at every cumulative split |
| `OrderedProbit` | Pooled ordered Probit | Common slopes at every cumulative split |
| `GeneralizedOrderedLogit` | Every slope may vary by split | Non-crossing is enforced on the estimation support, not globally |
| `PartialProportionalOdds` | Selected slopes vary by split | Pass varying DataFrame column names to `varying=` |
| `RandomEffectsOrderedLogit` | Static panel with an entity intercept | Uses non-adaptive Gaussian-Hermite quadrature |
| `RandomEffectsOrderedProbit` | Static panel with a Gaussian entity intercept | Shares the GHQ and posterior-prediction contract with the Logit model |
| `FixedEffectsOrderedLogit` | Static panel common slopes with unrestricted entity heterogeneity | BUC conditions out cutoffs/effects, so probabilities are not identified |
| `DynamicRandomEffectsOrderedLogit` | Panel state dependence with initial-conditions controls | Uses only the initial contiguous spell for each entity |
| `FixedEffectsOrderedProbit` *(experimental)* | Bias-corrected entity-FE Probit in a long balanced panel | Split-panel jackknife and bootstrap inference |
| `DynamicFixedEffectsOrderedLogit` *(experimental)* | Fixed-`T` state dependence with unrestricted entity heterogeneity | Narrow four-outcome-history conditional design |

## Common data contract

Pass a numeric NumPy array or pandas DataFrame as `X` and a
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

Pooled, flexible-slope, and random-effects results provide labeled common
parameters, covariance information, a coefficient summary, and probability
prediction under their documented target. Package-level `lincom()` and
`wald_test()` work when the result exposes a certified covariance.

Pooled and flexible-slope results also provide `marginal_effects()`,
`average_marginal_effects()`, and `margins()`. Probability and marginal-effect
plots support those result families when `limiteddepkit[plots]` is installed.

Static and dynamic random-effects models use panel-specific prediction instead.
See [Panel ordinal models](PANEL_ORDINAL.md) for population-averaged,
conditional, and posterior probabilities, and [Dynamic ordinal models](DYNAMIC_ORDINAL.md)
for the additional lagged-state and initial-condition inputs.

BUC fixed-effects Ordered Logit is the important exception: conditioning
removes cutoffs and entity effects, so only common slopes and their
entity-cluster covariance remain. Experimental fixed-effects Probit exposes
known-entity probabilities only as a mixed corrected/uncorrected diagnostic.
See [Fixed-effects ordinal panels](FIXED_EFFECTS_ORDINAL.md).

Dynamic fixed-effects Ordered Logit is a separate experimental fixed-`T`
conditional estimand. Its exact four-observation history, known-cutoff, and
discrete-stayer requirements are documented in the
[dynamic fixed-effects guide](DYNAMIC_FIXED_EFFECTS_ORDINAL.md).

Always inspect `converged` and `inference_valid` before interpreting estimates or
normal-approximation inference. Pooled and random-effects results expose raw
and scaled score diagnostics; constrained flexible-slope results expose
`scaled_kkt_residual`. Their optimization and certification thresholds are
capped independently of a user-supplied loose tolerance.
