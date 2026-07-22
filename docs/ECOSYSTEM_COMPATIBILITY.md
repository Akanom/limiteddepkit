# Ecosystem compatibility contract

`limiteddepkit` is methodologically separate from `systemgmmkit`, but its result
and post-estimation conventions intentionally follow the same ecosystem design.

All supported binary and ordinal result objects expose:

- labeled `params` or `all_params`;
- `standard_errors`, `zstats`, and `pvalues`;
- `covariance`, `vcov()`, `conf_int()`, and `summary_frame()`;
- `predict()` and model-appropriate `predict_proba()`;
- linear combinations and Wald tests through methods and package-level helpers.

Binary, pooled ordinal, and flexible-slope ordinal results expose
model-appropriate marginal effects and margins. The plotting helpers currently
accept pooled and flexible ordinal families. The Universal Output Hub adapter
currently covers the supported ordinal results. Random-effects
and dynamic results instead provide population-averaged, conditional, or
posterior prediction methods appropriate to their panel structure; they do not
currently expose the pooled-model margins or plotting interfaces.

Package-level post-estimation functions mirror the `systemgmmkit` calling style:

```python
from limiteddepkit import confint, margins, predict, vcov

prediction = predict(result, X_new)
covariance = vcov(result)
intervals = confint(result)
probability_margins = margins(result, X)
```

The package-root `varlist(data, variables, exclude=...)` helper provides a
deterministic string-based input-selection bridge without coupling estimator
calculation state to `systemgmmkit` or Universal Output Hub. Its wildcard and
ordering rules are documented in [Stata-style variable lists](DATA_CONTRACTS.md).
Future factor-variable support should compile to the same explicit DataFrame
design contract. Universal Output Hub remains a consumer of labeled fitted
parameters and does not own variable expansion or transformation state.

Linear-model concepts are not copied where they are not statistically natural.
For example, ordinal results expose category probabilities rather than pretending
that categorical outcomes have ordinary least-squares residuals.
