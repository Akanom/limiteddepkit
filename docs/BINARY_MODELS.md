# Binary Logit and Probit

`BinaryLogit` and `BinaryProbit` are stable root exports. Both estimate an
unpenalized maximum-likelihood model for outcomes coded exactly as `0` and `1`.

```python
from limiteddepkit import BinaryLogit, BinaryProbit

logit_result = BinaryLogit().fit(X, y)
probit_result = BinaryProbit().fit(X, y)

probabilities = logit_result.predict_proba(X_new)
classes = logit_result.predict(X_new, threshold=0.5)
```

Unlike pooled ordinal models, a binary design ordinarily includes an intercept
column when the specification requires one. `limiteddepkit` does not add an
intercept implicitly.

## Input and identification contract

- `X` must be finite and full rank, with more rows than columns. A
  one-dimensional array is accepted for a single feature; otherwise pass a
  two-dimensional array or DataFrame.
- DataFrame column names must be unique and prediction columns must match the
  fitted names and order.
- `y` must be one-dimensional, contain only `0` and `1`, and contain both
  classes.
- Complete or quasi-complete separation is rejected because the unpenalized
  finite MLE does not exist.
- `maxiter` and `tolerance` are explicit optimizer controls; optimization
  failures raise rather than returning a result with fabricated inference.
  Score certification is independently capped, so a deliberately loose
  tolerance cannot certify the zero starting vector.
- Binary Probit uses a damped Newton step with its analytical
  observed-information matrix and retains BFGS as a fallback. Final
  convergence is always certified from the analytical score, independent of
  the optimizer's status message.

## Result contract

Both result classes expose:

- labeled coefficients, observed-information covariance, standard errors,
  z-statistics, p-values, confidence intervals, AIC, and BIC;
- raw and per-observation scaled score diagnostics;
- `summary_frame()`, `vcov()`, `lincom()`, and `wald_test()`;
- schema-safe `predict_proba()` and threshold-controlled `predict()`;
- observation-level and average marginal effects for non-constant regressors;
- delta-method inference for average marginal effects; and
- `margins()` at the observed sample, covariate means, or a supplied
  representative-value mapping.

The same operations are available through package-level helpers where
applicable:

```python
from limiteddepkit import confint, margins, predict_proba, vcov

covariance = vcov(logit_result)
intervals = confint(logit_result)
average_probabilities = margins(logit_result, X, kind="probability")
new_probabilities = predict_proba(logit_result, X_new)
```

## Validation and current boundary

Maintained validation compares coefficients, log likelihood, covariance,
standard errors, probabilities, AIC/BIC, and average marginal effects with
Statsmodels Logit and Probit. Additional tests cover exact information-matrix
identities, finite-difference marginal effects, extreme indices, schema errors,
and complete and quasi-complete separation.

The stable contract currently provides model-based observed-information
covariance only. Outcome formulas, frequency/analytic weights, offsets, robust
or clustered covariance, penalized separation remedies, and bootstrap
inference are not yet part of these estimators. Explicit `i.`/`c.` preprocessing
is available through `FactorVariableCompiler` before fitting.
