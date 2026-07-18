# Small-sample binary models

`FirthBinaryLogit` is the stable bias-reduced binary-response estimator. It maximizes

```text
log L(beta) + 0.5 log |I(beta)|,
```

where `I(beta)` is the ordinary expected Fisher information. The Jeffreys penalty reduces
first-order coefficient bias and yields finite estimates for full-rank designs under
complete or quasi-complete separation. It is an explicit estimator choice; ordinary
`BinaryLogit` never switches to Firth fitting silently.

## Fitting and prediction

```python
import numpy as np

from limiteddepkit import FirthBinaryLogit

X = data[["const", "age", "exposure"]]  # include a constant explicitly
y = data["event"]                         # exactly 0/1

result = FirthBinaryLogit().fit(X, y)

print(result.params)
print(result.conf_int())                   # profile penalized-likelihood, default
print(result.conf_int(method="wald"))     # fast symmetric approximation

probability = result.predict_proba(X_new)[1]
classification = result.predict(X_new, threshold=0.5)
```

The design must be dense, finite, full column rank, and have more rows than columns.
Both outcome classes must be present. DataFrame feature names and order are retained and
must match at prediction time. A constant is not added automatically.

`predict_proba` returns labeled probabilities for outcomes 0 and 1. These are direct
plug-in Firth probabilities. Intercept corrections such as FLIC/FLAC are not implemented,
so bias reduction of coefficients should not be described as calibrated rare-event
prediction by itself.

## Inference

`conf_int()` defaults to asymmetric profile penalized-likelihood intervals. For every
coefficient and proposed bound, the named coefficient is fixed, all nuisance coefficients
are re-optimized, and the bound solves

```text
2 * (penalized_loglike_full - penalized_loglike_constrained)
    = chi_square_1(level).
```

Use `profile_penalized_loglike(parameter, values)` to inspect the constrained profile
directly. Profiling is slower than Wald inference and raises an explicit error if a
nuisance optimization or confidence-bound bracket cannot be established.

`covariance`, `standard_errors`, `zstats`, `pvalues`, and `summary_frame()` remain the
ordinary-Fisher Wald approximation evaluated at the bias-reduced estimate. They are
labelled separately from the default profile intervals. Penalized-likelihood-ratio
p-values are not yet exposed. The penalized objective is not an ordinary likelihood for
AIC/BIC comparison with unpenalized models.

## Validation boundary

Maintained tests require all of the following:

- the exact half-cell-correction coefficients for a completely separated 2x2 design;
- coefficient, maximized penalized-objective, and profile-interval parity with the NumPy
  backend of `firthmodels` 0.7.2 on a three-parameter fixture (coefficients within `2e-8`,
  objective within `2e-11`, bounds within `2e-6`);
- direct verification that every returned profile bound satisfies the requested
  one-degree-of-freedom penalized-deviance equation; and
- finite predictions, positive covariance diagonal, schema preservation, and failure
  checks for rank deficiency, invalid outcomes, controls, parameter names, and values.

Fit and constrained-profile certification use independently capped adjusted-score
thresholds. Constrained profiles use monotone, step-halved nuisance Fisher scoring,
so a tiny relative objective change on a mixed-scale design cannot certify an
unstationary profile. A deliberately loose optimizer tolerance likewise cannot
certify the zero starting iterate.

The separate promoted-family public-data suite independently solves the Firth adjusted
score on the empirical LBW application and compares coefficients, inverse ordinary-
Fisher covariance, penalized objective components, and probabilities. Its registered
Python/R checks passed within the suite's **120/120** result on 15 July 2026. This is an
independent score/likelihood implementation, not an industrial-package claim. The
manual Stata promoted run also passed Firth through optional `firthlogit`. See the
[promoted-family guide](../validation/promoted/README.md).

The current contract does not include offsets, observation weights, cluster/survey
covariance, formulas, missing-data handling, `p >= n`, rank-deficient ridge-like fitting,
multinomial/ordinal Firth estimators, FLIC/FLAC prediction correction, or penalized-
likelihood-ratio p-values.

## Why ridge remains provisional

`RidgeBinaryLogit` and `RidgeOrderedLogit` remain under `limiteddepkit.experimental`.
Their coefficient paths have useful scikit-learn/Statsmodels checks, but penalty selection
must be nested inside the validation design and their current sandwich covariances are
approximate penalized-estimating-equation quantities. Those choices do not yet define a
single stable inferential contract. Ridge is therefore not exported by
`limiteddepkit.small_sample` or the package root.
