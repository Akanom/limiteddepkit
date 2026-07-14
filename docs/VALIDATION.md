# Validation strategy

Validation follows the same layered philosophy used by `systemgmmkit`:

1. API and statistical-identity unit tests.
2. Numerical comparison with established implementations where equivalent
   reference estimators exist.
3. Deterministic simulation recovery for models without a maintained external
   Python reference implementation.
4. Explicit guards around constrained boundaries and unsafe extrapolation.

Binary Logit and Probit are compared with Statsmodels for coefficients,
log-likelihood, observed-information covariance, standard errors,
probabilities, information criteria, and average marginal effects. Maintained
tests also cover analytical scores, finite-difference effects, extreme linear
indices, schema enforcement, and complete and quasi-complete separation.

Ordered Logit and Ordered Probit are compared with
`statsmodels.miscmodels.ordinal_model.OrderedModel`. Generalized Ordered Logit
and Partial Proportional Odds are checked with replicated bounded-support data
whose thresholds and threshold-specific slopes are known. These simulation
checks are regression evidence, not a substitute for a full Monte Carlo study.

Random-effects Ordered Logit validation adds balanced and unbalanced panel
simulation recovery, Gaussian-Hermite scaling identities, row-order and entity-
label invariance, and quadrature-order convergence. Posterior entity log-
marginal contributions are also checked against the fitted integrated
log-likelihood.

Dynamic random-effects Ordered Logit is generated from the exact fitted
conditional specification: lag-category effects, initial-outcome effects,
initial covariates, post-initial covariate means, and a residual Gaussian entity
effect. Maintained recovery uses four deterministic panels and checks the mean
of every structural, state-dependence, initial-condition, correlated-effect,
threshold, and variance-component estimate.

The dynamic numerical certificate additionally compares 12- and 20-point
quadrature and checks invariance to row ordering, arbitrary entity relabeling,
and shifted time origins. Exact maintained tolerances are documented in
`DYNAMIC_ORDINAL_VALIDATION.md`.
