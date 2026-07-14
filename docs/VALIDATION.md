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

## Manual Stata parity gate

The stable binary and ordinal surface has two reproducible manual Stata tracks
under `validation/stata/`:

| Track | Purpose | Current status |
|---|---|---|
| Controlled deterministic fixtures | Strict, implementation-level release gate | Prepared; awaiting the manual Stata run |
| Downloaded Stata Press examples | Independent application check on non-simulated observations | Prepared; awaiting the manual Stata run |

The controlled track remains the certification benchmark. The application
track uses pinned, hash-verified `lbw`, `tvsfpors`, and `nlswork` files and must
not be used to broaden a benchmark-specific certification claim. The source
datasets are downloaded into the ignored working directory and are not
redistributed by the package.

Both tracks export raw Stata `e(b)` and full `e(V)` results, observation and
group counts, parameter counts, log likelihood, information criteria,
convergence state, and selected probabilities. The Python comparator applies
documented cutpoint, flexible-ordinal intercept, and random-effect scale
transformations; checks the prepared-file hashes; and writes CSV, Markdown, and
JSON evidence. Panel commands use nonadaptive Gauss-Hermite quadrature with the
same node count as the toolkit and compare fixed-part probabilities at a random
effect of zero.

No Stata parity result is claimed until the returned Stata files have been
compared successfully. See the [parity guide](../validation/stata/README.md) for
the complete commands, tolerances, data provenance, optional `gologit2` checks,
and allowed claim language.
