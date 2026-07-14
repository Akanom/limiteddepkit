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

## External-software parity gates

The stable binary and ordinal surface has aligned Stata and R tracks:

| Track | Purpose | Current status |
|---|---|---|
| Controlled deterministic fixtures — Stata | Strict, implementation-level release gate | **PASS — 82/82; all eight families** |
| Downloaded Stata Press examples — Stata | Independent application check on non-simulated observations | **PASS — 82/82; all eight families** |
| Controlled deterministic fixtures — R | Independent implementation check on the certification data | **PASS — 110/110; all eight families** |
| Downloaded Stata Press examples — R | Independent application check on the same frozen observations | **PASS — 110/110; all eight families** |

The controlled track remains the certification benchmark. The application
track uses pinned, hash-verified `lbw`, `tvsfpors`, and `nlswork` files and must
not be used to broaden a benchmark-specific certification claim. The source
datasets are downloaded into the ignored working directory and are not
redistributed by the package.

The Stata tracks export raw `e(b)` and full `e(V)` results, observation and
group counts, parameter counts, log likelihood, information criteria,
convergence state, and selected probabilities. The Python comparator applies
documented cutpoint, flexible-ordinal intercept, and random-effect scale
transformations; checks the prepared-file hashes; and writes CSV, Markdown, and
JSON evidence. Panel commands use nonadaptive Gauss-Hermite quadrature with the
same node count as the toolkit and compare fixed-part probabilities at a random
effect of zero.

The R tracks independently use `glm.fit`, `MASS::polr`, `VGAM::vglm`, and
`ordinal::clmm`. They export the same canonical evidence contract. Probit and
flexible ordinal covariance estimands are explicitly aligned to the toolkit's
observed information, and panel log-SD covariance is transformed with a full
Jacobian.

The completed comparisons used Stata 17 with `gologit2` 3.2.8 and a pinned R
4.5.1 environment on 14 July 2026. See the
[committed evidence index](../validation/PARITY_EVIDENCE.md) for the four
outcomes and exact manifest, report, and certificate digests; see the
[Stata parity guide](../validation/stata/README.md) and
[R parity guide](../validation/r/README.md) for commands, tolerances, provenance,
numerical envelopes, evidence files, and allowed claim language.
