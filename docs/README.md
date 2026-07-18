# limiteddepkit documentation

`limiteddepkit` currently has stable binary, small-sample Firth, ordinal,
Gaussian censoring, foundational count, and parametric duration families.
These notes document the supported contracts and validation evidence. Estimators
available only from `limiteddepkit.experimental` are provisional and are not covered by
the stable support claim; temporary compatibility aliases for promoted estimators do not
change those estimators' stable status.

## User and model guides

- [Package scope](PACKAGE_SCOPE.md) defines which model families stay, which
  are experimental, and which are extracted from the distribution.
- [Binary model guide](BINARY_MODELS.md) documents stable Logit/Probit inputs,
  prediction, inference, margins, and current limitations.
- [Small-sample binary guide](SMALL_SAMPLE_MODELS.md) documents stable Firth Logit,
  profile penalized-likelihood intervals, separation handling, and current boundaries.
- [Gaussian censoring model guide](CENSORING_MODELS.md) documents Tobit,
  truncated regression, interval regression, covariance choices, prediction targets,
  and current boundaries.
- [Count model guide](COUNT_MODELS.md) documents Poisson/NB2 offsets, exposure,
  weights, robust covariance, indexed prediction, and two-part-model boundaries.
- [Duration model guide](DURATION_MODELS.md) documents geometric, Exponential,
  Weibull, and Gamma duration likelihoods, delayed entry, covariance, and survival
  predictions.
- [Ordinal model guide](ORDINAL_MODELS.md) summarizes estimator selection,
  common input rules, supported post-estimation, and important boundaries.
- [Category ordering](CATEGORY_ORDER.md) explains explicit label order and
  ordered pandas categoricals.
- [Panel ordinal models](PANEL_ORDINAL.md) documents random-intercept Ordered
  Logit and Probit, quadrature, and posterior prediction.
- [Fixed-effects ordinal panels](FIXED_EFFECTS_ORDINAL.md) documents stable BUC
  Ordered Logit and the experimental bias-corrected Ordered Probit boundary.
- [Dynamic ordinal models](DYNAMIC_ORDINAL.md) documents the lagged-outcome and
  initial-conditions specification for stable dynamic random effects.
- [Dynamic fixed-effects Ordered Logit](DYNAMIC_FIXED_EFFECTS_ORDINAL.md) documents the
  experimental four-outcome-history MRV conditional estimator and its exact-stayer
  identification envelope.
- [Ecosystem compatibility](ECOSYSTEM_COMPATIBILITY.md) describes conventions
  shared with `systemgmmkit` and deliberate differences for discrete outcomes.
- [Experimental model status](EXPERIMENTAL_MODELS.md) records validation
  evidence, remaining promotion gates, and package-scope decisions.
- [Probability-aware validation workflows](ML_WORKFLOWS.md) document the
  experimental outcome-specific scoring, grouped and temporal splitting,
  out-of-fold prediction, and validity-gated comparison layer.
- [Open development](OPEN_DEVELOPMENT.md) explains how public issues,
  discussions, parity reports, and adoption notes should record future model
  development.
- [Public discussion drafts](PUBLIC_DISCUSSION_DRAFTS.md) provides starter
  GitHub Discussion posts for roadmap, validation, and adoption threads.

## Validation evidence

- [Cross-software parity evidence index](../validation/PARITY_EVIDENCE.md)
  records the four completed pre-expansion Stata/R outcomes and the separate
  promoted-family application result, with SHA-256 evidence identities.
- [Validation overview](VALIDATION.md) records the maintained reference and
  simulation-recovery boundary.
- [Stata parity harness](../validation/stata/README.md) provides controlled
  deterministic certification fixtures and a separate downloaded-real-data
  application track for the eight pre-expansion binary/ordinal estimators.
- [R parity harness](../validation/r/README.md) independently fits the same eight
  pre-expansion families on the controlled and real-data fixtures.
- [Promoted-family application harness](../validation/promoted/README.md) covers
  12 stable post-expansion fits. Python/R passed 120/120 registered checks on
  15 July 2026; the separate manual Stata comparison passed 140/140 required
  checks with one explicit Gamma skip.
- [Dynamic ordinal numerical validation](DYNAMIC_ORDINAL_VALIDATION.md) records
  quadrature-convergence and invariance certification.

Validation evidence is model-specific. Passing the package test suite does not
turn an estimator in `limiteddepkit.experimental` into a supported estimator.

## Project information

- [README](../README.md)
- [Changelog](../CHANGELOG.md)
- [Contributing guide](../CONTRIBUTING.md)
- [Release checklist](../RELEASING.md)
- [Security policy](../SECURITY.md)
- [Citation metadata](../CITATION.cff)
- [License](../LICENSE)
