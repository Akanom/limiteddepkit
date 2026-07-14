# limiteddepkit documentation

`limiteddepkit` is currently a binary-and-ordinal alpha. These notes document
the supported contracts and validation evidence; models in
`limiteddepkit.experimental` are provisional and are not covered by the same
support claim.

## User and model guides

- [Package scope](PACKAGE_SCOPE.md) defines which model families stay, which
  are experimental, and which are extracted from the distribution.
- [Binary model guide](BINARY_MODELS.md) documents stable Logit/Probit inputs,
  prediction, inference, margins, and current limitations.
- [Ordinal model guide](ORDINAL_MODELS.md) summarizes estimator selection,
  common input rules, supported post-estimation, and important boundaries.
- [Category ordering](CATEGORY_ORDER.md) explains explicit label order and
  ordered pandas categoricals.
- [Panel ordinal models](PANEL_ORDINAL.md) documents random-intercept Ordered
  Logit, quadrature, and posterior prediction.
- [Dynamic ordinal models](DYNAMIC_ORDINAL.md) documents the lagged-outcome and
  initial-conditions specification.
- [Ecosystem compatibility](ECOSYSTEM_COMPATIBILITY.md) describes conventions
  shared with `systemgmmkit` and deliberate differences for discrete outcomes.
- [Experimental model status](EXPERIMENTAL_MODELS.md) records validation
  evidence, remaining promotion gates, and package-scope decisions.
- [Probability-aware validation workflows](ML_WORKFLOWS.md) document the
  experimental outcome-specific scoring, grouped and temporal splitting,
  out-of-fold prediction, and validity-gated comparison layer.

## Validation evidence

- [Cross-software parity evidence index](../validation/PARITY_EVIDENCE.md)
  records the four completed Stata/R outcomes and the SHA-256 identities of
  their manifests, reports, and certificates.
- [Validation overview](VALIDATION.md) records the maintained reference and
  simulation-recovery boundary.
- [Stata parity harness](../validation/stata/README.md) provides controlled
  deterministic certification fixtures and a separate downloaded-real-data
  application track for the stable binary and ordinal estimators.
- [R parity harness](../validation/r/README.md) independently fits all eight
  maintained families on the same controlled and real-data fixtures.
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
