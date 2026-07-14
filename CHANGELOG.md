# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project intends to use [Semantic Versioning](https://semver.org/) after its
pre-release phase.

## [Unreleased]

### Added

- Pinned R 4.5.1 parity harness covering all eight stable binary and ordinal
  families on both controlled and public-data fixtures, with canonical
  covariance, fit, probability, metadata, report, and certificate exports.
- Committed cross-software evidence index recording the four completed parity
  outcomes and exact manifest, report, and certificate digests.

### Changed

- Completed both maintained Stata 17 parity tracks with `gologit2` 3.2.8; all
  eight families pass all declared checks on the controlled and real-data
  suites.
- Added Stata random-effect variance-to-standard-deviation canonicalization and
  tightened pooled ordered-model reference optimization.

## [0.1.0a1] - Release pending

### Added

- Ordinal-model alpha covering pooled, flexible-slope, random-effects, and
  dynamic random-effects ordinal estimators.
- Stable binary Logit and Probit with observed-information inference,
  prediction, margins, marginal effects, and Statsmodels parity.
- Shared prediction, inference, margins, plotting, simulation, and optional
  Output Hub workflows for supported ordinal results.
- Maintained numerical-reference, recovery, quadrature-convergence, and
  invariance validation tests.
- Deterministic manual Stata parity harness for the stable binary and ordinal
  estimators, including raw covariance and prediction exports.
- Separate hash-pinned Stata Press real-data application harness, with
  provenance-preserving downloads, full fit/covariance exports, and
  machine-readable comparison evidence.
- Independent, hash-pinned R parity workflows for the same eight models and
  datasets.
- Experimental namespace for provisional non-ordinal model families.
- Experimental fixed-boundary censored quantile regression with multi-start
  Powell check-loss fitting and opt-in pairs-bootstrap inference.
- Initial project governance, security, citation, and continuous-integration
  files.

### Changed

- Declared Python 3.10 as the minimum supported version.
- Limited the supported `0.1` alpha contract to the validated binary and
  ordinal stacks.
- Corrected likelihood, inference, identification, and prediction
  contracts across binary, choice, count, censoring, selection, and duration
  families, with independent or exact-manual validation tests.

### Removed

- Other provisional estimators from the stable top-level API.
- Linear 2SLS `TreatmentEffect` from the installed distribution because it is
  outside the limited-dependent-variable scope.
- Iid `GaussianMixtureRegression` and the misleading historical
  `SwitchingRegression` alias from the installed distribution; their source is
  staged outside `src/` for migration to a mixture/regime package.

This alpha is frozen locally; external repository and PyPI publication remain
separate release operations.
