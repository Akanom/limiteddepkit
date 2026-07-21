# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project intends to use [Semantic Versioning](https://semver.org/) after its
pre-release phase.

## [Unreleased]

### Added

- Stable parametric duration family at the package root and
  `limiteddepkit.duration`: explicit geometric discrete duration plus
  Exponential, Weibull, and Gamma likelihoods with right censoring, delayed
  entry, integer-frequency weights, observed/robust/cluster covariance,
  schema-safe survival/hazard/cumulative-hazard/quantile prediction, and
  temporary experimental compatibility aliases.
- Stable BUC `FixedEffectsOrderedLogit` with exact conditional-likelihood
  construction, entity-cluster composite covariance, Statsmodels parity, and
  an explicit slopes-only identification contract. Added experimental
  split-panel-jackknife `FixedEffectsOrderedProbit` with balanced-panel guards,
  entity-bootstrap inference, known-entity diagnostic probabilities, and
  unconditional likelihood parity.
- Experimental `DynamicFixedEffectsOrderedLogit` implementing the restricted
  four-outcome-history MRV conditional composite likelihood with exact discrete
  stayers, a known state cutoff, threshold normalization, entity-clustered
  Godambe inference, path-odds/recovery tests, and constructed-sample Statsmodels
  parity. It remains outside the stable root API and does not report category
  probabilities.
- Stable `FirthBinaryLogit` at the package root and in `limiteddepkit.small_sample`,
  with constrained profile penalized-likelihood confidence intervals, direct profile
  diagnostics, exact separated-table checks, and independent `firthmodels` coefficient,
  objective, and profile-bound parity. Experimental imports remain compatibility aliases;
  ridge Binary/Ordered Logit remain provisional.
- Stable `limiteddepkit.count` family and package-root exports for exposure/offset
  Poisson and NB2. The promoted contract adds exact integer-frequency and analytic
  estimating-equation weights, observed-information/HC0/HC1/cluster covariance,
  schema-safe indexed prediction, common fit diagnostics and information criteria,
  Statsmodels parity, and temporary `limiteddepkit.experimental` compatibility aliases.
  Zero-inflated and hurdle Poisson remain experimental.
- Stable `limiteddepkit.censoring` family and package-root exports for Gaussian Tobit,
  truncated regression, and interval regression. The promoted contract adds left/right
  Tobit and truncation rules, observed-information/robust/cluster covariance, common
  latent-distribution post-estimation, reflection and cross-likelihood identities, and
  temporary `limiteddepkit.experimental` compatibility aliases.
- Stable `RandomEffectsOrderedProbit` with the existing non-adaptive GHQ,
  observed-information, population/conditional prediction, posterior random-effect,
  posterior-prediction, simulation, and Output Hub contracts. Its conditional kernel is
  checked against Statsmodels and its marginal probabilities against the exact normal-
  convolution identity.
- Stable `limiteddepkit.ml` workflow submodule with dependency-light iid,
  stratified, complete-group, and forward-panel splitters; binary, multinomial,
  ordinal, grouped-choice, count, continuous, quantile, duration, and selection
  scores; out-of-fold predictions; and validity-gated model comparison.
- Limited-data validation extensions: stratified complete-group splitting with an exact
  MILP coverage fallback after greedy balancing, repeated splitters, fold-local
  preprocessing, pooled/weighted OOF aggregation, paired fold differences,
  observation/entity bootstrap intervals, and one-standard-error selection.
- Leakage-safe nested cross-validation for estimator and ridge-penalty selection.
- Held-out binary/ordinal calibration diagnostics with reliability tables, calibration
  intercept/slope, and an explicit grouped Brier decomposition remainder.
- Training-fold reverse-Kaplan-Meier censoring estimates with IPCW concordance,
  time-dependent and integrated Brier scores, and cumulative/dynamic AUC.
- Lazy optional bridges for scikit-learn, Statsmodels, and callback-driven external
  estimators with explicit prediction semantics and no invented validity diagnostics.
- Experimental ridge Binary/Ordered Logit estimators, with shrinkage and aligned
  Statsmodels/scikit-learn numerical evidence.
- Optional `[neural]` `ResidualBinaryMLP` advanced prediction challenger with an iid-only
  internal validation split, conservative completion/stabilization diagnostics,
  temperature scaling, and Monte Carlo dropout uncertainty output. All 21 neural tests
  passed in an isolated Python 3.13/PyTorch 2.13.0 runtime, and an optional neural CI job
  now maintains the path; no numerical-parity or inferential claim is made.
- Optional scikit-learn validation gates for matching prediction metrics,
  deterministic splitter contracts, and end-to-end held-out Binary Logit
  probabilities against Statsmodels and unpenalized scikit-learn.
- Pinned R 4.5.1 parity harness covering the eight previously certified binary and ordinal
  families on both controlled and public-data fixtures, with canonical
  covariance, fit, probability, metadata, report, and certificate exports.
- Committed cross-software evidence index recording the four completed parity
  outcomes and exact manifest, report, and certificate digests.
- Separate promoted-family public-data application harness covering 12 stable
  post-expansion fits. The 15 July 2026 Python/R run passed all 120/120
  registered checks using seven industrial-package fits, three independent
  likelihood or adjusted-score implementations, and two likelihood or
  pseudo-sample identities. The applications use empirical observations except
  for the official fictional `womenwage2` interval-regression software fixture;
  the manual Stata comparison passed 140/140 required checks with an explicit
  Gamma skip. This is application evidence,
  not an extension of the older controlled certification or a universal
  equality claim.

### Changed

- Documented explicit new-entity, known-entity future, and conditional panel
  prediction targets, including chronology and dynamic-lag leakage guards.
- Balanced stratified class remainders across folds so both total fold sizes and
  within-class counts differ by at most one.
- Replaced the common Binary Logit and Poisson optimization paths with analytical,
  damped Newton/IRLS steps plus BFGS fallbacks; finite-MLE separation safeguards and
  reference-package parameter parity remain enforced.
- Clarified that stable Firth and experimental ridge estimators require `n > p` and full column
  rank, and that native estimators require dense transformer output; sparse transformed
  designs remain available only to downstream estimators that explicitly accept them.
- Completed both maintained Stata 17 parity tracks with `gologit2` 3.2.8; all
  eight families pass all declared checks on the controlled and real-data
  suites.
- Added Stata random-effect variance-to-standard-deviation canonicalization and
  tightened pooled ordered-model reference optimization.

## [0.1.0a2] - Release pending

### Security

- Bounded every direct and optional dependency and added hash-verified,
  reproducible requirement sets without using OneDrive-incompatible `.lock`
  files.
- Added dependency auditing, distribution-content inspection, SBOM generation,
  dependency-review gates, immutable GitHub Actions, trusted PyPI publishing,
  and build-provenance attestations.
- Kept Matplotlib optional and made missing plotting support fail with an
  actionable `limiteddepkit[plots]` installation message.

## [0.1.0a1] - 2026-07-18

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
