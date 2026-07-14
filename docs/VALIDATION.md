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

## Python reference-package gates

The experimental `limiteddepkit.ml` layer is checked against scikit-learn for
every directly equivalent score: binary and multiclass log loss and Brier
score, accuracy, balanced accuracy, ROC AUC, ordinal encoded MAE, fixed-choice-
set scores, continuous/count MAE and RMSE, mean Poisson deviance, and quantile
pinball loss. A deterministic audit on 14 July 2026 used scikit-learn 1.9.0,
NumPy 2.4.6, seed `20260714`, 500 generated datasets per metric family, and
2,500 quantile comparisons. The maintained audit is implemented in
[the scikit-learn validation test](../tests/validation/test_ml_sklearn.py). The largest absolute difference was
`5.551115123125783e-16`, which is floating-point roundoff.

The maintained end-to-end gate materializes four identical folds and compares
`BinaryLogit` cross-validation with Statsmodels 0.14.6 and unpenalized
scikit-learn logistic regression. On the frozen seed-20260714 fixture, the
maximum absolute differences were:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Coefficients vs Statsmodels | `4.021255800568113e-09` |
| Held-out probabilities vs Statsmodels | `1.569703034220993e-09` |
| Held-out probabilities vs scikit-learn | `1.535356947135469e-08` |
| Fold scores vs scikit-learn scoring | `2.089673989402741e-10` |

Unshuffled `KFold` and `StratifiedKFold` matched scikit-learn's indices in
500/500 randomized valid designs. Both total stratified fold sizes and each
class's fold counts differed by at most one in 500/500 designs. Complete-group
splitting is compared by its estimand and invariants—complete test coverage,
no group leakage, and observation-count balance—because `EntityHoldoutSplit`
deliberately balances rows while scikit-learn can use different group ordering
and shuffle policies. `ForwardPanelSplit` has no direct scikit-learn analogue;
its gate is chronological leakage prevention and complete forecast windows.
`StratifiedGroupKFold` first uses greedy class/row balancing and local repair;
when coverage remains incomplete, an exact mixed-integer feasibility fallback
enforces every-class/every-fold coverage or rejects the design.

The following differences are intentional and tested or documented:

- Default log loss clips probabilities at `1e-15`; scikit-learn uses a dtype-
  dependent machine epsilon. Supplying the same `eps` makes float64 endpoint
  cases agree to float64 precision; lower-precision input arithmetic can still
  round differently.
- Multiclass and ordinal scores respect explicitly supplied category order or
  probability-frame columns. Reference probability columns must be reordered
  before comparing software that sorts labels.
- Randomized fold membership is reproducible within limiteddepkit but is not
  promised to reproduce scikit-learn's integer-seed indices, because the RNG
  and allocation implementations differ.
- Sparse categories are rejected when a class cannot occur in every
  stratified test fold. This is stricter than scikit-learn's warning behavior
  and protects probability-score interpretation.
- Fold-local preprocessing accepts sparse transformer output only as workflow
  plumbing for a downstream estimator that itself supports sparse fit and
  prediction. Native `limiteddepkit` estimators require dense transformed
  designs; the sparse contract test uses a purpose-built sparse-aware model.
- Ranked Probability Score, variable-size conditional-choice scores, zero-rate
  calibration, selection composites, and censoring-aware duration scores do
  not have one directly equivalent scikit-learn primitive. Their identity and
  edge-case tests remain separate. The original duration horizon Brier score
  is known-status and non-IPCW; the separate survival module now tests
  training-fold reverse-KM weights, Uno-style concordance, IPCW Brier curves,
  integrated Brier score, and cumulative/dynamic AUC by hand identities and
  censoring-heavy boundary cases.

## Limited-data and penalized-estimator gates

The experimental small-sample estimators have deliberately narrower claims:

| Estimator/diagnostic | Maintained reference evidence | Numerical boundary |
| --- | --- | --- |
| Firth Binary Logit | Exact 2x2 complete-separation half-cell correction | Coefficients within `3e-8`; Wald covariance is approximate and no profile interval is claimed |
| Ridge Binary Logit | scikit-learn `LogisticRegression` with aligned summed-log-likelihood scale `C = 1 / penalty` | Coefficients within `2e-6` |
| Near-unpenalized Ridge Ordered Logit | Statsmodels `OrderedModel` | Slopes/cutpoints within `3e-4`; log likelihood within `1e-5` |
| Binary calibration intercept/slope | Statsmodels Logit recalibration on the same held-out log odds | Parameters and covariance within `1e-7` on the deterministic fixture |

These checks do not make penalized likelihoods ordinary-MLE likelihoods. Firth
intervals remain inverse-ordinary-Fisher Wald approximations; ridge covariance
is an explicitly labelled penalized estimating-equation sandwich. Penalty and
model-family selection is tested through nested CV, including a transformer
spy that proves preprocessing is refitted only on the current inner/outer
training rows.

All maintained Firth/ridge implementations require `n > p` and full column
rank. Their gates cover separation resistance and shrinkage within an
identified low-dimensional specification; they do not certify high-dimensional
or rank-deficient designs and do not claim that ridge repairs exact
collinearity.

Repeated and grouped validation tests additionally cover paired observation
and entity bootstrap intervals, the one-standard-error rule, exact group
isolation, per-column support under heterogeneous nested predictions, duplicate
row labels, and weighted nonnumeric OOF modes. Overlapping repeated-fold
standard errors remain descriptive; they are not promoted as independent-fold
inference.

The optional bridges are contract tests, not estimator parity claims.
Scikit-learn estimators are cloned per fit, Statsmodels prediction semantics
must be declared as probability or value, invalid options fail before fitting,
and generic integrations import their dependency lazily. Direct randomized
parity against scikit-survival is not currently a maintained gate because that
optional dependency is not in the test environment; survival formulas are
therefore described only at their tested identity/edge-case boundary.

## Optional neural challenger status

`ResidualBinaryMLP` is installed through the optional `[neural]` extra and is
evaluated as a prediction challenger, not an inference or parity estimator.
Its result deliberately reports `inference_valid=False`. A returned finite
checkpoint sets `training_completed=True`, whereas `converged=True` requires
patience-based validation-loss stabilization; reaching the epoch limit alone
does not pass that conservative convergence gate. Its current internal split
is iid-stratified, so entity-grouped and chronological neural validation are
out of scope. Hyperparameters must be selected inside nested CV, temperature
calibration reuses the internal early-stopping partition, and Monte Carlo
dropout bands are only approximate conditional model uncertainty.

An isolated Python 3.13/PyTorch 2.13.0 run on 14 July 2026 passed all 21 neural
tests, including deterministic fitting, early stopping, temperature
calibration, nested CV, and Monte Carlo dropout. CI now has a dedicated Python
3.13 `[test,neural]` job so these paths are maintained instead of silently
skipped in the dependency-light matrix. This is runtime/contract evidence, not
a numerical-parity, recovery, uncertainty-coverage, or inferential claim.

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
