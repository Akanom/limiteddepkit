# R parity guide

## Current status

> **PASS — ALL EIGHT FAMILIES, CONTROLLED AND REAL-DATA TRACKS**

The maintained Python references were fitted independently in R 4.5.1 and
compared on 14 July 2026. The controlled synthetic suite passed 110 of 110
declared checks; the public-data application suite also passed 110 of 110.
There were no failures or skipped models.

The strongest supported statements are deliberately benchmark-specific:

> The maintained controlled fixtures passed all eight declared R parity checks
> within benchmark-specific tolerances.

> The maintained real-data R application checks passed for all eight families
> within benchmark-specific tolerances; this does not broaden the controlled
> certification claim.

These are not claims of universal equality across datasets, R versions,
optimizers, covariance estimands, quadrature implementations, starting values,
or preprocessing choices. Generated evidence is ignored by Git and must be
archived separately if it supports a release or publication.

The [committed evidence index](../PARITY_EVIDENCE.md) records the four final
outcomes and the exact manifest, report, and certificate digests for this run.
It does not replace the complete archival bundle.

## Why this harness exists

This is an independent implementation check, not an R wrapper around
`limiteddepkit`. R reads the exact analysis-ready CSVs registered in the Python
manifest, fits corresponding estimators, exports canonical results, and then a
Python comparator checks:

- parameter estimates and standard errors;
- the complete covariance matrix, including cross-covariances;
- observation, parameter, and group counts;
- log likelihood, AIC, and BIC;
- convergence and inference-validity markers;
- flexible-model constraint slack;
- selected category probabilities and their row sums; and
- dataset, model, software-version, quadrature, and completion metadata.

The same controlled and real-data inputs are used by the Stata harness. This
keeps the Python–R and Python–Stata claims aligned to one frozen benchmark rather
than quietly comparing different samples or transformations.

## Model mapping

| `limiteddepkit` family | Independent R estimator | Important alignment |
| --- | --- | --- |
| Binary Logit | `stats::glm.fit(..., binomial("logit"))` | Explicit constant is already in `X`; observed information |
| Binary Probit | `stats::glm.fit(..., binomial("probit"))` | Inverse-Mills observed information replaces R's default Fisher covariance |
| Ordered Logit | `MASS::polr(method="logistic")` | No ordinal constant; slopes followed by increasing thresholds |
| Ordered Probit | `MASS::polr(method="probit")` | Same category and threshold convention |
| Generalized Ordered Logit | `VGAM::vglm(cumulative(..., parallel=FALSE))` | R slopes are sign-flipped and reordered to threshold-major toolkit names |
| Partial Proportional Odds | `VGAM::vglm(cumulative(..., parallel=...))` | Common and varying slopes are mapped by name |
| Random-Effects Ordered Logit | `ordinal::clmm(nAGQ=-Q)` | Nonadaptive GH; log-SD is delta-transformed to `sigma_entity` |
| Dynamic Random-Effects Ordered Logit | `ordinal::clmm` on the exported augmented design | Exact lag/initial-condition design; fixed-`b=0` predictions |

### Binary Probit covariance

R's ordinary GLM summary uses expected/Fisher information for Probit. The
toolkit reports observed information. For signed index
`z_i = (2y_i - 1)x_i'beta` and inverse Mills ratio
`lambda_i = phi(z_i) / Phi(z_i)`, the R harness uses

```text
w_i = lambda_i (z_i + lambda_i)
V   = (X' diag(w) X)^(-1)
```

Without this alignment, coefficients and predictions agree but the covariance
comparison tests a different estimand.

### Flexible ordinal mapping

With `reverse=FALSE`, VGAM represents the cumulative equation as

```text
logit Pr(Y <= j) = alpha_j + x' gamma_j.
```

`limiteddepkit` uses `threshold_j - x' beta_j`. Therefore
`threshold_j = alpha_j` and `beta_j = -gamma_j`. VGAM's coefficient layout is
mapped by name into the toolkit's threshold-major layout.

VGAM's default covariance is Fisher/EIM based. The harness recomputes the same
central observed numerical Hessian used by `limiteddepkit`, then takes its
pseudoinverse. It also confirms that R probabilities equal probabilities
reconstructed from the canonical parameters and that the cumulative indices do
not cross on the estimation sample.

### Panel scale and prediction target

`ordinal::clmm` exposes thresholds, fixed slopes, and
`tau = log(sigma_entity)`. The helper constructs a full Jacobian that both
reorders parameters and applies

```text
d sigma_entity / d tau = sigma_entity.
```

The full covariance is `J V_raw J'`; off-diagonal terms are not discarded.
Negative `nAGQ` selects nonadaptive Gauss–Hermite quadrature in `ordinal`. Its
finite-node implementation is not algebraically identical to the toolkit's
node formula, so the panel result is numerical implementation parity within the
declared gate, not exact node-by-node identity.

Panel probabilities are conditional at random intercept `b=0`. They are not
empirical-Bayes, posterior-mean, or population-averaged probabilities.

## Pinned R environment

The validated environment is Windows R 4.5.1. Run the setup script from the
repository root:

```powershell
.\validation\r\setup_dependencies.ps1
```

If R is not on `PATH`, pass its executable explicitly:

```powershell
.\validation\r\setup_dependencies.ps1 `
  -Rscript 'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
```

The script downloads exact Windows binaries over HTTPS, verifies SHA-256 before
installation, and installs only into `validation/r/work/library`. It does not
modify the system R library. The runner restricts `.libPaths()` to that project
library plus R's own `.Library`; user and site libraries cannot override the
validated packages.

| Package | Version | SHA-256 of pinned binary |
| --- | --- | --- |
| MASS | 7.3-65 | `46f1a3d0991c8387411b23cc9faf657a5abfc5e93438546f8b042073d9988c14` |
| jsonlite | 2.0.0 | `4b9418cff57f2357fbf5d24b1a618f082310cb9d5b63af051bd8dd7f570e188a` |
| numDeriv | 2016.8-1.1 | `0df596925b695a2ba0bc327b71340921ba6550e8cbdc53e49024e41b50e2cdac` |
| ucminf | 1.2.3 | `335437fae88c185ae31142e7828ba1855b45e50524a5ac0bca17175d53d673e0` |
| ordinal | 2025.12-29 | `b27a83300c6664abe0b568fab39c962c4651e62d3be95bdfb552a15550789e9b` |
| VGAM | 1.1-14 | `752dd0d4012731a0e7b37bdf4a443631850d8b0263100dae1a877afae3a61bed` |

R 4.5.1's recommended `Matrix` 1.7-3 and `nlme` 3.1-168 packages are also
required, version-checked, and required to resolve from `.Library`. They are
not downloaded separately by the setup script.

The comparator refuses a different recorded version. A different environment
can be investigated, but it constitutes a new benchmark and should receive a
new manifest, tolerances, and evidence record.

CRAN's current Windows-contrib URLs are not permanent archive URLs. The hashes
prevent silent substitution, but a binary may eventually disappear upstream.
Archive the six verified ZIP files with a formal evidence bundle so the exact
environment remains reconstructable.

## Reproducing the controlled track

Use a dedicated R work directory when the Stata evidence must remain untouched:

```powershell
python validation/stata/prepare_parity.py `
  --output validation/r/work/synthetic

Rscript --vanilla validation/r/run_parity.R `
  validation/r/work/synthetic

python validation/r/compare_parity.py `
  validation/r/work/synthetic
```

The preparation script creates deterministic data, fits all eight Python
references, records hashes, and fixes 12 quadrature points for both panel
models. Pooled ordered references use `maxiter=5000` and optimizer tolerance
`1e-13`; panel references use tolerance `1e-12`. Before a new fit, the R runner
removes only its eight maintained result
and comparator artifacts; a failed run therefore cannot leave an older
completion marker or certificate looking current.

## Reproducing the real-data track

```powershell
python validation/stata/prepare_real_data.py `
  --output validation/r/work/real_data

Rscript --vanilla validation/r/run_parity.R `
  validation/r/work/real_data

python validation/r/compare_parity.py `
  validation/r/work/real_data
```

The Python preparation step downloads pinned Stata Press example datasets,
verifies their hashes, applies the documented transformations, and writes
analysis-ready CSVs. It uses 20 quadrature points for both panel applications.
Downloaded third-party data and generated outputs remain ignored by Git.

To compare against the exact shared work directories used in the completed
three-way run instead, use:

```powershell
Rscript --vanilla validation/r/run_parity.R validation/stata/work
python validation/r/compare_parity.py validation/stata/work

Rscript --vanilla validation/r/run_parity.R validation/stata/work/real_data
python validation/r/compare_parity.py validation/stata/work/real_data
```

Do not rerun either Stata preparation script in a work directory whose manual
Stata artifacts need to be preserved; preparation intentionally invalidates
older external-software evidence.

## Declared numerical gates

Absolute-difference tolerances were fixed by estimator family and are shared
with the Stata comparator.

| Family | Estimate | Standard error | Full covariance | Log likelihood | Probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binary | `2e-6` | `2e-6` | `3e-6` | `1e-7` | `2e-6` |
| Ordered | `5e-5` | `5e-5` | `1e-4` | `5e-6` | `5e-5` |
| Generalized/PPO | `2e-4` | `5e-4` | `1e-3` | `2e-5` | `2e-4` |
| Random-effects ordinal | `1e-3` | `2e-3` | `3e-3` | `1e-3` | `1e-3` |

AIC and BIC use twice the log-likelihood tolerance. Counts, dataset identities,
convergence, inference validity, and panel group counts must match exactly.
Probability row sums must be within `1e-10` of one.
For Generalized Ordered Logit and PPO, constraint slack uses the corresponding
`2e-4` probability tolerance and must remain positive enough for valid ordinary
inference.

## Completed-run numerical envelope

The largest absolute differences observed in the completed run were:

| Model | Controlled estimate | Real-data estimate |
| --- | ---: | ---: |
| Binary Logit | `1.40e-11` | `1.06e-9` |
| Binary Probit | `2.91e-9` | `2.40e-8` |
| Ordered Logit | `6.96e-8` | `7.57e-7` |
| Ordered Probit | `5.26e-7` | `4.36e-7` |
| Generalized Ordered Logit | `5.71e-7` | `4.42e-6` |
| Partial Proportional Odds | `4.14e-6` | `8.68e-7` |
| Static RE Ordered Logit | `6.16e-6` | `5.22e-5` |
| Dynamic RE Ordered Logit | `1.02e-5` | `3.19e-4` |

Across all models, the worst controlled differences were `1.26e-5` for a
standard error, `3.85e-6` for a covariance element, `1.53e-4` for log
likelihood, and `3.07e-6` for a probability. The corresponding real-data
maxima were `2.05e-4`, `2.29e-4`, `5.35e-4`, and `5.05e-5`. Each is inside its
predeclared family gate.

## Output and evidence contract

The R runner writes under `<workdir>/r/`:

```text
estimates.csv
covariance.csv
fit.csv
predictions.csv
metadata.csv
```

The comparator then adds:

```text
comparison_report.csv
comparison_summary.md
parity_certificate.json
```

The certificate records the suite, claim, model set, package versions,
quadrature, tolerances, Python-reference hashes, R-artifact hashes, check count,
and failure count. It is a structured provenance record, not a cryptographic
signature.

For archival, retain the preparation manifest, analysis-ready inputs or their
permitted provenance, Python references, all R outputs, comparator report,
summary, certificate, exact repository commit, and console log together. Never
combine artifacts from different manifests.

## Harness files

- [`setup_dependencies.ps1`](setup_dependencies.ps1) downloads, verifies, and
  locally installs the pinned R packages.
- [`run_parity.R`](run_parity.R) orchestrates all eight independent fits and
  canonical exports.
- [`flexible_models.R`](flexible_models.R) implements VGAM mapping,
  non-crossing checks, observed covariance, and probabilities.
- [`panel_models.R`](panel_models.R) implements `clmm` fitting, full Jacobian
  covariance mapping, and fixed-only predictions.
- [`compare_parity.py`](compare_parity.py) verifies hashes, schemas, metadata,
  numerical gates, and emits evidence.

## Authoritative R references

- [R `glm` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html)
- [MASS `polr` documentation](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/polr.html)
- [VGAM package and cumulative-model manual](https://cran.r-project.org/package=VGAM)
- [ordinal package](https://cran.r-project.org/package=ordinal)
- [ordinal reference manual](https://cran.r-project.org/web/packages/ordinal/ordinal.pdf)
