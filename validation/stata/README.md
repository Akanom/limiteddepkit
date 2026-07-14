# Stata parity and external-application guide

## Current status

> **PASS — ALL EIGHT FAMILIES, CONTROLLED AND REAL-DATA TRACKS**

The maintained suites were run manually in Stata 17 and compared on 14 July
2026. The controlled synthetic suite passed 82 of 82 declared checks and the
public-data application suite passed 82 of 82, with no failures or skips. All
eight families were included; the run log identifies `gologit2` 3.2.8. The
generated reports and certificates remain under the ignored work directories
and must be archived together if they support a release or publication claim.

This directory follows a two-track validation architecture:

1. **Controlled synthetic certification** aligns data, likelihoods,
   parameterizations, covariance estimators, quadrature, and predictions as
   tightly as possible. It is the only track that can support a strict,
   benchmark-specific parity claim.
2. **External public-data application** repeats the checks on downloaded Stata
   Press example data. It tests sample construction and practical behavior
   outside the controlled fixtures, but it does not broaden the controlled
   certification claim.

The scripts are argument-driven. They do not launch Stata, install community
commands, publish results, or modify an existing Stata installation. The user
runs Stata manually and decides whether to install `gologit2`.

Starting either preparation pass removes only the maintained prior Stata
exports, Stata log, canonical files, comparison report, summary, and certificate
from that target work directory, plus maintained R outputs if the directory is
shared with the R harness. The matching do-file repeats the Stata cleanup before
opening a new log. A completed do-file records its exact `suite` and
`run_completed=1` in `metadata.txt` only after every raw export succeeds; a
missing completion marker means the Stata run is incomplete and its partial
outputs must not be used as evidence.

## Claim boundary

Both maintained runs and their Python comparisons have completed. The current
evidence supports the two benchmark-specific statements below. It does not
support an unqualified claim that Python and Stata are universally identical.

After a successful controlled comparison, the strongest allowed statement is:

> The maintained controlled fixtures passed the declared Stata parity checks
> within benchmark-specific tolerances.

After a successful public-data comparison, the allowed statement is:

> The maintained real-data application checks passed within the declared
> benchmark tolerances; this does not broaden the controlled-fixture
> certification claim.

These statements are never claims of universal equality across datasets,
optimizers, starting values, quadrature rules, covariance choices, missing-data
patterns, active constraints, or preprocessing decisions. The generated
`parity_certificate.json` records the exact claim supported by a completed run.

## Stable-model coverage and status

All eight stable model families are prepared for both tracks. The six models
using official Stata commands are required by default. The two flexible ordinal
models depend on the community-contributed `gologit2` command and are optional
unless the comparator receives `--require-flexible`.

| Stable `limiteddepkit` model | Stata reference | Controlled fixture | Public-data application | Gate | Current status |
| --- | --- | --- | --- | --- | --- |
| `BinaryLogit` | `logit ..., noconstant vce(oim)` | 1,500-row binary Logit DGP | `lbw.dta` | Required | PASS — BOTH TRACKS |
| `BinaryProbit` | `probit ..., noconstant vce(oim)` | 1,500-row binary Probit DGP | `lbw.dta` | Required | PASS — BOTH TRACKS |
| `OrderedLogit` | `ologit ..., vce(oim)` | 1,500-row three-category Logit DGP | `tvsfpors.dta` | Required | PASS — BOTH TRACKS |
| `OrderedProbit` | `oprobit ..., vce(oim)` | 1,500-row three-category Probit DGP | `tvsfpors.dta` | Required | PASS — BOTH TRACKS |
| `GeneralizedOrderedLogit` | `gologit2 ..., npl` | 1,500-row nonparallel Logit DGP | `tvsfpors.dta` | Optional `gologit2` | PASS — BOTH TRACKS |
| `PartialProportionalOdds` | `gologit2 ..., npl(variable)` | Same flexible ordinal fixture | `tvsfpors.dta` | Optional `gologit2` | PASS — BOTH TRACKS |
| `RandomEffectsOrderedLogit` | `meologit ..., intmethod(ghermite)` | 80 groups × 6 periods | `tvsfpors.dta`, 28 schools | Required | PASS — BOTH TRACKS |
| `DynamicRandomEffectsOrderedLogit` | `meologit` on exported augmented design | 60 groups × 6 periods | balanced `nlswork.dta` subset | Required | PASS — BOTH TRACKS |

The built-in-command comparisons require Stata 15.1 or newer. If all eight
models are wanted, install `gologit2` manually from inside Stata:

```stata
ssc install gologit2
```

Neither do-file runs that command. If `gologit2` is absent, it records the
absence and skips the two flexible models.

## Source-software mapping

The reference commands are aligned to the toolkit estimands, not merely chosen
because their names are similar.

| Family | Controlled Stata specification | Public-data Stata specification |
| --- | --- | --- |
| Binary Logit | `logit y_logit intercept x1 x2, noconstant vce(oim)` | `logit y intercept x1 x2 x3 x4, noconstant vce(oim)` |
| Binary Probit | `probit y_probit intercept x1 x2, noconstant vce(oim)` | `probit y intercept x1 x2 x3 x4, noconstant vce(oim)` |
| Ordered Logit | `ologit y_ologit ox1 ox2, vce(oim)` | `ologit y ox1 ox2 ox3 ox4, vce(oim)` |
| Ordered Probit | `oprobit y_oprobit ox1 ox2, vce(oim)` | `oprobit y ox1 ox2 ox3 ox4, vce(oim)` |
| Generalized Ordered Logit | `gologit2 y_gologit gx1 gx2, npl` | `gologit2 y gx1 gx2 gx3 gx4, npl` |
| Partial Proportional Odds | `gologit2 y_gologit gx1 gx2, npl(gx1)` | `gologit2 y gx1 gx2 gx3 gx4, npl(gx4)` |
| Random-effects Ordered Logit | `meologit y x1 x2 \|\| entity:, intmethod(ghermite) intpoints(12) vce(oim)` | Same structure with `x1`–`x4` and 20 points |
| Dynamic random-effects Ordered Logit | `meologit` on `x1 state_1 state_2 initial_1 initial_2 initial_x1 mean_x1`, 12 points | Same augmented design, 20 points |

Every maintained `meologit` command also pins `iterate(2000)`,
`tolerance(1e-10)`, `ltolerance(1e-12)`, and `nrtolerance(1e-8)`.

The primary dynamic check passes the exact toolkit-exported augmented design to
Stata. This isolates likelihood parity from lag and initial-condition
preprocessing. The raw panel is also retained so lagged states,
initial-outcome indicators, initial covariates, and post-initial entity means
can be audited separately.

## Track A: controlled synthetic certification

### Fixture design

NumPy and the toolkit simulators generate every observation. Do not regenerate
the observations independently in Stata because the random-number streams will
not match.

| Fixture | Seed | Dimensions | Outcome/design | Models |
| --- | ---: | --- | --- | --- |
| Binary cross-section | 8,421 | 1,500 observations | Explicit `intercept`; correlated `x1` and `x2`; separate Logit and Probit outcomes | Binary Logit and Probit |
| Pooled ordinal cross-section | 4,102 | 1,500 observations, 3 categories | `ox1`, `ox2`; thresholds `(-0.70, 0.80)`; separate Logit and Probit outcomes | Ordered Logit and Probit |
| Flexible ordinal cross-section | 9,101 | 1,500 observations, 3 categories | `gx1`, `gx2`; thresholds `(-0.9, 0.9)`; split slopes `((0.85, -0.4), (0.3, -0.4))` | Generalized Ordered Logit and PPO varying `gx1` |
| Static random-effects panel | 8,821 | 80 groups × 6 periods = 480 observations | Three-category outcome, `x1`, `x2`, one random intercept | RE Ordered Logit |
| Dynamic random-effects panel | 8,263 | 60 groups × 6 periods = 360 raw observations; 300 post-initial rows | Three-category outcome, `x1`, exact seven-column augmented design | Dynamic RE Ordered Logit |

The controlled panel fits use 12-point nonadaptive Gauss–Hermite quadrature.
For every model, the first 25 stable observation IDs supply selected
probability comparisons.

### Step 1 — prepare inputs and Python references

From the repository root:

```powershell
python validation/stata/prepare_parity.py
```

To use a separate work directory:

```powershell
python validation/stata/prepare_parity.py --output "C:/parity/limiteddepkit-synthetic"
```

The default is `validation/stata/work/`. Preparation writes CSV and Stata
`.dta` datasets, Python reference results, and a hash manifest.
Pooled Ordered Logit/Probit references use `maxiter=5000` and optimizer
function tolerance `1e-13`; both controls are recorded in the manifest and
certificate. Panel references separately record optimizer tolerance `1e-12`.

### Step 2 — run Stata manually

Use an absolute path, forward slashes, and quotes. Quotes matter when the
repository path contains spaces.

```stata
do "C:/Users/omoko/OneDrive/Python packages/limiteddepkit/validation/stata/limiteddepkit_parity.do" ///
   "C:/Users/omoko/OneDrive/Python packages/limiteddepkit/validation/stata/work"
```

If Step 1 used a custom output directory, pass that exact directory as the
do-file argument.

### Step 3 — compare and generate evidence

For the default work directory:

```powershell
python validation/stata/compare_parity.py
```

For a custom work directory:

```powershell
python validation/stata/compare_parity.py "C:/parity/limiteddepkit-synthetic"
```

To make absence of either flexible ordinal model a failure:

```powershell
python validation/stata/compare_parity.py --require-flexible
```

The comparator verifies all prepared-input hashes before reading Stata output.
It exits with code 1 if a required check fails.

## Track B: public-data external application

This track uses three official Stata Press release-19 example datasets. Source
files are downloaded into the ignored work directory and hash-verified before
use.

### Step 1 — acquire and prepare the data

The simplest route lets Python download any missing pinned sources and then
prepare the analysis datasets and Python references:

```powershell
python validation/stata/prepare_real_data.py
```

For an explicit download step, use the PowerShell helper and point preparation
at the resulting cache:

```powershell
powershell -ExecutionPolicy Bypass -File validation/stata/download_real_data.ps1
python validation/stata/prepare_real_data.py --source-dir validation/stata/work/real_data/source
```

Both routes verify the pinned SHA-256 values documented below. The default work
directory is `validation/stata/work/real_data/`. A separate cache and output
directory can be selected:

```powershell
python validation/stata/prepare_real_data.py --source-dir "D:/validated-data/stata-r19" --output "C:/parity/limiteddepkit-real"
```

### Step 2 — run the public-data do-file manually

For the default work directory:

```stata
do "C:/Users/omoko/OneDrive/Python packages/limiteddepkit/validation/stata/limiteddepkit_real_data.do" ///
   "C:/Users/omoko/OneDrive/Python packages/limiteddepkit/validation/stata/work/real_data"
```

This do-file reads only the analysis-ready `.dta` files produced by Python. It
uses 20-point nonadaptive Gauss–Hermite quadrature for the panel applications.
It does not download sources, install community commands, or launch a process.

### Step 3 — run the same comparator

```powershell
python validation/stata/compare_parity.py validation/stata/work/real_data
```

To require both `gologit2` applications:

```powershell
python validation/stata/compare_parity.py validation/stata/work/real_data --require-flexible
```

The manifest identifies this suite as `real_data_application`, so the generated
certificate uses the narrower external-application claim.

## Public source data, provenance, and transformations

All URLs point to the official Stata Press release-19 data area. Hashes are
pinned in both [`prepare_real_data.py`](prepare_real_data.py) and
[`download_real_data.ps1`](download_real_data.ps1).

| Source | Official URL and pinned SHA-256 | Role | Maintained transformation and sample |
| --- | --- | --- | --- |
| `lbw.dta` | [Stata Press r19 `lbw.dta`](https://www.stata-press.com/data/r19/lbw.dta) — `00204ef3586836e56e49598cd9850148aea9058090a607e5bf20e12a6b0a58ee` | Binary Logit and Probit | Keep all 189 rows; `y=low`; `intercept=1`; `x1=age/10`; `x2=lwt/100`; `x3=smoke`; `x4=ht` |
| `tvsfpors.dta` | [Stata Press r19 `tvsfpors.dta`](https://www.stata-press.com/data/r19/tvsfpors.dta) — `50197a3e7b15809ed816b2846ca9dc1a4bc6aecac06ba75f4ae0312d7ceebfc8` | Pooled, flexible, and school-level RE ordinal models | Keep all 1,600 rows and 28 schools; `y=thk-1`; regressors `prethk`, `cc`, `tv`, and `cc*tv` |
| `nlswork.dta` | [Stata Press r19 `nlswork.dta`](https://www.stata-press.com/data/r19/nlswork.dta) — `b77bc182ac586205d769ad847e5e7cb0063c31be2c4bbef5f1ad16b74118c86f` | Dynamic RE ordinal application | Keep complete `idcode/year/ln_wage/tenure` rows in years 68–73 and entities with exactly six periods; 2,010 raw rows, 335 entities; bin `ln_wage` at 1.4 and 1.8; use `tenure` as `x1`; 1,675 post-initial rows |

These Stata Press files are example/reference datasets and may have been
altered, constructed, or otherwise prepared for software documentation. They
are engineering-only parity inputs, not a basis for substantive empirical
conclusions.

`limiteddepkit` claims no redistribution license for these datasets. The files
are downloaded to the ignored local `work/real_data/source/` directory and are
not distributed with the package. Do not commit or republish downloaded or
prepared copies without independently confirming the applicable rights and
source terms. Consult the
[official Stata 19 example-dataset index](https://www.stata-press.com/data/r19/)
and original documentation before any use beyond this engineering check.

## Exact public-data estimator designs

| Model | Outcome | Features and structure | Categories / integration |
| --- | --- | --- | --- |
| Binary Logit and Probit | `lbw.low` | Explicit intercept, `age/10`, `lwt/100`, `smoke`, `ht` | Binary; observed-information covariance |
| Ordered Logit and Probit | `tvsfpors.thk - 1` | `prethk`, `cc`, `tv`, `cc*tv` | Categories 0–3; common slopes |
| Generalized Ordered Logit | `tvsfpors.thk - 1` | Same four features, all split-specific | Three cumulative splits; positive constraint slack required |
| Partial Proportional Odds | `tvsfpors.thk - 1` | `prethk`, `cc`, `tv` common; `cc*tv` varying | Three cumulative splits; positive constraint slack required |
| RE Ordered Logit | `tvsfpors.thk - 1` | Same four features; school random intercept | 28 groups; 20-point nonadaptive GH |
| Dynamic RE Ordered Logit | Binned `nlswork.ln_wage` | `tenure`, lagged-state indicators, initial-outcome indicators, initial tenure, post-initial entity mean of tenure; entity random intercept | 335 groups; 1,675 rows; 20-point nonadaptive GH |

The dynamic augmented columns exported for Stata are `x1`, `state_1`,
`state_2`, `initial_1`, `initial_2`, `initial_x1`, and `mean_x1`. They map back
to the toolkit names `x1`, `state[1]`, `state[2]`, `initial[1]`,
`initial[2]`, `initial_x[x1]`, and `mean[x1]`.

## Parameterization and covariance canonicalization

The comparator does not assume that raw Stata coefficient names are already in
the toolkit's canonical parameterization.

### Binary models

Slope and explicit-intercept parameters use the identity map. Stata's
`noconstant` prevents an additional implicit intercept.

### Ordered Logit and Ordered Probit

Both implementations use:

```text
P(Y <= j | X) = F(cut_j - X beta)
```

Stata cutpoints therefore have the same sign as `limiteddepkit` thresholds and
are renamed to `threshold: j | j+1`.

### Generalized and partial proportional odds

`gologit2` expresses cumulative equations as `P(Y > j)`. Under the maintained
mapping:

- Stata slopes retain the toolkit slope sign.
- Each Stata equation intercept maps to the negative toolkit threshold.
- Common PPO slopes are retained once; declared varying slopes are mapped by
  split equation.

`limiteddepkit` imposes a minimum noncrossing gap over the observed covariate
support. `gologit2` does not impose that exact inequality. Preparation requires
the toolkit optimum to have positive constraint slack. A fit on an active
noncrossing boundary is not an exact external parity case and must not be forced
through this claim framework.

### Random-effect scale and full covariance

Stata releases expose the random-intercept scale under more than one raw
parameter name. For a log standard deviation `eta = log(sigma_entity)`, the
comparator uses:

```text
sigma_entity = exp(eta)
d sigma_entity / d eta = sigma_entity
```

The completed Stata 17 runs exported `/var(_cons[entity])` on the variance
scale. If `v = sigma_entity^2`, the comparator instead uses:

```text
sigma_entity = sqrt(v)
d sigma_entity / d v = 1 / (2 sigma_entity)
```

The recognized raw name determines the transformation; the comparator never
guesses from the numerical value. It applies the relevant derivative to the
standard error and the full covariance matrix. More generally, every canonical
covariance is calculated as `J V_stata J'`, including off-diagonal cells. It is
never reconstructed from reported standard errors.

## Quadrature and prediction conventions

The panel likelihoods use **nonadaptive Gauss–Hermite quadrature** in both
programs:

| Suite | Nodes |
| --- | ---: |
| Controlled synthetic | 12 |
| Public-data application | 20 |

Stata's default adaptive quadrature is not the same finite-node likelihood.
The do-files specify `intmethod(ghermite)` and `intpoints(...)` explicitly. The
do-files then capture Stata's actual `e(intmethod)` and `e(n_quad)` immediately
after each panel fit. The comparator rejects metadata unless both the static and
dynamic panel models report the matching method and node count.

Panel probability parity is conditional on a random effect of zero:

```text
Stata:         predict ..., conditional(fixedonly)
limiteddepkit: predict_proba(..., random_effects=0.0)
```

This is a fixed-component conditional probability. It is neither Stata's
default empirical-Bayes prediction nor a population-averaged probability
integrated over the random-effect distribution.

## Output and evidence schema

The controlled work directory has this shape. The real-data directory adds
`source/` and otherwise uses the same schema.

```text
work/
├── manifest.json
├── data/
│   ├── *.csv
│   └── *.dta
├── python/
│   ├── estimates.csv
│   ├── covariance.csv
│   ├── fit.csv
│   └── predictions.csv
├── stata/
│   ├── estimates_raw.csv
│   ├── covariance_raw.csv
│   ├── fit.csv
│   ├── predictions.csv
│   ├── metadata.txt
│   ├── stata_run.log
│   ├── estimates_canonical.csv
│   └── covariance_canonical.csv
├── comparison_report.csv
├── comparison_summary.md
└── parity_certificate.json
```

The evidence objects are:

- **Raw estimates:** Stata model, original `e(b)` position and name, estimate,
  and standard error.
- **Canonical estimates:** mapped toolkit parameter name, transformed estimate,
  and transformed standard error.
- **Full covariance:** every row-by-column entry from Stata `e(V)`, followed by
  the complete Jacobian-transformed canonical matrix.
- **Fit evidence:** observations, groups where applicable, parameter count, log
  likelihood, AIC, BIC, and Stata convergence. Python references also retain
  inference validity, quadrature nodes, and flexible-model constraint slack.
- **Selected predictions:** category probabilities keyed by stable `obs_id`.
  The comparator checks 25 Python reference rows per model and all categories
  for those rows.
- **Manifest and hashes:** suite identity, package/dependency versions, model
  maps, quadrature and sample assertions, public-source provenance, and SHA-256
  hashes for prepared inputs and Python references.
- **Metadata and log:** Stata version, `gologit2` availability and path, actual
  panel-model `e(intmethod)`/`e(n_quad)`, prediction mode, commands, warnings,
  convergence, suite identity, and the final `run_completed=1` marker.
- **Reports:** a row-level `comparison_report.csv`, human-readable
  `comparison_summary.md`, and machine-readable `parity_certificate.json`.

The certificate is a structured run record, not a cryptographic signature and
not a statement about specifications outside its manifest.

## Declared numerical tolerances

`compare_parity.py` uses maximum absolute differences. These are
implementation-parity tolerances, not simulation-recovery thresholds.

| Comparator family | Estimate | Standard error | Full covariance | Log likelihood | AIC/BIC | Probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary | `2e-6` | `2e-6` | `3e-6` | `1e-7` | `2e-7` | `2e-6` |
| Ordered | `5e-5` | `5e-5` | `1e-4` | `5e-6` | `1e-5` | `5e-5` |
| Generalized ordered | `2e-4` | `5e-4` | `1e-3` | `2e-5` | `4e-5` | `2e-4` |
| Partial proportional odds | `2e-4` | `5e-4` | `1e-3` | `2e-5` | `4e-5` | `2e-4` |
| Random-effects and dynamic RE ordered | `1e-3` | `2e-3` | `3e-3` | `1e-3` | `2e-3` | `1e-3` |

Observation count and parameter count must match exactly. Group count must
match exactly where applicable. Stata convergence must equal one. AIC and BIC
tolerances are twice the family log-likelihood tolerance because their only
floating component in this aligned comparison is `-2 * loglike`.

### Completed-run numerical envelope

Across all eight models, the largest controlled-suite differences were
`4.16e-6` for an estimate, `3.03e-7` for a standard error, `2.14e-7` for a
covariance element, `3.53e-9` for log likelihood, and `1.56e-6` for a
probability. The corresponding real-data maxima were `1.47e-5`, `2.11e-6`,
`2.36e-6`, `2.58e-9`, and `2.77e-6`. Every value is inside its predeclared
family gate. Each generated certificate records 82 passing checks, all eight
models available, and zero failures.

## Reading PASS, FAIL, and SKIP

- **PASS** means every aligned value for that model/statistic is finite,
  present, and within the declared maximum absolute tolerance.
- **FAIL** means a required model or key is missing, any aligned difference is
  non-finite, a numerical gate is exceeded, convergence is not confirmed, or
  an exact count differs. Any `FAIL` makes the comparator exit with status 1
  and prevents a parity claim.
- **SKIP** means an optional `gologit2` model was absent and
  `--require-flexible` was not supplied. A suite can exit successfully with
  these skips, but no claim may imply that all eight models were checked.

With `--require-flexible`, a missing generalized or partial proportional-odds
fit becomes `FAIL`. A clean exit remains benchmark-specific: it does not
certify unseen data, different covariates, active constraints, alternative
quadrature, robust/clustered covariance, marginal predictions, or different
preprocessing.

## Troubleshooting

### HTTPS, TLS, or proxy failure

Try [`download_real_data.ps1`](download_real_data.ps1), or obtain each official
file through an approved browser/network and place it in the directory supplied
to `--source-dir`. Do not bypass hash verification. If an official file's hash
changes, inspect the upstream change before deliberately updating the pin.

### Paths containing spaces

Use absolute paths, forward slashes, and double quotes in Stata. The work
directory is the do-file's first positional argument; running without it exits
with an error.

### `gologit2` is absent

This is expected on a clean Stata installation. Run the six built-in models and
accept `SKIP`, or install `gologit2` manually and rerun Stata. Use
`--require-flexible` only when all eight outputs are expected.

### Manifest or stale-hash error

Prepared input or Python reference files changed after the manifest was
written. Use a fresh work directory or rerun the matching preparation script.
Do not edit `manifest.json` to conceal a mismatch, and do not mix Stata outputs
from different prepared runs. Rerunning preparation intentionally invalidates
the maintained Stata and comparator evidence in that work directory; rerun the
matching do-file and comparator afterward.

### Quadrature metadata mismatch

Use the maintained do-file unchanged and confirm `metadata.txt` records
`random_effects_ordered_logit.intmethod=ghermite` and
`dynamic_random_effects_ordered_logit.intmethod=ghermite`, with each model's
`.n_quad` key equal to the suite's 12 or 20 points. These values are captured
from Stata's returned results, not copied from the requested command options.
Adaptive Gauss–Hermite results are not comparable under this gate.

### Nonconvergence or large panel differences

Inspect `stata_run.log` before changing settings. A modified optimizer, starting
value, quadrature method, or covariance estimator creates a new benchmark and
requires separately documented specifications and tolerances.

## Evidence archival

`validation/stata/work/` is ignored by Git because it contains generated files
and, for the real track, third-party data. After a completed run:

The [committed evidence index](../PARITY_EVIDENCE.md) records the final outcome
and manifest, report, and certificate digests for the 14 July 2026 run. It is
an audit pointer, not a replacement for the complete bundle described below.

1. Preserve the manifest, Python references, Stata raw exports, canonical
   exports, metadata, log, report, summary, and certificate together.
2. Record the exact commit, `limiteddepkit` version, Python environment, Stata
   version, and `gologit2` location/version evidence.
3. Preserve the do-file and comparator used for the run, or record their
   repository hashes.
4. Never combine artifacts from different manifests.
5. Keep downloaded real-data files in controlled local storage. Omit them from
   a public archive unless redistribution rights have been confirmed; retain
   the URLs and pinned hashes as provenance.
6. Do not promote a status table or release claim until the archived report and
   certificate support that exact wording.

## Extending the parity matrix

To add a model, dataset, or specification:

1. Define whether it belongs to controlled certification or external
   application.
2. Create deterministic analysis-ready data with stable observation IDs and an
   explicit category order.
3. Fit `limiteddepkit` and export canonical estimates, full covariance, fit
   statistics, and selected probabilities.
4. Add a manifest model specification with `kind`, feature order, required
   status, and any `varying` or `feature_map` declarations.
5. Align the Stata intercept, covariance, sample, quadrature, random-effect
   level, and prediction target explicitly.
6. Export raw `e(b)`, full `e(V)`, fit evidence, predictions, metadata, and log.
7. Add and test any new parameter/Jacobian mapping.
8. Declare tolerances before examining a desired result and justify any wider
   gate.
9. Run Stata manually, compare, archive the complete evidence, and only then
   update a validation claim.

An external application is supporting evidence, not a substitute for a
controlled benchmark. A new estimator family should normally receive both.

## Authoritative references

- [Stata Press release-19 example dataset index](https://www.stata-press.com/data/r19/)
- [Stata `meologit` reference manual](https://www.stata.com/manuals/memeologit.pdf)
- [Stata `logit` reference manual](https://www.stata.com/manuals/rlogit.pdf)
- [Stata `probit` reference manual](https://www.stata.com/manuals/rprobit.pdf)
- [Stata `ologit` reference manual](https://www.stata.com/manuals/rologit.pdf)
- [Stata `oprobit` reference manual](https://www.stata.com/manuals/roprobit.pdf)
- [Richard Williams, `gologit2` documentation](https://www.stata.com/meeting/4nasug/gologit2.pdf)

## Harness files

- [`prepare_parity.py`](prepare_parity.py): controlled fixtures and Python references.
- [`limiteddepkit_parity.do`](limiteddepkit_parity.do): manual controlled Stata run.
- [`prepare_real_data.py`](prepare_real_data.py): pinned acquisition,
  transformations, public-data references, and manifest.
- [`download_real_data.ps1`](download_real_data.ps1): optional explicit source download.
- [`limiteddepkit_real_data.do`](limiteddepkit_real_data.do): manual public-data Stata run.
- [`compare_parity.py`](compare_parity.py): shared canonicalization, numerical gates,
  reports, summaries, and certificates.
