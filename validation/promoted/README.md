# Promoted-family public-data parity

This directory contains the separate application harness for the stable model
families added after the original eight-model binary/ordinal certificate. It
does **not** replace or rewrite the frozen evidence under `validation/stata`
and `validation/r`.

The suite uses one prepared sample per estimand and records four different
evidence classes:

- **industrial-package parity**: an established R or Stata estimator fits the
  aligned likelihood;
- **numerical implementation parity**: the same integrated likelihood is fit
  with a different quadrature implementation;
- **pseudo-sample likelihood identity**: an exact grouped or conditional
  likelihood is represented by a standard binary/conditional command; and
- **independent-likelihood identity**: an independently coded likelihood is
  used only where the installed industrial packages have no exact estimator.

These labels matter. A likelihood identity is useful replication evidence, but
it must not be reported as if a different duration or panel estimator agreed by
accident.

## Model map

| `limiteddepkit` model | R reference | Stata reference | Evidence boundary |
| --- | --- | --- | --- |
| `FirthBinaryLogit` | Independent adjusted-score fit; `logistf` when available | Optional community `firthlogit` | Penalized objective; inverse-ordinary-Fisher covariance aligned separately |
| `PoissonRegressor` | `stats::glm(..., poisson)` | `poisson, exposure()` | Exact likelihood |
| `NegativeBinomialNB2` | `MASS::glm.nb` plus full observed Hessian | `nbreg, dispersion(mean) exposure()` | `log_alpha = -log(theta)` in R |
| `Tobit` | Gaussian `survival::survreg` | `tobit, ll(0)` | Scale transformed with the full Jacobian |
| `TruncatedRegression` | Independent truncated-normal likelihood | `truncreg, ll(0)` | R is an independent likelihood, not package parity |
| `IntervalRegression` | Gaussian interval `survival::survreg` | `intreg` | Open endpoints and scale are aligned explicitly |
| `GeometricDuration` | Grouped Binomial Logit | Expanded person-period `logit` | Exact pseudo-sample likelihood identity |
| `ExponentialDuration` | Exponential `survival::survreg` | `streg, distribution(exponential) time` | AFT scale |
| `WeibullDuration` | Weibull `survival::survreg` | `streg, distribution(weibull) time` | `shape = 1 / survreg_scale` |
| `GammaDuration` | Gamma GLM/full-likelihood identity on uncensored deaths | No exact ordinary-Gamma `streg` target | Predeclared Stata skip; generalized Gamma is not substituted |
| `RandomEffectsOrderedProbit` | `ordinal::clmm(link="probit", nAGQ=-20)` | `meoprobit`, 20 GH points | Numerical quadrature parity |
| `FixedEffectsOrderedLogit` | Blow-up `survival::clogit` | Blow-up `clogit` | BUC slopes/composite likelihood/entity sandwich only |

BUC conditions out thresholds and entity effects. Its application check must
not compare category probabilities, marginal effects, or ordinary ordered
information criteria, because those are not identified by this estimator.

## Frozen public applications

The preparation script reuses the hash-verified `lbw`, `tvsfpors`, and
`nlswork` analysis samples from the original real-data harness. Four additional
Stata Press release-19 files are downloaded over HTTPS and verified before any
fit:

| File | Application | SHA-256 |
| --- | --- | --- |
| [`rod93.dta`](https://www.stata-press.com/data/r19/rod93.dta) | Poisson and NB2 deaths with exposure | `023d1676ef716e320b49ebf0e0b31d259439161d3268da5b0b93022d138ddeab` |
| [`mroz87.dta`](https://www.stata-press.com/data/r19/mroz87.dta) | Left-censored and positive-sample hours | `2dbdabaad3f1c1a1239c1db4c01ca58b26bbfbbd7555bef5e2b07089bbda7c1c` |
| [`womenwage2.dta`](https://www.stata-press.com/data/r19/womenwage2.dta) | Official fictional interval-wage software example | `0ef5bad643a9b7562056d026fd1d2781c5bccb9d4c68a12fdc1b7ea216d9ea5a` |
| [`cancer.dta`](https://www.stata-press.com/data/r19/cancer.dta) | Right-censored drug-trial durations | `928e4449356bdd8d1466709c599a88f68e0aad2f60b3a3932869e3732fb962fb` |

Downloaded data and generated evidence live under
`validation/promoted/work/`, which is ignored by Git. Do not redistribute a
third-party source file merely because it was used in a local validation run.

The LBW, infant-mortality, Mroz labor-supply, cancer-duration, TVSFPORS, and
NLSWORK applications use empirical observations. Stata labels `womenwage2` as
fictional; it is retained only as the official native open-endpoint `intreg`
software example. The evidence index and any paper must preserve that
distinction rather than calling all twelve applications empirical.

## Python and R run

Install the Python validation extra and the four R dependencies first. When the
legacy R parity library exists, the runner prepends that pinned library automatically;
the generic command below is suitable for a fresh local check but may install newer R
package versions:

```powershell
python -m pip install -e ".[validation]"
Rscript --vanilla -e "install.packages(c('MASS','survival','numDeriv','ordinal'))"
```

An exact reproduction of the recorded evidence must retain the versions listed below,
not merely accept whatever versions are current at install time.

From the repository root:

```powershell
python validation/promoted/prepare_real_data.py `
  --output validation/promoted/work/real_data

Rscript --vanilla validation/promoted/run_parity.R `
  validation/promoted/work/real_data

python validation/promoted/compare_parity.py `
  validation/promoted/work/real_data
```

If the three legacy public-data fixtures are absent, first run the original
preparation step documented in `validation/stata/README.md`. A source cache can
be supplied to the promoted preparation script when network policy blocks a
fresh download:

```powershell
python validation/promoted/prepare_real_data.py `
  --output validation/promoted/work/real_data `
  --source-dir validation/promoted/work/source
```

Every cached source is still checked against its pinned SHA-256 before use.

### Recorded Python/R result

The run completed on 15 July 2026 with R 4.5.1, MASS 7.3-65,
survival 3.8-3, numDeriv 2016.8-1.1, and ordinal 2025.12-29. All 12 models fitted and all
**120 of 120** registered comparisons passed. The evidence comprised seven
industrial-package fits, three independent likelihood/score implementations,
and two likelihood or pseudo-sample identities.

| Quantity | Largest absolute Python/R difference |
| --- | ---: |
| Estimate | `5.0765757e-5` |
| Standard error | `2.5513079e-5` |
| Covariance entry | `8.7349028e-6` |
| Log likelihood | `4.6126814e-5` |
| Prediction | `1.5868239e-5` |

All differences were inside their preregistered, model-specific gates. The
manifest SHA-256 is
`86b589d3acfb245670e1317a05f9cc7542065ec69bde1ce8530f1d0d827b7516`;
the report, summary, and certificate hashes are, respectively,
`35178a26fa69a222100b829a0c0a99a45ff975386968f9c591e7767575e49833`,
`a8501fd02dad1b64689436858c92e3134bd4b9f0f74f4020e4f44ea5098083cd`,
and `bbd925871e37b2a5fffee31c40a4dce34bf249310cbfe7c1bda7aba07e699cae`.
Generated evidence remains outside version control and must be archived with
the exact repository revision when used in a release or paper.

## Manual Stata run

Stata is deliberately not installed or automated by the Python harness. After
preparation, run the promoted do-file manually and then the Stata comparator:

```stata
do "validation/promoted/limiteddepkit_real_data.do" ///
   "C:/path/to/limiteddepkit/validation/promoted/work/real_data"
```

```powershell
python validation/promoted/compare_stata.py `
  validation/promoted/work/real_data
```

Community commands are never installed automatically. Their absence is
recorded as a predeclared skip, not silently treated as a pass.

The completed manual Stata run passed **140/140** required checks with one
explicit skip: Gamma duration because Stata's generalized Gamma is not the
ordinary Gamma likelihood. Firth Binary Logit ran through optional `firthlogit`
and passed its aligned Stata checks. The Stata report, summary, and certificate hashes
are, respectively,
`483175035e64b96ebae834f2c2a69415ce4cd65b733b40ebe9f72f5cdd23a0e0`,
`c49d7a48d52cf91815e2decfe9624d0c4e538e166b1ee16851096ee749d32fd0`, and
`78c8745b535bcc0f1525f309ed084ac92cf6e9ca1e45eb6fee8130ca8f7dbd30`. The
[manual Stata guide](STATA.md) documents the raw exports, permitted skips,
canonical transformations, and restricted claims.

## Evidence contract

Preparation writes a manifest, analysis-ready CSV/DTA files, and canonical
Python estimates, covariance matrices, fit diagnostics, and selected
predictions. Each external runner invalidates only its own promoted-suite
outputs before fitting. A comparator must verify the manifest hashes and exact
model/parameter schema before checking numerical tolerances.

A successful comparator writes a detailed CSV report, a Markdown summary, and
a JSON certificate. The certificate is a provenance record, not a
cryptographic signature and not a universal claim about every dataset,
optimizer, covariance choice, or software version. Archive the manifest,
permitted data provenance, all generated results, console/Stata logs, package
versions, repository revision, report, and certificate together.
