# Cross-software parity evidence index

This is the committed, data-free index for the completed `limiteddepkit`
Python–Stata and Python–R comparisons regenerated on 14 July 2026. It records
the exact outcome and SHA-256 identity of the generated evidence without
redistributing the ignored work directories or third-party data.

## Certified outcome

| Track | Implementation | Models | Result | Checks | Failures | Skips |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Controlled synthetic certification | Stata | 8 | **PASS** | 82 | 0 | 0 |
| Controlled synthetic certification | R | 8 | **PASS** | 110 | 0 | 0 |
| Frozen real-data application | Stata | 8 | **PASS** | 82 | 0 | 0 |
| Frozen real-data application | R | 8 | **PASS** | 110 | 0 | 0 |

All four comparisons cover Binary Logit, Binary Probit, Ordered Logit,
Ordered Probit, Generalized Ordered Logit, Partial Proportional Odds, static
random-effects Ordered Logit, and dynamic random-effects Ordered Logit.

Only the controlled track supports the stated benchmark-specific parity
claim. The real-data track is application evidence on frozen public example
data and does not broaden that claim. Neither track establishes universal
equality for other datasets, specifications, estimands, optimizers, covariance
targets, or quadrature implementations.

## Evidence identities

The report and certificate paths below are relative to each ignored work root.
The certificates additionally contain hashes for every Python reference,
prepared input, and external-software export.

### Controlled synthetic certification

Manifest SHA-256:
`b74f790dac0d25c3d0ef872ed43c5941b4559e5974d412bd447a85ce06906d38`

| Implementation | Report SHA-256 | Certificate SHA-256 |
| --- | --- | --- |
| Stata | `89f62fc52201f760609c43e293eeffe56e1e137c56290b7934d7dbecdbd408df` | `aa51c2dd03afc3548baf2053851eccd15836c408bebda6348102c9bfbd7d05dd` |
| R | `0b18681b41c286c94a6592060d3e7bce3635244c23a5240773aae194356775d5` | `6b4001596c8597ddafd296d3622adfc53ec91cbacc13ac239d1c4fadbfb059bd` |

### Frozen real-data application

Manifest SHA-256:
`2780339b9e02d6b8917c9c33edad10422811edbe61b7d4ccdfb9cda8d143cea3`

| Implementation | Report SHA-256 | Certificate SHA-256 |
| --- | --- | --- |
| Stata | `51e8ca82fe0fa936ca8aeebb43bcf501377641f492cd8cc49df2a2915d053cfb` | `8f7d4647a4796f169c02ffc5013d41daebee873c7621cb91f4c542c63b628651` |
| R | `a65566f6c2094fd0c6e4d2f582ecde6ff3084b2ce8c41d0b3841f436f2dbf6cb` | `1333bff29b5024462d427ce23215b9cdc080b3589bfc04e3bcf2333e21dfa5d1` |

## Observed numerical envelope

These are the largest absolute differences across all eight models in each
completed report. The declared gates remain in the Stata and R harness guides;
every value below is inside its model-family gate.

| Track | Implementation | Estimate | Standard error | Covariance | Log likelihood | Probability |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Controlled | Stata | `4.16e-6` | `3.03e-7` | `2.14e-7` | `3.53e-9` | `1.56e-6` |
| Controlled | R | `1.02e-5` | `1.26e-5` | `3.85e-6` | `1.53e-4` | `3.07e-6` |
| Real data | Stata | `1.47e-5` | `2.11e-6` | `2.36e-6` | `2.58e-9` | `2.77e-6` |
| Real data | R | `3.19e-4` | `2.05e-4` | `2.29e-4` | `5.35e-4` | `5.05e-5` |

## Recorded environment

- Package: `limiteddepkit` 0.1.0a1.
- Stata: release 17 with `gologit2` 3.2.8.
- R: 4.5.1 with MASS 7.3-65, Matrix 1.7-3, VGAM 1.1-14,
  jsonlite 2.0.0, nlme 3.1-168, numDeriv 2016.8-1.1, ordinal
  2025.12-29, and ucminf 1.2.3.
- Quadrature: 12 points for controlled panel models and 20 points for the
  real-data panel applications.
- Pooled ordered optimization: `maxiter=5000`, function tolerance `1e-13`.
- Panel optimizer tolerance: `1e-12`.

## Archival boundary

This index is not a cryptographic signature and is not a substitute for the
complete evidence bundle. A release or publication archive must retain the
matching manifest, analysis-ready data or permitted provenance, Python
references, Stata log and exports, R exports, comparator reports and
certificates, exact dependency binaries where redistribution permits, and the
repository revision containing this file. Never combine artifacts from the two
manifest hashes above or from a later run.

Reproduction commands, parameter mappings, estimand alignment, tolerances, and
the full evidence contract are documented in the [Stata guide](stata/README.md)
and [R guide](r/README.md).

## Separate promoted-family application suite

The later promotion suite is intentionally separate from the eight-model
certificate above. On 15 July 2026, its Python/R public-data application track
passed **120 of 120** declared checks across 12 models:

- Firth Binary Logit;
- Poisson and NB2;
- Tobit, truncated regression, and interval regression;
- Geometric, Exponential, Weibull, and Gamma duration;
- Random-effects Ordered Probit; and
- BUC Fixed-effects Ordered Logit.

| Implementation | Models | Result | Checks | Failures |
| --- | ---: | --- | ---: | ---: |
| R 4.5.1 | 12 | **PASS** | 120 | 0 |
| Stata | 11 exact/aligned runs plus one explicit Gamma skip | **PASS** | 140 | 0 |

The R evidence is classified per estimand: seven industrial-package fits,
three independent likelihood/score implementations, and two exact likelihood/pseudo-sample
identities. It is application evidence, not a new controlled certification
claim. The LBW, infant-mortality, Mroz labor-supply, cancer-duration,
TVSFPORS, and NLSWORK inputs are empirical. Stata's official `womenwage2`
interval-regression fixture is explicitly fictional and must not be described
as respondent data.

The largest observed Python/R differences were `5.0765757e-5` for an
estimate, `2.5513079e-5` for a standard error, `8.7349028e-6` for a covariance
entry, `4.6126814e-5` for log likelihood, and `1.5868239e-5` for a prediction.
All were within the model-specific gates registered before comparison.

The promoted Stata pass has one explicit skip: Gamma duration because Stata
`streg, distribution(ggamma)` is not the ordinary Gamma likelihood implemented
by `GammaDuration`. Firth Binary Logit ran through the optional `firthlogit`
command and passed its aligned coefficient, prediction, and inverse-ordinary-
Fisher covariance checks. Exponential and Weibull `streg`
checks certify coefficients, covariance, event/sample counts, convergence, and
declared prediction targets; Stata's survival-time log-likelihood constants are
not used as strict fit-statistic parity targets.

Promoted-suite manifest SHA-256:
`86b589d3acfb245670e1317a05f9cc7542065ec69bde1ce8530f1d0d827b7516`

| Artifact | SHA-256 |
| --- | --- |
| R comparison report | `35178a26fa69a222100b829a0c0a99a45ff975386968f9c591e7767575e49833` |
| R comparison summary | `a8501fd02dad1b64689436858c92e3134bd4b9f0f74f4020e4f44ea5098083cd` |
| R parity certificate | `bbd925871e37b2a5fffee31c40a4dce34bf249310cbfe7c1bda7aba07e699cae` |
| Stata comparison report | `483175035e64b96ebae834f2c2a69415ce4cd65b733b40ebe9f72f5cdd23a0e0` |
| Stata comparison summary | `c49d7a48d52cf91815e2decfe9624d0c4e538e166b1ee16851096ee749d32fd0` |
| Stata parity certificate | `78c8745b535bcc0f1525f309ed084ac92cf6e9ca1e45eb6fee8130ca8f7dbd30` |

The complete reproduction and evidence contract is documented in the
[promoted-family guide](promoted/README.md). Generated data and evidence remain
ignored by Git and must be archived separately with the repository revision.
