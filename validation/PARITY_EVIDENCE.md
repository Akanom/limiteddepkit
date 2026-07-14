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
