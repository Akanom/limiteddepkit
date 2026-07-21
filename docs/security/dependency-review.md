# Dependency Review

Review date: 2026-07-21

Owner: Maintainer

Reassessment: every dependency change and at least quarterly

## Runtime dependencies

| Package | Purpose | Decision |
|---|---|---|
| NumPy | Numerical arrays and linear algebra | Mandatory; `>=1.23,<3` |
| pandas | Labelled data and result tables | Mandatory; `>=1.5,<4` |
| SciPy | Optimization and statistical distributions | Mandatory; `>=1.9,<2` |
| Matplotlib | Ordinal-model plots | Optional `plots` extra with lazy import |

Scikit-learn and Statsmodels remain validation extras; PyTorch remains an
explicit neural extra; Universal Output Hub remains an output integration
extra. Core package import does not require any of them.

An OSV-backed audit of the latest resolvable runtime graph found no known
vulnerability on 2026-07-21. This point-in-time result is enforced in CI and
must be reassessed when dependency resolution changes.

## Expected capabilities

Native NumPy/SciPy code is expected. Network access exists only in controlled
validation-data preparation scripts with hash and provenance checks. PyTorch's
`network.eval()` is model mode selection, not Python dynamic execution;
`ast.literal_eval()` in tests is constrained parsing, not built-in `eval()`.
