# Dependency review

Review date: 2026-07-28

Owner: Maintainer

Reassessment: every dependency change and at least quarterly

## Dependency boundaries

| Package | Purpose | Installed scope |
|---|---|---|
| NumPy | Numerical arrays and linear algebra | Mandatory runtime; `>=1.23,<3` |
| pandas | Labelled data and results | Mandatory runtime; `>=1.5,<4` |
| SciPy | Optimization and distributions | Mandatory runtime; `>=1.9,<2` |
| Matplotlib | Diagnostic and marginal-effect plots | Optional `plots` and test/development only |
| Ruff | Linting and formatting | Test/development only |
| build | PEP 517 distribution construction | Release/development only |

Scikit-learn and Statsmodels remain validation extras; PyTorch is an explicit
neural extra; Universal Output Hub is an output integration extra. None is
required for the core package import.

## Scanner findings reviewed

The screenshots are capability heuristics rather than confirmed vulnerabilities
in limiteddepkit. Because the dashboard did not expose the exact downloaded hash,
no version-specific suppression is authorized. Official PyPI hashes are recorded
below so the artifact can be matched during scanner review.

| Finding | Exact artifact | Official SHA-256 | Disposition and containment |
|---|---|---|---|
| Opaque/obfuscated archive | `numpy-2.5.1.tar.gz` | `a48a113e6afea91f5608793bafa7ef2ad481fefbda87ec5069f483de61cb9fa3` | `MONITOR`. Nested archives are upstream Meson test fixtures. Prefer official wheels and do not suppress before hash comparison. |
| Potential `f2py` `eval` path | `numpy-2.5.1-cp312-cp312-macosx_10_13_x86_64.whl` | `2c889b56fe48b1018f764b0eec8df59ab654e9148aa91faa12596043500de277` | `REVIEWED`. limiteddepkit does not call F2PY. Never execute attacker-writable F2PY configuration. |
| Shell access | `build-1.5.0-py3-none-any.whl` | `13f3eecb844759ab66efec90ca17639bbf14dc06cb2fdf37a9010322d9c50a6f` | `EXPECTED, RELEASE-ONLY`. Backend invocation is the build frontend's declared function; it is absent from runtime requirements. |
| Install scripts | `ruff-0.16.0.tar.gz` | `e460aafd5495ec89efaa6ced2e4a9a581116451e1c88b9d37ef497e0f8e93982` | `EXPECTED, DEVELOPMENT-ONLY`. The flagged Rust build files and test fixture occur in the sdist. Prefer the official wheel. |
| Network access | `matplotlib-3.11.1-cp311-cp311-macosx_10_12_x86_64.whl` | `b7cf158e7add54a8d51ac9b5a84abd6d4e13ed4951b4f25f1c5139f41c2addb2` | `EXPECTED, OPTIONAL`. Plotting is lazy and local; estimator and package imports do not fetch remote resources. |

The screenshots' `pydynpd`, `linearmodels`, and mypy artifacts apply to
systemgmmkit's optional development/backend graph, not limiteddepkit's declared
dependencies. They are reviewed in that repository rather than being added here.
The permissive development bounds may resolve newer versions than the currently
committed hash-locked graphs (NumPy 2.5.1 versus the current 2.4.6 lock); lock
updates remain separate reviewed changes.

## Controls

- Reusable governance workflows run dependency audit, secret scan, distribution
  inspection, and hashed-lock verification.
- Runtime, validation, neural, plotting, output, and release capabilities remain
  separate extras or requirement graphs.
- Native or source artifacts must match official PyPI hashes; wheels are preferred
  when no source build is required.
- No scanner control is disabled and no finding is recorded as an accepted risk.
- Core import and estimator execution require neither network nor shell access.
