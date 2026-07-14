# Extraction staging

This directory preserves code removed from the installable `limiteddepkit`
distribution because it is not a limited-dependent-variable model.

- `treatment_effect.py` is a legacy homoskedastic linear 2SLS implementation.
  Its eventual destination is a causal/IV package.
- `gaussian_mixture_regression.py` is an iid Gaussian finite-mixture
  regression. Its eventual destination is a mixture/regime-model package.

The files are snapshots, not importable `limiteddepkit` modules, and are
excluded from wheels and source distributions. Their former tests are retained
under `tests/` as migration evidence but are not part of the maintained test
suite. A destination package should adapt imports, establish its own result
contract, and re-enable those tests before exposing either estimator.
