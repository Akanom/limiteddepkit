# Extraction staging

This directory preserves code removed from the installable `limiteddepkit`
distribution because it is not a limited-dependent-variable model.

- `gaussian_mixture_regression.py` is an iid Gaussian finite-mixture
  regression. Its eventual destination is a mixture/regime-model package.

The remaining files are snapshots, not importable `limiteddepkit` modules, and
are excluded from wheels and source distributions. Their former tests are
retained under `tests/` as migration evidence but are not part of the maintained
test suite. The legacy 2SLS snapshot and test were deleted after the maintained,
independently contracted IV implementation was established in CauseKit.
