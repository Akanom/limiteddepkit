# Extraction staging

This directory preserves code removed from the installable `limiteddepkit`
distribution because it is not a limited-dependent-variable model.

- `gaussian_mixture_regression.py` is an iid Gaussian finite-mixture
  regression. Its eventual destination is a mixture/regime-model package.

The remaining file is a snapshot, not an importable `limiteddepkit` module, and is
excluded from wheels and source distributions. Its former tests are retained under
`tests/` as migration evidence but are not part of the maintained test suite.

The ordinary homoskedastic 2SLS `TreatmentEffect` migration is complete. CauseKit owns
the maintained `IV2SLS` replacement and its numerical migration contract, so the obsolete
LimitedDepKit source and test snapshots were removed rather than duplicated.
