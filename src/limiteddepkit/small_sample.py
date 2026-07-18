"""Stable small-sample and separation-resistant estimators.

Only Firth's mean-bias-reduced Binary Logit currently satisfies the stable
inferential contract. Ridge response estimators remain under
``limiteddepkit.experimental`` because their penalty-selection and covariance
contracts are still provisional.
"""

from ._small_sample import FirthBinaryLogit, FirthBinaryLogitResult

__all__ = ["FirthBinaryLogit", "FirthBinaryLogitResult"]
