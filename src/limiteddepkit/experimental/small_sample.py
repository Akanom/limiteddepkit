"""Compatibility and provisional small-sample response estimators.

Firth Binary Logit is retained here as a compatibility alias after promotion
to :mod:`limiteddepkit.small_sample`. Ridge estimators remain provisional.
"""

from .._small_sample import (
    FirthBinaryLogit,
    FirthBinaryLogitResult,
    RidgeBinaryLogit,
    RidgeBinaryLogitResult,
    RidgeOrderedLogit,
    RidgeOrderedLogitResult,
)

__all__ = [
    "FirthBinaryLogit",
    "FirthBinaryLogitResult",
    "RidgeBinaryLogit",
    "RidgeBinaryLogitResult",
    "RidgeOrderedLogit",
    "RidgeOrderedLogitResult",
]
