"""Stable count-regression family.

Poisson and NB2 have completed the package's foundational promotion gates:
industrial coefficient/covariance parity, offset and exposure support, explicit
weight semantics, robust covariance, schema-safe prediction, and the common
result contract. Zero-inflated and hurdle models remain available from
``limiteddepkit.experimental`` while their weighted/covariance gates are open.
"""

from ..negative_binomial import (
    NegativeBinomial,
    NegativeBinomialNB2,
    NegativeBinomialResult,
)
from ..poisson import PoissonRegressor, PoissonResult

__all__ = [
    "NegativeBinomial",
    "NegativeBinomialNB2",
    "NegativeBinomialResult",
    "PoissonRegressor",
    "PoissonResult",
]
