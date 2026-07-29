"""Smoke-test an installed limiteddepkit distribution and its engine contracts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from typing import Any

import numpy as np
import pandas as pd

import limiteddepkit as ldk
from limiteddepkit.experimental import ConditionalLogit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    return parser.parse_args()


def _assert_exact(left: Any, right: Any) -> None:
    for name in ("all_params", "params", "covariance", "standard_errors"):
        if hasattr(left, name):
            np.testing.assert_array_equal(
                np.asarray(getattr(left, name)),
                np.asarray(getattr(right, name)),
            )
    for name in ("loglike", "converged", "inference_valid", "information_rank"):
        if hasattr(left, name):
            assert getattr(left, name) == getattr(right, name)


def _poisson_smoke(rng: np.random.Generator) -> None:
    design = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=160),
            "x2": rng.normal(size=160),
        }
    )
    mean = np.exp(design.to_numpy() @ np.array([0.1, 0.3, -0.2]))
    response = rng.poisson(mean)
    reference = ldk.PoissonRegressor().fit(design, response, engine="reference")
    accelerated = ldk.PoissonRegressor().fit(design, response, engine="accelerated")
    _assert_exact(reference, accelerated)


def _conditional_logit_smoke(rng: np.random.Generator) -> None:
    n_groups = 24
    n_alternatives = 4
    groups = np.repeat(np.arange(n_groups), n_alternatives)
    design = pd.DataFrame(
        {
            "price": rng.normal(size=len(groups)),
            "quality": rng.normal(size=len(groups)),
        }
    )
    utility = design.to_numpy() @ np.array([-0.8, 0.45]) + rng.gumbel(size=len(groups))
    chosen = np.zeros(len(groups), dtype=int)
    for group in range(n_groups):
        rows = np.flatnonzero(groups == group)
        chosen[rows[np.argmax(utility[rows])]] = 1
    estimator = ConditionalLogit(n_alts=n_alternatives)
    reference = estimator.fit(design, chosen, groups=groups, engine="reference")
    accelerated = estimator.fit(design, chosen, groups=groups, engine="accelerated")
    _assert_exact(reference, accelerated)
    np.testing.assert_array_equal(
        reference.predict_proba(design, groups=groups),
        accelerated.predict_proba(design, groups=groups),
    )


def _random_effects_smoke(rng: np.random.Generator, estimator: Any) -> None:
    n_entities = 16
    n_periods = 4
    entity = np.repeat(np.arange(n_entities), n_periods)
    design = pd.DataFrame(
        {
            "x1": rng.normal(size=len(entity)),
            "x2": rng.normal(size=len(entity)),
        }
    )
    effects = np.repeat(rng.normal(scale=0.45, size=n_entities), n_periods)
    latent = design.to_numpy() @ np.array([0.55, -0.3]) + effects + rng.logistic(size=len(entity))
    response = np.digitize(latent, bins=np.array([-0.5, 0.65]))
    kwargs = {"entity": entity, "quadrature_points": 6}
    reference = estimator.fit(design, response, engine="reference", **kwargs)
    accelerated = estimator.fit(design, response, engine="accelerated", **kwargs)
    _assert_exact(reference, accelerated)


def main() -> None:
    args = _parse_args()
    distribution_version = importlib.metadata.version("limiteddepkit")
    assert distribution_version == args.expected_version
    assert ldk.__version__ == args.expected_version

    rng = np.random.default_rng(20260729)
    binary_design = pd.DataFrame({"const": 1.0, "x": rng.normal(size=120)})
    binary_response = rng.binomial(
        1,
        1.0 / (1.0 + np.exp(-(binary_design.to_numpy() @ np.array([-0.2, 0.6])))),
    )
    binary = ldk.BinaryLogit().fit(binary_design, binary_response)
    assert binary.converged and binary.inference_valid

    ordered_design = pd.DataFrame({"x1": rng.normal(size=140), "x2": rng.normal(size=140)})
    ordered_latent = ordered_design.to_numpy() @ np.array([0.5, -0.25]) + rng.logistic(size=140)
    ordered_response = np.digitize(ordered_latent, bins=np.array([-0.6, 0.7]))
    ordered = ldk.OrderedLogit().fit(ordered_design, ordered_response)
    assert ordered.converged and ordered.inference_valid

    _poisson_smoke(rng)
    _conditional_logit_smoke(rng)
    _random_effects_smoke(rng, ldk.RandomEffectsOrderedLogit())
    _random_effects_smoke(rng, ldk.RandomEffectsOrderedProbit())

    print(
        json.dumps(
            {
                "result": "PASS",
                "version": distribution_version,
                "module": str(ldk.__file__),
                "checks": [
                    "binary_logit",
                    "ordered_logit",
                    "poisson_reference_accelerated",
                    "conditional_logit_reference_accelerated",
                    "random_effects_ordered_logit_reference_accelerated",
                    "random_effects_ordered_probit_reference_accelerated",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
