import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logsumexp, softmax

from benchmarks.benchmark_estimators import _environment
from limiteddepkit import RandomEffectsOrderedLogit, RandomEffectsOrderedProbit
from limiteddepkit import __version__ as limiteddepkit_version
from limiteddepkit._performance import rowwise_logsumexp
from limiteddepkit.experimental import ConditionalLogit


def _assert_exact_result(left, right) -> None:
    for name in (
        "all_params",
        "params",
        "covariance",
        "standard_errors",
        "zstats",
        "pvalues",
    ):
        if hasattr(left, name):
            np.testing.assert_array_equal(
                np.asarray(getattr(left, name)),
                np.asarray(getattr(right, name)),
            )
    for name in (
        "loglike",
        "converged",
        "inference_valid",
        "information_rank",
        "score_norm",
        "scaled_score_norm",
    ):
        if hasattr(left, name):
            assert getattr(left, name) == getattr(right, name)
    for name in ("x", "fun", "jac", "nit", "nfev", "njev", "status", "success"):
        if hasattr(left.optimizer_result, name):
            np.testing.assert_array_equal(
                np.asarray(getattr(left.optimizer_result, name)),
                np.asarray(getattr(right.optimizer_result, name)),
            )


def _conditional_problem(
    *, seed: int = 4_701, sizes: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if sizes is None:
        sizes = np.repeat(4, 180)
    groups = np.concatenate([np.repeat(group, size) for group, size in enumerate(sizes)])
    design = pd.DataFrame(
        {
            "price": rng.normal(size=len(groups)),
            "quality": rng.normal(size=len(groups)),
        }
    )
    choice = np.zeros(len(groups), dtype=int)
    offset = 0
    for size in sizes:
        rows = np.arange(offset, offset + size)
        probabilities = softmax(design.iloc[rows].to_numpy() @ np.array([-0.7, 0.45]))
        choice[rows[rng.choice(size, p=probabilities)]] = 1
        offset += size
    return design, choice, groups


@pytest.mark.parametrize(
    "sizes",
    [np.repeat(4, 180), np.resize(np.array([2, 3, 5]), 180)],
)
def test_conditional_logit_accelerated_engine_is_exact(sizes):
    design, choice, groups = _conditional_problem(sizes=sizes)
    estimator = ConditionalLogit(4 if np.all(sizes == 4) else None)

    default = estimator.fit(design, choice, groups=groups)
    reference = estimator.fit(design, choice, groups=groups, engine="reference")
    accelerated = estimator.fit(design, choice, groups=groups, engine="accelerated")

    _assert_exact_result(default, reference)
    _assert_exact_result(reference, accelerated)
    np.testing.assert_array_equal(
        reference.predict_proba(design, groups=groups).to_numpy(),
        accelerated.predict_proba(design, groups=groups).to_numpy(),
    )
    assert default.engine == reference.engine == "reference"
    assert accelerated.engine == "accelerated"


def _panel_ordinal_problem(
    *, seed: int = 9_771, n_entities: int = 24, n_periods: int = 4
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    design = pd.DataFrame(
        {
            "x1": rng.normal(size=len(entity)),
            "x2": rng.normal(size=len(entity)),
        }
    )
    random_intercepts = rng.normal(scale=0.6, size=n_entities)
    linear = design.to_numpy() @ np.array([0.65, -0.4]) + random_intercepts[entity]
    cumulative = expit(np.array([-0.6, 0.7])[None, :] - linear[:, None])
    probabilities = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)[:, 0], 1.0 - cumulative[:, 1]]
    )
    response = np.array([rng.choice(3, p=row) for row in probabilities])
    return design, response, entity


@pytest.mark.parametrize("estimator_type", [RandomEffectsOrderedLogit, RandomEffectsOrderedProbit])
def test_random_effects_ordered_accelerated_engine_is_exact(estimator_type):
    design, response, entity = _panel_ordinal_problem()
    estimator = estimator_type()
    kwargs = {"entity": entity, "quadrature_points": 6}

    default = estimator.fit(design, response, **kwargs)
    reference = estimator.fit(design, response, engine="reference", **kwargs)
    accelerated = estimator.fit(design, response, engine="accelerated", **kwargs)

    _assert_exact_result(default, reference)
    _assert_exact_result(reference, accelerated)
    assert default.engine == reference.engine == "reference"
    assert accelerated.engine == "accelerated"


def test_random_effects_acceleration_is_exact_for_unbalanced_unsorted_entities():
    design, response, entity = _panel_ordinal_problem(n_entities=30, n_periods=5)
    retained = np.ones(len(entity), dtype=bool)
    retained[::11] = False
    permutation = np.random.default_rng(581).permutation(np.flatnonzero(retained))
    design = design.iloc[permutation].reset_index(drop=True)
    response = response[permutation]
    entity = np.asarray([f"entity-{value}" for value in entity[permutation]])
    estimator = RandomEffectsOrderedLogit()
    kwargs = {"entity": entity, "quadrature_points": 6}

    reference = estimator.fit(design, response, engine="reference", **kwargs)
    accelerated = estimator.fit(design, response, engine="accelerated", **kwargs)

    _assert_exact_result(reference, accelerated)


def test_rowwise_logsumexp_is_exactly_scipy_rowwise():
    rng = np.random.default_rng(63)
    for width in (2, 3, 4, 8, 16):
        values = rng.normal(size=(200, width)) * 25.0
        expected = np.array([logsumexp(row) for row in values])
        np.testing.assert_array_equal(rowwise_logsumexp(values), expected)


def test_benchmark_environment_reports_source_version():
    assert _environment()["limiteddepkit"] == limiteddepkit_version


@pytest.mark.parametrize(
    "estimator,fit_args",
    [
        (
            ConditionalLogit(4),
            lambda: (*_conditional_problem(),),
        ),
        (
            RandomEffectsOrderedLogit(),
            lambda: (*_panel_ordinal_problem(),),
        ),
    ],
)
def test_grouped_estimators_reject_unknown_engine(estimator, fit_args):
    design, response, third = fit_args()
    kwargs = (
        {"groups": third}
        if isinstance(estimator, ConditionalLogit)
        else {"entity": third, "quadrature_points": 5}
    )
    with pytest.raises(ValueError, match="engine"):
        estimator.fit(design, response, engine="auto", **kwargs)
    with pytest.raises(TypeError, match="engine"):
        estimator.fit(design, response, engine=None, **kwargs)
