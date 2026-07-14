import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from limiteddepkit.experimental import ConditionalLogit


def make_conditional_data(seed=42, n_choice_sets=1_200, n_alts=4):
    rng = np.random.default_rng(seed)
    n_rows = n_choice_sets * n_alts
    groups = np.repeat([f"chooser-{index}" for index in range(n_choice_sets)], n_alts)
    alternatives = np.tile([f"alternative-{index}" for index in range(n_alts)], n_choice_sets)
    X = pd.DataFrame(
        {
            "price": rng.normal(size=n_rows),
            "quality": rng.normal(size=n_rows),
        },
        index=pd.Index([f"row-{index}" for index in range(n_rows)]),
    )
    coefficients = np.array([-0.70, 0.45])
    utilities = (X.to_numpy() @ coefficients).reshape(n_choice_sets, n_alts)
    probabilities = softmax(utilities, axis=1)
    selected = np.array([rng.choice(n_alts, p=row) for row in probabilities])
    choice = np.zeros(n_rows, dtype=int)
    choice[np.arange(n_choice_sets) * n_alts + selected] = 1
    return X, choice, groups, alternatives, coefficients


def test_conditional_logit_recovers_slopes_and_observed_information():
    X, choice, groups, alternatives, coefficients = make_conditional_data()
    result = ConditionalLogit(n_alts=4).fit(
        X,
        choice,
        groups=groups,
        alternatives=alternatives,
    )

    assert result.converged
    assert result.inference_valid
    assert result.nobs == result.n_choice_sets == 1_200
    assert result.n_rows == 4_800
    assert result.n_alts == 4
    assert result.params.to_numpy() == pytest.approx(coefficients, abs=0.08)
    assert result.information_rank == result.n_params == 2
    assert result.covariance.to_numpy() == pytest.approx(
        result.covariance.to_numpy().T, abs=1e-12
    )
    assert np.all(np.isfinite(result.standard_errors))
    assert np.all(result.standard_errors > 0)
    assert result.summary_frame().shape == (2, 4)
    assert result.conf_int().shape == (2, 2)


def test_conditional_probabilities_and_predictions_use_arbitrary_labels():
    X, choice, groups, alternatives, _ = make_conditional_data(n_choice_sets=350, n_alts=3)
    result = ConditionalLogit(n_alts=3).fit(
        X, choice, groups=groups, alternatives=alternatives
    )

    probabilities = result.predict_proba(X, groups=groups)
    predictions = result.predict(X, groups=groups, alternatives=alternatives)

    assert probabilities.index.equals(X.index)
    probability_sums = probabilities.groupby(pd.Series(groups, index=X.index)).sum()
    assert probability_sums.to_numpy() == pytest.approx(1.0, abs=1e-12)
    assert predictions.index.tolist() == list(dict.fromkeys(groups))
    assert set(predictions) <= set(alternatives)


def test_conditional_logit_supports_unequal_choice_set_sizes():
    rng = np.random.default_rng(720)
    sizes = np.resize(np.array([2, 3, 5]), 450)
    groups = np.concatenate(
        [np.repeat(f"set-{group}", size) for group, size in enumerate(sizes)]
    )
    alternatives = np.concatenate(
        [[f"option-{position}" for position in range(size)] for size in sizes]
    )
    X = pd.DataFrame(
        {"cost": rng.normal(size=len(groups)), "comfort": rng.normal(size=len(groups))}
    )
    coefficients = np.array([-0.55, 0.35])
    choice = np.zeros(len(groups), dtype=int)
    offset = 0
    for size in sizes:
        index = np.arange(offset, offset + size)
        probabilities = softmax(X.iloc[index].to_numpy() @ coefficients)
        choice[index[rng.choice(size, p=probabilities)]] = 1
        offset += size

    result = ConditionalLogit().fit(
        X, choice, groups=groups, alternatives=alternatives
    )

    assert result.converged
    assert result.n_alts is None
    assert result.params.to_numpy() == pytest.approx(coefficients, abs=0.12)
    assert result.predict(X, groups=groups, alternatives=alternatives).shape == (450,)


def test_conditional_logit_enforces_choice_set_and_identification_contracts():
    X, choice, groups, alternatives, _ = make_conditional_data(n_choice_sets=80, n_alts=3)

    no_choice = choice.copy()
    no_choice[:3] = 0
    with pytest.raises(ValueError, match="exactly one chosen"):
        ConditionalLogit(3).fit(X, no_choice)

    two_choices = choice.copy()
    two_choices[:3] = [1, 1, 0]
    with pytest.raises(ValueError, match="exactly one chosen"):
        ConditionalLogit(3).fit(X, two_choices)

    with pytest.raises(ValueError, match="not identified"):
        ConditionalLogit(3).fit(X.assign(constant=1.0), choice)

    duplicate_alternatives = alternatives.copy()
    duplicate_alternatives[:3] = "duplicate"
    with pytest.raises(ValueError, match="unique within"):
        ConditionalLogit(3).fit(
            X,
            choice,
            groups=groups,
            alternatives=duplicate_alternatives,
        )

    with pytest.raises(ValueError, match="groups is required"):
        ConditionalLogit().fit(X, choice)
    with pytest.raises(ValueError, match="at least two"):
        ConditionalLogit(1)

    result = ConditionalLogit(3).fit(X, choice)
    with pytest.raises(ValueError, match="columns must match"):
        result.predict(X[["quality", "price"]])
