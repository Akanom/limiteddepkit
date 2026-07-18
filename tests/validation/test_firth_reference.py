"""Frozen parity evidence against firthmodels 0.7.2's NumPy backend."""

import numpy as np
import pandas as pd
import pytest

from limiteddepkit.small_sample import FirthBinaryLogit


@pytest.mark.validation
def test_firth_coefficients_objective_and_profiles_match_firthmodels() -> None:
    x = np.array(
        [-2.4, -2.0, -1.7, -1.4, -1.1, -0.9, -0.7, -0.4, -0.2, 0.0,
         0.2, 0.4, 0.7, 0.9, 1.1, 1.3, 1.6, 1.8, 2.1, 2.5]
    )
    z = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    outcomes = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    design = pd.DataFrame({"const": 1.0, "x": x, "z": z})

    result = FirthBinaryLogit().fit(design, outcomes, tolerance=1e-10)

    expected_coefficients = np.array([0.02655061, 1.31419500, -0.30429194])
    expected_intervals = np.array(
        [
            [-1.65917446, 1.73495957],
            [0.41578602, 2.84103243],
            [-2.75683245, 1.93721490],
        ]
    )
    assert result.params.to_numpy() == pytest.approx(expected_coefficients, abs=2e-8)
    assert result.penalized_loglike == pytest.approx(-6.820606164775911, abs=2e-11)
    assert result.conf_int(tolerance=1e-8).to_numpy() == pytest.approx(
        expected_intervals, abs=2e-6
    )
