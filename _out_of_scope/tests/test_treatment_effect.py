import numpy as np
import pandas as pd
import pytest

from limiteddepkit.treatment_effect import TreatmentEffect


def test_treatment_effect_fits_and_predicts():
    rng = np.random.default_rng(45)
    n_obs = 400

    # Exogenous instruments and covariates
    Z = pd.DataFrame({
        "const": 1.0,
        "z1": rng.normal(size=n_obs),
        "x1": rng.normal(size=n_obs),
    })

    # Treatment equation: T = 1 if Z*gamma + error > 0
    gamma = np.array([0.0, 0.4, 0.2])
    treatment_latent = Z.to_numpy() @ gamma + rng.normal(0, 1.0, n_obs)
    T = (treatment_latent > 0).astype(int)

    # Outcome equation: Y = T*beta1 + X*beta2 + error
    beta1 = 2.5  # treatment effect
    beta2_x1 = 0.8
    Y = beta1 * T + beta2_x1 * Z["x1"].to_numpy() + rng.normal(0, 0.6, n_obs)

    # Prepare data for 2SLS
    X_endog = pd.DataFrame({"T": T})  # Endogenous treatment
    X_exog = pd.DataFrame({
        "const": 1.0,
        "x1": Z["x1"],
    })
    Z_instr = pd.DataFrame({
        "const": 1.0,
        "z1": Z["z1"],
        "x1": Z["x1"],
    })

    with pytest.warns(FutureWarning, match="outside limiteddepkit"):
        model = TreatmentEffect()
    result = model.fit(Y, X_endog, X_exog, Z_instr)

    assert result.converged
    assert result.nobs == n_obs
    predictions = result.predict(X_endog, X_exog)
    assert predictions.shape == (n_obs,)
    assert np.all(np.isfinite(predictions.to_numpy()))


def test_treatment_effect_is_not_part_of_the_experimental_namespace():
    import limiteddepkit.experimental as experimental

    assert "TreatmentEffect" not in experimental.__all__
    assert not hasattr(experimental, "TreatmentEffect")
