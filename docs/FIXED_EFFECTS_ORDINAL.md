# Fixed-effects ordinal panels

`limiteddepkit` separates three static-panel estimands that are often blurred together:

| Estimator | Link and method | What is identified |
|---|---|---|
| `FixedEffectsOrderedLogit` | Logit; blow-up-and-cluster conditional composite likelihood | Common slopes under arbitrary correlation between regressors and time-invariant entity heterogeneity |
| `FixedEffectsOrderedProbit` | Probit; unconditional entity effects with split-panel-jackknife bias correction | Bias-corrected common slopes and thresholds in balanced large-`N`, large-`T` panels |
| `RandomEffectsOrderedLogit` / `RandomEffectsOrderedProbit` | Gaussian random intercept integrated by quadrature | Slopes, thresholds, and a random-effect distribution under the stated distributional assumptions |

The fixed-effects estimators are not aliases for random intercepts or for an
ordered model with a large dummy matrix.

Dynamic fixed-effects Ordered Logit is a fourth, separate estimand with an
exact four-observation conditional-history design. It remains experimental and
is documented in the
[dynamic fixed-effects guide](DYNAMIC_FIXED_EFFECTS_ORDINAL.md).

## Fixed-effects Ordered Logit

The stable `FixedEffectsOrderedLogit` estimator implements the
blow-up-and-cluster (BUC) estimator of Baetschmann, Staub, and Winkelmann. It
dichotomizes the ordered response at every cutoff, conditions each binary
Logit clone on its entity success count, sums the conditional log likelihoods,
and clusters the composite-likelihood covariance by the original entity.

```python
from limiteddepkit import FixedEffectsOrderedLogit

result = FixedEffectsOrderedLogit().fit(
    X,
    y,
    entity=person_id,
    category_order=["low", "middle", "high"],
)
print(result.params)
print(result.odds_ratios())
```

This procedure conditions out the entity effects and cutoffs. It consequently
does **not** identify their values, category probabilities, marginal effects,
or an ordinary ordered-model likelihood/AIC. `linear_index(X)` returns only
`X beta`; it is not a fitted latent index containing entity effects. Constants,
time-invariant regressors, and other within-entity collinearity are rejected.
Entities that never cross a particular cutoff do not contribute to that clone.
`score_norm` and `scaled_score_norm` expose BUC stationarity. The optimizer
gradient tolerance and the independent certification threshold are capped, so
a deliberately loose user tolerance cannot certify the starting coefficients.

## Fixed-effects Ordered Probit

There is no Ordered-Probit counterpart to the Logit conditional likelihood.
`FixedEffectsOrderedProbit` is therefore deliberately experimental. It fits
the unconditional entity-effects Ordered Probit to the full panel and its two
time halves, then reports

```text
2 * full_panel - (first_half + second_half) / 2
```

for common slopes and thresholds. This split-panel jackknife removes the
leading incidental-parameter bias under its large-panel assumptions; it does
not make short-panel nuisance effects consistently estimable.

```python
from limiteddepkit.experimental import FixedEffectsOrderedProbit

result = FixedEffectsOrderedProbit().fit(
    X,
    y,
    entity=person_id,
    time=period,
    category_order=["low", "middle", "high"],
    bootstrap_repetitions=200,
    random_state=2026,
)
```

The implementation currently requires a balanced common time grid, an even
panel length of at least six, within-entity design rank, and every category in
both halves. These are guardrails, not a claim that six periods are generally
enough for reliable large-`T` approximation. Inference remains unavailable
unless at least 20 entity-bootstrap replications were requested and at least
80 percent converge (with at least 20 successful draws).

It also rejects any entity observed only in the lowest or highest category in
the full panel or either time half. Such a nuisance fixed effect has no finite
unconditional MLE; retaining it would make thresholds depend on an arbitrary
optimizer bound and can break label/order invariance.

Known-entity `predict_proba` is a diagnostic plug-in: it combines corrected
common parameters with the uncorrected full-sample nuisance entity effect.
It cannot predict a new entity and is not a fully bias-corrected predictive
distribution. Always compare the corrected and `uncorrected_params`, inspect
`score_norms`, `bootstrap_successes`, `converged`, and `inference_valid`, and
run a longer-panel sensitivity analysis. Its optimizer and certification score
thresholds are capped independently for the same reason.

## Evidence and reference boundary

Maintained BUC tests compare its elementary conditional likelihood and score
with exact enumeration and its fitted slopes with a Statsmodels conditional-
Logit construction. Maintained Probit tests compare the uncorrected entity-
dummy likelihood, slope, and threshold spacings with Statsmodels, verify the
jackknife identity and label invariances, exercise an actual entity bootstrap,
and check deterministic recovery.

For stable BUC Ordered Logit only, the separate promoted-family application suite also
compares the exact blow-up conditional likelihood and entity-sandwich covariance with
R's `survival::clogit` on empirical NLSWORK observations. Its registered Python/R checks
passed within the suite's **120/120** result on 15 July 2026; the Stata run remains
pending. This is application evidence, not identification of fixed effects, thresholds,
or category probabilities. Experimental fixed-effects Ordered Probit is not covered by
that suite. See the [promoted-family guide](../validation/promoted/README.md).

Method references:

- Baetschmann, Staub, and Winkelmann (2015), “Consistent Estimation of the
  Fixed Effects Ordered Logit Model,” *Journal of the Royal Statistical
  Society: Series A* 178(3), 685–703,
  <https://doi.org/10.1111/rssa.12090>.
- Fernández-Val and Weidner (2016), “Individual and Time Effects in Nonlinear
  Panel Models with Large N, T,” *Journal of Econometrics* 192(1), 291–312,
  <https://doi.org/10.1016/j.jeconom.2015.12.014>.
- Dhaene and Jochmans (2015), “Split-panel Jackknife Estimation of Fixed-effect
  Models,” *Review of Economic Studies* 82(3), 991–1030,
  <https://doi.org/10.1093/restud/rdv007>.
