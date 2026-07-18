# Panel ordinal models

## Random-effects Ordered Logit and Probit

`RandomEffectsOrderedLogit` and `RandomEffectsOrderedProbit` use a shared normally
distributed intercept for all observations belonging to an entity. The entity
likelihood is integrated using non-adaptive Gaussian-Hermite quadrature in log
space. Both are stable, first-class estimators and expose the same result contract.

```python
result = RandomEffectsOrderedLogit().fit(X, y, entity=entity)

probit_result = RandomEffectsOrderedProbit().fit(X, y, entity=entity)
```

`predict_proba(X)` returns population-averaged probabilities by integrating over
the fitted random-effect distribution. Conditional probabilities require an
explicit scalar, row-level vector, or entity-keyed random effect:

```python
conditional = result.predict_proba(
    X_new,
    random_effects=entity_effects,
    entity=new_entity,
)
```

For observed entities, posterior summaries and exact quadrature posterior-
predictive probabilities are available for either link:

```python
posterior = result.posterior_random_effects(X, y, entity=entity)
prediction = result.posterior_predict_proba(
    X_new,
    entity=new_entity,
    posterior=posterior,
)
```

The reported posterior SD describes uncertainty in an entity effect conditional
on fitted model parameters. It is not a frequentist parameter standard error.
Posterior prediction integrates over retained quadrature weights; evaluating at
`posterior_mean` is a separate empirical-Bayes plug-in calculation.

For the Probit model, population-averaged probabilities also have a useful exact
identity. Integrating a normal random intercept with standard deviation
`sigma_entity` rescales each cumulative index by
`sqrt(1 + sigma_entity**2)`. The test suite checks the numerical quadrature output
against this closed form and checks the conditional likelihood directly against
Statsmodels' Ordered Probit kernel.

These are random-intercept models, not entity fixed-effects estimators. They require
at least two entities and repeated observations within at least one entity. Refit
important analyses with more quadrature points and confirm that estimates and the
log likelihood are stable.

Both links expose `score_norm` and `scaled_score_norm`. Convergence and
observed-information inference require an independently capped scaled-score
check, a finite covariance, and a scale-invariant positive-information check;
an overly loose optimizer tolerance cannot certify the pooled initialization.

## Fixed-effects alternatives

Use stable `FixedEffectsOrderedLogit` when time-invariant entity heterogeneity
may be arbitrarily correlated with the regressors and the Logit link and common
slopes are appropriate. Its BUC conditional likelihood identifies slopes only;
it does not estimate cutoffs, entity effects, or probabilities.

The Probit and dynamic fixed-effects alternatives have materially narrower
contracts and remain experimental. Their assumptions, panel-shape restrictions,
prediction limits, and validation evidence are documented separately in
[Fixed-effects ordinal panels](FIXED_EFFECTS_ORDINAL.md) and
[Dynamic fixed-effects Ordered Logit](DYNAMIC_FIXED_EFFECTS_ORDINAL.md). Do not choose between
fixed and random effects by treating their predictions or likelihoods as the
same estimand.
