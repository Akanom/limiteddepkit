# Panel ordinal models

## Random-effects Ordered Logit

`RandomEffectsOrderedLogit` uses a shared normally distributed intercept for all
observations belonging to an entity. The entity likelihood is integrated using
non-adaptive Gaussian-Hermite quadrature in log space.

```python
result = RandomEffectsOrderedLogit().fit(X, y, entity=entity)
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
predictive probabilities are available:

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
