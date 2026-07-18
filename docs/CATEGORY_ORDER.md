# Ordered outcome labels

Every ordinal estimator accepts an explicit `category_order` argument:

```python
result = OrderedLogit().fit(
    X,
    y,
    category_order=["low", "medium", "high"],
)
```

The same option is available for Ordered Probit, Generalized Ordered Logit,
Partial Proportional Odds, random-effects Ordered Logit/Probit, and dynamic random-
effects Ordered Logit. The fitted order is retained by thresholds, probability
columns, predictions, state-dependence indicators, and initial-condition
controls.

An ordered pandas categorical is recognized automatically:

```python
y = pd.Series(
    pd.Categorical(
        labels,
        categories=["low", "medium", "high"],
        ordered=True,
    )
)
result = OrderedLogit().fit(X, y)
```

Unordered pandas categoricals require `category_order`. All listed categories
must be observed exactly once in the ordering contract; unused, duplicated, or
missing labels are rejected because their thresholds are not identified.

When neither mechanism is used, ordinary sortable numeric or string labels keep
the historical sorted-label behavior. For substantive ordinal scales, explicit
ordering is recommended.
