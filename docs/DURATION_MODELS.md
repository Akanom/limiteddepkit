# Duration models

The stable duration family contains four right-censored likelihoods:

| Estimator | Distribution or hazard | Primary scale |
|---|---|---|
| `GeometricDuration` | Constant within-spell logit hazard | Per-period hazard |
| `ExponentialDuration` | Exponential AFT | `scale = exp(X beta)` |
| `WeibullDuration` | Weibull AFT | `scale = exp(X beta)` plus shape |
| `GammaDuration` | Gamma AFT | `scale = exp(X beta)` plus shape |

`DiscreteTimeDuration` is retained as a compatibility alias for
`GeometricDuration`. The explicit geometric name matters: it is not a general
person-period baseline-hazard model and it does not silently add period
indicators.

## Fitting

```python
from limiteddepkit.duration import WeibullDuration

result = WeibullDuration().fit(
    X,
    duration,
    event,
    entry=entry,                       # optional delayed entry
    frequency_weights=frequency,       # optional likelihood replication weights
    covariance_type="cluster",
    clusters=entity,
)
```

`duration` is the observed event or right-censoring time and `event` is one for
an observed event and zero for right censoring. Continuous models require
`0 <= entry < duration`; the geometric model uses integer `entry_period` under
the same strict ordering. The likelihood is conditional on survival through
entry, so delayed entry is not treated as ordinary follow-up from time zero.

Frequency weights are non-negative **integer** likelihood-replication weights.
Zero-weight rows cannot supply events, event-free exposure, design rank, or
clusters. They are not survey probability weights. `effective_nobs`, residual
degrees of freedom, BIC, HC1, and CR1 use the replicated count. The covariance
choices are:

- `"observed"`: inverse observed information;
- `"robust"`: observation-level finite-sample sandwich;
- `"cluster"`: CR1 cluster sandwich, requiring at least two clusters.

## Prediction

Every result exposes:

- `predict_mean(X)` and its compatibility alias `predict(X)`;
- `predict_survival(X, times)`;
- `predict_hazard(X, times)`;
- `predict_cumulative_hazard(X, times)`;
- `predict_quantile(X, probability)`.

A scalar time returns a row-labelled `Series`; a time grid returns a
`DataFrame` whose columns carry the requested times. DataFrame prediction
columns must match the fitted schema and order.

An extreme geometric quantile whose finite integer period exceeds machine
range is returned as positive infinity rather than silently wrapped to an
integer. Gamma cumulative hazards and hazards evaluate the upper incomplete
gamma in log space, including tails where ordinary survival probabilities
underflow to zero.

## Evidence and boundaries

The exponential likelihood and observed covariance are compared with its
Poisson-exposure representation in Statsmodels. Weibull and Gamma likelihoods
are compared observation by observation with SciPy distribution densities and
survival functions. Maintained tests additionally cover delayed-entry
conditioning, frequency-weight identities, robust and clustered sandwich
identities against literal row expansion for all four estimators, positive-
weight identification, stationarity certification, schema enforcement, Gamma
upper-tail accuracy, and survival/cumulative-hazard consistency. Inspect both
`converged` and `scaled_score_norm`; optimizer objective stopping by itself is
not accepted as convergence.

The separate promoted-family public-data suite adds R application evidence on the
empirical cancer trial: `survival::survreg` for Exponential and Weibull, an exact grouped
Binomial likelihood identity for Geometric, and an independently implemented ordinary
Gamma likelihood on uncensored deaths. These applications passed their registered
Python/R gates within the suite's **120/120** result on 15 July 2026. The promoted Stata
run passed the aligned Geometric, Exponential, and Weibull checks. Stata has no exact
ordinary-Gamma `streg` target, so Gamma is a predeclared Stata skip rather than a
generalized-Gamma substitution. See the
[promoted-family guide](../validation/promoted/README.md).

The current family is parametric and single-record-per-spell. It does not claim
support for interval censoring, recurrent events, competing risks, frailty, or
time-varying covariate histories. Use the separate limited-outcome validation
layer for censoring-aware held-out scores; those scores do not change the
fitted likelihood or turn a misspecified duration distribution into a valid
one.
