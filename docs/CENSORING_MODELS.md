# Gaussian censoring and truncation models

The stable Gaussian censoring family is available from the package root or the focused
`limiteddepkit.censoring` namespace:

```python
from limiteddepkit import IntervalRegression, Tobit, TruncatedRegression
```

All three estimators model a latent response

\[
y_i^* = x_i'\beta + \varepsilon_i, \qquad
\varepsilon_i \sim N(0, \sigma^2),
\]

using a homoskedastic Gaussian maximum likelihood. No constant is added automatically;
include one explicitly in `X` when the specification requires an intercept.

## Choosing the observation rule

| Estimator | Data supplied | Observation rule |
| --- | --- | --- |
| `Tobit(censoring_point=c, side="left")` | Recorded `y >= c` | `y = max(c, y*)` |
| `Tobit(censoring_point=c, side="right")` | Recorded `y <= c` | `y = min(c, y*)` |
| `TruncatedRegression(truncation_point=a, side="left")` | Retained `y > a` only | Sample observed only when `y* > a` |
| `TruncatedRegression(truncation_point=a, side="right")` | Retained `y < a` only | Sample observed only when `y* < a` |
| `IntervalRegression` | One lower and upper bound per row | Exact, finite interval, or either open tail |

Censoring keeps an observation at a boundary; truncation removes observations outside
the retained support. They are different likelihoods and should not be interchanged.
For interval regression, exact observations use `lower == upper`, left-censored rows use
`lower=-inf`, right-censored rows use `upper=inf`, and a finite nonzero-width pair is an
interval observation.

```python
tobit_result = Tobit(censoring_point=0.0).fit(X, observed_y)

interval_result = IntervalRegression().fit(
    X,
    lower=reported_lower,
    upper=reported_upper,
)
```

## Result and prediction contract

Results expose labeled `params`, positive `sigma`, `all_params`, `covariance`,
`standard_errors`, `zstats`, `pvalues`, `loglike`, `aic`, `bic`, `df_resid`,
`converged`, `inference_valid`, `summary_frame()`, `conf_int()`, and `vcov()`.
Prediction DataFrames must retain the fitted feature names and order.
`score_norm` and `scaled_score_norm` expose likelihood stationarity. Optimizer
and certification tolerances are capped independently, so a deliberately loose
user tolerance cannot certify the initialization.

The common distribution methods are:

- `predict_latent(X)` for `X beta`;
- `predict_latent_cdf(X, values)` for `P(y* <= value | X)`; and
- `predict_latent_interval(X, level=...)` for a latent-outcome predictive interval.

`Tobit.predict()` defaults to the mean of the recorded censored response and also accepts
`which="latent"` or `which="censoring_probability"`. `TruncatedRegression.predict()`
defaults to the conditional mean in the retained sample and also accepts
`which="latent"` or `which="selection_probability"`. `IntervalRegression.predict()` is
the latent mean; `predict_interval()` remains an alias for its latent predictive interval.
Predictive intervals describe a future latent outcome conditional on fitted parameters;
they are not confidence intervals for a coefficient or mean.

## Covariance choices

The default is `covariance_type="observed-information"`. Two score-sandwich alternatives
are supported:

```python
robust = Tobit().fit(X, y, covariance_type="robust")
clustered = Tobit().fit(
    X,
    y,
    covariance_type="cluster",
    clusters=entity_id,
)
```

`robust` uses the outer product of observation-level likelihood scores. `cluster` sums
scores within each supplied group and applies the conventional finite-sample CR1 factor.
Clusters affect covariance only; they do not introduce random effects or change the
conditional likelihood. With few clusters, normal-reference inference can remain poor.

## External application evidence

The separate promoted-family public-data suite compares Tobit and interval regression
with Gaussian `survival::survreg` fits and truncated regression with an independently
coded truncated-normal likelihood. All registered Python/R checks for these applications
passed on 15 July 2026 as part of the suite's **120/120** result. The promoted Stata run
also passed the aligned Tobit, truncated-regression, and interval-regression checks. The
Mroz censoring and truncation application is empirical; Stata's official `womenwage2`
interval-regression software fixture is explicitly fictional.
See the [promoted-family guide](../validation/promoted/README.md) for transformations,
tolerances, provenance, and the exact evidence boundary.

## Current boundaries

The stable contract does not currently include heteroskedastic scale equations,
observation-specific Tobit/truncation points, endogenous sample selection, weights,
random effects, finite mixtures, or a two-boundary Tobit convenience class. General
left/right mixtures can be encoded by `IntervalRegression`; Gaussian endogenous
selection remains a separate provisional model. A robust covariance does not repair a
misspecified conditional mean, Gaussian distribution, or observation rule.
