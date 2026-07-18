# Count model guide

`limiteddepkit.count` is the stable namespace for the foundational count
models:

```python
from limiteddepkit.count import NegativeBinomialNB2, PoissonRegressor
```

`NegativeBinomial` remains an equivalent compatibility name for
`NegativeBinomialNB2`. The response must be a one-dimensional, non-negative
integer vector with at least one positive count. The design must be finite and
full rank. An intercept is not inserted automatically; include a constant
column when the specification requires one.

## Mean specifications

Poisson uses

\[
Y_i \mid X_i \sim \operatorname{Poisson}(\mu_i), \qquad
\log(\mu_i) = X_i\beta + o_i + \log(e_i).
\]

NB2 uses the same conditional mean and

\[
\operatorname{Var}(Y_i \mid X_i) = \mu_i + \alpha\mu_i^2,
\qquad \alpha > 0.
\]

Pass an additive log-scale term with `offset=` and a strictly positive
count-scale term with `exposure=`. When both are supplied they are combined.
The NB2 result reports `alpha` for interpretation and estimates
`log_alpha` for unconstrained inference.

```python
result = NegativeBinomialNB2().fit(
    X,
    y,
    offset=log_baseline_risk,
    exposure=person_time,
    cov_type="cluster",
    clusters=person_id,
)
```

## Weight semantics

Only one weight type may be supplied:

- `freq_weights` must be non-negative integers. A row with weight `m` has
  exactly the same likelihood, information, and ordinary robust-covariance
  meaning as `m` repeated rows. Zero-weight rows do not identify parameters or
  clusters. All rows must still contain a valid response, finite design,
  offset, and exposure; only an inactive cluster label may be missing. AIC and
  BIC use the replicated sample size.
- `analytic_weights` must be strictly positive. They scale the estimating
  equations and robust score contributions. Because their overall scale is
  not a replicated sampling likelihood, `aic` and `bic` deliberately return
  `NaN`; the weighted objective remains available as `loglike` for numerical
  diagnostics, not likelihood-based cross-weight comparisons.

Analytic weights are not probability or survey-design weights. Complex survey
strata, finite-population corrections, and multistage primary sampling units
are outside this contract.

## Covariance choices

The default `cov_type="nonrobust"` is the inverse observed information under
the specified conditional count distribution. `"HC0"` and `"HC1"` are
observation-level sandwich estimators. `"cluster"` aggregates score vectors
by the labels supplied in `clusters=`. HC1 and cluster finite-sample factors
are applied by default and can be disabled with `use_correction=False`.

Cluster covariance changes inference, not the conditional independence used
by the likelihood or the mean specification. It requires at least two active
clusters. It is not a random-effects, fixed-effects, GEE, or conditional panel
count estimator.

## Result and prediction contract

Both results expose:

- labelled `params`, `all_params`, `standard_errors`, `zstats`, `pvalues`, and
  `covariance`;
- `vcov()`, `conf_int()`, `summary_frame()`, `aic`, `bic`, `df_resid`, and
  `n_params`;
- `converged`, `inference_valid`, `score_norm`, `information_condition`,
  `scaled_score_norm`, `pearson_chi2`, `deviance`, and `diagnostics()`;
- indexed in-sample `fitted_values`; and
- `predict(X, offset=..., exposure=...)` for expected counts.

DataFrame predictions retain their index and require exactly the fitted
column names in the fitted order. Arrays are checked by width. Offset,
exposure, response, weight, and cluster Series must have exactly the design
index; the estimator never silently reorders labelled inputs.

Prediction-time offset and exposure are explicit. Training values are not
silently reused for new rows. This prevents an old exposure vector from being
accidentally applied to a new population.

`score_norm` is the raw summed score and therefore changes with regressor and
weight scale. `scaled_score_norm` divides each summed component by the
empirical root-sum-square of its likelihood-score contributions (using exact
row-replication scaling for frequency weights). Convergence and inferential
certification require this scale-invariant stationarity check; a generic
optimizer success flag alone is insufficient. The certification threshold is
also capped independently of the requested optimizer tolerance.

## Promotion and validation boundary

Poisson and NB2 are promoted because maintained tests compare coefficients,
likelihoods, observed-information covariance, HC0 covariance, cluster
covariance, offsets, exposure, and frequency/analytic-weight behavior with
Statsmodels. Frequency-weight NB2 is additionally checked against literal
row expansion because Statsmodels' discrete NB2 estimator has no frequency
weight interface.

The separate promoted-family public-data suite also compares Poisson with R's
`stats::glm` and NB2 with `MASS::glm.nb`, including full canonical covariance
transformations, on the empirical `rod93` infant-mortality application. Both passed their
registered Python/R gates within the suite's **120/120** result on 15 July 2026. The
corresponding Stata promoted run also passed its required Poisson and NB2 checks; see the
[promoted-family guide](../validation/promoted/README.md).

`ZeroInflatedPoisson` and `HurdlePoisson` remain in
`limiteddepkit.experimental`. Their unweighted two-part likelihoods and
predictions remain compatible, but they have not inherited the stable
foundations' offset/exposure, weight, and robust-covariance contract. Promotion
of Poisson and NB2 must not be read as promotion of those mixture models.

Current foundations also do not provide formula parsing, automatic exposure
construction, NB1/generalized-Poisson families, panel fixed/random effects,
GEE, endogenous regressors, survey variance, or a zero-truncated NB2 model.
