# Dynamic fixed-effects Ordered Logit

`DynamicFixedEffectsOrderedLogit` is an experimental fixed-*T* estimator for a
short dynamic ordinal panel with unrestricted entity effects. It implements the
composite conditional maximum-likelihood construction of Muris, Raposo, and
Vandoros (2025), not a dummy-variable ordered likelihood and not static BUC with
a lagged outcome appended to `X`.

The model is

\[
Y^*_{it}=\alpha_i+X_{it}'\beta
  +\rho\,1\{Y_{i,t-1}\ge k\}-U_{it},\qquad t=1,2,3,
\]

with logistic innovations, ordered thresholds, and an unmodelled initial
outcome at `t=0`. The cutoff `k` is fixed and known. The estimator conditions
out `alpha_i` and identifies `beta`, `rho`, and all thresholds relative to the
normalization `gamma[k] = 0`.

Primary reference: [Muris, Raposo, and Vandoros, “A Dynamic Ordered Logit Model
with Fixed Effects,” *Review of Economics and Statistics* 107(4),
1104–1114](https://doi.org/10.1162/rest_a_01336). The
[open accepted manuscript](https://discovery.ucl.ac.uk/id/eprint/10173423/1/muris%20et%20al%20restat%202023.pdf)
contains the conditioning events, likelihood, identification assumptions, and
simulation study. The more general functional-differencing GMM model of
[Honoré, Muris, and Weidner (2025)](https://doi.org/10.3982/QE2052) permits
level-specific lag effects and four or more periods; it is not implemented here.

## Supported identification envelope

The current implementation intentionally supports only the paper's discrete-
regressor, exact-stayer case:

- exactly four consecutive outcome observations per entity;
- at least three observed ordered categories;
- one known state cutoff supplied as a category label;
- state dependence restricted to one coefficient on
  `1(lagged_y >= state_cutoff)`;
- discrete regressors with positive mass on exact vector equality `X[2] == X[3]`;
- strict exogeneity of the complete regressor path under the model;
- enough conditional histories and variation in `X[1] - X[2]` to identify every
  common parameter;
- independent sampling across entities and logistic, serially independent
  innovations conditional on the fixed effect and history.

Continuous-regressor kernel matching, arbitrary category-specific lag effects,
longer-panel functional-differencing GMM, unbalanced panels, and bootstrap
bandwidth selection are not implemented. An exact float equality is not a
surrogate for continuous matching; continuous covariates will usually produce
no stayers and the fit will stop with an explicit error.

## Conditional likelihood

For every threshold pair `j <= k <= l`, the method selects histories that move
up between periods 1 and 2 or move down between those periods. Conditional on
that pair, the initial and terminal binary states, and `X[2] == X[3]`, the odds
of the down history relative to the up history no longer contain `alpha_i`.
Under the package's displayed structural convention
`Y* = alpha + X beta + ... - U`, their log odds use

\[
(X_{i1}-X_{i2})'\beta
+\rho(D_{i0}-D_{i3,jl})
+(1-D_{i3,jl})\gamma_l+D_{i3,jl}\gamma_j.
\]

`conditional_sample_frame()` exposes every binary contribution and its design
columns for auditing. `entity_score_frame()` exposes the entity-aggregated
scores used in the Godambe sandwich covariance.

Sign-convention note: the accepted manuscript prints
`Delta X = X[2] - X[1]` next to a likelihood whose response is the down history.
Directly taking the down/up path-probability ratio under its displayed latent
model gives `X[1] - X[2]`; this is also the orientation that recovers the
structural coefficient in simulation. The implementation uses that structural-
sign orientation and locks it down with an exact path-probability identity test.

## Usage

```python
from limiteddepkit.experimental import DynamicFixedEffectsOrderedLogit

result = DynamicFixedEffectsOrderedLogit().fit(
    X,
    y,
    entity=person_id,
    time=period,
    state_cutoff="good",
    category_order=["poor", "fair", "good", "excellent"],
)

print(result.summary_frame())
print(result.params)                 # structural slopes beta
print(result.state_dependence)       # rho
print(result.thresholds)             # includes the normalized zero threshold
print(result.conditional_sample_frame())
```

`time` must be numeric and each entity must have four observations separated by
`time_step` (one by default). Input row order and entity labels do not affect the
fit. Explicit `category_order` remains important for semantic labels.

## Inference and post-estimation

Point estimation maximizes the stacked conditional log likelihood while linear
constraints enforce ordered thresholds around `gamma[k] = 0`. The reported
covariance is an entity-clustered Godambe sandwich with the usual CR1 finite-
cluster correction. In the discrete exact-stayer case this is the standard
clustered binary-likelihood covariance for the auditable pseudo-sample.

The model reports inference only when optimization succeeds, a capped scaled
KKT-stationarity check passes, the information matrix is positive definite and
well conditioned on a relative scale, threshold constraints are interior, the
covariance is finite, and the number of contributing entities exceeds the
number of common parameters. `scaled_kkt_residual` exposes the independent
stationarity check; deliberately loose optimizer tolerances cannot certify
inference. `n_entities`/`n_groups` count the input panel, whereas
`n_inference_groups` counts only entities contributing clusters to the
Godambe covariance.
Before optimization, a homogeneous linear-program check rejects complete or
quasi-complete separation because the unpenalized conditional likelihood then
has no finite maximizer.

The fixed effects are conditioned out, not estimated. Therefore:

- `common_index(X, lagged_y=...)` returns only
  `X beta + rho * 1(lagged_y >= k)`;
- `predict_proba()` raises `NotImplementedError`;
- category probabilities, entity effects, marginal effects, ordinary AIC/BIC,
  and multi-step forecasts are unavailable.

## Validation evidence

The dedicated tests cover:

- a direct path-probability identity showing that the conditional odds are
  invariant to the entity effect;
- deterministic recovery of slopes, state dependence, and thresholds from a
  four-category fixed-effects data-generating process;
- exact coefficient and entity-cluster covariance parity with Statsmodels GLM
  after constructing the same MRV binary conditional sample;
- threshold normalization/order, semantic category labels, row-order and entity-
  label invariance, schema checks, identification failures, and refusal to
  fit a separated conditional sample or certify a nonstationary fit returned
  under a deliberately loose tolerance.

Statsmodels parity validates numerical optimization and sandwich assembly after
conditioning. It is not an independent implementation of the MRV history
construction. No maintained Stata or R command providing this exact restricted
CCMLE was found, so the estimator remains experimental pending independent
replication against author code or another implementation.
