# limiteddepkit

`limiteddepkit` is a Python toolkit for limited-dependent-variable econometrics. Alpha
release `0.1.0a1` supports a narrow, reviewable binary and ordinal core.

> **Alpha warning:** APIs may change before a stable release. The maintained Python, Stata,
> and R validation suites pass, but every claim remains specification- and benchmark-specific.

The package is intended for empirical researchers working with binary choices, ordered
outcomes, and static or dynamic ordinal panels. Its result conventions align with
`systemgmmkit` where statistically natural, while preserving nonlinear identification and
probability-scale interpretation.

## Why limiteddepkit?

Applied limited-outcome work often combines an estimator from one library, hand-written
category coding, separate marginal-effect calculations, and software-specific reporting.
That fragmentation can change the estimand unintentionally or hide incompatible data
contracts.

`limiteddepkit` follows five principles:

1. **Explicit identification.** No constant is added silently. Binary models accept a
   user-supplied constant; ordinal models reject constants because thresholds identify
   location.
2. **Ordered labels remain ordered.** Category order is retained in thresholds,
   probabilities, predictions, state indicators, and initial-condition controls.
3. **Estimator-specific post-estimation.** The API reports binary probabilities, ordinal
   category probabilities, category-specific effects, and posterior panel predictions—not
   artificial linear-model residual concepts.
4. **Unsafe inference is visible.** Results expose `converged` and `inference_valid`.
   Flexible ordinal fits suppress ordinary Hessian inference on an active non-crossing
   boundary.
5. **Claims remain model-specific.** Reference comparisons, deterministic recovery, and
   real-data applications provide different evidence and are reported separately.

## Quick user path

Most users can start from the root namespace:

```python
from limiteddepkit import BinaryLogit, OrderedLogit, margins

binary = BinaryLogit().fit(X_binary, y_binary)  # X_binary contains an explicit constant
ordinal = OrderedLogit().fit(
    X_ordinal,                                # no constant
    y_ordinal,
    category_order=["low", "medium", "high"],
)

print(binary.summary_frame())
print(ordinal.predict_proba(X_ordinal_new))
print(margins(ordinal, X_ordinal, kind="probability"))
```

Use DataFrames when possible: fitted names are retained and prediction columns must match
the fitted names and order.

## Installation

Python 3.10 or newer is required. From a local checkout:

```bash
python -m pip install -e .
```

Optional dependency groups are intentionally separate:

```bash
python -m pip install -e ".[plots]"       # Matplotlib helpers
python -m pip install -e ".[validation]"  # Statsmodels comparisons
python -m pip install -e ".[outputhub]"   # Universal Output Hub adapter
python -m pip install -e ".[test]"        # full maintained test environment
python -m pip install -e ".[dev]"         # test, lint, build, and twine tools
```

## Stable alpha coverage

Only imports from the package root are part of the supported alpha surface.

| Family | Stable estimator | Structure | Supported use |
| --- | --- | --- | --- |
| Binary | `BinaryLogit` | Logistic CDF | Pooled binary response |
| Binary | `BinaryProbit` | Normal CDF | Pooled binary response |
| Ordinal | `OrderedLogit` | Proportional-odds Logit | Pooled ordered response |
| Ordinal | `OrderedProbit` | Ordered Probit | Pooled ordered response |
| Flexible ordinal | `GeneralizedOrderedLogit` | Split-specific Logit slopes | Every slope may vary by split |
| Flexible ordinal | `PartialProportionalOdds` | Common plus split-specific Logit slopes | Named columns vary by split |
| Panel ordinal | `RandomEffectsOrderedLogit` | Gaussian random-intercept Logit | Static grouped outcomes |
| Dynamic panel ordinal | `DynamicRandomEffectsOrderedLogit` | Lag-category and initial-condition controls | Contiguous ordinal panels |

The root also exports shared post-estimation, nested ordinal comparison, plotting,
simulation, posterior random-effect, and optional Output Hub helpers.

### Experimental boundary

These in-scope families remain under `limiteddepkit.experimental` and are not covered by
the stable API promise:

| Family | Experimental coverage | Main promotion boundary |
| --- | --- | --- |
| Finite choice | Multinomial, Conditional, Sequential Logit | API, covariance, and failure contracts |
| Counts | Poisson, NB2, zero-inflated, hurdle Poisson | Exposure/offset and covariance contracts |
| Censoring/truncation | Tobit, truncated Gaussian, interval regression | Current Gaussian forms are narrow |
| Selection | Gaussian sample selection | FIML-only implementation |
| Duration | Discrete-time, Exponential, Weibull, Gamma | Survival and censoring contracts |
| Quantiles | Fixed-boundary censored quantile regression | Non-convex fit; bootstrap-only inference |

Experimental numerical tests are evidence, not promotion. Import those estimators only by
their experimental paths and expect change. See
[Experimental model status](docs/EXPERIMENTAL_MODELS.md).

## Data and identification

### Binary response

- `X` must be finite and full rank, with more rows than columns. A one-dimensional array
  is accepted for a single feature; use a two-dimensional array or DataFrame otherwise.
- `y` must be one-dimensional, coded exactly `0` and `1`, and contain both classes.
- Add a constant column explicitly when an intercept is required.
- DataFrame names must be unique; prediction names and order must match the fit.
- Complete or quasi-complete separation is rejected because the unpenalized finite MLE
  does not exist.

### Pooled and flexible ordinal response

- Do **not** include a constant: thresholds identify location.
- Supply `category_order=` for substantive labels unless `y` is an ordered pandas
  categorical.
- Unordered categoricals require an explicit order; every declared category must be
  observed.
- Without either mechanism, sortable numeric or string labels use sorted order.
- Prediction DataFrames must reproduce fitted feature names and order.

See [Category ordering](docs/CATEGORY_ORDER.md) for unused, missing, and duplicated-label
rules.

### Panel identifiers and time

Static `RandomEffectsOrderedLogit` takes one non-missing `entity` label per row. Labels need
not be numeric, unbalanced groups are supported, and the static estimator has no `time`
argument.

Dynamic `DynamicRandomEffectsOrderedLogit` requires non-missing `entity` and numeric `time`,
with unique entity-time pairs. Rows are sorted internally. `time_step` defaults to `1.0` and
defines an exact one-period transition.

The dynamic likelihood uses only the initial contiguous spell. The first row conditions the
specification; after the first internal time gap, later rows are excluded rather than
restarted. Inspect `dropped_initial`, `dropped_nonconsecutive`, `estimation_index`, and
`nobs` before reporting a fit.

### Shared limits

Formula parsing is not part of `0.1.0a1`. Prepare dummies, interactions, scales, and other
transformations before fitting, then reproduce the design at prediction time. Stable
estimators currently use model-based observed-information covariance. Weights, offsets,
robust/clustered covariance, and penalized separation remedies are not general stable-root
features.

## Model conventions

Binary Logit/Probit use

```text
Pr(y_i = 1 | x_i) = F(x_i' beta),
```

where `F` is the logistic or standard-normal CDF. An intercept exists only when `x_i`
contains a constant.

Pooled ordinal models use

```text
Pr(y_i <= j | x_i) = F(kappa_j - x_i' beta),
```

with strictly ordered thresholds. Generalized Ordered Logit replaces `beta` by `beta_j`;
Partial Proportional Odds varies selected components. Non-crossing is enforced over the
observed estimation support, not globally over arbitrary future covariates.

The static random-intercept model uses

```text
Pr(y_it <= j | x_it, u_i) = Lambda(kappa_j - x_it' beta - u_i),
u_i ~ Normal(0, sigma_entity^2).
```

Its likelihood uses **non-adaptive Gaussian-Hermite quadrature** in log space, with 12
points by default. Replications should set `quadrature_points` and align both method and
node count across software.

The dynamic model adds previous-category indicators, initial-category indicators,
initial-period covariates, and post-initial entity means. State-dependence parameters are
category-relative latent-index shifts, not a scalar autoregressive coefficient or direct
probability changes.

## Runnable API examples

The examples below share deterministic data. Run this setup once:

```python
import numpy as np
import pandas as pd
from scipy.special import expit

from limiteddepkit import (
    simulate_dynamic_random_effects_ordered_logit,
    simulate_generalized_ordered_logit,
    simulate_random_effects_ordered_logit,
)

rng = np.random.default_rng(2026)
nobs = 1_500
x1, x2 = rng.normal(size=(2, nobs))
X_binary = pd.DataFrame({"intercept": 1.0, "x1": x1, "x2": x2})
y_binary = pd.Series(rng.binomial(1, expit(-0.3 + 0.8 * x1 - 0.5 * x2)))

ordinal_data = simulate_generalized_ordered_logit(nobs=nobs, seed=9_101)
X_ordinal, y_ordinal = ordinal_data.X, ordinal_data.y
order = [0, 1, 2]
```

### Binary Logit and Probit

```python
from limiteddepkit import BinaryLogit, BinaryProbit

logit_result = BinaryLogit().fit(X_binary, y_binary)
probit_result = BinaryProbit().fit(X_binary, y_binary)

probabilities = logit_result.predict_proba(X_binary.iloc[:10])
classes = probit_result.predict(X_binary.iloc[:10], threshold=0.4)
logit_ame = logit_result.average_marginal_effects(X_binary)
probit_ame_inference = probit_result.average_marginal_effects_inference(X_binary)
```

Binary effects are derivatives of `Pr(y=1)` for continuous, non-constant regressors.

### Ordered Logit and Probit

```python
from limiteddepkit import OrderedLogit, OrderedProbit

ordered_logit = OrderedLogit().fit(X_ordinal, y_ordinal, category_order=order)
ordered_probit = OrderedProbit().fit(X_ordinal, y_ordinal, category_order=order)

probabilities = ordered_logit.predict_proba(X_ordinal.iloc[:10])
categories = ordered_probit.predict(X_ordinal.iloc[:10])
representative = ordered_logit.margins(
    X_ordinal, at={"x1": 0.5}, kind="probability"
)
po_test = ordered_logit.proportional_odds_test(X_ordinal, y_ordinal)
```

### Generalized Ordered Logit and Partial Proportional Odds

```python
from limiteddepkit import (
    GeneralizedOrderedLogit,
    PartialProportionalOdds,
    likelihood_ratio_test,
)

generalized = GeneralizedOrderedLogit().fit(
    X_ordinal, y_ordinal, category_order=order
)
partial = PartialProportionalOdds(varying=["x1"]).fit(
    X_ordinal, y_ordinal, category_order=order
)
restricted = OrderedLogit().fit(X_ordinal, y_ordinal, category_order=order)
comparison = likelihood_ratio_test(restricted, partial)

print(generalized.threshold_slopes)
print(generalized.constraint_slack, generalized.inference_valid)
print(comparison.statistic, comparison.p_value, comparison.note)
```

Prediction raises when new covariates make cumulative indices cross; it does not clip
invalid probabilities. If `constraint_slack <= 1e-5`, `inference_valid` is false and
ordinary covariance, z, Wald, interval, and chi-square LR inference is unavailable. Use a
suitable constrained bootstrap if inference is required.

### Random-effects Ordered Logit

```python
from limiteddepkit import RandomEffectsOrderedLogit

panel = simulate_random_effects_ordered_logit(
    n_entities=80, n_periods=6, seed=8_821
)
re_result = RandomEffectsOrderedLogit().fit(
    panel.X,
    panel.y,
    entity=panel.entity,
    category_order=order,
    quadrature_points=12,
)

population = re_result.predict_proba(panel.X.iloc[:12])
fixed_only = re_result.predict_proba(panel.X.iloc[:12], random_effects=0.0)
posterior = re_result.posterior_random_effects(panel.X, panel.y, entity=panel.entity)
posterior_probability = re_result.posterior_predict_proba(
    panel.X.iloc[:12], entity=panel.entity.iloc[:12], posterior=posterior
)
```

The posterior SD is uncertainty in an entity effect conditional on fitted parameters, not a
frequentist parameter standard error. Exact posterior prediction integrates quadrature
weights; plugging in `posterior_mean` is a different estimand.

### Dynamic random-effects Ordered Logit

```python
from limiteddepkit import DynamicRandomEffectsOrderedLogit

dynamic_panel = simulate_dynamic_random_effects_ordered_logit(
    n_entities=60, n_periods=6, seed=8_263
)
dynamic_result = DynamicRandomEffectsOrderedLogit().fit(
    dynamic_panel.X,
    dynamic_panel.y,
    entity=dynamic_panel.entity,
    time=dynamic_panel.time,
    category_order=order,
    quadrature_points=12,
    maxiter=800,
)

print(dynamic_result.structural_params)
print(dynamic_result.state_dependence_params)
print(dynamic_result.initial_condition_params)
print(dynamic_result.correlated_effects_params)

rows = np.flatnonzero(dynamic_panel.time.to_numpy() > 0)[:12]
one_step = dynamic_result.predict_proba(
    dynamic_panel.X.iloc[rows],
    entity=dynamic_panel.entity.iloc[rows],
    lagged_y=dynamic_panel.y.iloc[rows - 1],
)
dynamic_posterior = dynamic_result.posterior_random_effects()
posterior_one_step = dynamic_result.posterior_predict_proba(
    dynamic_panel.X.iloc[rows],
    entity=dynamic_panel.entity.iloc[rows],
    lagged_y=dynamic_panel.y.iloc[rows - 1],
    posterior=dynamic_posterior,
)
```

One-step prediction requires a lagged category. Multi-step propagation of the entire
predicted category distribution is not part of this alpha API.

## Results and post-estimation

### Shared result contract

Supported results expose a model-appropriate subset of labeled `params`/`all_params`,
`standard_errors`, `zstats`, `pvalues`, `covariance`, `converged`, `inference_valid`,
`loglike`, `nobs`, `n_params`, `summary_frame()`, `vcov()`, `conf_int()`, `predict()`, and
`predict_proba()`. Panel results also identify their `backend` and `covariance_type`.

Check `converged` and `inference_valid` before interpreting a table. Point estimates can be
available when ordinary normal-approximation inference is not.

### Function-style facade

```python
from limiteddepkit import (
    confint, lincom, marginal_effects, margins, predict, predict_proba,
    summary_frame, vcov, wald_test,
)

table = summary_frame(ordered_logit)
covariance = vcov(ordered_logit)
intervals = confint(ordered_logit)
probabilities = predict_proba(ordered_logit, X_ordinal.iloc[:10])
labels = predict(ordered_logit, X_ordinal.iloc[:10])
effects = marginal_effects(ordered_logit, X_ordinal)
average = margins(ordered_logit, X_ordinal, at="overall")
difference = lincom(ordered_logit, {"x1": 1.0, "x2": -1.0})
joint = wald_test(ordered_logit, [{"x1": 1.0}, {"x2": 1.0}])
```

Use names from `result.all_params.index`; helpers reject unknown parameters. Wald and
linear-combination calculations inherit the fitted covariance's validity.

### Prediction and effects targets

| Family | Default `predict_proba()` target | Effects support |
| --- | --- | --- |
| Binary | `Pr(y=0)` and `Pr(y=1)` | Binary probability margins and continuous effects |
| Pooled/flexible ordinal | One probability per category | Category probabilities and category-specific effects |
| Static RE ordinal | Population average unless `random_effects=` is supplied | Panel-specific prediction, no pooled margins |
| Dynamic RE ordinal | Population-average one-step probability with explicit lag | Panel-specific prediction, no pooled margins |

`margins(..., at="overall")` averages rows; `at="mean"` evaluates one mean row; a mapping
overrides selected means. Report which estimand you use.

Nested `likelihood_ratio_test()` supports aligned Ordered Logit, Partial Proportional Odds,
and Generalized Ordered Logit pairs. Fits must use the same observations, ordered labels,
features, and feature order; the pooled restricted model must use Logit.

For static panels, package-level `posterior_random_effects()` and
`posterior_predict_proba()` mirror result methods. Dynamic results use their own methods
because lag and initial-condition inputs are also required. Keep population-average,
conditional `u=0`, posterior-mean plug-in, and exact posterior-predictive targets distinct.

### Plotting and Output Hub

Pooled and flexible ordinal fits support probability and marginal-effect plots with
`limiteddepkit[plots]`:

```python
from limiteddepkit import plot_marginal_effects, plot_probabilities

ax1 = plot_probabilities(ordered_logit, X_ordinal, feature="x1")
ax2 = plot_marginal_effects(ordered_logit, X_ordinal, feature="x1")
```

Other covariates are held at their means. Binary and panel results are not covered by these
plot helpers.

All supported ordinal results can be converted to Universal Output Hub models:

```python
from limiteddepkit import add_to_outputhub, to_outputhub_model
from universal_output_hub import OutputHub

model = to_outputhub_model(ordered_logit, name="Ordered rating", depvar="rating")
hub = OutputHub("Ordinal analysis")
add_to_outputhub(hub, ordered_logit, name="Ordered rating", X=X_ordinal)
```

Supplying `X` adds delta-method AME inference for pooled/flexible ordinal fits. Panel models
can be exported as coefficient models but do not yet provide that AME table.

## Validation philosophy and status

Validation proceeds from API and identity tests, to equivalent Python references, to
deterministic simulation and invariance checks, and finally to manual cross-software
comparison. Each layer answers a different question.

| Validation slice | Status | Claim boundary |
| --- | --- | --- |
| Deterministic Python test suite | **PASS** | Maintained repository tests pass |
| Binary Logit/Probit vs Statsmodels | **PASS** | Coefficients, likelihood, inference, predictions, criteria, AMEs |
| Ordered Logit/Probit vs Statsmodels | **PASS** | Maintained aligned pooled specifications |
| Flexible ordinal recovery/safeguards | **PASS** | Known bounded-support DGPs and non-crossing behavior |
| Static RE ordinal recovery/numerics | **PASS** | Balanced/unbalanced panels, quadrature, invariance, posterior identities |
| Dynamic RE ordinal recovery/numerics | **PASS** | Exact specification, quadrature, invariance, trimming rules |
| Controlled synthetic Stata certification | **PASS — 82/82** | All eight families; Stata 17 and `gologit2` 3.2.8 |
| Downloaded real-data Stata application | **PASS — 82/82** | Applied robustness; does not broaden controlled certification |
| Controlled synthetic R parity | **PASS — 110/110** | All eight families; pinned R 4.5.1 environment |
| Downloaded real-data R application | **PASS — 110/110** | Applied robustness; independent R estimators on frozen data |

The **controlled synthetic track** fixes the DGP, mappings, prediction target, quadrature,
and tolerances; it is the strict certification design. The **downloaded real-data track**
adds external data provenance and applied data handling, but cannot establish parameter
recovery and does not replace the controlled gate.

The manual Stata runs and independent R runs completed on 14 July 2026. Both controlled and
real-data tracks included Binary Logit/Probit, Ordered Logit/Probit, Generalized Ordered
Logit, Partial Proportional Odds, static RE Ordered Logit, and dynamic RE Ordered Logit.
See the [Stata parity harness](validation/stata/README.md) and
[R parity harness](validation/r/README.md) for commands, mappings, declared tolerances,
software versions, result envelopes, and evidence boundaries. The
[cross-software evidence index](validation/PARITY_EVIDENCE.md) records the
four completed outcomes and their exact manifest, report, and certificate
digests.

Parity requires an explicit binary constant, no ordinal constant, aligned cutpoint signs,
observed-information covariance where declared, full random-effect scale Jacobians, aligned
nonadaptive quadrature, and the same prediction target. In particular, conditional `u=0`
probabilities are not interchangeable with default empirical-Bayes predictions.

Validation claims apply only to maintained specifications. Users must still assess their
sample, category support, convergence, identification, extrapolation, and estimand.

## Practical guidance and reporting

- Start with the simplest defensible family. Use a pooled ordered model before relaxing
  slopes or adding panel heterogeneity.
- Treat category order as data, not decoration. Reversing it changes cumulative
  probabilities and coefficient interpretation.
- Never put a constant in ordinal `X`; do add an explicit binary constant when required.
- Use theory, the proportional-odds diagnostic, and aligned nested comparisons before
  adding threshold-specific slopes.
- For flexible fits, inspect `minimum_index_gap`, `constraint_slack`, and
  `inference_valid`; do not report ordinary inference on an active boundary.
- Name the prediction target: population averaged, conditional at a stated random effect,
  posterior-mean plug-in, or exact posterior predictive.
- Refit important panel models with more quadrature points and compare likelihoods,
  coefficients, `sigma_entity`, and probabilities.
- Treat dynamic state dependence as specification-dependent association, not automatic
  causal evidence.
- Save category maps, dummy levels, scales, interactions, and feature order because the
  package does not store a formula pipeline.

For publication or review, report:

- exact package version, estimator, link, outcome coding, and ordinal category order;
- sample construction, exclusions, missing-data handling, features, and transformations;
- explicit binary-intercept choice or confirmation of no ordinal constant;
- observations, entities, convergence, optimizer controls, covariance type, and confidence
  level;
- coefficient/threshold sign convention and the probability or effect estimand;
- flexible-model constraint diagnostics and inference validity;
- panel `sigma_entity`, quadrature method/points, and prediction target;
- dynamic initial-condition controls, `time_step`, and dropped initial/gap rows; and
- parity track, software versions, mappings, and tolerance if cross-software evidence is
  reported.

Archive the manifest, Python references, Stata log and exports, R exports, comparator
reports, software versions, and certificates with any parity claim. Do not carry the result
to a changed dataset, specification, optimizer, covariance target, or quadrature rule.

## Package boundary and roadmap

`limiteddepkit` contains models whose response, observation rule, or choice mechanism is
intrinsically limited. It is not a general home for every microeconometric method.

Ordinary linear IV/treatment-effect 2SLS, iid Gaussian mixtures and generic switching
regression, ordinary quantile regression, and a generic GAM are deliberately outside the
installed package. See [Package scope](docs/PACKAGE_SCOPE.md) for the keep/extract rationale.

Near-term work remains conservative:

1. archive the completed Stata and R evidence with the exact repository commit;
2. rerun both external-software tracks after estimator or reference changes;
3. broaden covariance and data contracts only with tests and explicit estimands;
4. promote an experimental family only after likelihood, inference, post-estimation,
   failure modes, recovery, and reference evidence are reviewed; and
5. avoid expanding the stable namespace merely to increase model count. No experimental
   estimator has a promised promotion date.

## Documentation, citation, and license

Start with the [documentation index](docs/README.md), then use the guides for
[binary](docs/BINARY_MODELS.md), [ordinal](docs/ORDINAL_MODELS.md),
[panel](docs/PANEL_ORDINAL.md), and [dynamic](docs/DYNAMIC_ORDINAL.md) models. Separate notes
cover [category ordering](docs/CATEGORY_ORDER.md), the
[dynamic numerical certificate](docs/DYNAMIC_ORDINAL_VALIDATION.md),
[ecosystem compatibility](docs/ECOSYSTEM_COMPATIBILITY.md),
[validation](docs/VALIDATION.md), [experimental status](docs/EXPERIMENTAL_MODELS.md), and the
[cross-software evidence index](validation/PARITY_EVIDENCE.md),
[Stata harness](validation/stata/README.md), and
[R harness](validation/r/README.md). Project processes are in
[CONTRIBUTING.md](CONTRIBUTING.md), [RELEASING.md](RELEASING.md),
[CHANGELOG.md](CHANGELOG.md), and [SECURITY.md](SECURITY.md).

If you use `limiteddepkit`, cite the software and exact version. Machine-readable metadata
is in [CITATION.cff](CITATION.cff). Suggested interim citation:

```text
Akanbi, Oluwajuwon Mayomi. limiteddepkit: Limited-dependent-variable models for Python.
Version 0.1.0a1, 2026. Alpha software.
```

No repository URL, DOI, or archival identifier is asserted before one is assigned. Add the
permanent identifier when a public archive exists.

`limiteddepkit` is distributed under the MIT License. See [LICENSE](LICENSE).
