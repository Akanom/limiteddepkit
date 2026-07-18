# limiteddepkit

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Development status](https://img.shields.io/badge/status-alpha-orange.svg)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Akanom/limiteddepkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Akanom/limiteddepkit/actions/workflows/ci.yml)

`limiteddepkit` is a Python toolkit for limited-dependent-variable and
microeconometric workflows.

It provides a unified, explicit interface for:

- Binary Logit and Probit;
- Firth bias-reduced Binary Logit;
- pooled, generalized, and partial-proportional-odds ordinal models;
- static random-effects Ordered Logit and Probit;
- static fixed-effects Ordered Logit through blow-up-and-cluster estimation;
- dynamic random-effects Ordered Logit with initial-condition controls;
- Poisson and negative-binomial NB2 count models;
- Gaussian censoring, truncation, and interval regression;
- geometric, Exponential, Weibull, and Gamma duration models;
- model-appropriate prediction and post-estimation;
- probability-aware validation and ML-style diagnostics; and
- reproducible comparison and reporting workflows.

The package is intended for applied researchers working with binary choices,
ordered outcomes, counts, censored or truncated responses, event durations, and
static or dynamic ordinal panels.

The objective is not to collect as many estimators as possible. The objective is to make
the observation rule, identification assumptions, prediction target, inferential status,
and validation evidence visible enough for replication, review, and applied use.

> **Alpha warning:** the current package version is `0.1.0a1`. APIs can change before a
> stable release. Validation claims are model- and benchmark-specific; they do not imply
> universal equivalence across datasets, specifications, optimizers, covariance targets,
> quadrature rules, or software defaults.

---

# Why limiteddepkit?

Limited-outcome analysis often combines an estimator from one library, hand-written
category coding, separate marginal-effect calculations, and software-specific reporting.
That fragmentation can silently change the estimand or obscure an incompatible data
contract.

`limiteddepkit` is built around six principles.

## 1. Explicit model specification

Important assumptions remain visible in code:

- binary intercepts are supplied explicitly;
- ordinal designs reject constants because thresholds identify location;
- substantive category order is supplied or retained explicitly;
- censoring and truncation sides are separate observation rules;
- offsets, exposure, weights, and covariance choices are estimator-specific;
- panel entity and time identifiers are passed explicitly; and
- quadrature order and prediction targets are not hidden in reporting helpers.

## 2. Identification-aware model boundaries

Models that look similar can identify different quantities. The package therefore keeps
the following distinctions explicit:

- random effects versus fixed effects;
- static versus dynamic state dependence;
- censoring versus truncation;
- population-average versus conditional versus posterior prediction;
- ordinary maximum likelihood versus Firth or ridge penalization; and
- an econometric estimator versus a prediction challenger.

For example, stable BUC fixed-effects Ordered Logit identifies common slopes but
conditions out cutoffs and entity effects. It therefore does not pretend to provide
category probabilities or an ordinary ordered-model AIC.

## 3. Prediction is a first-class estimand

Prediction is not treated as a generic `X @ beta` afterthought. Depending on the fitted
family, results can expose:

- binary event probabilities;
- one probability per ordered category;
- population-averaged random-effects probabilities;
- probabilities conditional on supplied random effects;
- posterior random-effect summaries and posterior-predictive probabilities;
- expected counts under new offsets or exposure;
- latent, recorded, selection, or censoring targets for Gaussian observation rules; and
- duration means, survival, hazards, cumulative hazards, and quantiles.

These targets are labelled separately because they answer different questions.

## 4. Unsafe inference remains visible

Results expose `converged` and `inference_valid` where applicable. Flexible ordinal models
do not report ordinary Hessian inference when a non-crossing constraint is active.
Experimental fixed-effects Probit requires successful entity-bootstrap inference.
Conditional fixed-effects models do not manufacture unidentified thresholds or
probabilities.

## 5. Verification is layered

Validation proceeds through:

1. API and exact statistical identities;
2. aligned comparisons with established Python implementations;
3. deterministic simulation recovery and invariance checks;
4. explicit guards at identification and optimization boundaries; and
5. maintained Stata/R comparison harnesses where an equivalent specification exists.

Controlled certification and real-data application evidence are reported separately.

## 6. Prediction diagnostics complement econometrics

The experimental `limiteddepkit.ml` layer adds leakage-aware splitting,
outcome-appropriate scores, calibration, nested model selection, uncertainty-aware
comparisons, censoring-aware duration diagnostics, and optional external-estimator
bridges. It evaluates fitted models; it does not relax identification assumptions or turn
predictive performance into causal or inferential validity.

---

# Current Development Focus

The current development line consolidates the major limited-outcome families behind
reviewable contracts.

Recent stable promotions include:

- Gaussian Tobit, truncated regression, and interval regression;
- exposure/offset Poisson and NB2 with explicit weight and covariance semantics;
- geometric, Exponential, Weibull, and Gamma duration models;
- Random-effects Ordered Probit;
- BUC Fixed-effects Ordered Logit; and
- Firth Binary Logit with profile penalized-likelihood confidence intervals.

Two research estimators deliberately remain experimental:

- `FixedEffectsOrderedProbit`, using unconditional entity effects and a split-panel
  jackknife; and
- `DynamicFixedEffectsOrderedLogit`, using the restricted four-outcome-history
  conditional estimator of Muris, Raposo, and Vandoros.

The separate promoted-family application suite now supplies additional Python/R evidence
for these stable additions. That evidence remains distinct from the earlier eight-family
controlled certificates and does not turn application agreement into universal equality.
The corresponding promoted-family Stata run remains a manual release gate.

Public roadmap, parity, and adoption discussions are part of the development
record. See [Open development](docs/OPEN_DEVELOPMENT.md) and the GitHub
Discussion templates for model-family proposals, validation reports, and use
cases.

---

# Quick User Path

Most stable estimators are available from the package root:

```python
import limiteddepkit as ldk

binary = ldk.BinaryLogit().fit(X_binary, y_binary)

ordinal = ldk.OrderedLogit().fit(
    X_ordinal,
    y_ordinal,
    category_order=["low", "medium", "high"],
)

count = ldk.PoissonRegressor().fit(
    X_count,
    y_count,
    exposure=person_time,
    cov_type="cluster",
    clusters=person_id,
)

censored = ldk.Tobit(censoring_point=0.0, side="left").fit(
    X_censored,
    y_censored,
    covariance_type="robust",
)

print(binary.summary_frame())
print(ordinal.predict_proba(X_ordinal_new))
print(count.predict(X_count_new, exposure=new_person_time))
print(censored.predict(X_censored_new, which="censoring_probability"))
```

Use pandas DataFrames when possible. Fitted feature names are retained, and prediction
DataFrames must reproduce those names and their order.

Focused stable namespaces are also available:

```python
from limiteddepkit.censoring import IntervalRegression, Tobit, TruncatedRegression
from limiteddepkit.count import NegativeBinomialNB2, PoissonRegressor
from limiteddepkit.duration import GammaDuration, GeometricDuration, WeibullDuration
from limiteddepkit.small_sample import FirthBinaryLogit
```

---

# Installation

Python 3.10 or newer is required.

The repository does not currently claim a public PyPI release. Install the current
development version from a checkout:

```bash
git clone https://github.com/Akanom/limiteddepkit.git
cd limiteddepkit
python -m pip install -e .
```

Or install directly from the repository:

```bash
python -m pip install git+https://github.com/Akanom/limiteddepkit.git
```

Optional dependency groups are separate:

```bash
python -m pip install -e ".[plots]"       # Matplotlib probability/effect plots
python -m pip install -e ".[validation]"  # Statsmodels/scikit-learn reference checks
python -m pip install -e ".[outputhub]"   # Universal Output Hub adapter
python -m pip install -e ".[test]"        # maintained test environment
python -m pip install -e ".[dev]"         # test, lint, build, and release tools
python -m pip install -e ".[neural]"      # optional PyTorch prediction challenger
```

Check the installed version:

```python
import limiteddepkit

print(limiteddepkit.__version__)
```

---

# Current Feature Coverage

## Stable binary and small-sample models

- `BinaryLogit`
- `BinaryProbit`
- `FirthBinaryLogit`
- observed-information inference for ordinary binary MLEs
- profile penalized-likelihood intervals for Firth Logit
- schema-safe probability and class prediction
- binary marginal effects and representative-value margins
- linear combinations and Wald tests where covariance is certified

## Stable pooled and flexible ordinal models

- `OrderedLogit`
- `OrderedProbit`
- `GeneralizedOrderedLogit`
- `PartialProportionalOdds`
- explicit semantic category order
- threshold-specific slopes
- observed-support non-crossing constraints
- proportional-odds diagnostics
- aligned nested likelihood-ratio comparisons
- category probabilities, predictions, margins, and marginal effects

## Stable ordinal panel models

- `RandomEffectsOrderedLogit`
- `RandomEffectsOrderedProbit`
- `FixedEffectsOrderedLogit`
- `DynamicRandomEffectsOrderedLogit`
- non-adaptive Gaussian-Hermite quadrature for random-intercept models
- population-average, conditional, and posterior probability targets
- posterior random-effect summaries
- BUC common-slope estimation under unrestricted time-invariant heterogeneity
- lag-category state dependence and explicit initial-condition controls

## Stable count models

- `PoissonRegressor`
- `NegativeBinomialNB2`
- `NegativeBinomial` compatibility name
- additive offsets and positive exposure
- integer frequency weights and analytic estimating-equation weights
- observed-information, HC0, HC1, and cluster covariance
- expected-count prediction with explicit prediction-time exposure
- deviance, Pearson, information, and stationarity diagnostics

## Stable censoring and truncation models

- left- or right-censored Gaussian Tobit
- left- or right-truncated Gaussian regression
- exact, finite-interval, left-tail, and right-tail interval regression
- observed-information, robust, and cluster covariance
- latent means and latent predictive intervals
- recorded-response, selection-probability, and censoring-probability targets

## Stable duration models

- `GeometricDuration`
- `DiscreteTimeDuration` compatibility name
- `ExponentialDuration`
- `WeibullDuration`
- `GammaDuration`
- right-censored single-spell likelihoods
- delayed entry
- integer frequency weights
- observed, robust, and cluster covariance
- mean, survival, hazard, cumulative-hazard, and quantile prediction

## Post-estimation and reporting

- labelled result objects
- covariance and confidence intervals
- coefficient summaries
- prediction and probability helpers
- marginal effects and margins where identified
- posterior panel prediction where identified
- ordinal probability and effect plots
- Universal Output Hub conversion for supported ordinal results

## Experimental workflow layer

- outcome-aware scoring for binary, ordinal, count, continuous, quantile, duration,
  grouped-choice, and selection outcomes
- iid, stratified, complete-group, entity-holdout, and forward-panel splitters
- cross-validation and validity-gated model comparison
- nested tuning and one-standard-error selection
- calibration and reliability diagnostics
- paired fold and observation/entity bootstrap comparisons
- IPCW survival diagnostics
- lazy Statsmodels, scikit-learn, and generic estimator bridges
- optional residual neural binary challenger

---

# Model Selection Guide

| Research setting | Recommended starting point | Main boundary |
| --- | --- | --- |
| Binary response, finite ordinary MLE | `BinaryLogit` or `BinaryProbit` | Explicit constant; separation is rejected |
| Binary response with separation or small-sample bias concern | `FirthBinaryLogit` | Full rank and `n > p`; not high-dimensional ridge |
| Ordered response with proportional slopes | `OrderedLogit` or `OrderedProbit` | No constant; category order matters |
| Ordered response with non-proportional slopes | `PartialProportionalOdds` or `GeneralizedOrderedLogit` | Inspect non-crossing constraints and inference validity |
| Static ordinal panel with a Gaussian entity intercept | `RandomEffectsOrderedLogit` or `RandomEffectsOrderedProbit` | Distributional random-effects assumption and quadrature sensitivity |
| Static ordinal panel with unrestricted time-invariant heterogeneity | `FixedEffectsOrderedLogit` | Common slopes only; no thresholds or probabilities |
| Dynamic ordinal panel with correlated random effects | `DynamicRandomEffectsOrderedLogit` | One-step state dependence and explicit initial-condition specification |
| Equidispersed counts | `PoissonRegressor` | Conditional Poisson mean/variance specification |
| Quadratically overdispersed counts | `NegativeBinomialNB2` | NB2 variance, not NB1 or a panel count model |
| Boundary-recorded Gaussian response | `Tobit` | Censoring keeps boundary observations |
| Sample retained only beyond a boundary | `TruncatedRegression` | Truncation removes observations outside support |
| Exact, grouped, or tail-bounded Gaussian response | `IntervalRegression` | Homoskedastic Gaussian latent response |
| Integer-period constant hazard | `GeometricDuration` | Not a general time-varying baseline hazard |
| Parametric continuous duration | Exponential, Weibull, or Gamma | Right censoring and one record per spell |

Experimental SPJ Probit and MRV dynamic fixed-effects Logit should be chosen only when
their narrower panel and identification requirements match the research design.

---

# Quick Start

## Binary Logit and Probit

Binary outcomes must be coded exactly as 0 and 1. Add a constant explicitly when the
model requires an intercept.

```python
from limiteddepkit import BinaryLogit, BinaryProbit

logit_result = BinaryLogit().fit(X_binary, y_binary)
probit_result = BinaryProbit().fit(X_binary, y_binary)

probabilities = logit_result.predict_proba(X_binary_new)
classes = probit_result.predict(X_binary_new, threshold=0.4)
effects = logit_result.average_marginal_effects(X_binary)
effect_inference = probit_result.average_marginal_effects_inference(X_binary)
```

Ordinary binary MLEs reject complete or quasi-complete separation rather than returning
misleading finite coefficients.

## Firth Binary Logit

Use Firth explicitly when the identified low-dimensional specification requires
mean-bias reduction or finite estimates under separation.

```python
from limiteddepkit import FirthBinaryLogit

firth_result = FirthBinaryLogit().fit(X_binary, y_binary)

profile_intervals = firth_result.conf_int()
wald_intervals = firth_result.conf_int(method="wald")
profile = firth_result.profile_penalized_loglike(
    "treatment",
    values=[-1.0, 0.0, 1.0],
)
```

`conf_int()` defaults to profile penalized-likelihood intervals. The covariance,
standard errors, z statistics, p-values, and `summary_frame()` remain labelled
ordinary-Fisher Wald approximations. FLIC/FLAC prediction correction and penalized-
likelihood-ratio p-values are not currently implemented.

## Ordered Logit and Probit

Do not include a constant in ordinal `X`; ordered thresholds identify location.

```python
from limiteddepkit import OrderedLogit, OrderedProbit

order = ["low", "medium", "high"]

ordered_logit = OrderedLogit().fit(
    X_ordinal,
    y_ordinal,
    category_order=order,
)
ordered_probit = OrderedProbit().fit(
    X_ordinal,
    y_ordinal,
    category_order=order,
)

probabilities = ordered_logit.predict_proba(X_ordinal_new)
categories = ordered_probit.predict(X_ordinal_new)
po_test = ordered_logit.proportional_odds_test(X_ordinal, y_ordinal)
```

## Generalized Ordered Logit and Partial Proportional Odds

```python
from limiteddepkit import (
    GeneralizedOrderedLogit,
    OrderedLogit,
    PartialProportionalOdds,
    likelihood_ratio_test,
)

generalized = GeneralizedOrderedLogit().fit(
    X_ordinal,
    y_ordinal,
    category_order=order,
)

partial = PartialProportionalOdds(varying=["income"]).fit(
    X_ordinal,
    y_ordinal,
    category_order=order,
)

restricted = OrderedLogit().fit(X_ordinal, y_ordinal, category_order=order)
comparison = likelihood_ratio_test(restricted, partial)

print(generalized.threshold_slopes)
print(generalized.constraint_slack, generalized.inference_valid)
print(comparison.statistic, comparison.p_value, comparison.note)
```

Prediction raises if new covariates make cumulative indices cross. It does not clip an
invalid ordered probability vector. When the non-crossing boundary is active, ordinary
Hessian inference is unavailable.

## Poisson and NB2 counts

```python
from limiteddepkit import NegativeBinomialNB2, PoissonRegressor

poisson = PoissonRegressor().fit(
    X_count,
    y_count,
    offset=log_baseline_risk,
    exposure=person_time,
    cov_type="HC1",
)

nb2 = NegativeBinomialNB2().fit(
    X_count,
    y_count,
    exposure=person_time,
    cov_type="cluster",
    clusters=entity_id,
)

expected_count = nb2.predict(
    X_count_new,
    exposure=new_person_time,
)
print(nb2.alpha)
print(nb2.diagnostics())
```

Frequency weights are exact row-replication weights. Analytic weights define estimating
equations and therefore do not receive ordinary likelihood AIC/BIC interpretation.

## Tobit, truncated regression, and interval regression

```python
from limiteddepkit import IntervalRegression, Tobit, TruncatedRegression

tobit = Tobit(censoring_point=0.0, side="left").fit(
    X,
    observed_y,
    covariance_type="robust",
)

truncated = TruncatedRegression(truncation_point=0.0, side="left").fit(
    X_retained,
    retained_y,
    covariance_type="cluster",
    clusters=cluster_id,
)

interval = IntervalRegression().fit(
    X,
    lower=reported_lower,
    upper=reported_upper,
)

recorded_mean = tobit.predict(X_new)
censoring_probability = tobit.predict(X_new, which="censoring_probability")
selection_probability = truncated.predict(X_new, which="selection_probability")
latent_interval = interval.predict_latent_interval(X_new, level=0.95)
```

Censoring and truncation are different likelihoods. Robust covariance changes inference;
it does not repair a misspecified Gaussian latent distribution or observation rule.

## Duration models

```python
from limiteddepkit import WeibullDuration

duration_result = WeibullDuration().fit(
    X_duration,
    duration,
    event,
    entry=entry_time,
    frequency_weights=frequency,
    covariance_type="cluster",
    clusters=entity_id,
)

mean_duration = duration_result.predict_mean(X_duration_new)
survival = duration_result.predict_survival(X_duration_new, times=[1.0, 3.0, 5.0])
hazard = duration_result.predict_hazard(X_duration_new, times=[1.0, 3.0, 5.0])
median = duration_result.predict_quantile(X_duration_new, probability=0.5)
```

The stable duration family is parametric, right-censored, and single-record-per-spell. It
does not claim recurrent-event, competing-risk, frailty, interval-censoring, or
time-varying-covariate support.

---

# Ordinal Panel Models

## Random-effects Ordered Logit and Probit

The static random-intercept models use

```text
Pr(y_it <= j | x_it, u_i) = F(kappa_j - x_it' beta - u_i),
u_i ~ Normal(0, sigma_entity^2).
```

`F` is logistic for `RandomEffectsOrderedLogit` and standard normal for
`RandomEffectsOrderedProbit`. Both use non-adaptive Gaussian-Hermite quadrature in log
space.

```python
from limiteddepkit import RandomEffectsOrderedLogit, RandomEffectsOrderedProbit

re_logit = RandomEffectsOrderedLogit().fit(
    X_panel,
    y_panel,
    entity=entity_id,
    category_order=order,
    quadrature_points=12,
)

re_probit = RandomEffectsOrderedProbit().fit(
    X_panel,
    y_panel,
    entity=entity_id,
    category_order=order,
    quadrature_points=12,
)

population_average = re_logit.predict_proba(X_panel_new)
conditional_at_zero = re_logit.predict_proba(X_panel_new, random_effects=0.0)

posterior = re_logit.posterior_random_effects(
    X_panel,
    y_panel,
    entity=entity_id,
)
posterior_prediction = re_logit.posterior_predict_proba(
    X_panel_new,
    entity=new_entity_id,
    posterior=posterior,
)
```

The posterior SD describes uncertainty in an entity effect conditional on fitted model
parameters. It is not a frequentist parameter standard error. Refit important models with
more quadrature points and inspect coefficient, variance, likelihood, and probability
sensitivity.

## BUC Fixed-effects Ordered Logit

`FixedEffectsOrderedLogit` implements the blow-up-and-cluster estimator. It dichotomizes
the response at each cutoff, conditions each binary clone on its entity success count,
sums the conditional log likelihoods, and clusters the composite covariance by entity.

```python
from limiteddepkit import FixedEffectsOrderedLogit

fe_logit = FixedEffectsOrderedLogit().fit(
    X_panel,
    y_panel,
    entity=entity_id,
    category_order=order,
)

print(fe_logit.params)
print(fe_logit.odds_ratios())
print(fe_logit.linear_index(X_panel_new))
```

BUC identifies common slopes. It does not identify cutoffs, entity effects, category
probabilities, marginal effects, or an ordinary ordered-model likelihood/AIC.

## Dynamic random-effects Ordered Logit

The stable dynamic random-effects model adds lag-category indicators, initial-outcome
indicators, initial-period covariates, post-initial entity means, and a remaining Gaussian
entity intercept.

```python
from limiteddepkit import DynamicRandomEffectsOrderedLogit

dynamic = DynamicRandomEffectsOrderedLogit().fit(
    X_dynamic,
    y_dynamic,
    entity=entity_id,
    time=period,
    category_order=order,
    quadrature_points=12,
)

print(dynamic.structural_params)
print(dynamic.state_dependence_params)
print(dynamic.initial_condition_params)
print(dynamic.correlated_effects_params)

one_step = dynamic.predict_proba(
    X_dynamic_new,
    entity=new_entity_id,
    lagged_y=lagged_category,
)
```

Only the initial contiguous spell for each entity enters the likelihood. A time gap does
not silently start a new spell. One-step prediction requires an explicit lagged category;
multi-step propagation of the complete predicted category distribution is not currently
exposed.

State-dependence coefficients are category-relative latent-index shifts, not a scalar
autoregressive coefficient and not automatically causal effects.

---

# Experimental Research Estimators

Experimental estimators are importable from `limiteddepkit.experimental`. Their APIs and
inferential contracts can change before promotion.

## SPJ Fixed-effects Ordered Probit

There is no Ordered-Probit equivalent of the Logit conditional likelihood. The
experimental estimator fits unconditional entity-effects Ordered Probit to the full panel
and two time halves, then applies

```text
2 * full_panel - (first_half + second_half) / 2
```

to common slopes and thresholds.

```python
from limiteddepkit.experimental import FixedEffectsOrderedProbit

fe_probit = FixedEffectsOrderedProbit().fit(
    X_panel,
    y_panel,
    entity=entity_id,
    time=period,
    category_order=order,
    bootstrap_repetitions=200,
    random_state=2026,
)
```

The current contract requires a balanced common time grid, even `T >= 6`, category
support in both halves, finite nuisance effects, and successful entity-bootstrap
inference. Six periods is a computational guardrail, not a general claim that the
large-`T` approximation is reliable. Known-entity probabilities combine corrected common
parameters with uncorrected nuisance effects and are diagnostic only.

## MRV Dynamic Fixed-effects Ordered Logit

`DynamicFixedEffectsOrderedLogit` implements a narrow fixed-`T` conditional estimator,
not an ordered dummy-variable likelihood and not static BUC with a lag appended.

```python
from limiteddepkit.experimental import DynamicFixedEffectsOrderedLogit

dynamic_fe = DynamicFixedEffectsOrderedLogit().fit(
    X_dynamic_fe,
    y_dynamic_fe,
    entity=entity_id,
    time=period,
    state_cutoff="good",
    category_order=["poor", "fair", "good", "excellent"],
)

print(dynamic_fe.params)
print(dynamic_fe.state_dependence)
print(dynamic_fe.thresholds)
print(dynamic_fe.conditional_sample_frame())
```

The implementation requires exactly four consecutive outcome observations per entity,
discrete exact stayers, a known state cutoff, restricted state dependence, and sufficient
conditional-history variation. Fixed effects are conditioned out. Category probabilities,
entity effects, marginal effects, ordinary AIC/BIC, and multi-step forecasts are not
available.

Its Statsmodels comparison validates optimization and sandwich assembly after the same
conditional pseudo-sample has been constructed. It is not an independent replication of
the MRV history construction, and no Stata/R parity certificate is claimed.

## Other provisional families

- ridge Binary and Ordered Logit;
- Multinomial, Conditional, and Sequential Logit;
- zero-inflated and hurdle Poisson;
- Gaussian sample selection; and
- fixed-boundary censored quantile regression.

See [Experimental model status](docs/EXPERIMENTAL_MODELS.md) for current evidence and
remaining promotion gates.

---

# Data and Identification Conventions

## Binary models

- `y` must be one-dimensional, contain both classes, and be coded exactly 0/1.
- Add an explicit constant column when an intercept is required.
- Ordinary MLE models reject separation; Firth is an explicit alternative.
- Firth and ridge require full rank and `n > p`; they are not high-dimensional methods.

## Ordinal models

- Do not include a constant in `X`.
- Supply `category_order=` for substantive labels unless `y` is an ordered pandas
  categorical.
- Every declared category must be observed.
- Save category maps because reversing the order changes the model.

## Panel models

- Entity labels must be non-missing but need not be numeric.
- Dynamic time must be numeric with unique entity-time pairs.
- Static random-effects models support unbalanced groups.
- BUC uses within-entity information and rejects time-invariant or collinear regressors.
- SPJ Probit requires a balanced common time grid.
- MRV dynamic fixed effects requires its exact four-observation design.

## Shared design rules

- Native estimators accept dense finite numeric arrays or DataFrames.
- DataFrame feature names must be unique.
- Prediction DataFrames must match fitted columns and order.
- Formula parsing and automatic categorical encoding are not part of `0.1.0a1`.
- Weight, covariance, offset, exposure, and cluster support are family-specific.
- The package does not silently impute missing values or rebuild a preprocessing pipeline.

See [Category ordering](docs/CATEGORY_ORDER.md) and the individual model guides for the
complete input contracts.

---

# Results and Post-Estimation

Supported result objects expose a model-appropriate subset of:

- `params` and `all_params`;
- `standard_errors`, `zstats`, and `pvalues`;
- `covariance`, `vcov()`, and `conf_int()`;
- `loglike`, `aic`, and `bic` where meaningful;
- `converged`, `inference_valid`, and score diagnostics;
- `summary_frame()`;
- `predict()` and `predict_proba()`; and
- family-specific effect, posterior, distribution, or survival methods.

Check `converged` and `inference_valid` before interpreting a coefficient table. A result
can contain useful point estimates while ordinary normal-approximation inference is
unavailable.

## Function-style facade

```python
from limiteddepkit import (
    confint,
    lincom,
    marginal_effects,
    margins,
    predict,
    predict_proba,
    summary_frame,
    vcov,
    wald_test,
)

table = summary_frame(ordered_logit)
covariance = vcov(ordered_logit)
intervals = confint(ordered_logit)
probabilities = predict_proba(ordered_logit, X_ordinal_new)
labels = predict(ordered_logit, X_ordinal_new)
effects = marginal_effects(ordered_logit, X_ordinal)
average = margins(ordered_logit, X_ordinal, at="overall")
difference = lincom(ordered_logit, {"income": 1.0, "age": -1.0})
joint = wald_test(ordered_logit, [{"income": 1.0}, {"age": 1.0}])
```

Helpers reject unknown parameter names and inherit the fitted covariance's validity.

---

# Prediction Targets

| Family | Main prediction target | Important distinction |
| --- | --- | --- |
| Binary/Firth | `Pr(y=0)` and `Pr(y=1)` | Firth plug-in probabilities are not FLIC/FLAC calibration |
| Pooled/flexible ordinal | One probability per category | Effects are category-specific |
| Static RE ordinal | Population average by default | Conditional and posterior targets are separate |
| BUC FE Ordered Logit | Common index `X beta` only | Thresholds and probabilities are not identified |
| Dynamic RE ordinal | One-step population-average probabilities | Explicit lagged category is required |
| Count | Expected count | New offset/exposure must be supplied explicitly |
| Tobit | Mean recorded response by default | Latent mean and censoring probability are separate |
| Truncated regression | Mean conditional on selection | Latent mean and selection probability are separate |
| Interval regression | Latent mean | Predictive intervals are for future latent outcomes |
| Duration | Mean, survival, hazard, cumulative hazard, quantile | Time grids and spell assumptions must be reported |

For random effects, do not interchange:

- population-average prediction;
- conditional prediction at `u = 0` or another supplied effect;
- posterior-mean plug-in prediction; and
- exact posterior-predictive integration.

---

# Plotting and Reporting

Pooled and flexible ordinal results support probability and marginal-effect plots when
the `[plots]` extra is installed:

```python
from limiteddepkit import plot_marginal_effects, plot_probabilities

ax1 = plot_probabilities(ordered_logit, X_ordinal, feature="income")
ax2 = plot_marginal_effects(ordered_logit, X_ordinal, feature="income")
```

Other covariates are held at their means. These helpers do not currently cover binary or
panel results.

Supported ordinal results can be adapted to `universal-output-hub`:

```python
from limiteddepkit import add_to_outputhub, to_outputhub_model
from universal_output_hub import OutputHub

model = to_outputhub_model(
    ordered_logit,
    name="Ordered rating",
    depvar="rating",
)

hub = OutputHub("Ordinal analysis")
add_to_outputhub(
    hub,
    ordered_logit,
    name="Ordered rating",
    X=X_ordinal,
)
```

Supplying `X` adds delta-method average-marginal-effect inference for supported pooled or
flexible ordinal models. Panel models can be exported as coefficient models but do not
receive a pooled AME table.

---

# Probability-Aware ML and Diagnostic Layer

`limiteddepkit.ml` is experimental and dependency-light. It operates around fitted
econometric models without changing their likelihoods or identification assumptions.

## Outcome-aware scores

| Outcome | Available diagnostics |
| --- | --- |
| Binary | log loss, Brier score, ROC AUC, accuracy, balanced accuracy |
| Multinomial | multiclass log loss and Brier score |
| Ordinal | multiclass scores, Ranked Probability Score, ordered-category MAE |
| Grouped choice | choice-set log loss, Brier score, hit rate |
| Continuous/censored mean | MAE, RMSE, prediction bias |
| Count | MAE, RMSE, Poisson deviance, zero-rate calibration |
| Quantile | check loss |
| Duration | concordance, IPCW Brier curves, integrated Brier, dynamic AUC |
| Selection | selection scores plus selected-outcome RMSE |

Probability scores are primary for categorical models. Accuracy is a secondary
description, not a replacement for calibration or proper scoring rules.

## Leakage-aware splitting and comparison

```python
from limiteddepkit import GeneralizedOrderedLogit, OrderedLogit
from limiteddepkit.ml import StratifiedKFold, compare_models

comparison = compare_models(
    {
        "ordered_logit": OrderedLogit,
        "generalized": GeneralizedOrderedLogit,
    },
    X_ordinal,
    y_ordinal,
    splitter=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2026,
    ),
    outcome="ordinal",
)

print(comparison.table)
print(comparison.best_model)
```

Available split designs include independent K-fold, stratified K-fold, complete-group
holdout, entity holdout, stratified group splitting, repeated designs, and forward panel
windows. Compared models reuse the same materialized folds. Fold-local preprocessing must
be supplied as a factory so each transformer is fitted only on training rows.

Models are ranked only when configured econometric gates pass. Non-convergence,
invalid inference, non-crossing failure, or non-finite primary scores remain visible and
can exclude a candidate from automatic ranking.

## Nested tuning and uncertainty-aware comparison

The layer supports nested cross-validation, the one-standard-error rule, paired fold
differences, and observation- or entity-bootstrap intervals. Ridge strengths and neural
hyperparameters should be selected inside the inner loop, never on the outer test folds.

## Optional residual neural challenger

`ResidualBinaryMLP` is a prediction challenger, not an econometric estimator.

```python
from limiteddepkit.ml import ResidualBinaryMLP, StratifiedKFold, cross_validate

neural_cv = cross_validate(
    lambda: ResidualBinaryMLP(
        hidden_width=32,
        n_blocks=2,
        dropout=0.15,
        weight_decay=1e-3,
        patience=30,
        temperature_scaling=True,
        random_state=2026,
    ),
    X_binary,
    y_binary,
    splitter=StratifiedKFold(5, shuffle=True, random_state=90210),
    outcome="binary",
    require_inference_valid=False,
)
```

The model uses residual GELU/LayerNorm blocks, AdamW, gradient clipping, train-only
standardization, early stopping, optional temperature scaling, and Monte Carlo dropout.
Its result intentionally reports `inference_valid=False`. The current internal validation
split is iid-stratified; grouped and chronological neural validation are not claimed.

See [Probability-aware validation workflows](docs/ML_WORKFLOWS.md) for splitter,
alignment, calibration, survival, and uncertainty contracts.

---

# Verification Philosophy

Verification is a core design principle of `limiteddepkit`.

Where equivalent estimands exist, maintained tests compare with established Python, R,
and Stata implementations. Where no equivalent external estimator exists, the package
uses exact likelihood identities, independent kernels, deterministic recovery,
invariance checks, and explicit refusal tests.

The purpose is not to label every numerical check “parity.” The purpose is to state what
was compared, under which specification, and what remains unverified.

---

# Current Validation Status

| Component | Status | Evidence boundary |
| --- | --- | --- |
| Binary Logit/Probit | Maintained Python reference gate | Statsmodels coefficients, likelihood, covariance, probabilities, criteria, AMEs |
| Firth Binary Logit | Maintained Python reference gate; separate R application pass | Exact separated table, `firthmodels` profile checks, and independent R adjusted-score replication |
| Ordered Logit/Probit | Maintained Python reference gate | Aligned Statsmodels `OrderedModel` specifications |
| Generalized/PPO Logit | Maintained recovery and constraint gates | Known bounded-support DGPs and non-crossing behavior |
| Static RE Ordered Logit | Maintained recovery/numerical gate; included in prior Stata/R certificates | Quadrature, invariance, posterior identities, aligned certified benchmark |
| Static RE Ordered Probit | Maintained Python identity/recovery gate; separate R application pass | Conditional kernel, normal-convolution identity, and aligned `ordinal::clmm` application |
| BUC FE Ordered Logit | Maintained conditional-likelihood Python gate; separate R application pass | Exact enumeration, Statsmodels construction, and R `survival::clogit` likelihood identity |
| Dynamic RE Ordered Logit | Maintained recovery/numerical gate; included in prior Stata/R certificates | Exact specification, quadrature, invariance, initial/gap rules |
| Poisson/NB2 | Maintained Statsmodels gate; separate R application pass | Coefficients, likelihood, dispersion, offsets/exposure, weights, covariance, and R package replication |
| Gaussian censoring family | Maintained identity/Statsmodels gate; separate R application pass | Likelihood identities, reflections, OLS submodel, robust covariance, and aligned R applications |
| Duration family | Maintained Statsmodels/SciPy/identity gate; separate R application pass | Poisson-exposure, distribution contributions, entry, weights, covariance, tails, and aligned R applications |
| SPJ FE Ordered Probit | Experimental Python reference/recovery gate | Uncorrected dummy likelihood, jackknife identity, bootstrap path; no Stata/R certificate |
| MRV dynamic FE Ordered Logit | Experimental conditional-sample Python gate | Path odds, recovery, Statsmodels pseudo-sample optimization/covariance; no independent construction parity |
| ML metrics and ordinary splitters | Maintained scikit-learn comparison gate | Directly equivalent definitions plus documented intentional differences |
| Residual neural challenger | Maintained optional runtime/contract gate | Training/calibration/MC-dropout tests; no numerical-parity or inference claim |

Newer promotions are not automatically covered by older certificates. Read each family
guide and [Validation strategy](docs/VALIDATION.md) for the exact claim boundary.

---

# Eight-Family Stata and R Certificates

The pre-expansion binary/ordinal surface has four completed external-software tracks.

The eight models are:

1. Binary Logit;
2. Binary Probit;
3. Ordered Logit;
4. Ordered Probit;
5. Generalized Ordered Logit;
6. Partial Proportional Odds;
7. static Random-effects Ordered Logit; and
8. dynamic Random-effects Ordered Logit.

| Track | Purpose | Recorded result |
| --- | --- | --- |
| Controlled synthetic — Stata 17 | Strict benchmark-specific certification | **PASS — 82/82** |
| Frozen public-data application — Stata 17 | Applied data-handling check | **PASS — 82/82** |
| Controlled synthetic — pinned R 4.5.1 | Independent implementation check | **PASS — 110/110** |
| Frozen public-data application — pinned R 4.5.1 | Applied check on the same observations | **PASS — 110/110** |

The controlled track is the certification benchmark. The real-data track is an
application check and does not broaden the certification claim. The comparisons were
completed on 14 July 2026; Stata used `gologit2` 3.2.8, and R used aligned `glm.fit`,
`MASS::polr`, `VGAM::vglm`, and `ordinal::clmm` specifications.

Observed maximum absolute differences across the completed reports were:

| Track | Implementation | Estimate | Standard error | Covariance | Log likelihood | Probability |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Controlled | Stata | `4.16e-6` | `3.03e-7` | `2.14e-7` | `3.53e-9` | `1.56e-6` |
| Controlled | R | `1.02e-5` | `1.26e-5` | `3.85e-6` | `1.53e-4` | `3.07e-6` |
| Real data | Stata | `1.47e-5` | `2.11e-6` | `2.36e-6` | `2.58e-9` | `2.77e-6` |
| Real data | R | `3.19e-4` | `2.05e-4` | `2.29e-4` | `5.35e-4` | `5.05e-5` |

These values are inside their declared model-family gates; they are not universal
tolerances for future analyses.

The certificates do **not** cover Firth, censoring, counts, duration,
Random-effects Ordered Probit, BUC Fixed-effects Ordered Logit, SPJ Fixed-effects Ordered
Probit, or MRV Dynamic Fixed-effects Ordered Logit. A separate application suite now adds
external Python/R evidence for the stable promoted families; it is not an extension of
the controlled certificate. The two experimental fixed-effects estimators retain only
their model-specific Python/reference evidence summarized above.

See the [evidence index](validation/PARITY_EVIDENCE.md),
[Stata harness](validation/stata/README.md), and
[R harness](validation/r/README.md) for hashes, mappings, tolerances, commands, and
allowed claim language.

---

# Promoted-Family Python/R Application Evidence

The stable families added after the original eight-model surface have a separate
public-data application harness. On 15 July 2026, all 12 Python and R fits completed and
all **120/120** registered comparisons passed their model-specific gates.

The evidence comprises:

- seven comparisons with industrial R packages;
- three independent likelihood or adjusted-score implementations; and
- two exact likelihood or pseudo-sample identities.

These labels are deliberate: identity and independently coded likelihood checks are
useful replication evidence, but they are not package-to-package agreement. The
applications use empirical LBW, infant-mortality, labor-supply, cancer-duration,
TVSFPORS, and NLSWORK observations. The interval-regression check instead uses the
official fictional `womenwage2` software fixture because it exercises native open
endpoints; it must not be described as empirical data.

The corresponding manual Stata application run also passed its required checks:
**140/140**, with one explicit skip for Gamma duration because Stata's generalized
Gamma is not the same ordinary Gamma likelihood. Firth Binary Logit ran through
optional `firthlogit` and passed its aligned Stata checks. This result is application evidence only.
It neither extends the older controlled certification nor claims universal
numerical equality across datasets, specifications, optimizers, covariance
estimands, or software versions.

See the [promoted-family application harness](validation/promoted/README.md) and
[cross-software evidence index](validation/PARITY_EVIDENCE.md) for the model map,
provenance, commands, tolerances, recorded hashes, and allowed claim language.

---

# Practical Modelling Guidance

## Start with the simplest defensible specification

A useful ordinal progression is:

1. pooled Ordered Logit or Probit;
2. inspect category order and the proportional-odds restriction;
3. relax named slopes with Partial Proportional Odds;
4. use Generalized Ordered Logit only when the broader flexibility is justified;
5. add random effects or fixed effects only when the panel estimand is required; and
6. introduce dynamic state dependence only with an explicit initial-condition strategy.

## Keep observation rules separate

- Use Tobit when observations remain at a censoring boundary.
- Use truncated regression when observations outside support are absent.
- Use interval regression for exact, grouped, or open-tail bounds.
- Use duration likelihoods for event-time processes, not merely because a continuous
  response is positive.

## Name the prediction target

Report whether predictions are population averaged, conditional on a supplied random
effect, posterior-mean plug-ins, exact posterior predictive, latent, recorded, selected,
or survival-scale quantities.

## Treat dynamic state dependence cautiously

Lagged outcome association can reflect genuine state dependence, unobserved heterogeneity,
initial conditions, or specification error. A fitted state coefficient is not automatic
causal evidence.

## Preserve the preprocessing design

Save category maps, dummy levels, interactions, scales, feature order, offsets, exposure,
entry times, and sample filters. The package does not store a formula or preprocessing
pipeline for later reconstruction.

---

# Recommended Reporting

For publication or review, report:

- exact package version and estimator class;
- outcome coding and ordinal category order;
- sample construction, exclusions, and missing-data handling;
- design columns and transformation rules;
- explicit binary intercept choice or confirmation that ordinal `X` has no constant;
- observations, entities/clusters, effective weighted observations, and contributing
  conditional groups where relevant;
- optimizer controls, convergence, score diagnostics, and `inference_valid`;
- covariance type, finite-sample correction, confidence level, and bootstrap details;
- coefficient, threshold, dispersion, scale, shape, or random-effect parameterization;
- offset, exposure, weight, censoring, truncation, delayed-entry, or duration convention;
- quadrature method and point count for random-effects models;
- prediction target and whether random effects are integrated, supplied, or posterior;
- dynamic time-step, initial-condition controls, lag-category specification, and rows
  removed after gaps;
- BUC/SPJ/MRV identification limitations when those estimators are used; and
- exact external-software track, versions, mappings, data manifest, and tolerance when a
  parity statement is reported.

Archive the manifest, prepared data or permitted provenance, Python references, Stata/R
exports, logs, comparator reports, software versions, and certificates with any external
parity claim.

---

# Package Scope

`limiteddepkit` contains models whose response, observation rule, duration, or choice
mechanism is intrinsically limited. It is not a general home for every econometric model.

The following were deliberately kept outside the installed package:

- ordinary linear IV/treatment-effect 2SLS;
- iid Gaussian mixtures and generic switching regression;
- ordinary uncensored quantile regression; and
- a generic generalized additive model.

The historical 2SLS and mixture sources are retained under `_out_of_scope/` for possible
migration to causal/IV and regime-model packages. Future nonlinear basis support should
serve an in-scope outcome family rather than create a generic `GAM` label without a
limited-response estimand.

See [Package scope](docs/PACKAGE_SCOPE.md) for the complete keep/extract rationale.

---

# Roadmap

Near-term work remains evidence-driven:

- archive completed external-software evidence with an exact repository revision;
- rerun Stata/R tracks after estimator, mapping, or reference changes;
- add independent replication for newly promoted panel families where equivalent
  software exists;
- strengthen SPJ Probit recovery and bootstrap evidence on longer panels;
- independently replicate the MRV conditional-history construction before considering
  promotion;
- broaden weight, covariance, exposure, entry, and prediction contracts only with an
  explicit estimand and maintained tests;
- improve reporting around model-specific convergence and identification diagnostics;
  and
- preserve the stable/experimental boundary instead of expanding the root merely to
  increase model count.

No experimental estimator has a promised promotion date.

---

# Documentation

Start with the [documentation index](docs/README.md).

Model guides:

- [Binary models](docs/BINARY_MODELS.md)
- [Small-sample binary models](docs/SMALL_SAMPLE_MODELS.md)
- [Ordinal models](docs/ORDINAL_MODELS.md)
- [Category ordering](docs/CATEGORY_ORDER.md)
- [Panel ordinal models](docs/PANEL_ORDINAL.md)
- [Fixed-effects ordinal panels](docs/FIXED_EFFECTS_ORDINAL.md)
- [Dynamic random-effects ordinal models](docs/DYNAMIC_ORDINAL.md)
- [Dynamic fixed-effects Ordered Logit](docs/DYNAMIC_FIXED_EFFECTS_ORDINAL.md)
- [Gaussian censoring models](docs/CENSORING_MODELS.md)
- [Count models](docs/COUNT_MODELS.md)
- [Duration models](docs/DURATION_MODELS.md)

Workflow and evidence guides:

- [Probability-aware validation and ML workflows](docs/ML_WORKFLOWS.md)
- [Validation strategy](docs/VALIDATION.md)
- [Experimental model status](docs/EXPERIMENTAL_MODELS.md)
- [Ecosystem compatibility](docs/ECOSYSTEM_COMPATIBILITY.md)
- [Dynamic ordinal numerical validation](docs/DYNAMIC_ORDINAL_VALIDATION.md)
- [Cross-software evidence index](validation/PARITY_EVIDENCE.md)
- [Stata parity harness](validation/stata/README.md)
- [R parity harness](validation/r/README.md)
- [Promoted-family application harness](validation/promoted/README.md)

Project processes:

- [Contributing](CONTRIBUTING.md)
- [Release checklist](RELEASING.md)
- [Changelog](CHANGELOG.md)
- [Security policy](SECURITY.md)

---

# Citation

If you use `limiteddepkit`, cite the software and the exact version used.
Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

Suggested interim citation:

```text
Akanbi, Oluwajuwon Mayomi.

limiteddepkit: Limited-dependent-variable models for Python.

Version 0.1.0a1, 2026.
```

No DOI or archival identifier is asserted before one is assigned. Add the permanent
identifier when a public archive exists.

---

# License

MIT License.

See [LICENSE](LICENSE) for details.
