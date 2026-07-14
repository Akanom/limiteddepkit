# Probability-aware validation and ML-style workflows

`limiteddepkit.ml` is an **experimental** workflow layer around the package's fitted
econometric models. Its one optional neural challenger is prediction-only; it does not
replace identification, likelihood, inference, or cross-software validation. The layer's
main purpose is to make out-of-sample probability evaluation, leakage-safe splitting,
and model comparison reproducible.

The module has no scikit-learn dependency.

## Outcome-aware scores

| Outcome | Scores |
| --- | --- |
| Binary | Log loss, Brier score, ROC AUC, accuracy, balanced accuracy |
| Multinomial | Multiclass log loss and Brier score |
| Ordinal | Multiclass scores, Ranked Probability Score, ordered-category MAE |
| Grouped conditional choice | Choice-set log loss, Brier score, and hit rate |
| Continuous/censored mean | MAE, RMSE, prediction bias |
| Count | MAE, RMSE, Poisson deviance, zero-rate calibration where available |
| Quantile | Quantile/check loss |
| Duration | Harrell concordance plus explicit training-fold IPCW concordance, Brier, integrated Brier, and dynamic AUC diagnostics |
| Selection | Selection log loss/Brier score and selected-outcome RMSE |

Probability scores are primary for categorical models. Classification accuracy is a
secondary descriptive measure and should not replace calibration or proper scoring rules.

## Splitters and aggregation

- `KFold` is for independent, exchangeable rows.
- `StratifiedKFold` retains every observed outcome class in each categorical test fold.
- `StratifiedGroupKFold` starts with deterministic greedy class/row balancing while
  keeping every entity intact. If local repair cannot give every fold every category,
  an exact mixed-integer coverage fallback searches for a feasible minimum-move
  assignment; structurally infeasible or solver-unresolved designs are rejected.
- `EntityHoldoutSplit`/`GroupKFold` hold out complete entities or choice sets.
- `ForwardPanelSplit` uses expanding calendar windows and validates exact within-entity
  continuity.
- `RepeatedKFold`, `RepeatedStratifiedKFold`, and `RepeatedGroupKFold` repeat only the
  split design; they do not make overlapping fold scores independent.

Do not use iid row splitting for repeated entities, long-format choices, or dynamic
panels. Compared models reuse one materialized fold design even when a shuffled splitter
has no seed. Summary means are unweighted macro-averages across folds; inspect the fold
table whenever test-fold sizes differ materially. Use `weighted_summary_frame()` for a
test-size-weighted descriptive summary and `pooled_out_of_fold_predictions()` to average
repeated predictions once per original row. The row-position key keeps duplicated pandas
index labels distinct. When nested count selections expose different optional prediction
components, each pooled column reports its own `__count` and `__weight_sum` support.

### Fold-local preprocessing

Pass a factory—not an already fitted transformer—when scaling, encoding, or imputing is
needed. A fresh transformer is fitted on each training partition and stored with its fold
evidence:

```python
from sklearn.preprocessing import StandardScaler

scaled_cv = cross_validate(
    BinaryLogit,
    X,
    y,
    splitter=StratifiedKFold(5, shuffle=True, random_state=2026),
    outcome="binary",
    # with_mean=False preserves an explicitly supplied constant column
    transformer_factory=lambda: StandardScaler(with_mean=False),
)
```

The transformed design must preserve row counts. Native `limiteddepkit` estimators expect
dense numeric designs, so their transformer must also return dense output. The workflow
can carry a SciPy sparse matrix only when the downstream estimator and its prediction
methods explicitly accept sparse input; sparse workflow plumbing does not add sparse
support to native estimators. Feature construction that depends on entity histories or
time ordering still needs a purpose-built fold-safe callback.

### Uncertainty-aware comparisons

Fold means alone do not show whether a small difference is stable. The uncertainty helpers
keep the comparison paired and define a positive difference as favoring the candidate:

```python
from limiteddepkit.ml import (
    binary_log_loss,
    paired_bootstrap_interval,
    paired_fold_score_differences,
)

fold_differences = paired_fold_score_differences(
    candidate_log_loss,
    reference_log_loss,
    higher_is_better=False,
)

interval = paired_bootstrap_interval(
    candidate_probabilities,
    reference_probabilities,
    y_true=y,
    scorer=binary_log_loss,
    clusters=entity,  # omit only for genuinely independent rows
    random_state=2026,
)
```

The percentile interval resamples complete entities when `clusters=` is supplied. A
weighted fold standard error assumes independent fold scores and is not valid inferential
evidence for overlapping repeated CV; use the paired observation/entity bootstrap instead.

### Alignment with scikit-learn conventions

The module has no runtime scikit-learn dependency, but maintained validation compares
every directly equivalent metric and ordinary splitter with scikit-learn. Unshuffled
`KFold` and `StratifiedKFold` follow the same fold allocation on aligned inputs;
stratified total fold sizes and within-class counts each differ by at most one.

There are deliberate boundaries:

- integer seeds are reproducible within limiteddepkit, but shuffled indices are not
  promised to equal scikit-learn's because the RNG implementations differ;
- `EntityHoldoutSplit` greedily balances observation counts and keeps entities intact,
  so its group assignment and tie-breaking can differ from scikit-learn `GroupKFold`;
- stratification rejects a category with fewer observations than folds instead of
  allowing probability-scoring folds that omit that category;
- log loss uses a configurable default clipping value of `1e-15`, while scikit-learn
  uses dtype-dependent machine precision; align `eps` and dtype precision when testing
  exact endpoints; and
- category order comes from explicit `labels` or probability DataFrame columns and must
  be aligned before comparing a reference implementation that sorts labels.

See [Validation strategy](VALIDATION.md#python-reference-package-gates) for numerical
results and the end-to-end Statsmodels/scikit-learn probability check.

## Pooled binary and ordinal validation

```python
from limiteddepkit import GeneralizedOrderedLogit, OrderedLogit
from limiteddepkit.ml import StratifiedKFold, compare_models

comparison = compare_models(
    {
        "ordered_logit": OrderedLogit,
        "generalized": GeneralizedOrderedLogit,
    },
    X,
    y,
    splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=2026),
    outcome="ordinal",
)

print(comparison.table)
print(comparison.best_model)
```

Models are ranked only when every fold satisfies the configured econometric gates.
Non-convergence, invalid ordinary inference, and non-crossing failure remain visible in
the fold evidence and exclude the model from automatic ranking. The selected primary
metric must also be finite in every fold; for example, ROC AUC cannot rank a validation
design containing a one-class test fold.

## Small-sample models and nested tuning

Ordinary `BinaryLogit` remains the stable MLE reference. Experimental
`FirthBinaryLogit` provides finite bias-reduced estimates under separation;
`RidgeBinaryLogit` and `RidgeOrderedLogit` provide explicit L2 regularization with
approximate penalized-estimator covariance. Penalized likelihoods are not ordinary-MLE
likelihoods and are not interchangeable inputs to AIC/BIC comparisons.

All three experimental estimators require more observations than fitted parameters
(`n > p`) and a full-column-rank design. They are therefore small-sample/separation
tools, not high-dimensional estimators and not remedies for exact collinearity. Reduce or
re-specify a design that has `p >= n` or rank-deficient columns before fitting them.

Penalty strengths must be chosen inside nested CV. In the example below, lower
`complexity` means simpler/stronger regularization; each outer test fold remains untouched
until the inner one-standard-error decision is complete:

```python
from limiteddepkit import BinaryLogit
from limiteddepkit.experimental import RidgeBinaryLogit
from limiteddepkit.ml import (
    StratifiedKFold,
    TuningCandidate,
    nested_cross_validate,
)

candidates = {
    "ridge_10": TuningCandidate(
        RidgeBinaryLogit,
        fit_kwargs={"penalty": 10.0},
        complexity=0.0,
    ),
    "ridge_1": TuningCandidate(
        RidgeBinaryLogit,
        fit_kwargs={"penalty": 1.0},
        complexity=1.0,
    ),
    "mle_reference": TuningCandidate(BinaryLogit, complexity=2.0),
}

nested = nested_cross_validate(
    candidates,
    X,
    y,
    outer_splitter=StratifiedKFold(5, shuffle=True, random_state=2026),
    inner_splitter_factory=lambda: StratifiedKFold(
        4, shuffle=True, random_state=90210
    ),
    outcome="binary",
    primary_metric="log_loss",
    selection_rule="one_se",
)

print(nested.fold_frame())
print(nested.selected_models)
```

Pre-specify the candidate path and complexity ordering before looking at outer scores.
Firth has no ridge-strength hyperparameter; it can be included as a pre-specified model
candidate when the scientific comparison calls for it, but it should not become an
automatic fallback chosen after observing failed outer folds.

## Optional residual neural prediction challenger

`ResidualBinaryMLP` is the deliberately separate, optional advanced nonlinear prediction
challenger. Install it with `pip install limiteddepkit[neural]`. PyTorch is imported only
when `fit()` is called; the ordinary package and all econometric estimators remain
dependency-light.

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
    X,
    y,
    splitter=StratifiedKFold(5, shuffle=True, random_state=90210),
    outcome="binary",
    require_inference_valid=False,
)
```

The model uses residual GELU/LayerNorm blocks, AdamW, gradient clipping, train-only
standardization, deterministic internal early stopping, and optional scalar temperature
scaling. The unweighted binary loss is the default because log-loss/Brier evaluation needs
event probabilities; opt-in class weighting targets a cost-weighted probability that
temperature scaling need not fully correct.

Every returned result has `training_completed=True`, meaning training produced a finite
checkpoint. The more conservative `converged` flag is true only when patience-based
early stopping records validation-loss stabilization; merely reaching `max_epochs` does
not count as convergence. Generic CV therefore still applies its convergence gate. If a
prediction-only sensitivity analysis deliberately sets `require_converged=False`, report
that choice and inspect the training history rather than relabelling completion as
convergence.

The fitted result exposes `predict_proba_uncertainty()` using Monte Carlo dropout. Its
bands are approximate conditional model uncertainty—not econometric confidence or
prediction intervals—and omit sampling, model-selection, and calibration uncertainty.
`inference_valid=False` is intentional, so prediction-only CV must opt out of the
inference gate explicitly as above. Architecture, regularization, early-stopping, and
calibration choices belong inside `nested_cross_validate`; only untouched outer-fold
predictions are performance evidence.

This first neural implementation is restricted to independent rows. It rejects
`entity=` and `time=` because its internal validation split is iid-stratified; grouped or
chronological data need a future group/time-aware neural trainer. The same internal
validation partition is used for epoch selection and optional temperature fitting, so its
calibrated validation loss is tuning evidence rather than held-out performance.

An isolated Python 3.13/PyTorch 2.13.0 run on 14 July 2026 passed all 21 neural tests,
including deterministic fitting, calibration, nested CV, and Monte Carlo dropout. CI has
a dedicated Python 3.13 `[test,neural]` job so the optional training path is maintained.
This is runtime and contract evidence; it is not a neural numerical-parity, recovery,
uncertainty-coverage, or inferential claim.

## Calibration diagnostics

Held-out probabilities can be accurate in rank yet poorly calibrated. Binary diagnostics
include a reliability table, grouped Murphy Brier decomposition, and logistic calibration
intercept/slope. Ordinal calibration repeats the intercept/slope diagnostic for each
cumulative event `Y <= category`:

```python
from limiteddepkit.ml import (
    binary_brier_decomposition,
    binary_calibration_intercept_slope,
    binary_reliability_table,
    ordinal_cumulative_calibration,
)

binary_calibration = binary_calibration_intercept_slope(y_test, probability_one)
reliability = binary_reliability_table(
    y_test, probability_one, n_bins=8, strategy="quantile"
)
brier_parts = binary_brier_decomposition(y_test, probability_one, n_bins=8)
ordinal_calibration = ordinal_cumulative_calibration(
    y_ordinal_test,
    ordinal_probabilities,
)
```

Perfect binary recalibration has intercept zero and slope one. Estimate these diagnostics
from held-out/OOF probabilities, not the observations used to fit the prediction model.
The grouped Brier decomposition reports its binning residual explicitly rather than
pretending the binned reconstruction is always the raw score.

## Grouped conditional choice

Conditional Logit uses long-format alternative rows, so complete choice sets—not
individual alternatives—are the independent units. The same group labels are passed to
the estimator, predictor, and choice-set scorer:

```python
from limiteddepkit.experimental import ConditionalLogit
from limiteddepkit.ml import GroupKFold, cross_validate

choice_cv = cross_validate(
    ConditionalLogit,
    X_long,
    chosen,
    splitter=GroupKFold(n_splits=5, shuffle=True, random_state=2026),
    entity=choice_set,
    outcome="choice",
    prediction_target="new_entity",
    fit_context={"groups": choice_set},
    predict_context={"groups": choice_set},
    score_context={"groups": choice_set},
)
```

The scorer requires exactly one chosen alternative and probabilities summing to one
inside every held-out choice set.

## Two different panel prediction questions

Panel validation must state the target explicitly.

### New entities

Use complete-entity holdout and population-average probabilities when the question is:
"How well will this model predict an entity that was not used for estimation?"

```python
from limiteddepkit import RandomEffectsOrderedLogit
from limiteddepkit.ml import EntityHoldoutSplit, cross_validate

new_entity_cv = cross_validate(
    RandomEffectsOrderedLogit,
    X,
    y,
    splitter=EntityHoldoutSplit(n_splits=5),
    entity=entity,
    prediction_target="new_entity",
    outcome="ordinal",
)
```

### Future observations for known entities

Use an expanding time split and posterior probabilities based only on the training
history when the question is: "How well will this model predict a later observation for
an entity already observed?"

```python
from limiteddepkit.ml import ForwardPanelSplit

known_entity_cv = cross_validate(
    RandomEffectsOrderedLogit,
    X,
    y,
    splitter=ForwardPanelSplit(
        n_splits=4,
        min_train_periods=3,
        test_periods=1,
        time_step=1,
    ),
    entity=entity,
    time=time,
    prediction_target="known_entity_future",
    outcome="ordinal",
)
```

`ForwardPanelSplit` rejects duplicate, out-of-order, and noncontiguous entity histories.
It never joins observations across an internal time gap. Known-entity posterior
evaluation also verifies, entity by entity, that every training time precedes every test
time; a random within-entity split cannot be mislabeled as a forecast.

## Dynamic ordered panels

Dynamic posterior forecasting additionally needs the lagged category for every test row.
Provide it as row-aligned prediction context; it is sliced by fold:

```python
dynamic_cv = cross_validate(
    DynamicRandomEffectsOrderedLogit,
    X,
    y,
    splitter=ForwardPanelSplit(n_splits=4, min_train_periods=4),
    entity=entity,
    time=time,
    outcome="ordinal",
    prediction_target="known_entity_future",
    predict_context={"lagged_y": lagged_y},
)
```

Default dynamic posterior validation permits only a one-period test window. It verifies
that each supplied lag equals that entity's last observed training outcome and rejects an
embargo gap, so future or embargoed outcomes cannot become predictors. Initial outcomes,
initial covariates, and entity means come from each training fit and cannot be replaced by
full-sample summaries. Use an explicit `predict=` callback for recursive multi-step
forecasts; the callback must define how predicted states feed later horizons.

## Duration example

Row-aligned fit and scoring context are split without leakage:

```python
from limiteddepkit.experimental import WeibullDuration
from limiteddepkit.ml import StratifiedKFold, cross_validate

duration_cv = cross_validate(
    WeibullDuration,
    X,
    duration,
    splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=2026),
    split_y=event,
    outcome="duration",
    fit_context={"event": event},
    score_context={"event": event},
)
```

The generic CV score retains Harrell concordance and its explicitly known-status horizon
Brier diagnostic. For censoring-aware comparisons, estimate reverse-Kaplan-Meier weights
on each training fold and pass that object to the IPCW functions:

```python
from limiteddepkit.ml import (
    cumulative_dynamic_auc,
    fit_censoring_distribution,
    integrated_brier_score,
    ipcw_concordance_index,
    time_dependent_brier_scores,
)

censoring = fit_censoring_distribution(train_duration, train_event)
uno_c = ipcw_concordance_index(
    test_duration,
    test_event,
    test_risk,
    censoring=censoring,
    tau=5.0,
)
brier_curve = time_dependent_brier_scores(
    test_duration,
    test_event,
    test_survival_matrix,
    times=evaluation_times,
    censoring=censoring,
)
ibs = integrated_brier_score(
    test_duration,
    test_event,
    test_survival_matrix,
    times=evaluation_times,
    censoring=censoring,
)
dynamic_auc = cumulative_dynamic_auc(
    test_duration,
    test_event,
    test_risk,
    times=evaluation_times,
    censoring=censoring,
)
```

The censoring distribution refuses times beyond training follow-up and zero censoring
survival at a required weight. This prevents silent extrapolation and test-fold leakage.
These metrics strengthen evaluation of the existing duration families; they do not by
themselves justify adding another parametric duration distribution.

## Censored quantiles and interval outcomes

The fitted quantile is inherited from a `CensoredQuantileRegressionResult`, so it need not
be duplicated in score context:

```python
from limiteddepkit.experimental import CensoredQuantileRegression
from limiteddepkit.ml import KFold, cross_validate

quantile_cv = cross_validate(
    lambda: CensoredQuantileRegression(quantile=0.25, lower=0.0),
    X,
    y,
    splitter=KFold(5, shuffle=True, random_state=2026),
    outcome="auto",
)
```

Check loss at different quantiles is not one common ranking target, so comparison rejects
models fitted at different quantiles. Censored-quantile inference is often invalid unless
its bootstrap succeeds; scores remain inspectable, but such folds are not eligible for
automatic ranking under the default gate.

Interval regression is deliberately not scored against an interval's lower bound as if
that bound were an observed response. Default CV therefore rejects it. Validation needs a
custom fit callback, an explicit point target available in the application or simulation,
and `outcome="continuous"`, or a future purpose-built interval scoring rule.

## Sample-selection example

`Z` and the selection indicator can be supplied as row-aligned context:

```python
from limiteddepkit.experimental import SampleSelection

selection_cv = cross_validate(
    SampleSelection,
    X,
    y,
    splitter=StratifiedKFold(n_splits=5),
    outcome="selection",
    fit_context={"Z": Z, "selection": selected},
    predict_context={"Z": Z},
    score_context={"selection": selected},
    split_y=selected,
)
```

If an estimator has a genuinely different multi-equation interface, pass explicit `fit=`
and `predict=` callbacks. The workflow fails rather than guessing an unidentified target.
Row-aligned arrays belong in `fit_context`, `predict_context`, or `score_context`; they are
sliced by fold. Unsupported user keywords are rejected so a typo cannot silently switch
an estimator back to a default specification.

## Optional industrial-package bridges

The bridge module normalizes prediction methods while importing optional packages only
when `fit()` is called. It does not claim that differently specified models share an
estimand. Statsmodels prediction semantics are therefore explicit:

```python
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from limiteddepkit.ml import sklearn_bridge, statsmodels_bridge

statsmodels_logit = statsmodels_bridge(
    sm.Logit,
    prediction_kind="probability",
    add_constant=True,
    fit_options={"disp": False},
)
sklearn_logit = sklearn_bridge(
    LogisticRegression(max_iter=2_000),
    prediction_method="predict_proba",
)
```

Scikit-learn estimators are cloned by default so a fold cannot mutate and reuse a previous
fit. Generic callbacks support model-specific lifelines, scikit-survival, Biogeme, Bambi,
or other ecosystems without pretending their fit/prediction signatures are universal.
Optional libraries remain optional; importing `limiteddepkit.ml` does not import them.

Bridges never invent econometric validity. Supply an explicit diagnostics callback when
the external result has reliable convergence/inference flags, or set
`require_converged=False` and `require_inference_valid=False` for a deliberately
prediction-only exercise. The latter makes a model scoreable; it does not certify its
inference.

## Experimental contract

- `limiteddepkit.ml` is outside the stable root API for `0.1.0a1`.
- `prediction_target="conditional"` requires a custom `predict=` callback that estimates
  effects from each training fold or uses effects known independently of held-out outcomes;
  externally supplied full-sample posterior effects are not accepted by the default CV path.
- Scores assess predictive behavior; they do not certify estimator correctness.
- Cross-validation does not justify choosing regressors, restrictions, or random-effects
  assumptions without substantive identification arguments.
- Stable promotion requires deterministic tests, documented estimands, result-contract
  compatibility, and cross-software evidence where a reference implementation exists.
