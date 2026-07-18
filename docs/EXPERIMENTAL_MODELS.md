# Experimental model status

The stable package root contains binary, ordinal, Gaussian censoring,
foundational count, parametric duration, and Firth small-sample families. The genuinely
provisional estimators in the table below are available only from
`limiteddepkit.experimental`; later sections also record promoted names retained there
temporarily as compatibility aliases.

## Validated experimental slices

| Family | Current evidence | Main remaining boundary |
|---|---|---|
| Ridge Binary/Ordinal | ridge Binary Logit parity with scikit-learn; near-unpenalized ridge Ordered Logit parity with Statsmodels | Requires `n > p` and full column rank; covariance is an approximate sandwich; penalties must be selected by nested CV |
| Multinomial, Conditional, and Sequential Logit | Exact Statsmodels likelihood/inference decompositions and arbitrary-label tests | No robust/cluster covariance or separation remedy; each model retains its standard IIA/choice-set/ordering assumptions |
| Zero-inflated and Hurdle Poisson | Tested two-part likelihoods and component-level Statsmodels parity | No offset/exposure, weights, robust covariance, or non-Poisson count component |
| Gaussian sample selection | Correct full-sample Heckman likelihood and correlated-error recovery | Gaussian homoskedastic FIML only; no two-step/robust/cluster alternative |
| Fixed-effects Ordered Probit | Unconditional entity-effect likelihood parity with Statsmodels, split-panel identity/recovery, invariance guards, and entity-bootstrap tests | Balanced even common grid; large-`N`, large-`T` approximation; diagnostic known-entity prediction only |
| Dynamic fixed-effects Ordered Logit | Fixed-`T` conditional composite-likelihood identities, recovery, and Statsmodels parity on the constructed binary sample | Exactly four outcome observations, discrete exact-stayer support, a known state cutoff, restricted state dependence, and no probability prediction |
| Fixed-boundary censored quantile regression | Powell check-loss recovery, inactive-censoring reduction to Statsmodels QuantReg, and explicit pairs-bootstrap tests | Non-convex local optimization; fixed known censoring bounds only; no inference unless bootstrap is requested and succeeds |

Passing these tests does not promote a model automatically. Promotion also
requires a settled shared result contract, documented failure modes, broader
covariance choices where relevant, and a deliberate compatibility decision.

The separate experimental `limiteddepkit.ml` workflow can cross-validate these
families with outcome-appropriate scores. Predictive evidence does not promote
an estimator or replace likelihood, inference, recovery, and cross-software
validation; see [Probability-aware validation workflows](ML_WORKFLOWS.md).

## Promoted Gaussian censoring family

`Tobit`, `TruncatedRegression`, and `IntervalRegression` have moved to the package root
and `limiteddepkit.censoring`. Their shared stable contract includes labeled regression
and scale parameters, observed-information, likelihood-score robust, and clustered
sandwich covariance, normal-approximation inference, AIC/BIC, schema-safe prediction,
and latent Gaussian CDF and predictive-interval methods. Tobit and truncated regression
support both left and right observation rules; interval regression combines exact,
finite-interval, left-censored, and right-censored observations.

The old `limiteddepkit.experimental` names remain compatibility aliases during the alpha
series, but do not indicate provisional status. Promotion does not imply a general
censoring engine: current models assume a homoskedastic Gaussian latent error, one
censoring/truncation point for Tobit/truncated regression, exogenous regressors, and
independent likelihood contributions. Cluster covariance changes inference, not the
conditional-mean or observation-rule specification.

## Promoted foundational count family

`PoissonRegressor` and NB2 (`NegativeBinomialNB2`, with `NegativeBinomial` retained as an
equivalent name) have moved to the package root and `limiteddepkit.count`. Their shared
stable contract includes additive offsets, positive exposure, exact integer-frequency
weights, analytic estimating-equation weights, observed-information/HC0/HC1/cluster
covariance, schema-safe indexed prediction, likelihood information criteria where they
are defined, and common summaries and diagnostics. Maintained gates compare Poisson and
NB2 coefficients, likelihood, dispersion, and covariance with Statsmodels. Frequency-
weighted NB2 is checked against literal replicated Statsmodels rows.

The former `limiteddepkit.experimental` Poisson/NB2 names remain compatibility aliases
during the alpha series. `ZeroInflatedPoisson` and `HurdlePoisson` are still genuinely
experimental: their unweighted likelihood and component-prediction tests remain valid,
but they do not yet implement the promoted offset/exposure, weights, and robust-
covariance contract. See [Count model guide](COUNT_MODELS.md).

## Promoted duration family

`GeometricDuration` (with `DiscreteTimeDuration` retained as an equivalent
historical name), `ExponentialDuration`, `WeibullDuration`, and `GammaDuration`
have moved to the package root and `limiteddepkit.duration`. The stable contract
includes right censoring, delayed entry, exact integer-frequency weights,
observed-information/robust/cluster covariance, schema-safe mean, survival,
hazard, cumulative-hazard, and quantile prediction, and maintained likelihood
identities against SciPy and Statsmodels foundations.

The historical experimental names remain compatibility aliases during the
alpha series. This is a parametric, one-record-per-spell family. It does not
claim a general time-varying discrete baseline, interval censoring, recurrent
events, competing risks, frailty, or time-varying covariate histories. See the
[duration model guide](DURATION_MODELS.md).

## Fixed-effects ordinal boundary

BUC `FixedEffectsOrderedLogit` is stable because its conditional composite
likelihood eliminates unrestricted entity effects and cutoffs and its
entity-cluster covariance and reference construction are maintained. Only
common slopes are identified; the stable result intentionally has no category
probabilities, fixed-effect estimates, thresholds, marginal effects, or
ordinary ordered-model AIC.

`FixedEffectsOrderedProbit` remains experimental. No Probit conditional
likelihood eliminates entity effects, so it uses unconditional nuisance effects
and a balanced-panel split-panel jackknife. Dynamic fixed-effects Ordered Logit
also remains experimental under its much narrower fixed-`T` identification
design and entity-clustered Godambe inference. See
[fixed-effects ordinal panels](FIXED_EFFECTS_ORDINAL.md) and the separate
[dynamic fixed-effects guide](DYNAMIC_FIXED_EFFECTS_ORDINAL.md).

## Promoted Firth and provisional ridge models

`FirthBinaryLogit` is stable at the package root and in `limiteddepkit.small_sample`;
its former experimental name remains a temporary compatibility alias. It permits complete
or quasi-complete separation, reports the Jeffreys penalty explicitly, and defaults to
profile penalized-likelihood confidence intervals. See the
[small-sample binary guide](SMALL_SAMPLE_MODELS.md).

`RidgeBinaryLogit` and `RidgeOrderedLogit` remain deliberately experimental. Their ridge
fitting uses a summed-log-likelihood penalty of
`0.5 * penalty * ||beta||^2`; constant columns are excluded by default in the binary
model and thresholds are never penalized in the ordered model.

Each estimator requires `n > p` and a full-column-rank design. Firth and ridge can make
specified low-dimensional models more usable under separation or limited information,
but these implementations are not high-dimensional estimators and do not identify
exactly collinear regressors. Ridge here must not be presented as permission to fit a
rank-deficient or `p >= n` design.

Do not compare penalized likelihoods, effective degrees of freedom, or approximate
penalized covariance as though they were ordinary-MLE AIC/BIC and covariance. If a
ridge strength or estimator family is chosen from predictive performance, use
`limiteddepkit.ml.nested_cross_validate`; selecting it on the same folds later reported
as test evidence is optimistic.

## Optional neural prediction challenger

`limiteddepkit.ml.ResidualBinaryMLP` is a separate advanced prediction challenger, not an
econometric estimator. It requires the `[neural]` extra, reports
`inference_valid=False`, and currently permits only independent rows because its internal
training/validation split is iid-stratified. `training_completed=True` means a finite
checkpoint was returned; `converged=True` is reserved for patience-based stabilization.
Its hyperparameters require nested CV, its temperature calibration uses the same internal
validation partition as early stopping, and Monte Carlo dropout bands are approximate
conditional model uncertainty rather than confidence or prediction intervals.

An isolated Python 3.13/PyTorch 2.13.0 run on 14 July 2026 passed all 21 neural tests,
including training, calibration, nested CV, and Monte Carlo dropout. A dedicated optional
neural CI job maintains that runtime path. This contract evidence is not neural numerical
parity, recovery, uncertainty-coverage, or inferential certification.

## Scope decisions

- Binary Logit and Probit completed their promotion gate and now belong to the
  stable root rather than this namespace.
- Ordinary linear 2SLS `TreatmentEffect` and the iid Gaussian mixture formerly
  called `SwitchingRegression` were extracted from the installed package.
  Their snapshots remain under `_out_of_scope/` pending migration to causal/IV
  and mixture/regime packages, respectively.
- Generalized additive models and ordinary quantile regression are not planned
  for this package merely because they are econometric models. The in-scope
  quantile addition is fixed-boundary `CensoredQuantileRegression`; it remains
  experimental because Powell's objective is non-convex and its inference is
  bootstrap-only.

## Method references

- James L. Powell (1986), “Censored Regression Quantiles,” *Journal of
  Econometrics* 32(1), 143–155,
  <https://doi.org/10.1016/0304-4076(86)90016-3>.
- Victor Chernozhukov and Han Hong (2002), “Three-Step Censored Quantile
  Regression and Extramarital Affairs,” *JASA* 97(459), 872–882,
  <https://www.mit.edu/~vchern/papers/Chernozhukov%20and%20Hong%20%28JASA%202002%29%20Three%20Step%20Censored%20Quantile%20Regression.pdf>.
