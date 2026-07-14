# Experimental model status

The stable package root contains the binary-and-ordinal alpha. The estimators below are
available only from `limiteddepkit.experimental`; their numerical tests are now
substantive, but their APIs can still change before promotion.

## Validated experimental slices

| Family | Current evidence | Main remaining boundary |
|---|---|---|
| Small-sample Binary/Ordinal | Firth Binary Logit half-cell identity; ridge Binary Logit parity with scikit-learn; near-unpenalized ridge Ordered Logit parity with Statsmodels | Requires `n > p` and full column rank; Firth intervals are approximate Wald rather than profile penalized-likelihood; ridge covariance is an approximate sandwich; penalties must be selected by nested CV |
| Multinomial, Conditional, and Sequential Logit | Exact Statsmodels likelihood/inference decompositions and arbitrary-label tests | No robust/cluster covariance or separation remedy; each model retains its standard IIA/choice-set/ordering assumptions |
| Poisson and NB2 | Coefficients, likelihood, covariance, and dispersion checks against Statsmodels | No exposure/offset, weights, or robust covariance interface yet |
| Zero-inflated and Hurdle Poisson | Stable two-part likelihoods and component-level Statsmodels parity | No robust covariance; only Poisson count components |
| Tobit, truncated Gaussian, and interval regression | Exact manual likelihood checks, uncensored-Gaussian reductions, recovery, and tail-stability tests | One-sided Gaussian specifications only; no heteroskedastic or robust-covariance variants |
| Gaussian sample selection | Correct full-sample Heckman likelihood and correlated-error recovery | Gaussian homoskedastic FIML only; no two-step/robust/cluster alternative |
| Geometric discrete duration, Exponential, Weibull, and Gamma duration | Person-period/Poisson equivalences, SciPy likelihood checks, and recovery tests | Time-varying discrete hazards and richer censoring/covariance schemes are not implemented |
| Fixed-boundary censored quantile regression | Powell check-loss recovery, inactive-censoring reduction to Statsmodels QuantReg, and explicit pairs-bootstrap tests | Non-convex local optimization; fixed known censoring bounds only; no inference unless bootstrap is requested and succeeds |

Passing these tests does not promote a model automatically. Promotion also
requires a settled shared result contract, documented failure modes, broader
covariance choices where relevant, and a deliberate compatibility decision.

The separate experimental `limiteddepkit.ml` workflow can cross-validate these
families with outcome-appropriate scores. Predictive evidence does not promote
an estimator or replace likelihood, inference, recovery, and cross-software
validation; see [Probability-aware validation workflows](ML_WORKFLOWS.md).

## Small-sample and separation-resistant models

`FirthBinaryLogit`, `RidgeBinaryLogit`, and `RidgeOrderedLogit` are deliberately
experimental. Ordinary `BinaryLogit` remains the stable, unpenalized MLE reference.
Firth fitting permits complete or quasi-complete separation and reports the Jeffreys
penalty explicitly. Ridge fitting uses a summed-log-likelihood penalty of
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
