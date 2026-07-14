# Experimental model status

The stable package root contains the binary-and-ordinal alpha. The estimators below are
available only from `limiteddepkit.experimental`; their numerical tests are now
substantive, but their APIs can still change before promotion.

## Certified experimental slices

| Family | Current evidence | Main remaining boundary |
|---|---|---|
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
