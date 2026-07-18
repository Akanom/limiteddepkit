# Package scope

`limiteddepkit` contains models whose response, observation rule, or choice
mechanism is intrinsically limited. A method does not belong merely because it
is used in microeconometrics.

## Stable package root

The stable root exports the families that have completed their promotion gate:

- binary Logit and Probit;
- Firth bias-reduced Binary Logit for identified low-dimensional designs;
- Gaussian Tobit, truncated regression, and interval regression;
- exposure/offset Poisson and negative-binomial NB2;
- geometric discrete duration and Exponential, Weibull, and Gamma parametric duration;
- pooled Ordered Logit and Probit;
- Generalized Ordered Logit and Partial Proportional Odds;
- BUC fixed-effects Ordered Logit common slopes;
- random-effects Ordered Logit/Probit and dynamic random-effects Ordered Logit; and
- maintained inference, prediction, and margins interfaces where applicable.

The ordinal stack additionally retains plotting, simulation, posterior panel
prediction, and Output Hub integration.

## Retained under `limiteddepkit.experimental`

These families are in scope but retain provisional APIs:

| Family | Why it belongs |
|---|---|
| Ridge Binary and Ordered Logit | Explicit shrinkage belongs in limited-data workflows, but penalty selection and approximate covariance remain provisional |
| Multinomial, Conditional, and Sequential Logit | The response is a finite choice |
| Zero-inflated Poisson and hurdle Poisson | The response has an explicit structural-zero or two-part mechanism; weighted and robust-covariance contracts remain open |
| Gaussian sample selection | A binary selection rule determines whether the continuous outcome is observed |
| Fixed-effects Ordered Probit | Entity effects are explicit and common parameters receive a split-panel correction, but the estimator requires balanced large panels and successful entity-bootstrap inference |
| Dynamic fixed-effects Ordered Logit | State dependence and unrestricted time-invariant heterogeneity belong here, but the four-outcome MRV conditional estimator requires discrete exact stayers, a known state cutoff, and restricted state dependence |
| Fixed-boundary censored quantile regression | The conditional quantile is observed through a known censoring boundary |

See [fixed-effects ordinal panels](FIXED_EFFECTS_ORDINAL.md) for the stable BUC
Logit and experimental SPJ Probit distinction, and
[dynamic fixed-effects Ordered Logit](DYNAMIC_FIXED_EFFECTS_ORDINAL.md) for the
separate fixed-`T` conditional design.

## Extracted from the installed package

- `TreatmentEffect` is ordinary linear 2SLS. Endogeneity alone does not make a
  dependent variable limited. Its source is preserved in `_out_of_scope/` for
  later migration to a causal/IV package.
- `GaussianMixtureRegression` and the historical `SwitchingRegression` alias
  describe an iid continuous Gaussian mixture. That is a latent-class model,
  not a limited-response model. Their source is preserved in
  `_out_of_scope/` for a future mixture/regime package.

A future endogenous switching regression could fit this package only if its
limited selection equation and regime-specific observation mechanism are
specified explicitly. It would not reuse the extracted iid mixture under a new
name.

## Deliberately not added

- Ordinary quantile regression remains with general regression packages.
- A generic generalized additive model is a functional-form framework, not an
  outcome family. Future nonlinear support should be a spline/basis layer used
  by retained binary, count, censoring, or duration estimators rather than a
  standalone `GAM` estimator here.
- General linear IV, treatment-effect, finite-mixture, and Markov-switching
  models belong in causal/IV or regime-model packages.
