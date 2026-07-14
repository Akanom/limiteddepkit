# Dynamic ordinal models

The initial dynamic estimator is a random-intercept Ordered Logit with:

- category indicators for the one-period lagged outcome;
- initial-outcome category indicators;
- initial-period covariates;
- entity means of the retained post-initial covariate history;
- a remaining normally distributed entity intercept.

The initial-outcome, initial-covariate, and post-initial entity-mean controls implement one explicit
conditional-random-effects initial-conditions specification. This document does
not use “Wooldridge correction” as an unqualified label: alternative
specifications may additionally condition on initial-period covariates or use a
different representation of the full covariate history.

Only the contiguous spell beginning at each entity's initial observation enters
the dynamic likelihood. The spell is truncated at the first internal time gap;
later observations are not silently restarted as a new conditional likelihood.
Lagged outcomes are represented by category indicators rather than imposing a
cardinal linear effect on ordinal labels.

`fitted_probabilities` are population-averaged over the fitted residual random-
effect distribution. One-step prediction requires an explicit lagged category.
Multi-step forecasting would require propagation of the entire predicted
category distribution and is not yet exposed by this estimator.

State-dependence coefficients are category-relative latent-index shifts. They
are neither a scalar autoregressive coefficient nor direct changes in category
probabilities.

This is a standard dynamic ordinal specification, not the proposed BDCPM. It
models observed state dependence but does not introduce an evolving latent
behavioral state.
