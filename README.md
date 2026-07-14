# limiteddepkit

`limiteddepkit` is a developing Python toolkit for limited-dependent-variable
models. The `0.1.0a1` candidate is a **binary-and-ordinal alpha**: the maintained public
surface covers binary Logit/Probit and the ordinal regression stack, with result and post-estimation
conventions aligned with `systemgmmkit` where those conventions make sense for
discrete outcomes.

The package is pre-release software. APIs can change before the first stable
release, and users should validate estimates for consequential applications.

## Supported alpha scope

The supported core currently provides:

- binary Logit and Probit with observed-information inference, probability
  prediction, representative margins, and marginal effects;
- pooled Ordered Logit and Ordered Probit;
- Generalized Ordered Logit and Partial Proportional Odds models;
- random-intercept Ordered Logit using Gaussian-Hermite quadrature;
- dynamic random-effects Ordered Logit with explicit initial-conditions
  controls;
- explicit category ordering, probability and category prediction;
- covariance estimates, confidence intervals, Wald tests, linear combinations,
  margins, and category-specific marginal effects where supported;
- posterior random-effect summaries and posterior prediction for panel models;
- plotting helpers and an optional Universal Output Hub adapter; and
- deterministic simulation, numerical-invariance, and reference-software
  validation tests.

Model-specific inference safeguards matter. For example, ordinary Hessian
inference is not reported for flexible ordinal fits on an active non-crossing
constraint boundary.

## Experimental models

Other provisional estimators live under `limiteddepkit.experimental`.
They are retained for correction and validation work, but are **not part of the
supported alpha API**. They may have incomplete likelihoods, inference,
post-estimation methods, or external validation, and may move to a different
package. Import them only by their experimental path and do not rely on API
stability.

An estimator is promoted from `experimental` only after its likelihood,
failure modes, shared result contract, reference comparison, and simulation
recovery have been reviewed.

The [experimental model status](docs/EXPERIMENTAL_MODELS.md) records the
completed certification slices, remaining limitations, and models deliberately
removed or renamed on scope grounds.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

Optional dependency groups are available for plotting, external validation,
Output Hub integration, testing, and development:

```bash
python -m pip install -e ".[plots,validation]"
python -m pip install -e ".[test]"
```

The `test` and `dev` groups include all optional dependencies exercised by the
maintained test suite. Install only `.[outputhub]` when an application needs the
Output Hub adapter without the development tools.

Python 3.10 or newer is required.

## Minimal example

```python
from limiteddepkit import BinaryLogit, OrderedLogit

binary_result = BinaryLogit().fit(X_binary, y_binary)
binary_probabilities = binary_result.predict_proba(X_binary_new)

ordinal_result = OrderedLogit().fit(
    X_ordinal, y_ordinal, category_order=["low", "medium", "high"]
)
ordinal_probabilities = ordinal_result.predict_proba(X_ordinal_new)
print(ordinal_result.summary_frame())
```

Binary designs may include an explicit constant. Do not include one in a pooled
ordinal design: threshold parameters identify the ordinal model's location.

## Documentation

The [package scope](docs/PACKAGE_SCOPE.md) records the firm keep/extract
boundary. The [documentation index](docs/README.md) links the model contracts, category
ordering rules, panel and dynamic specifications, and current validation
evidence. [CONTRIBUTING.md](CONTRIBUTING.md) describes the checks required for
changes, and [RELEASING.md](RELEASING.md) records the first-release checklist.

## License

`limiteddepkit` is distributed under the MIT License. See [LICENSE](LICENSE).
