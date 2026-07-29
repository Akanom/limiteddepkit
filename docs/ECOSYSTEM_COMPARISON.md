# Ecosystem workflow comparison

This page compares three end-to-end research workflows in the Akanom Python
econometrics ecosystem with their closest established alternatives. The comparison is
about complete research paths—data contract, estimation, diagnostics, validation, and
reporting—not the length of an estimator list.

The evidence is intentionally separated into four classes:

- **real-data application**: a pinned public dataset exercises the workflow;
- **controlled runtime**: a deterministic benchmark measures a declared implementation;
- **parity**: an aligned external implementation targets the same estimand and options; and
- **capability comparison**: a nearby package solves a similar problem but is not assumed
  to be numerically interchangeable.

Runtime rows are not a league table. Hardware, dependency versions, samples, instrument
construction, covariance corrections, and timed regions must be identical before two
packages can support a speed ranking. Where that experiment has not been performed, the
page says **not measured** rather than inventing one.

## Evidence snapshot

| Component | Evidence revision | Distribution status | Role on this page |
| --- | --- | --- | --- |
| `limiteddepkit` | `v0.1.0a4` / `96677fc` | [PyPI](https://pypi.org/project/limiteddepkit/0.1.0a4/) | Limited outcomes and count workflow |
| `systemgmmkit` | `v0.5.13` / `2d17710` | [PyPI](https://pypi.org/project/systemgmmkit/0.5.13/) | Static panels, IV, and dynamic-panel GMM |
| CauseKit | `0.6.0a4` development candidate / `6c12022` | New distribution name not yet on PyPI | Causal treatment-effect workflow |
| `universal-output-hub` | `v0.2.4` | [PyPI](https://pypi.org/project/universal-output-hub/0.2.4/) | Model-neutral result collection and export |

CauseKit is the renamed successor to the earlier `causalkit` development distribution.
The old name must not be used for new installation or comparison instructions.

## At-a-glance decision table

| Research task | Ecosystem starting point | Closest alternative | Main ecosystem distinction | Main alternative advantage |
| --- | --- | --- | --- | --- |
| Limited or count outcome | `limiteddepkit` | [Statsmodels discrete/count models](https://www.statsmodels.org/stable/discretemod.html) | Stata-oriented contracts, explicit identification boundaries, panel ordinal families, maintained Stata/R evidence, and reference/accelerated engines | Larger community, broader general statistics surface, and mature formula/result ecosystem |
| Dynamic panel with endogenous regressors | `systemgmmkit` | [`pydynpd`](https://github.com/dazhwu/pydynpd) | Integrated OLS/FE/RE/panel-IV/GMM workflow, explicit instrument architecture, diagnostic reporting, Output Hub integration, and controlled accelerators | Concise Stata-like command string and a focused Difference/System-GMM implementation |
| Treatment effect under a declared design | CauseKit | [DoWhy](https://www.pywhy.org/dowhy/) | Applied-econometric estimands, refusal boundaries, covariance contracts, matching/DiD/IV diagnostics, and aligned R/Stata checks | Graph-based identification, refutation workflows, and a larger causal-inference ecosystem |

These alternatives overlap; none is described as a drop-in replacement. For example,
`linearmodels.system.IVSystemGMM` estimates systems of linear IV equations and is not by
itself an `xtabond2`-style dynamic-panel System-GMM substitute.

---

## Workflow 1: limited outcome with exposure

### Applied question and data

The maintained application uses Stata Press `rod93`, an infant-mortality table with
deaths, exposure, age, and birth cohort. The file is downloaded only with explicit
authorization, cached outside the repository, and checked against SHA-256
`023d1676ef716e320b49ebf0e0b31d259439161d3268da5b0b93022d138ddeab` before it is
read. The example fits exposure-adjusted Poisson and NB2 models. It reports conditional
rate associations, not causal birth-cohort effects.

### Ecosystem path

```python
from limiteddepkit import NegativeBinomialNB2, PoissonRegressor

poisson = PoissonRegressor().fit(
    X,
    deaths,
    exposure=person_time,
    engine="accelerated",
)
nb2 = NegativeBinomialNB2().fit(X, deaths, exposure=person_time)

print(poisson.summary_frame())
print(poisson.deviance, poisson.pearson_chi2, poisson.converged)
```

The corresponding Statsmodels comparison starts from the same analysis matrix and
exposure vector:

```python
import statsmodels.api as sm

reference = sm.GLM(
    deaths,
    X,
    family=sm.families.Poisson(),
    exposure=person_time,
).fit()
```

This is an aligned Poisson likelihood starting point, not an assertion that every default,
covariance correction, diagnostic, or NB2 parameterization is identical.

### Runtime, diagnostics, and parity

| Evidence | Result | Boundary |
| --- | --- | --- |
| Controlled Poisson runtime | 10,000 rows: reference `4.313 s`, accelerated `0.875 s`; `4.93x` relative speed and `79.7%` reduction | Windows, CPython 3.13.7, NumPy 2.5.1, pandas 3.0.5, SciPy 1.18.0; machine-specific |
| Numerical identity | Parameters, covariance, standard errors, fitted values, likelihood, score diagnostics, deviance, Pearson statistic, and optimizer counts were bitwise equal | Reference versus accelerated `limiteddepkit`, not versus Statsmodels |
| Applied diagnostics | Convergence, score norm, deviance, Pearson statistic, dispersion estimate for NB2, fitted means, and explicit exposure semantics | Diagnostics do not repair a misspecified conditional mean or identify causality |
| Real-data parity | `rod93` Poisson/NB2 are included in the 12-model promoted application: R passed `120/120`; manual Stata passed `140/140` required checks with one unrelated predeclared Gamma skip | Application evidence; model-specific gates, not universal equality |
| Alternative runtime | Not measured under a common environment and timed region | No Statsmodels speed claim is made |

Reproduce the application and timing evidence:

```powershell
python examples/real_world_count_workflow.py --download
python examples/real_world_count_workflow.py --poisson-engine reference
python benchmarks/benchmark_estimators.py --case poisson --engine both --repetitions 3
```

The complete evidence boundary is in [controlled performance](PERFORMANCE.md) and the
[promoted-family parity guide](../validation/promoted/README.md).

---

## Workflow 2: dynamic-panel System GMM

### Applied question and data

SystemGMMKit's real-data application uses NASA CMAPSS FD001 with `unit` as the entity and
`cycle` as time. A representative model relates risk to lagged risk, degradation and
sensor summaries, and operating settings. The application validates sample formation,
variable-role-specific lag windows, compact instrument counts, and Difference/System-GMM
command construction.

```python
import systemgmmkit as sgk

result = sgk.system_gmm(
    data=fd001,
    entity="unit",
    time="cycle",
    dependent="risk",
    regressors=[
        "degradation_index",
        "sensor_mean_z",
        "pc2",
        "op_setting1",
        "op_setting2",
    ],
    predetermined=["degradation_index", "sensor_mean_z", "pc2"],
    exogenous=["op_setting1", "op_setting2"],
    lagged_dependent=1,
    gmm_lags_by_role={"endogenous": (2, 2), "predetermined": (2, 2)},
    collapse=True,
    backend="native",
    windmeijer=True,
)
```

The easy facade constructs `L1_risk` and assigns it the default endogenous role. The
controlled acceleration benchmark calls the lower-level
`run_native_dynamic_panel_gmm(..., preparation_engine="accelerated")` path so the
preparation engine is explicit without adding a facade option that does not exist.

The closest focused Python alternative is `pydynpd`, whose comparable interface is a
Stata-like command string:

```python
from pydynpd import regression

fit = regression.abond(
    "risk L(1:1).risk degradation_index sensor_mean_z pc2 op_setting1 op_setting2 "
    "| gmm(risk, 2:2) gmm(degradation_index sensor_mean_z pc2, 2:2) "
    "iv(op_setting1 op_setting2) | collapse",
    fd001,
    ["unit", "cycle"],
)
```

That code illustrates interface shape only. A numerical comparison requires identical
equation scope, trimming, instruments, finite-sample corrections, and covariance scaling.

### Runtime, diagnostics, and parity

| Evidence | Result | Boundary |
| --- | --- | --- |
| Controlled System-GMM runtime | 1,344 rows/96 entities: reference `1.3019 s`, accelerated `0.3895 s`; `3.34x` relative speed and `70.1%` reduction | Windows, CPython 3.13.7, NumPy 2.5.1, pandas 3.0.5, SciPy 1.18.0 |
| Numerical identity | Coefficients, covariance-derived standard errors, residuals, instrument counts, and diagnostics are exactly equal on the maintained preparation benchmark | Reference versus accelerated SystemGMMKit |
| Applied FD001 diagnostics | Actual versus expected compact instrument counts passed for six Difference/System-GMM lag-window scenarios; the application also records Hansen, Sargan, AR(1), and AR(2) outputs | FD001 is external application evidence, not the strict certificate |
| Strict `xtabond2` parity | Certified specifications cover coefficients, Windmeijer standard errors, Hansen p-value, observations/groups, and instruments under declared tolerances | AR(1)/AR(2) parity is tracked separately and is not part of the conservative strict certificate |
| `pydynpd` comparison | Difference and System GMM both ran in the pinned compatibility environment; status remains `REVIEW` because construction and corrections were not aligned within the auxiliary band | It is not relabelled as failed parity and does not support a speed ranking |
| Alternative runtime | Not measured under the same interpreter, dependencies, data, and correction contract | No `pydynpd` speed claim is made |

The strict System-GMM tolerance contract is `1e-6` maximum absolute coefficient
difference, `0.001` maximum relative Windmeijer-SE difference, `1e-6` maximum absolute
Hansen-p difference, and exact instrument and observation counts.

Reproduce the released timing evidence:

```powershell
python benchmarks/benchmark_native_gmm.py --suite quick --repetitions 3
python benchmarks/benchmark_native_gmm.py --suite full --repetitions 5
```

Released evidence:

- [SystemGMMKit controlled performance](https://github.com/Akanom/systemgmmkit/blob/v0.5.13/docs/PERFORMANCE.md)
- [Conservative System-GMM parity matrix](https://github.com/Akanom/systemgmmkit/blob/v0.5.13/docs/parity/system_gmm_parity_matrix.md)
- [SystemGMMKit v0.5.13 source](https://github.com/Akanom/systemgmmkit/tree/v0.5.13)

---

## Workflow 3: causal treatment effect

### Applied question and data

CauseKit's development-candidate application uses four hash-pinned sources rather than
pretending one design can validate every estimator:

| Data | Design exercised | Primary diagnostics |
| --- | --- | --- |
| 1980 Census `hsng` | Robust IV/2SLS | First-stage excluded-instrument relevance and overidentification where identified |
| National Supported Work `nsw_mixtape` | Randomized ATE with Lin adjustment | Arm counts and standardized baseline differences |
| Stata Press `cattaneo2` | Supplied-nuisance IPW/AIPW and nearest-neighbor matching | Propensity support, effective sample size, reuse, ties, and balance |
| Stata Press `hospdd` | Conventional and PT-All efficient DiD | Group-time effects, event-time effects, anticipation and parallel-trends boundaries |

For the randomized experiment:

```python
from causekit import RandomizedATE

result = RandomizedATE(adjustment="lin", covariance="robust").fit(
    nsw["re78"],
    treatment=nsw["treat"],
    covariates=nsw[baseline_columns],
)

print(result.summary_frame())
print(result.balance)
```

DoWhy is the closest end-to-end causal alternative when the workflow begins with a
causal graph and proceeds through identification, estimation, and refutation. It is not
used as the numerical parity target here because similarly named estimators can differ in
estimand, nuisance fitting, support rules, aggregation, and covariance. CauseKit instead
uses aligned R and Stata implementations where the exact target exists.

### Runtime, diagnostics, and parity

| Evidence | Result | Boundary |
| --- | --- | --- |
| Controlled matching runtime snapshot | 100,000-observation balanced ATE matching: five-run median `3.6664 s` (`3.1845–4.3879 s`) | Local audit on commit `6c12022`, CPython 3.14.6, NumPy 2.4.6, pandas 3.0.3, one BLAS/OpenMP thread; not yet a released performance claim |
| R real-data parity | All 14 Python assertions passed; maximum absolute differences below `8.7e-7` IV, `8.1e-7` randomized ATE, `1.4e-5` IPW/AIPW, `1e-9` matching, and `2e-14` conventional/efficient DiD | R 4.5.1 with pinned comparator checkouts |
| Stata real-data parity | Every comparable family and the overall certificate passed; maxima `8.65e-7` IV, `8.07e-7` randomized ATE, `5.64e-7` IPW/AIPW, `7.39e-13` matching, and `1.11e-16` conventional DiD | Efficient PT-All DiD has no aligned Stata target and remains explicitly unavailable |
| Identification diagnostics | IV first stage cannot establish exclusion; observational overlap does not establish unconfoundedness; matching may refuse unsupported inference; efficient DiD requires the stronger PT-All restriction | Diagnostics expose assumptions but do not prove them |
| DoWhy runtime/parity | Not measured as an aligned numerical target | No DoWhy speed or parity claim is made |

Reproduce the committed CauseKit candidate evidence from its repository checkout:

```powershell
python examples/real_world_causal_workflow.py --download
python benchmarks/prepare_real_data.py --download
python benchmarks/benchmark_matching.py --scenario balanced_ate --n 100000
python -m pytest tests/validation/test_real_data_parity.py
```

The comparison must continue to label CauseKit as a development candidate until the
renamed distribution, repository, release tag, and public evidence bundle are available.

---

## One reporting path

The three workflows converge on labeled parameter, inference, fit, and diagnostic
records. Universal Output Hub consumes those records without owning model calculation
state:

```python
from causekit import add_to_outputhub
from universal_output_hub import OutputHub

hub = OutputHub("Econometrics robustness bundle")
hub.add_model(limited_result, name="Poisson")
hub.add_model(gmm_result, name="System GMM")
add_to_outputhub(hub, causal_result, name="Treatment effect")

hub.export_bundle("results")
hub.export_html_report("results")
```

CauseKit's adapter preserves treatment-effect diagnostics and auxiliary tables while the
hub's built-in adapters handle the limited-outcome and GMM results. A release workflow
must still test the exact objects it exports. The reporting layer may normalize labels
and diagnostics; it must not change an estimate, covariance matrix, identification
status, or active model state.

## What this comparison supports

It supports the bounded claim that the ecosystem provides a consistent, Stata-oriented
path across limited outcomes, dynamic panels, causal designs, and reporting, with
model-specific validation and explicit diagnostic boundaries.

It does **not** support claims that:

- every estimator is unique to this ecosystem;
- every alternative implements the same estimand or defaults;
- a parity pass on one fixture proves universal equality;
- a controlled acceleration benchmark is a competitor speed ranking; or
- diagnostics convert association into causal identification.

The defensible distinction is the integration of declared data contracts, refusal
boundaries, cross-software evidence, reference execution paths, controlled acceleration,
and model-neutral reporting.
