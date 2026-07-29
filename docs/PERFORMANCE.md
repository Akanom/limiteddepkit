# Controlled performance benchmarks

Performance changes must preserve the estimator, likelihood, inference, missing-
data, indexing, and convergence contracts. Reference implementations remain the
default, and no performance result authorizes weaker numerical tolerances.

## Stage 1 scope

Profiling identified repeated boolean slicing and repeated evaluation of the
data-only `gammaln(y + 1)` term inside the Poisson optimizer. The opt-in
`engine="accelerated"` path prepares contiguous active-row arrays and the log-
factorial term once per fit. It uses the same starting values, Newton/BFGS
fallback, likelihood and gradient expressions, information matrix, covariance,
diagnostics, and reporting path as `engine="reference"`.

On Windows with CPython 3.13.7, NumPy 2.5.1, pandas 3.0.5, and SciPy 1.18.0, a
deterministic 10,000-observation Poisson case with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1` produced these committed-harness warm-fit medians:

| Engine | Median | Relative speed | Reduction |
|---|---:|---:|---:|
| `reference` | 4.313 s | 1.00x | - |
| `accelerated` | 0.875 s | 4.93x | 79.7% |

The fitted parameters, covariance matrix, standard errors, fitted values,
likelihood, score diagnostics, deviance, Pearson statistic, and optimizer counts
were bitwise equal. Timings remain machine- and environment-specific rather than
a universal guarantee.

The same harness run recorded a 1,789,588-byte reference versus 1,730,186-byte
accelerated `tracemalloc` peak, a 3.3% reduction. This is Python-managed
allocation evidence, not total process resident memory.

## Grouped likelihood acceleration

Profiling of the remaining quick suite identified repeated Python calls to
SciPy's stable `logsumexp` as the dominant cost in grouped Conditional Logit and
the non-adaptive quadrature likelihood for static random-effects Ordered Logit
and Probit. The opt-in accelerated paths batch equal-width reductions through
the same SciPy implementation. They preserve group order, within-group row
order, quadrature nodes and weights, likelihood accumulation order, optimizer
settings, numerical Hessian construction, covariance, convergence, and result
contracts.

On the same controlled Windows environment, three-repetition warm-fit medians
were:

| Workload | Reference | Accelerated | Relative speed | Reduction |
|---|---:|---:|---:|---:|
| Conditional Logit, 600 four-alternative sets | 0.638 s | 0.084 s | 7.57x | 86.8% |
| RE Ordered Logit, 300 rows/60 entities/8 GHQ points | 1.399 s | 0.182 s | 7.69x | 87.0% |
| RE Ordered Probit, 300 rows/60 entities/8 GHQ points | 1.849 s | 0.332 s | 5.57x | 82.1% |

All three cases reported exact reference/accelerated identity for fitted
parameters, covariance, standard errors, likelihood, inference state, and
optimizer diagnostics. Exact tests also cover unequal choice-set sizes and
unbalanced, unsorted panels. The Conditional Logit quick case increased the
separately measured Python-allocation peak from 345,142 to 502,060 bytes because
it prepares batched group indices; the 156,918-byte absolute increase is the
documented tradeoff for the 86.8% runtime reduction. The two panel cases added
less than 10 KB each.

## Reproduce

From an isolated development environment at the repository root:

```powershell
python benchmarks/benchmark_estimators.py --suite quick --engine both --repetitions 3
python benchmarks/benchmark_estimators.py --suite full --engine both --repetitions 5 --output benchmark-results/estimators.json
python benchmarks/benchmark_estimators.py --case conditional_logit --engine both --profile-case conditional_logit --profile-limit 40
```

The harness covers Binary Logit, Binary Probit, Ordered Logit, Ordered Probit,
Generalized Ordered Logit, Multinomial Logit, Poisson, NB2, zero-inflated and
hurdle Poisson, Tobit, truncated regression, sample selection, grouped
conditional logit, and random-effects Ordered Logit and Probit. It measures full fit,
prediction, post-estimation (including marginal effects where supported), and a
separately measured `tracemalloc` Python-allocation peak. Data generation is
deterministic and excluded from timed regions. The allocation peak is not total
process resident memory. Fractional response is omitted because no such
estimator is currently implemented; duration and other promoted families can be
added as later benchmark coverage without changing the controlled acceleration
scope.

The quick accelerator smoke command is:

```powershell
python benchmarks/benchmark_estimators.py --case conditional_logit --engine accelerated --repetitions 1
python benchmarks/benchmark_estimators.py --case random_effects_ordered_logit --engine accelerated --repetitions 1
python benchmarks/benchmark_estimators.py --case poisson --engine accelerated --repetitions 1
```

## Boundaries

- `reference` remains the default and rollback path.
- The optimization adds no optional compiler or runtime dependency.
- Binary, pooled ordinal, dynamic panel, duration, censoring, and other count-
  family optimization opportunities remain profiling targets. No additional
  execution path is added when measured runtime does not justify it.
- Benchmark JSON and profiler output are generated evidence and should not be
  committed unless a review explicitly requests a stable artifact.
