# Manual Stata promoted-family parity

This runner is separate from the frozen certificate under `validation/stata`.
It targets Stata 17 or newer and consumes the hash-verified contract produced
by `prepare_real_data.py`.

From the repository root, prepare the suite first:

```powershell
python validation/promoted/prepare_real_data.py `
  --output validation/promoted/work/real_data
```

Then run the do-file manually in Stata:

```stata
do "validation/promoted/limiteddepkit_real_data.do" ///
   "C:/path/to/limiteddepkit/validation/promoted/work/real_data"
```

Finally, return to PowerShell and compare the raw Stata export:

```powershell
python validation/promoted/compare_stata.py `
  validation/promoted/work/real_data
```

The Stata run writes raw `e(b)` and `e(V)`, fit statistics, selected
predictions, model statuses, metadata, and a text log under `work/real_data/stata`.
The comparator verifies the prepared manifest and hashes, applies the complete
raw-to-canonical covariance Jacobian, and writes canonical estimates plus
`comparison_report.csv`, `comparison_summary.md`, and
`parity_certificate.json`. It exits nonzero for missing artifacts, undeclared
skips, malformed schemas, provenance failures, or numerical failures.

`firthlogit` is optional and is never installed by the do-file. If it is
already installed, coefficients and predictions are checked; covariance is
aligned to limiteddepkit as inverse ordinary Fisher information evaluated at
the Stata bias-reduced coefficients. If it is absent, its exact
`optional_command_not_installed` status is the only permitted Firth skip.

Gamma duration is the other predeclared Stata skip. `streg` generalized Gamma
is not the ordinary Gamma likelihood implemented by `GammaDuration`, so the
runner does not substitute it and create misleading evidence.

Three models have deliberately narrow claims:

- geometric duration is an exact person-period Logit likelihood identity;
- fixed-effects Ordered Logit is the exact BUC blow-up/conditional-Logit
  identity, clustered by original entity, and compares only slopes, composite
  log likelihood, and audit counts; and
- random-effects Ordered Probit is numerical quadrature parity at 20
  nonadaptive Gauss-Hermite points with prediction conditional on random
  effect zero. The do-file supplies deterministic Python-derived starting
  values because Stata can otherwise stop at a non-converged boundary solution
  on this application fixture.

For Exponential and Weibull duration models, the Stata comparison certifies
coefficients, covariance, event/sample counts, convergence, and declared
prediction targets. It does not compare `streg` log likelihood, AIC, or BIC
because Stata's survival-time likelihood includes a software-specific constant
relative to the package's maintained likelihood target.

The `womenwage2` interval-regression fixture is Stata's official fictional
open-endpoint software example. It must not be described as empirical data.
All generated evidence and downloaded source data remain ignored local work
artifacts and should not be committed or redistributed.
