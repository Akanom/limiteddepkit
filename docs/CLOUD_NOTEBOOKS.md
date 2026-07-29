# Kaggle and Google Colab usage

`limiteddepkit` examples may be run in Kaggle or Google Colab notebooks when
the notebook validates `limiteddepkit` model families only. Cloud notebooks are
reproducibility aids; they are not a separate unrelated-package comparison workflow.

## Package boundary

- Use `limiteddepkit` notebooks for limited-dependent-variable models:
  binary, ordinal, count, censoring, duration, panel ordinal, post-estimation,
  prediction, marginal effects, and package-scoped validation examples.
- Do not host unrelated package evidence inside a `limiteddepkit` notebook.
- If a notebook compares against Stata, R, Statsmodels, or scikit-learn, the
  comparison must be tied to an aligned `limiteddepkit` estimator, fixture,
  estimand, mapping, and tolerance.

## Credentials and data

- Never commit `kaggle.json`, Google credentials, API tokens, cookies, or local
  notebook secrets.
- Store Kaggle credentials in the runtime secret manager or in
  `~/.kaggle/kaggle.json` on the machine running the notebook.
- Keep downloaded third-party datasets and generated validation work directories
  out of Git unless their license and provenance permit redistribution.
- Record dataset source URLs, hashes, package versions, random seeds, and the
  exact `limiteddepkit` commit or release used for any reported result.

## Minimal cloud setup

```python
!python -m pip install limiteddepkit

import limiteddepkit as ldk

print(ldk.__version__)
```

For development checkouts, install from the repository URL or upload a built
wheel. Pin the package version for any notebook intended as evidence.

## Repository notebook artifact

This repository includes a Kaggle-ready quickstart notebook at
`notebooks/kaggle/limiteddepkit_quickstart.ipynb` with Kaggle metadata in
`notebooks/kaggle/kernel-metadata.json`.

To publish or update the public Kaggle notebook from an authenticated machine:

```powershell
cd notebooks/kaggle
kaggle kernels push -p .
```

After Kaggle finishes processing the upload, the public notebook should be
available under the kernel id recorded in `kernel-metadata.json`:
`akanom/limiteddepkit-quickstart`.

For Google Colab, open the tracked notebook from GitHub after the branch is
pushed, or upload the same `.ipynb` file directly into Colab.

## Evidence rule

Cloud notebook output can support examples and adoption notes, but formal
validation claims remain the package-scoped Stata/R/Python harnesses documented
in the validation guides.
