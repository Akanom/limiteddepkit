# Contributing to limiteddepkit

Thank you for helping improve `limiteddepkit`. The package is in a binary-and-ordinal
alpha phase, so correctness evidence and explicit scope are more important
than adding model names quickly.

## Development setup

Use Python 3.10 or newer and install the development dependencies from the
repository root:

```bash
python -m pip install -e ".[dev]"
```

Run the standard checks before submitting a change:

```bash
python -m ruff check .
python -m pytest
python -m build
python -m twine check --strict dist/*
```

Keep changes focused. Do not reformat or rewrite unrelated user work, and add
or update documentation whenever behavior or the public API changes.

## Econometric changes

A new estimator is not ready for the supported API based on smoke tests alone.
A promotion proposal should include:

1. the formal model, likelihood, parameterization, and identification
   conditions;
2. input validation and tests for expected failure modes;
3. result-contract parity for prediction and inference where meaningful;
4. comparison with an independent maintained implementation or published
   benchmark;
5. deterministic simulation-recovery evidence; and
6. numerical-stability checks, including boundaries and poorly conditioned
   designs.

New or materially unverified model families should begin in
`limiteddepkit.experimental`. Experimental status must be visible in imports
and documentation and should not be described as validated.

## Tests and style

- Put fast behavioral tests under `tests/`.
- Mark external-reference checks with `@pytest.mark.validation`.
- Mark recovery or Monte Carlo checks with `@pytest.mark.simulation`.
- Use deterministic random seeds and explain statistical tolerances.
- Keep public behavior compatible with Python 3.10.
- Follow the configured Ruff rules and 100-character line length.

## Reporting bugs

Include a minimal reproducer, Python and dependency versions, the observed and
expected behavior, and any warnings or convergence information. Do not include
confidential data. For vulnerabilities, follow [SECURITY.md](SECURITY.md)
instead of opening a public issue.
