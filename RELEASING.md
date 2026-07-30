# Releasing limiteddepkit

The current release target is `0.1.0a5`. Version metadata, the changelog,
tests, and the signed Git tag must remain synchronized. Release artifacts are
built by GitHub Actions and published to PyPI with a Trusted Publisher; never
publish a locally built artifact or store a long-lived PyPI token.

1. Merge the release changes to `main` only after every required CI and
   dependency-review check succeeds.
2. Confirm that `0.1.0a5` is synchronized in `pyproject.toml`,
   `src/limiteddepkit/__init__.py`, `CITATION.cff`, the README, tests, and
   changelog. The release tag must be exactly `v0.1.0a5`.
3. Run the maintained Python and live R parity gates. Preserve the frozen Stata
   evidence and its exact provenance; do not describe it as a new execution
   unless Stata was actually rerun. Controlled tracks are benchmark-specific
   certification evidence, while public-data and promoted-family tracks are
   application evidence rather than universal software equality.
4. Run the checks from a clean environment:

   ```bash
   python -m pip install -e ".[dev]"
   python -m ruff check .
   python -m pytest
   python -m build --outdir <temporary-directory>
   python -m twine check --strict <temporary-directory>/*
   ```

5. Inspect the wheel and source archive. Confirm that the wheel imports without
   the source tree on `PYTHONPATH` and that the source archive includes the
   license, changelog, citation metadata, performance documentation, example,
   and validation guides.
6. Confirm the PyPI Trusted Publisher and protected `pypi` GitHub environment
   before publishing. Keep human approval enabled for the deployment job.
7. Create and push a signed `v0.1.0a5` tag at the exact merged `main` commit,
   then publish the matching GitHub release to trigger the release workflow.
8. Approve the protected `pypi` deployment only after its freshly built
   artifacts and attestations pass all preceding workflow jobs.
9. Download the published wheel from PyPI into a fresh environment, verify its
   project/version metadata, and run Binary Logit, Ordered Logit, Poisson,
   Conditional Logit, and static random-effects Ordered Logit/Probit
   reference/accelerated smoke tests before announcing the release. The
   maintained command is `python scripts/release_smoke.py --expected-version 0.1.0a5`.

Do not reuse a version already uploaded to PyPI. Correct a failed release with a
new pre-release version.
