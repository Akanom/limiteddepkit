# Releasing limiteddepkit

The source tree is frozen at `0.1.0a1`, but it is not yet publishable. Version,
classifier, citation metadata, changelog, tests, and local artifact checks are
complete. Use the remaining checklist for external publication.

1. Create the source repository and add its Homepage, Repository, and Issues
   URLs under `[project.urls]` in `pyproject.toml`. Do not publish placeholder
   links. Replace relative links in the package README with absolute repository
   links so they resolve from the PyPI project page.
2. Confirm that `0.1.0a1` remains synchronized in `pyproject.toml`,
   `src/limiteddepkit/__init__.py`, and `CITATION.cff`, and that the Git tag is
   exactly `0.1.0a1` or `v0.1.0a1`.
3. Replace `Release pending` in the `0.1.0a1` changelog heading with the actual
   publication date, then review the supported-versus-experimental API list.
   Do not add another model family to this release candidate.
4. Reconfirm and archive the external-software evidence. The 14 July 2026
   Stata 17 runs with `gologit2` 3.2.8 passed all eight models in both tracks;
   the pinned R 4.5.1 runs also passed all eight. Before publication, rerun both
   comparators against the unchanged manifests and archive each manifest,
   external-software log/metadata, canonical result CSV, comparison report,
   Markdown summary, JSON certificate, and exact commit. Treat only the
   controlled tracks as certification evidence; the real-data tracks are
   application checks. Separately, rerun the promoted-family preparation, R
   runner, and Python/R comparator and require all 12 fits and all 120/120
   registered checks to pass. Its recorded 15 July 2026 result is application
   evidence, not an extension of the controlled certificate or a universal
   equality claim. Before making any promoted-family Stata claim, manually run
   `validation/promoted/limiteddepkit_real_data.do` and then
   `validation/promoted/compare_stata.py`; that Stata result is currently
   pending and cannot be inferred from the Python/R pass. Archive the promoted
   manifest, permitted provenance, canonical exports, logs, reports,
   certificates, software versions, and exact commit together.
5. Run the checks from a clean environment:

   ```bash
   python -m pip install -e ".[dev]"
   python -m ruff check .
   python -m pytest
   python -m build
   python -m twine check --strict dist/*
   ```

6. Inspect the wheel and source archive. Confirm that the wheel imports without
   the source tree on `PYTHONPATH` and that the source archive includes the
   license, changelog, citation metadata, and `docs/` guides.
7. Configure a PyPI Trusted Publisher for the repository and the `pypi` GitHub
   environment before enabling the publish workflow. Apply environment approval
   rules if a human release gate is desired.
8. Publish a GitHub release for the exact version. The publish workflow builds
   fresh artifacts, checks them with Twine, and authenticates to PyPI through
   OpenID Connect; no long-lived PyPI token is required.
9. Install the published wheel into a fresh environment and rerun minimal
   Binary Logit and Ordered Logit fits before announcing the release.

Do not reuse a version already uploaded to PyPI. Correct a failed release with a
new pre-release version.
