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
4. Run the checks from a clean environment:

   ```bash
   python -m pip install -e ".[dev]"
   python -m ruff check .
   python -m pytest
   python -m build
   python -m twine check --strict dist/*
   ```

5. Inspect the wheel and source archive. Confirm that the wheel imports without
   the source tree on `PYTHONPATH` and that the source archive includes the
   license, changelog, citation metadata, and `docs/` guides.
6. Configure a PyPI Trusted Publisher for the repository and the `pypi` GitHub
   environment before enabling the publish workflow. Apply environment approval
   rules if a human release gate is desired.
7. Publish a GitHub release for the exact version. The publish workflow builds
   fresh artifacts, checks them with Twine, and authenticates to PyPI through
   OpenID Connect; no long-lived PyPI token is required.
8. Install the published wheel into a fresh environment and rerun minimal
   Binary Logit and Ordered Logit fits before announcing the release.

Do not reuse a version already uploaded to PyPI. Correct a failed release with a
new pre-release version.
