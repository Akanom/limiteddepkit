# Release Integrity

The security and build jobs call `Akanom/python-package-governance` at immutable
commit `256d69be94f67d22548e9f1431b3f9902f235a50`; dependency updates must review
and deliberately advance that SHA.

Every release must pass supported-Python tests, minimum-version tests,
dependency consistency and advisory audits, optional-extra tests, strict Twine
validation, distribution inspection, wheel-only installation, and separate
sdist installation. CI stores the artifact inventory and CycloneDX SBOM.

The build job uploads immutable distributions. The protected publication job
downloads those exact artifacts, creates a provenance attestation, and
publishes through PyPI Trusted Publishing. After publication, an operator must
download the PyPI wheel, compare hashes, install it with
`--only-binary=:all:`, and run the documented smoke test.
