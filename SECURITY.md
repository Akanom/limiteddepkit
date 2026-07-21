# Security Policy

## Supported versions

`limiteddepkit` has not made a stable release. Security fixes are provided for
the latest prerelease and current `main` branch on Python 3.10–3.13.

## Reporting a vulnerability

Use GitHub's private security-advisory form. Do not publish exploit details,
credentials, private datasets, or unpublished research material in a public
issue. Include the affected version or commit, impact, reproduction steps, and
any suggested mitigation.

The maintainer aims to acknowledge reports within five business days and give
an initial status update within ten business days. Coordinated disclosure is
preferred. Statistical defects without security impact belong in the normal
issue tracker.

## Dependency and release policy

Confirmed critical and high vulnerabilities block release unless an owner
approves a documented, time-bounded exception. Behavioural and heuristic
scanner alerts require package-, version-, artifact-, and file-level evidence.
Runtime and optional dependencies use tested compatibility ranges; CI audits
resolved dependencies, validates artifacts, exercises optional extras, and
tests clean wheel and source-distribution installation.

PyPI publication uses GitHub OIDC Trusted Publishing from the protected `pypi`
environment. Release jobs use immutable action revisions, least-privilege
permissions, artifact inspection, SBOM generation, and provenance
attestations. Repository administrators must retain environment protection and
the PyPI trusted-publisher mapping.
