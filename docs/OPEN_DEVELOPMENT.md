# Open Development

`limiteddepkit` is developed in public so model design choices, validation evidence,
limitations, and adoption can be inspected before release.

## Why this exists

Limited-dependent-variable software is useful only when the model boundary is clear.
Public discussions and issues should therefore record more than feature requests. They
should show how a model was specified, what evidence supports it, which external
implementations it was compared against, and where the estimator should not be used.

## Public discussion tracks

Use GitHub Discussions for work that benefits from early public input:

- model-family proposals before implementation;
- parity and validation reports against Stata, R, Python, or published examples;
- documentation examples and applied use cases;
- roadmap priorities for future releases;
- adoption notes from papers, teaching material, replication packages, or external
  projects.

Ready-to-post starter threads are kept in
[Public discussion drafts](PUBLIC_DISCUSSION_DRAFTS.md).

Use GitHub Issues for concrete, actionable work:

- reproducible bugs;
- failing validation checks;
- documentation gaps;
- scoped model improvements;
- release blockers.

## Model promotion record

A model family should not move into the stable API without a visible promotion record.
That record should include:

1. the likelihood or estimating equation;
2. the data contract and identification assumptions;
3. prediction targets and inferential quantities;
4. tests for expected failure modes;
5. independent reference evidence where a comparable implementation exists;
6. simulation or identity checks where external software parity is not meaningful; and
7. documentation describing current limitations.

## Evidence that helps future review

The most useful public evidence is specific:

- links to scripts, datasets, and hashes used for validation;
- comparison tables with tolerances and software versions;
- issue threads showing design tradeoffs;
- pull requests that connect implementation, tests, and documentation;
- examples of external use beyond the maintainer's own research workflow.

Keeping the repository public for a period of time is not enough by itself. The public
record should make it easy for a reviewer to see active development, scrutiny, and use.
