# Public Discussion Drafts

These drafts are ready to post as GitHub Discussions. They are written to create a
public development trail around roadmap decisions, validation evidence, and adoption.

## Discussion 1: Stable model-family roadmap after alpha expansion

Category: Ideas

Labels: `roadmap`, `model-proposal`

Body:

`limiteddepkit` has expanded from binary and ordinal limited-dependent-variable models
to a broader alpha toolkit covering Firth Logit, count models, censoring/truncation,
interval regression, parametric duration, and panel ordinal estimators.

This thread is for public discussion of the next stable model-family priorities. The
current rule is conservative: new families should not enter the stable API until the
likelihood or estimating equation, identification assumptions, prediction targets,
failure modes, and validation route are documented.

Useful proposals should include:

- the model family and applied use case;
- reference implementations in Stata, R, Python, or published examples;
- required prediction and inference outputs;
- known identification or small-sample limitations;
- whether the model should begin in `limiteddepkit.experimental`.

Current candidates for future work include zero-inflated and hurdle count models,
sample-selection models, switching regressions, generalized additive limited-outcome
models, and additional panel limited-dependent-variable estimators.

## Discussion 2: Promoted-family parity and validation evidence

Category: Show and tell

Labels: `validation`, `parity`

Body:

This thread tracks public validation evidence for the post-expansion stable model
families in `limiteddepkit`.

The current promoted-family application harness compares 12 stable fits on public or
officially distributed data using Python and R reference routes. Evidence is intentionally
reported as model-specific application evidence, not as a universal equivalence claim.

Useful follow-up evidence includes:

- Stata output for the promoted-family harness;
- additional public datasets with reproducible hashes;
- independent R or Python reference implementations;
- failures where defaults differ across software;
- examples where agreement depends on covariance, quadrature, optimizer, or data coding.

Please include software versions, commands, data hashes, compared quantities, tolerances,
and any warnings or convergence messages.

## Discussion 3: Adoption, teaching, and replication examples

Category: Show and tell

Labels: `adoption`

Body:

This thread records public uses of `limiteddepkit` in applied work, teaching, replication
packages, technical reports, and examples.

Helpful notes include:

- which model family was used;
- whether the use was exploratory, teaching, replication, or publication support;
- links to public notebooks, repositories, papers, or course material;
- any limitations encountered;
- features that would make the package easier to adopt.

This adoption record helps the project understand where the software is useful and where
the documentation or API needs more work.

