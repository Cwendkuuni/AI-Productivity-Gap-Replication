# SLR Measurement-Boundary Coding Protocol v0.1

## Objective

This secondary coding analyzes how far through the software-engineering workflow productivity or effort is empirically observed in the 39 primary studies included in the Mohamed et al. SLR artifact.

The analysis does not infer workflow boundaries from SPACE dimensions. SPACE characterizes what dimension of productivity is studied; this coding characterizes where the measured effort or outcome stops in the workflow.

## Stage 1 - Eligibility

### ELIGIBLE
The study reports at least one empirical productivity, effort, time, work output, or task outcome from which a software-workflow measurement boundary can be identified.

### NOT_ELIGIBLE
The study reports only perceptions, attitudes, adoption, satisfaction, opinions, forecasts, theory, or other evidence without an observable workflow-level productivity/effort outcome.

### UNCLEAR
The available artifact is insufficient to determine eligibility. The primary study must be inspected.

## Stage 2 - Measurement Boundary

The boundary is the farthest workflow stage actually included in an
empirical productivity/effort outcome.

### B1 - Local coding / AI interaction
Examples:
- code completion
- code generation
- prompting or AI interaction
- local editing activity

### B2 - Completed implementation
Examples:
- completed programming task
- working solution
- implementation completion
- pull request ready for review

### B3 - Review / verification / correction
Examples:
- review
- validation
- testing as part of the developer workflow
- correction or rework
- acceptance following verification

### B4 - Integration / delivery
Examples:
- merge
- integration
- CI/CD completion
- deployment
- production delivery

## Decision Rules

R1. Code the farthest workflow stage actually included in the measured productivity/effort outcome, not the farthest stage merely discussed.

R2. If multiple outcomes are reported, record them and assign the maximum boundary supported by a productivity/effort outcome.

R3. Perception-only evidence does not establish a workflow boundary unless it reports a concrete behavioral or effort quantity tied to a defined stage.

R4. Quality measures do not automatically establish a broader boundary; the point in the workflow at which quality is assessed must be determined.

R5. Testing is B3 only when testing or validation forms part of the measured developer workflow. Tests used solely as an offline correctness oracle do not automatically move the boundary to B3.

R6. Repository-level context does not imply B4. Merge, integration, delivery, or deployment must actually be observed.

R7. Do not infer a boundary from title, strategy, or procedure alone. Ambiguous cases are coded UNCLEAR and checked against the primary paper.

## Evidence Requirement

Every final B1--B4 assignment must record:
- measured outcome;
- evidence source;
- supporting quotation or precise paraphrase;
- page/section where available;
- confidence level.

## Coding and review procedure (as performed)

The first author conducted eligibility screening, primary-source evidence extraction, and initial workflow-boundary coding for the studies. A second author reviewed the eligibility decisions, assigned boundaries, and supporting evidence, with particular attention to ambiguous cases. This was a review/adjudication procedure, **not** a fully independent second coding exercise. No inter-rater agreement statistic is reported because a separate, independently coded validation sample was not completed.

The final codes are in `verified_eligibility` and `verified_boundary`. The `evidence_checked_by` field records the initial source checker and must not be interpreted as the complete record of second-author review. No empty `coder2`
field should be treated as evidence of independent coding. Any further claim about second-author review records should be supported by a review log or author notes.

## Verification Status

A pilot boundary is not considered a study result until the primary source has been checked.

Each study receives one of the following statuses:

- VERIFIED: the measured outcome and workflow boundary are directly supported by the primary study.
- DOWNGRADED: the pilot boundary was broader than the evidence supports.
- UPGRADED: the primary study supports a broader boundary than the pilot coding suggested.
- NOT_ELIGIBLE: no empirical productivity/effort outcome with an identifiable workflow boundary is reported.
- UNRESOLVED: the available source remains insufficient for a defensible
  code.

## Evidence Standard

A VERIFIED boundary requires:

1. identification of the measured productivity/effort outcome;
2. identification of the unit or operational measure;
3. identification of the farthest workflow stage included in that outcome;
4. a supporting quotation or precise source-grounded paraphrase;
5. page, section, table, or figure location where available.

Mentions of review, testing, deployment, or production context do not by themselves justify B3 or B4. The activity must contribute to the measured productivity/effort outcome.

## Conservative Coding Rule

When evidence supports multiple plausible boundaries, assign the narrower boundary unless the broader stage is explicitly included in the measured outcome.
