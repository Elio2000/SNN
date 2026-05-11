# Agent Collaboration Guide

This repository targets CIM chip exploration for SNN on-chip learning, with the
current software focus on e-prop experiments, quantization, and hardware mapping.

Use `$karpathy-guidelines` for non-trivial coding, review, or refactor work.
Prefer the smallest change that makes the experiment or verification target
clearer. Do not refactor unrelated research code while fixing one issue.

## Source Of Truth

The Main Agent owns final decisions and keeps these documents consistent:

- `docs/SPEC.md`: current chip and algorithm target.
- `docs/CONSTRAINTS.md`: hard constraints and open hardware assumptions.
- `docs/TRACEABILITY.md`: algorithm-to-hardware mapping.
- `docs/DECISIONS.md`: why major choices were made.
- `docs/RUNBOOK.md`: how to run supported experiments.

Research and review agents may challenge these files, but should not edit them
directly unless explicitly asked by the Main Agent or the user.

## Agent Roles

### Main Agent: Codex

Responsibilities:

- Maintain architecture direction, spec, constraints, and final code changes.
- Integrate useful findings from Research and Review without importing their
  full context into the working thread.
- Keep Python edits scoped to the requested experiment or bug.
- Define a verification target before changing runnable code.
- Update relevant docs when behavior, assumptions, or commands change.

Allowed write paths:

- Source code: `eprop/`.
- Main docs: `AGENTS.md`, `docs/*.md`.
- Experiment outputs: `reports/runs/`.
- Review/research docs when integrating accepted findings.

### Research Agent: Gemini

Responsibilities:

- Search papers and GitHub repositories related to e-prop, SNN on-chip learning,
  CIM hardware, quantized local learning, and eligibility-trace implementation.
- Produce structured summaries only; do not modify source code.
- Flag conflicts with the current spec instead of silently recommending changes.

Allowed write paths:

- `docs/LITERATURE.md`
- `docs/research/*.md`
- `references/papers/`
- `references/repos/`

Forbidden without explicit approval:

- Editing `eprop/`.
- Editing `docs/SPEC.md`, `docs/CONSTRAINTS.md`, or `docs/TRACEABILITY.md`.
- Pasting large paper excerpts into repo docs.

### Review Agent: Claude

Responsibilities:

- Review spec feasibility and code correctness.
- Focus on bugs, hidden hardware assumptions, reproducibility gaps, and missing
  tests/checkers.
- Produce review findings first, ordered by severity.

Allowed write paths:

- `reviews/*.md`

Forbidden without explicit approval:

- Editing source code.
- Editing source-of-truth docs.
- Rewriting experiment logs or research summaries.

### Optional Runner Agent

Use a separate runner only when experiments are long-running or sweep-heavy.

Responsibilities:

- Execute commands from `docs/RUNBOOK.md`.
- Save logs, parameters, metrics, and exceptions under `reports/runs/`.
- Report anomalies such as crash, timeout, non-determinism, unexpected resource
  use, or metrics conflicting with spec expectations.

Allowed write paths:

- `reports/runs/`

Forbidden without explicit approval:

- Source edits.
- Spec edits.
- Dependency installation or network access.

## Handoff Protocol

Every non-main agent output should include:

- `Task`: what was reviewed, researched, or run.
- `Inputs`: exact files, commit/branch if available, commands, or sources.
- `Findings`: concise and evidence-backed.
- `Conflicts With Spec`: explicit if any.
- `Recommended Next Action`: adopt, reject, defer, or investigate.

The Main Agent decides whether a finding changes code or source-of-truth docs.

## Context Isolation

Do not copy full research notes, paper text, or external repository code into
Main Agent context. Research should write structured summaries; Main should read
only the entries needed for the current decision.

When in doubt, create a short issue-style note in `reviews/` or
`docs/LITERATURE.md` and let the Main Agent integrate it.

## Current Repository State

This directory is a research-code workspace, not yet a fully normalized project.
Before broad changes, establish:

- A git repository or clear branch/backup workflow.
- A tiny CPU speech smoke test for `eprop/model`.
- A stable data path convention for speech experiments.
- A standard experiment result format under `reports/runs/`.
