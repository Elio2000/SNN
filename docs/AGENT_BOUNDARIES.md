# Agent Boundaries

This file defines operational boundaries for the planned multi-CLI workflow.

## Boundary Summary

| Agent | Primary Purpose | Can Edit Code? | Can Edit Spec? | Primary Output |
| --- | --- | --- | --- | --- |
| Codex Main | Architecture, integration, code changes | Yes | Yes | Source, `docs/*.md` |
| Gemini Research | Papers and GitHub scan | No | No | `docs/LITERATURE.md`, `docs/research/*.md` |
| Claude Review | Spec/code review | No by default | No | `reviews/*.md` |
| Runner | Long experiments | No | No | `reports/runs/*` |

## Codex Main Rules

- Owns final code and document changes.
- Reads review/research outputs selectively.
- Keeps `SPEC.md`, `CONSTRAINTS.md`, `TRACEABILITY.md`, and `DECISIONS.md`
  internally consistent.
- Must not accept research recommendations without a local experiment,
  hardware-feasibility rationale, or explicit decision entry.

## Gemini Research Rules

Gemini should optimize for breadth, but its output must be narrow and structured.

Required format for each entry:

```text
Source:
Claim:
Evidence:
Hardware implication:
Conflict with current spec:
Possible experiment:
Adopt / Reject / Defer:
```

Research must distinguish:

- Directly supported claims from paper or repository evidence.
- Inferences made by the agent.
- Ideas that require local experiments before adoption.

## Claude Review Rules

Claude should act as a critical reviewer, not a co-implementer.

Review output order:

1. Findings by severity.
2. Open questions or assumptions.
3. Suggested verification.
4. Short summary.

Findings should include file and line references when reviewing code.

Review categories:

- Correctness bug.
- Reproducibility risk.
- Hardware infeasibility.
- Spec conflict.
- Missing test or checker.
- Documentation ambiguity.

## Runner Rules

Runner should treat every experiment as immutable evidence.

Each run folder should include:

```text
reports/runs/YYYYMMDD-HHMMSS-short-name/
  command.txt
  environment.txt
  params.json
  metrics.json
  stdout.log
  stderr.log
  notes.md
```

Runner must flag:

- Crash or timeout.
- NaN or unstable training.
- Accuracy/resource numbers outside expected range.
- Non-reproducible results under the same seed.
- GPU/CPU mismatch.

## Conflict Resolution

If agents disagree:

1. Preserve both claims in the relevant review or literature file.
2. Main Agent adds or updates an entry in `docs/DECISIONS.md`.
3. If a spec change is accepted, Main Agent updates `SPEC.md`,
   `CONSTRAINTS.md`, and `TRACEABILITY.md` as needed.

