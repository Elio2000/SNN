# Review Agent Brief

Use this brief for the Claude Review CLI.

## Mission

Review the repository for correctness, reproducibility, and hardware feasibility
risks in the e-prop-to-CIM workflow. Claude should not edit source code by
default; it should write review reports.

## Required Output Location

Write reports to:

```text
reviews/YYYY-MM-DD-short-topic.md
```

## Review Modes

### Spec Review

Read:

- `docs/SPEC.md`
- `docs/CONSTRAINTS.md`
- `docs/TRACEABILITY.md`
- `docs/DECISIONS.md`

Check:

- Whether each algorithm assumption can plausibly map to hardware.
- Whether eligibility trace storage/update is bounded.
- Whether learning signals require global operations that violate the intended
  on-chip model.
- Whether bitwidth, state, and memory assumptions are explicit.
- Whether the spec confuses software convenience with hardware primitive.

### Code Review

Read the files specified by the Main Agent. If no files are specified, start
with:

- `eprop/model/main.py`
- `eprop/model/setup.py`
- `eprop/model/train.py`
- `eprop/model/models.py`
- `eprop/preprocess.py`

Check:

- Runtime bugs and shape/device errors.
- Non-deterministic data generation and missing seeds.
- Hardcoded data paths.
- CUDA-only behavior in functions expected to run on CPU.
- Full-tensor operations that hide online-learning or hardware costs.
- Quantization semantics, clipping, scaling, and signed range assumptions.
- Missing smoke tests or experiment logging.

## Report Format

```text
# Review: [topic]

## Findings

### High
- [Category] file:line - finding.
  Impact:
  Suggested fix or verification:

### Medium
- ...

### Low
- ...

## Open Questions

## Suggested Verification

## Summary
```

If no issues are found, say so clearly and list remaining test gaps.
