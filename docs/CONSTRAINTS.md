# Hardware And Software Constraints

Status: draft v0.1
Owner: Main Agent / Codex

This file separates hard constraints, working assumptions, and open questions.

## Hard Constraints

These should not be violated without an explicit decision in `DECISIONS.md`.

- Research and review agents do not edit core Python code by default.
- Source-of-truth docs are owned by the Main Agent.
- Experiments must record enough parameters to be reproducible.
- Hardware mapping must account for eligibility trace storage and update cost.
- Claims from papers or repositories require a source link or citation.

## Working Hardware Assumptions

These are placeholders until replaced by a more concrete chip target:

- The chip target is CIM-oriented and should exploit in-memory MAC where useful.
- On-chip learning should avoid full BPTT.
- Learning updates should be local or decomposable into local state plus a
  bounded learning signal.
- Bitwidth is a first-class design variable, not only a software compression
  knob.
- Weight writes are expensive and must be measured or estimated.

## Working Software Assumptions

- `eprop/` is the first maintained speech e-prop code area.
- The older e-prop baseline has been moved out of the workspace to a temporary
  backup at `/private/tmp/SNN_old_eprop_20260511-171229`.
- CPU smoke tests are required before relying on GPU-only behavior.
- Speech data paths should become explicit CLI arguments.
- Experiment outputs should be written under `reports/runs/`.

## Known Current Risks

- Some scripts assume local working directories and hardcoded data filenames.
- Some quantization functions are CUDA-specific.
- The e-prop implementation computes full eligibility tensors with PyTorch
  vectorized operations, which may hide online hardware costs.
- Random data selection and Poisson generation need explicit seed handling for
  reproducibility.
- There is no established checker for trace memory, weight traffic, or update
  primitive count.

## Constraint Checklist For New Experiments

Before accepting an experiment result, check:

- Was the exact command recorded?
- Was the data path recorded?
- Were random seeds recorded?
- Were bitwidth parameters recorded?
- Was device CPU/GPU recorded?
- Were crash/timeout/NaN conditions recorded?
- Are reported metrics enough to compare against hardware assumptions?
