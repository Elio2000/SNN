# SPEC: SNN CIM Chip For E-prop On-chip Learning

Status: draft v0.1
Owner: Main Agent / Codex

This document is the current source of truth for the chip/software target. Do
not treat open questions as design decisions.

## Goal

Explore a CIM-oriented SNN chip design for on-chip learning, using e-prop or an
e-prop-derived local learning rule as the main algorithmic target.

The immediate repository goal is to make the Python e-prop code runnable,
measurable, and traceable to hardware primitives.

## Current Software Baseline

Main code areas:

- `eprop/`: maintained speech e-prop implementation target.
- old `eprop/`: removed from the workspace; temporary backup is under
  `/private/tmp/SNN_old_eprop_20260511-171229`.
- removed `speech/`: PyGeNN/STDP reference code moved out of the workspace to
  `/private/tmp/SNN_speech_20260511-172237`.

Current baseline status:

- No top-level git repository was detected during initial inspection.
- No top-level test framework is defined.
- `eprop` contains the most relevant speech e-prop implementation, but it
  needs cleanup before it is reliable.
- Speech experiments currently depend on local data-path conventions.

## Target Workloads

Primary:

- E-prop training on speech-like SNN classification.

Secondary:

- Small synthetic speech-like tasks for smoke tests and regression checks.
- Small synthetic tasks for verifying trace and quantization behavior.

## Algorithm Scope

Initial in scope:

- LIF recurrent SNN.
- Input, recurrent, and output weights.
- Eligibility traces for input and recurrent weights.
- Output-layer trace/update.
- Quantized weights and neuron states.
- Online or near-online learning analysis.

Out of scope until explicitly accepted:

- Full BPTT as the chip learning algorithm.
- ALIF or more complex neuron models.
- Large external frameworks as mandatory runtime dependencies.
- Unbounded tensor-level operations that cannot be mapped to chip resources.

## Chip-Oriented Metrics

Every serious experiment should try to report:

- Accuracy or task loss.
- Weight bitwidth.
- Neuron-state bitwidth.
- Spike rate.
- Number of neurons and synapses.
- Eligibility trace state size.
- Weight read/write count.
- Local memory footprint.
- Required multiply/add/shift/LUT operations.
- Whether the update can be performed online.

## Open Questions

These must be resolved through experiments, review, or design decisions:

- Which speech subset or synthetic speech-like task is the first tapeout-
  relevant benchmark?
- Is full e-prop trace storage feasible, or must traces be approximated?
- Are multipliers allowed in the learning datapath, or only add/shift/LUT?
- What CIM primitive is assumed: SRAM, RRAM, MRAM, or abstract MAC array?
- What is the write endurance and write energy model?
- Which optimizer is allowed on chip: SGD-like local update only, or something
  more complex?
- Should output use membrane potential, spike count, or another observable?
- What accuracy degradation is acceptable under quantization?

## Acceptance Criteria For Mainline Changes

A code change can be accepted into the mainline if it:

- Improves reproducibility, measurability, or hardware traceability.
- Has a small verification target: smoke test, unit test, or logged experiment.
- Does not hide hardware cost behind a larger software-only operation.
- Updates docs when assumptions or commands change.
