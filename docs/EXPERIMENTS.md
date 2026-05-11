# Experiments

Status: active log template
Owner: Main Agent / Experiment workflow

Do not record claimed results here unless the run artifacts exist under
`reports/runs/` or the command output is otherwise preserved.

## Experiment Entry Template

```text
## EXP-YYYYMMDD-NNN: [short name]

Question:
Hypothesis:
Code version:
Command:
Run directory:
Dataset:
Parameters:
Metrics:
Anomalies:
Interpretation:
Spec impact:
Next action:
```

## Planned Baseline Experiments

### EXP-TODO-001: eprop CPU speech smoke test

Question:

Can `eprop/model/main.py` run a minimal CPU-only speech experiment to
completion?

Expected value:

This establishes a safe baseline for future `eprop` cleanup.

Known blockers:

- `main.py` currently crashes at the final undefined-variable print.
- Current CLI dataset choices need review before passing `--dataset`.
- Visualization defaults to on and is hard to disable from CLI.

### EXP-TODO-002: quantization sweep scaffold

Question:

Can `SynapseNoB` and `NeuronNoB` be swept while producing machine-readable
metrics?

Expected value:

This is the first bridge from algorithm behavior to CIM bitwidth tradeoffs.

### EXP-TODO-003: eligibility trace resource estimate

Question:

How much state is implied by input, recurrent, and output traces for the current
network sizes?

Expected value:

This identifies whether the current e-prop implementation is directly
hardware-feasible or needs approximation.
