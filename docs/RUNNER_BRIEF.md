# Runner Agent Brief

Use this brief only when a separate CLI is assigned to long-running experiments.

## Mission

Run commands exactly as requested by the Main Agent, capture evidence, and report
anomalies. Do not edit source or spec files.

## Output Directory

Each run gets a new directory:

```text
reports/runs/YYYYMMDD-HHMMSS-short-name/
```

Required files:

```text
command.txt
environment.txt
params.json
metrics.json
stdout.log
stderr.log
notes.md
```

## Metrics To Capture

For e-prop/CIM experiments, capture when available:

- Accuracy or loss.
- Dataset and sample counts.
- Random seed.
- `SynapseNoB`, `NeuronNoB`, and quantization scheme.
- `n_rec`, `n_inputs`, `n_classes`, `n_steps`.
- Spike rate.
- Eligibility trace tensor/state size.
- Number of weight reads/writes per sample or timestep.
- Runtime and device.
- Crash, timeout, NaN, or out-of-memory status.

## Anomaly Rules

Flag the run as anomalous if:

- The command crashes or times out.
- Metrics are missing or unparsable.
- Accuracy changes materially under the same seed.
- CPU/GPU results differ unexpectedly.
- Resource use is inconsistent with `docs/CONSTRAINTS.md`.

