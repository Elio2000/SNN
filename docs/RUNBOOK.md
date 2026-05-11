# Runbook

Status: draft
Owner: Main Agent / Runner

This file records supported commands. Commands marked "planned" may require code
cleanup before they work reliably.

## Repository Preparation

Recommended before multi-CLI work:

```bash
git init
git status --short
```

The initial inspection did not detect a top-level `.git` directory.

## Environment Discovery

Use these before running experiments:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Planned Smoke Test: eprop Speech

Purpose:

Run a small speech experiment after the data path is made explicit.

Planned command shape:

```bash
cd eprop
python model/main.py --cpu --epochs 1 --train-len 2 --test-len 2 --batch-size 1 --test-batch-size 1 --n-rec 4
```

Known issue:

This command currently starts training but crashes at the final undefined
`synapse_nob`/`neuron_nob` print in `eprop/model/main.py`. The current
speech loader also hardcodes `./data/data.npy` and `./data/label.npy`; these
should become CLI parameters before experiments are considered reproducible.

## Run Artifact Convention

For any non-trivial run, save:

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

## Before Accepting A Result

Check:

- Command recorded.
- Dataset recorded.
- Random seed recorded.
- Device recorded.
- Bitwidth parameters recorded.
- Metrics recorded in machine-readable form.
- Anomalies recorded.
