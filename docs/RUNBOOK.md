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

A top-level git repository is present as of 2026-05-11. Always check
`git status --short` before edits or experiment runs.

## Environment Discovery

Use these before running experiments:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Preferred maintained environment:

```bash
uv --cache-dir /private/tmp/uv-cache sync
uv --cache-dir /private/tmp/uv-cache run python --version
uv --cache-dir /private/tmp/uv-cache run python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"
```

Legacy local conda environment on this workstation:

```bash
/opt/anaconda3/envs/snn_eprop/bin/python --version
/opt/anaconda3/envs/snn_eprop/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"
```

Observed on 2026-05-11:

- `uv`: `/Users/lixiangting/.local/bin/uv`, version `0.8.9`
- uv environment Python: `.venv/bin/python3`, Python 3.12.12.
- uv environment PyTorch version: 2.11.0.
- uv environment MPS status: built and available; `device='mps'` tensor
  creation succeeds.
- `snn_eprop` Python: `/opt/anaconda3/envs/snn_eprop/bin/python`
- `snn_eprop` Python version: 3.8.20
- `snn_eprop` PyTorch version: 2.4.1
- CUDA unavailable on this Mac.
- Apple M1 Max Metal hardware is present.
- Legacy conda environments currently report MPS as unavailable; use the uv
  environment for Apple GPU runs.
- In the Codex shell, `conda run -n snn_eprop ...` can fail from `eprop/` and
  fall back to `/opt/local/bin/python`; use the absolute environment Python or
  an already activated shell.

## Smoke Test: eprop Speech

Purpose:

Run a small speech experiment with explicit seed and data path.

Legacy conda CPU command:

```bash
cd eprop
/opt/anaconda3/envs/snn_eprop/bin/python model/main.py --device cpu --epochs 1 --train-len 2 --test-len 2 --batch-size 1 --test-batch-size 1 --n-rec 4 --seed 123 --data-dir ./data
```

Preferred uv MPS command from the repository root, after `uv sync`:

```bash
uv --cache-dir /private/tmp/uv-cache run python eprop/model/main.py --device mps --epochs 1 --train-len 2 --test-len 2 --batch-size 1 --test-batch-size 1 --n-rec 4 --seed 123 --data-dir eprop/data
```

Expected behavior:

- Uses deterministic train/test pools from `--seed`.
- Uses `./data/data.npy` and `./data/label.npy` unless `--data-file` and
  `--label-file` are supplied.
- Does not save a checkpoint unless `--checkpoint-path` is supplied.
- Repeated runs with the same command should produce the same selected samples
  and losses.

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
