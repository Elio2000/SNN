# Repository Cleanup Plan

Status: proposed
Owner: Main Agent / Codex

This plan organizes the current research workspace without losing useful older
experiments.

## Principles

- Do not delete or move data before git/backup exists.
- Keep `eprop/` as the first maintained speech e-prop line because cue
  accumulation is not part of the target workflow.
- The older baseline `eprop/` has been moved out of the workspace to a temporary
  backup at `/private/tmp/SNN_old_eprop_20260511-171229`.
- `speech/` has been moved out of the workspace because it was STDP/PyGeNN
  reference code and duplicated data.
- Keep generated artifacts out of source control.

## Proposed Final Layout

```text
eprop/                   # cleaned mainline PyTorch speech e-prop
  model/
    main.py
    models.py
    setup.py
    train.py
  preprocess.py

experiments/
  eprop_original/            # optional future archive if old baseline is restored
  speech_pygenn_stdp/        # optional future archive if old speech backup is restored

data/
  raw/
    fsdd_recordings/         # single canonical copy of 3000 wav files
  processed/
    speech_40x5/
      data.npy
      frames.npy
      label.npy

docs/
reports/
reviews/
references/
```

This layout is a target, not a command sequence. The old baseline currently
exists only as a temporary backup outside the workspace.

## Phase 1: Safe Metadata Cleanup

Actions:

- Add `.gitignore` for OS noise, Python caches, generated data, checkpoints, and
  compiled artifacts.
- Keep source folders untouched except for the completed rename to `eprop/`.
- Record inventory in `docs/REPO_AUDIT.md`.

Status:

- `.gitignore` added.
- `docs/REPO_AUDIT.md` added.

## Phase 2: Establish `eprop` Mainline Smoke Test

Actions:

- Fix only the minimum needed in `eprop/model` for a tiny CPU speech run to
  complete.
- Make dataset choices valid CLI choices for speech.
- Disable visualization by default.
- Pass the real test loader to training/evaluation.
- Remove the undefined variable print at the end of `main.py`.
- Record the successful command in `docs/RUNBOOK.md`.

Verification target:

```bash
cd eprop
python model/main.py --cpu --epochs 1 --train-len 2 --test-len 2 --batch-size 1 --test-batch-size 1 --n-rec 4
```

## Phase 3: Clean `eprop` Feature Flags And Data Handling

Clean these existing `eprop` features:

- label-aware speech preprocessing,
- `label.npy`-based speech targets,
- device-aware quantization assignment,
- circuit-style LIF/reset and output-spike options,
- explicit checkpoint output path.

Rewrite:

- Undefined variable usage in `eprop/model/main.py`.
- Hardcoded `./data/*.npy` paths.
- Unconditional checkpoint writes.
- CLI arguments that are declared but ignored.
- Boolean CLI arguments using `type=bool`.

## Phase 4: Consolidate Data

After mainline code accepts explicit data paths:

- Move one copy of recordings to `data/raw/fsdd_recordings/`.
- Move accepted processed arrays to `data/processed/speech_40x5/`.
- Remove duplicate recordings and stale `.npy` files only after verifying hashes
  or regenerating them from raw data.

Expected removable duplicates:

- One of `eprop/recordings/` or `speech/recordings/`.
- One of `speech/data.npy` or `eprop/data/data.npy`, after confirming which
  one is canonical.

## Phase 5: External Reference Code

Status:

- Completed by moving `speech/` out of the workspace to
  `/private/tmp/SNN_speech_20260511-172237`.

Notes:

- Restore only if the PyGeNN/STDP reference or old preprocessed arrays are needed
  for comparison.
- The current `eprop/preprocess.py` covers the useful wav feature extraction and
  adds `label.npy` export.

## Immediate Do-not-do List

- Do not delete either recordings folder yet.
- Do not reintroduce cue-accumulation cleanup as a priority.
- Do not vendor more external repositories.
- Do not rewrite all scripts before a smoke test exists.
