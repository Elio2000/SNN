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

### EXP-20260511-001: eprop CPU speech reproducibility smoke

Question:

Can `eprop/model/main.py` run a minimal CPU-only speech experiment with explicit
seed and data path, without overwriting the existing checkpoint?

Hypothesis:

Adding explicit `--seed` and `--data-dir` should make the tiny run repeatable.

Code version:

Working tree after P1 reproducibility changes to `eprop/model/main.py`,
`eprop/model/setup.py`, and `eprop/model/train.py`.

Command:

```bash
cd eprop
/opt/anaconda3/envs/snn_eprop/bin/python model/main.py --device cpu --epochs 1 --train-len 2 --test-len 2 --batch-size 1 --test-batch-size 1 --n-rec 4 --seed 123 --data-dir ./data
```

Run directory:

None. Command output is preserved in the Codex session; future non-trivial runs
should use `reports/runs/`.

Dataset:

`eprop/data/data.npy`, `eprop/data/label.npy`.

Parameters:

`device=cpu`, `seed=123`, `train_len=2`, `test_len=2`, `batch_size=1`,
`test_batch_size=1`, `n_rec=4`, `test_split=0.2`, `SynapseNoB=4`,
`NeuronNoB=7`.

Metrics:

Two repeated runs produced the same values:

- Train score: `0/2`, loss `0.7393251657485962`.
- Test score: `0/2`, loss `1.5268422365188599`.
- Train/test pool sizes: `2400/600`.

Anomalies:

- The local `snn_eprop` environment works through
  `/opt/anaconda3/envs/snn_eprop/bin/python`; `conda run -n snn_eprop ...`
  failed from `eprop/` in the Codex shell and fell back to `/opt/local/bin/python`.
- Legacy conda PyTorch reports MPS unavailable despite Apple M1 Max Metal
  hardware being present. This was resolved by the uv environment in
  `EXP-20260511-002`.
- Matplotlib/fontconfig emitted cache warnings because user cache directories
  are not writable from the sandbox.

Interpretation:

The tiny CPU speech run is reproducible under the explicit seed/data path
interface. It is only a smoke test, not an accuracy result.

Spec impact:

Supports the repository goal of making e-prop experiments runnable and
reproducible before hardware mapping claims.

Next action:

Add machine-readable run artifacts and metrics under `reports/runs/` before
accepting larger experiments.

### EXP-20260511-002: uv MPS speech smoke

Question:

Can the maintained uv environment use Apple GPU/MPS for the same tiny speech
smoke test?

Hypothesis:

Using a fresh uv environment with a current PyTorch wheel should make MPS
available on Apple M1 Max.

Code version:

Working tree with `pyproject.toml`, `.python-version`, and `uv.lock`.

Command:

```bash
uv --cache-dir /private/tmp/uv-cache sync
uv --cache-dir /private/tmp/uv-cache run python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available()); print(torch.ones(1, device='mps'))"
uv --cache-dir /private/tmp/uv-cache run python eprop/model/main.py --device mps --epochs 1 --train-len 2 --test-len 2 --batch-size 1 --test-batch-size 1 --n-rec 4 --seed 123 --data-dir eprop/data
```

Run directory:

None. Command output is preserved in the Codex session; future non-trivial runs
should use `reports/runs/`.

Dataset:

`eprop/data/data.npy`, `eprop/data/label.npy`.

Parameters:

`device=mps`, `seed=123`, `train_len=2`, `test_len=2`, `batch_size=1`,
`test_batch_size=1`, `n_rec=4`, `test_split=0.2`, `SynapseNoB=4`,
`NeuronNoB=7`.

Metrics:

- PyTorch version: `2.11.0`.
- MPS status: built `True`, available `True`.
- Train score: `0/2`, loss `0.7393252849578857`.
- Test score: `0/2`, loss `1.5268425941467285`.
- Train/test pool sizes: `2400/600`.

Anomalies:

- `uv run` needs elevated execution in the Codex sandbox because uv's macOS
  system-configuration check panics under the default sandbox. This is a Codex
  execution constraint, not a project runtime issue.

Interpretation:

The uv environment supports Apple GPU execution for the current tiny smoke
test. CPU and MPS losses match within expected floating-point differences.

Spec impact:

Future experiment commands should prefer uv and can use `--device mps` after
recording the MPS availability check.

Next action:

Use uv for the first real run artifact under `reports/runs/`, including
`uv.lock`, environment details, command, stdout/stderr, and metrics.

### EXP-TODO-001: eprop CPU speech smoke test

Status:

Completed by `EXP-20260511-001` for a tiny CPU-only reproducibility smoke.
Future work should move from command-output-only smoke tests to run directories
under `reports/runs/`.

Question:

Can `eprop/model/main.py` run a minimal CPU-only speech experiment to
completion?

Expected value:

This establishes a safe baseline for future `eprop` cleanup.

Resolved blockers:

- `main.py` no longer crashes at the final bitwidth print.
- `--dataset speech` is now a valid CLI choice.
- Visualization is disabled by default and accepts explicit boolean values.

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
