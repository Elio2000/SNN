# Review: Current `eprop/`

Date: 2026-05-11
Owner: Main Agent / Codex

Scope:

- Compare the current `eprop/model` code against the older e-prop baseline that
  was present before the directory rename.
- Ignore cue-accumulation support, per user direction.
- Identify functional differences and likely bugs before cleaning the maintained
  mainline.

## Summary

The current `eprop/` is the newer and more relevant branch for the speech-recognition
task. It adds label-aware speech data, circuit-oriented neuron/output options,
device-safe quantized weight assignment, and checkpoint saving.

It is not yet clean enough to use as-is. The most important issues are:

- `main.py` crashes after training because it prints undefined variables.
- The true test loader is ignored; current "test" can run on the training
  loader.
- New `LIF_type` and `Output_type` options are overwritten to `True`, so the
  circuit modes cannot be enabled from CLI.
- Boolean CLI arguments are parsed incorrectly.
- Default visualization is enabled and hard to disable from CLI.
- Speech train/test sampling is random, with replacement, and not a fixed split.

## Functional Differences Besides Cue Accumulation

| Area | Older baseline | Current `eprop/` | Impact |
| --- | --- | --- | --- |
| Speech labels | Infers target with `selected_speech // 300` | Loads `data/label.npy` | Current `eprop/` is more correct for real filename labels |
| Speech data files | Loads `./data.npy` | Loads `./data/data.npy` and `./data/label.npy` | Better structure, still hardcoded |
| Preprocessing | No current label-producing preprocess in `eprop` | `preprocess.py` extracts labels and writes `label.npy` | Useful and should be kept |
| Weight quantization device | Uses `.cuda()` in quantization assignment | Uses `.to(self.device)` | Current `eprop/` is better for CPU/GPU portability |
| Log quantization helper | Has unused `t2()` | Removed `t2()` | No loss if unused |
| LIF dynamics | Paper-style subtractive reset only | Adds paper-style and circuit-style reset modes | Useful for hardware mapping |
| Output dynamics | Output membrane `vo` only | Adds optional output spike/reset state `zo` | Useful but incomplete |
| Visualization | Default `False` | Default `True` | Current `eprop/` is worse for headless/runnable experiments |
| Checkpoint | No unconditional save | Saves `model.pth` unconditionally | Useful, but output path must be controlled |
| Model checkpoint present | None | `model.pth` exists with 6 tensors | Useful artifact, should be treated as generated |
| Test script | Old/incomplete | Still old/incomplete | `eprop/model/test.py` needs rewrite or removal |

## Confirmed Runtime Check

Syntax check passed:

```bash
python -m py_compile eprop/model/main.py eprop/model/setup.py eprop/model/models.py eprop/model/train.py eprop/preprocess.py
```

Tiny CPU run command:

```bash
cd eprop
python model/main.py --cpu --epochs 1 --train-len 2 --test-len 2 --batch-size 1 --test-batch-size 1 --n-rec 4
```

Observed:

- Training and evaluation started.
- A `model.pth` checkpoint was written before final crash.
- The run crashed at the final print in `main.py`:
  `NameError: name 'synapse_nob' is not defined`.

## Findings

### High: `main.py` crashes after training

File:

- `eprop/model/main.py`

Problem:

`main.py` prints `synapse_nob` and `neuron_nob`, but those variables do not
exist in this version.

Impact:

Every normal run crashes at the end, even if training completed.

Suggested fix:

Print `args.SynapseNoB` and `args.NeuronNoB`, or remove the print.

### High: Test loader is ignored

File:

- `eprop/model/main.py`

Problem:

`setup.setup(args)` returns `test_loader`, but `main.py` passes `train_loader` as
the `test_loader` argument to `train.train(...)`.

Impact:

Reported "test set" accuracy can actually be training-loader accuracy. If
`train_len` and `test_len` differ, the score denominator can also be wrong.

Suggested fix:

Pass the real `test_loader` returned by setup.

### High: Circuit options are defined but disabled by `main.py`

Files:

- `eprop/model/main.py`
- `eprop/model/models.py`

Problem:

`main.py` declares `--LIF_type` and `--Output_type`, then immediately forces:

```python
args.LIF_type = True
args.Output_type = True
```

Impact:

The new circuit-style LIF branch and output-spike branch in `models.py` cannot
be selected through the main script.

Suggested fix:

Remove the forced assignments and replace `type=bool` with explicit
`store_true` / `store_false` flags or string choices.

### High: Boolean CLI parsing is wrong

File:

- `eprop/model/main.py`

Problem:

Several arguments use `type=bool`. In argparse, values like `--visualize False`
parse to `True` because non-empty strings are truthy.

Affected options:

- `--shuffle`
- `--visualize`
- `--visualize-light`
- `--LIF_type`
- `--Output_type`

Impact:

Users cannot reliably disable visualization or choose model variants from CLI.

Suggested fix:

Use `action="store_true"` / `action="store_false"` or explicit choices such as
`choices=["paper", "circuit"]`.

### Medium: Dataset choices are malformed

File:

- `eprop/model/main.py`

Problem:

The choices list is:

```python
choices = ['cue_accumulation, speech']
```

This is one string, not two choices. Default `speech` works only because argparse
does not validate the default when the option is omitted. Passing
`--dataset speech` fails.

Impact:

Normal CLI usage is broken.

Suggested fix:

Since cue accumulation is no longer needed, use `choices=["speech"]` or remove
the dataset argument until another dataset exists.

### Medium: `setup.py` still references removed cue loader

File:

- `eprop/model/setup.py`

Problem:

The branch for `args.dataset == "cue_accumulation"` calls
`load_dataset_cue_accumulation`, but that function no longer exists.

Impact:

Not relevant if cue accumulation is abandoned, but the dead branch should be
removed to avoid confusing future agents.

Suggested fix:

Remove cue-accumulation branch and make speech the only supported dataset.

### Medium: Speech train/test split is not a real split

File:

- `eprop/model/setup.py`

Problem:

Each `SpeechDataset` randomly samples examples with replacement from 3000
recordings. Train and test datasets are generated independently and can overlap.

Impact:

Accuracy is not a stable benchmark and may overestimate generalization.

Suggested fix:

Create a deterministic index split with a seed, then sample from disjoint train
and test index lists.

### Medium: Randomness is not reproducible

Files:

- `eprop/model/setup.py`
- `eprop/model/train.py`

Problem:

`torch.manual_seed(42)` is set inside training, but NumPy random sampling and
Poisson spike generation use unseeded random states.

Impact:

Repeated runs with the same CLI parameters can produce different datasets and
spikes.

Suggested fix:

Add a `--seed` argument and use it for Python, NumPy, torch, dataset sampling,
and Poisson generation.

### Medium: Output-spike mode is incomplete

File:

- `eprop/model/models.py`

Problems:

- When `Output_type == False`, final classification still uses
  `softmax(self.vo)`, not `zo` or spike counts.
- In the circuit-style LIF branch, output reset checks `self.zo[t+1]` before it
  is computed, so reset likely never triggers at the intended time.
- `neuron_quan()` quantizes `v`, `z`, and `vo`, but not `zo`.

Impact:

The output-spike path is not a reliable hardware-equivalent output model yet.

Suggested fix:

Define whether output classification should use membrane, output spikes, or
spike counts. Then update the output reset order and quantization/checking.

### Medium: `preprocess.py` ignores CLI arguments

File:

- `eprop/preprocess.py`

Problem:

`parser.parse_args()` is commented out, and the script always uses
`Dataset('./recordings', 52000.0, 52.0)`.

Impact:

The script is not reusable from another working directory or data root.

Suggested fix:

Use parsed args and explicit output paths.

### Medium: `preprocess.py` assumes exactly 3000 recordings

File:

- `eprop/preprocess.py`

Problem:

It reshapes preprocessed data to `(1500, 400)`, which assumes exactly 3000 files
and exactly two 200-element samples per row.

Impact:

Any dataset-size change breaks preprocessing or silently invalidates indexing.

Suggested fix:

Derive shape from the number of processed files and store one sample per row, or
document the two-sample packing explicitly and validate file count.

### Low: Default visualization is unsuitable for batch runs

File:

- `eprop/model/main.py`

Problem:

`--visualize` defaults to `True`. The tiny CPU run produced matplotlib cache and
macOS GUI-related warnings.

Impact:

Runs are slower and noisier, and can be fragile in headless environments.

Suggested fix:

Default visualization to `False`.

### Low: `model.pth` is overwritten unconditionally

File:

- `eprop/model/train.py`

Problem:

Every training run writes `model.pth` in the current working directory.

Impact:

Previous checkpoints are overwritten and experiment outputs are not organized.

Suggested fix:

Add `--save-model` and `--model-path`, or write under `reports/runs/...`.

### Low: `test.py` is stale

File:

- `eprop/model/test.py`

Problem:

It appears to expect an older `train.train` return signature and an older model
forward signature.

Impact:

It should not be treated as a valid test harness.

Suggested fix:

Rewrite as a real evaluation script after the main training path is cleaned, or
remove/archive it.

## Recommended Fix Order

1. Make `eprop/model/main.py` run to completion:
   - remove undefined print,
   - pass real `test_loader`,
   - set visualization default off.
2. Fix CLI semantics:
   - dataset choices,
   - boolean/model-mode flags,
   - seed argument,
   - data path arguments.
3. Stabilize speech dataset:
   - deterministic split,
   - no hardcoded `3000`,
   - explicit sample index recording.
4. Decide output semantics:
   - membrane output,
   - output spike count,
   - or both as separate modes.
5. Move outputs to `reports/runs/` and stop overwriting `model.pth`.
