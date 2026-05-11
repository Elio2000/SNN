# Repository Audit: e-prop Speech Code

Date: 2026-05-11
Owner: Main Agent / Codex

This audit originally inspected the three pre-cleanup code/data folders:

- old `eprop/`
- `eprop_new/`
- `speech/`

After the audit:

- `eprop_new/` was renamed to `eprop/`.
- The older `eprop/` directory was moved out of the workspace to
  `/private/tmp/SNN_old_eprop_20260511-171229`.
- `speech/` was moved out of the workspace to
  `/private/tmp/SNN_speech_20260511-172237`.

## Executive Summary

The current `eprop/` contains the newest e-prop speech-recognition experiment
work, but it is not a clean runnable mainline yet. It adds important newer ideas
such as real speech labels, circuit-style LIF/reset options, output-spike mode,
CPU-safe quantization assignment, and a saved checkpoint. It also contains
regressions and unfinished edits.

The old `eprop/` was older and is now only a temporary backup. Cue accumulation
is not part of the target workflow.

`speech/` was not an e-prop implementation. It was a separate "Speech to
Spiking Signatures" / PyGeNN supervised-STDP project plus raw/preprocessed
speech data that later e-prop experiments reused. Its useful wav preprocessing
logic is covered by current `eprop/preprocess.py`, and its recordings were
duplicated byte-for-byte in `eprop/recordings/`.

Recommended mainline strategy after user clarification:

Keep the current `eprop/` as the maintained speech e-prop line and fix it in
place. The old `speech/` tree is no longer part of the workspace.

## Folder Classification

| Folder | Role | Status | Keep As |
| --- | --- | --- | --- |
| old `eprop/` | Original PyTorch e-prop baseline plus early speech adaptation | Older baseline | Temporary backup outside workspace |
| current `eprop/` | Newer speech/eprop experiment branch | Newest but unfinished | Mainline after cleanup |
| old `speech/` | PyGeNN supervised-STDP speech project and duplicated data source | Removed from workspace | Temporary backup outside workspace |

## Size And Data Summary

Observed sizes:

- old `eprop/`: about 688 KB.
- current `eprop/`: about 36 MB.
- old `speech/`: about 65 MB.
- current `eprop/recordings/`: about 26 MB.
- `speech/recordings/`: about 26 MB.
- `speech/genn-master/`: about 20 MB.

Recording duplication:

- current `eprop/recordings/` has 3000 wav files.
- `speech/recordings/` has 3000 wav files.
- File names match exactly.
- Byte-for-byte comparison found `checked=3000 mismatches=0`.

So the recordings are duplicated and should eventually be consolidated into one
data location.

## Data Files

Observed `.npy` shapes:

| File | Shape | Notes |
| --- | --- | --- |
| `speech/data.npy` | `(1500, 400)` | Speech features, no labels |
| `speech/data/frames.npy` | `(3000, 40, 5)` | Frame-level speech features |
| `eprop/data/data.npy` | `(1500, 400)` | Speech features |
| `eprop/data/frames.npy` | `(3000, 40, 5)` | Frame-level speech features |
| `eprop/data/label.npy` | `(3000,)` | Digit labels 0-9 |
| `speech/data/data_52000.npy` | object dict | Keys: `data`, `labels`; data shape `(1500, 200)` |
| `speech/data/data_10000.npy` | object dict | Keys: `data`, `labels`; data shape `(1500, 200)` |
| `speech/data/new_file.npy` | object dict | Same hash as `data_52000.npy` |

`eprop/data/label.npy` is a meaningful improvement over the older speech
data path because it preserves labels extracted from filenames.

## Code Version Findings

### Old `eprop/`

Strengths:

- Keeps the original cue-accumulation dataset.
- Keeps the original PyTorch e-prop structure.
- Has the clearest upstream provenance from `README.md`.

Risks:

- Speech loader uses `./data.npy` directly.
- Speech labels are synthetic: `selected_speech // 300`, not label-file based.
- Quantization assignment uses `.cuda()` in `models.py`, which blocks CPU smoke
  tests.
- Experiment sweep is hardwired inside `main.py`.

### Current `eprop/`

Useful newer changes:

- `preprocess.py` extracts labels from wav filenames and writes `label.npy`.
- `model/setup.py` loads `./data/label.npy` and `./data/data.npy`.
- `model/models.py` changes quantized weight assignment from `.cuda()` to
  `.to(self.device)`, which is better for CPU/GPU portability.
- `model/models.py` adds circuit-style LIF/reset and output-spike options.
- `model/train.py` saves `model.pth`.
- Existing `model.pth` contains six tensors:
  `w_in`, `w_in_quan`, `w_rec`, `w_rec_quan`, `w_out`, `w_out_quan`.

Known problems:

- `model/main.py` prints `synapse_nob` and `neuron_nob` without defining them.
- `model/setup.py` still has a stale cue-accumulation branch that references a
  removed loader.
- `model/setup.py` still hardcodes relative data files.
- Random sampling is not controlled by a reproducible seed.
- `preprocess.py` declares CLI args but ignores `parser.parse_args()` and uses
  hardcoded `./recordings`.
- `model/train.py` saves `model.pth` unconditionally in the working directory.

Conclusion:

The current `eprop/` is the selected main speech e-prop line, but it needs
cleanup before it is reliable. Detailed code findings are in
`docs/EPROP_REVIEW.md`.

### Old `speech/`

Purpose:

- This folder implements the paper "A Spiking Network that Learns to Extract
  Spike Signatures from Speech Signals".
- Its training path uses PyGeNN and supervised STDP, not e-prop.
- It contains raw recordings, preprocessed features, a vendored or extracted
  `genn-master/`, generated GeNN C++ output, and notebooks.

Evidence:

- `speech/README.md` describes PyGeNN installation and STDP training.
- `speech/src/train.py` imports `pygenn` and defines custom Izhikevich neurons
  and supervised STDP.
- `speech/speech_recognition_CODE/` contains generated/compiled GeNN artifacts,
  including `.cc`, `.o`, `.d`, and `librunner.so`.

Conclusion:

`speech/` did not need to remain in the workspace. The STDP/PyGeNN logic is out
of scope, and the wav preprocessing in current `eprop/preprocess.py` is the
same feature extraction pipeline plus label export.

## Generated Or Low-value Files

Candidates to ignore or archive after backup:

- `.DS_Store`
- `__pycache__/`
- `speech/speech2spike.ipynb` because it is empty.
- `speech/speech_recognition_CODE/*.o`
- `speech/speech_recognition_CODE/*.d`
- `speech/speech_recognition_CODE/librunner.so`
- Duplicate recordings in either `speech/recordings/` or `eprop/recordings/`
- Duplicate generated `.npy` files after a canonical data path exists.
- `speech/genn-master.zip` and `speech/genn-master/` unless PyGeNN source needs
  to stay vendored.

## Recommended Next Engineering Step

Before moving files:

1. Initialize git or make a filesystem backup.
2. Create a canonical layout plan.
3. Establish a tiny CPU speech smoke test for `eprop/model/main.py`.
4. Clean `eprop` in place:
   - label-aware speech dataset,
   - device-safe quantization,
   - explicit data path arguments,
   - circuit LIF/output options,
   - controlled checkpoint output.
5. Move duplicate data only after code uses the canonical data path.
