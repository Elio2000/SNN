# Traceability: Algorithm To Hardware

Status: draft v0.1
Owner: Main Agent / Codex

This table maps software concepts to hardware implications. It should be updated
whenever the algorithm or hardware assumptions change.

| Algorithm Item | Current Software Location | Hardware Primitive | State / Storage | Risk | Status |
| --- | --- | --- | --- | --- | --- |
| Input spikes `x[t]` | `eprop/model/setup.py`, `eprop/model/models.py` | Spike buffer / input encoder | `n_steps * n_inputs` if buffered | Full buffering may be unnecessary | Open |
| Recurrent spikes `z[t]` | `eprop/model/models.py` | Spike register / event buffer | `n_rec` per timestep if online | Full history used by vectorized trace code | Open |
| Membrane voltage `v` | `eprop/model/models.py` | Neuron state SRAM/register | `n_rec * state_bits` | Quantization range unclear | Open |
| Output voltage `vo` | `eprop/model/models.py` | Output accumulator | `n_out * state_bits` | Output type not finalized | Open |
| Output spikes `zo` | `eprop/model/models.py` | Output spike register / counter | `n_out` per timestep if online | Current output-spike mode is incomplete | Open |
| Input weights `w_in` | `eprop/model/models.py` | CIM weight array | `n_rec * n_in * weight_bits` | Write/update cost | Open |
| Recurrent weights `w_rec` | `eprop/model/models.py` | CIM or local SRAM array | `n_rec * n_rec * weight_bits` | Self-connection masking and dense cost | Open |
| Output weights `w_out` | `eprop/model/models.py` | Digital/SRAM/CIM array | `n_out * n_rec * weight_bits` | Learning signal depends on these weights | Open |
| Surrogate derivative `h` | `eprop/model/models.py` | LUT / comparator / piecewise logic | transient or per-neuron state | Requires abs, threshold, scale | Open |
| Input trace | `eprop/model/models.py` | Eligibility trace state | potentially `n_rec * n_in` | May exceed local memory | High risk |
| Recurrent trace | `eprop/model/models.py` | Eligibility trace state | potentially `n_rec * n_rec` | Major storage/update cost | High risk |
| Output trace | `eprop/model/models.py` | Output eligibility state | `n_out * n_rec` or `n_rec` | Mapping depends on output update | Open |
| Learning signal `L` | `eprop/model/models.py` | Broadcast/error path | `n_rec` per timestep/sample | Global error path may be costly | Open |
| Weight update | `eprop/model/models.py`, optimizer step | Local write/update circuit | write traffic per weight | Optimizer must be hardware-feasible | Open |
| Quantization | `eprop/model/models.py` | ADC/DAC precision or digital fixed point | weight/state bitwidth | Current semantics need review | Open |

## Required Traceability Additions

For each accepted experiment, add or update:

- Which algorithm variables were measured.
- Which hardware primitive they imply.
- Whether storage is per-neuron, per-synapse, per-timestep, or global.
- Whether the operation is online or batch/offline.
- Whether the implementation uses software convenience that must be replaced.
