"""
LI (Leaky Integrator) output readout from post_spike event records.

Hardware correspondence: w_out fan-in is computed on the FPGA — each recurrent
post_spike export is multiplied by w_out[k, j] and accumulated into an output
register. The register is read at the end of each software timestep (word period
boundary) to produce logits, then optionally decayed before the next timestep.

This module is a post-processing pass over EventRecord lists. It does not modify
HWSim or inject callbacks; all hardware-side logic is captured after the fact from
the spike record.

Timestep assignment rule
------------------------
A post_spike at hardware time t is assigned to software timestep:
    ts = min(floor(t / dt_per_ts), n_ts - 1)

So a spike at t = k * dt_per_ts falls at the START of ts=k (not the end of ts=k-1).
A recurrent carry-over spike from ts=k fires at t = k*dt_per_ts + dt_hw; with
dt_per_ts = dt_hw = 1.0 this lands in ts=k+1 (the next timestep). The last-ts cap
(n_ts-1) ensures any tail spikes past the final boundary are counted in the last ts.

output_trace sampling
---------------------
output_trace[ts] is the accumulated output_v at the END of software_ts ts, BEFORE
applying the inter-timestep leak. leak=1.0 (default) disables decay.
"""

from __future__ import annotations

import numpy as np


def compute_output_trace(
    spike_records: list,
    w_out: np.ndarray,
    n_ts: int,
    dt_per_ts: float = 1.0,
    leak: float = 1.0,
) -> np.ndarray:
    """
    Compute LI readout trace from a post_spike EventRecord list.

    Parameters
    ----------
    spike_records : list of EventRecord (kind == "post_spike" entries are used)
    w_out : (n_out, n_rec) float weight matrix
    n_ts : number of software timesteps
    dt_per_ts : hardware time units per software timestep
    leak : multiplicative decay applied to output_v after each timestep readout.
           1.0 = no leak (default). Must be in (0, 1].

    Returns
    -------
    output_trace : np.ndarray, shape (n_ts, n_out)
        output_trace[ts] = output_v at end of software timestep ts (before leak).
    """
    if n_ts <= 0:
        raise ValueError(f"n_ts must be > 0, got {n_ts}")
    n_out = w_out.shape[0]
    output_v = np.zeros(n_out)
    output_trace = np.zeros((n_ts, n_out))

    # Group post-spike sources by software_ts (stable sort by t handles ordering)
    by_ts: list[list[int]] = [[] for _ in range(n_ts)]
    for r in spike_records:
        if r.kind == "post_spike":
            ts = min(int(r.t / dt_per_ts), n_ts - 1)
            by_ts[ts].append(r.src)

    for ts in range(n_ts):
        for j in by_ts[ts]:
            output_v += w_out[:, j]
        output_trace[ts] = output_v.copy()
        output_v *= leak

    return output_trace
