"""
Input spike encoder for the EventHWNet (event-driven control path) only.

Used exclusively by EventHWNet / HWSim. TSHWNet uses TSScheduler's randomized
AER arbitration instead and does NOT call this module.

Converts a software-timestep raster to a flat hardware event list for HWSim.
Within each timestep, channels are enumerated in ascending index order — this is
a property of the continuous-time HWSim path, not a hardware-accuracy claim.
"""

from __future__ import annotations

import numpy as np


def raster_to_events(
    raster: np.ndarray,
    dt_per_ts: float = 1.0,
) -> list[tuple[float, int]]:
    """
    Convert a [T, n_in] spike raster to a hardware event list.

    Parameters
    ----------
    raster : bool/int array, shape (T, n_in)
        raster[ts, i] = 1 means input neuron i spikes at software timestep ts.
    dt_per_ts : hardware time units per software timestep (default 1.0).
        Hardware time for a spike at ts is t_hw = ts * dt_per_ts.
        Set t_max = T * dt_per_ts when calling HWSim.run().

    Returns
    -------
    List of (t_hw, input_idx) pairs in ascending (t_hw, input_idx) order.
    For hardware-accurate arbitration use TSHWNet/TSScheduler instead.
    """
    raster = np.asarray(raster, dtype=bool)
    if raster.ndim != 2:
        raise ValueError(f"raster must be 2-D [T, n_in], got shape {raster.shape}")
    T, _n_in = raster.shape
    events: list[tuple[float, int]] = []
    for ts in range(T):
        t_hw = float(ts) * dt_per_ts
        for i in range(_n_in):
            if raster[ts, i]:
                events.append((t_hw, i))
    return events
