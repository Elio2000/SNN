"""
CNM-side LIF neuron array with per-event compare/reset.

Hardware flow per input event (TRACEABILITY.md):
    input event -> leak -> capacitive injection -> membrane update
               -> threshold compare
               -> pre-reset overshoot capture    (if spiking)
               -> hard reset of membrane node    (if spiking)
               -> export post-spike + overshoot to FPGA

Leak is applied per-neuron based on elapsed time since the neuron's last event,
not a global synchronous timestep.

Note on GLOBAL_STALL: the simulator (simulator.py) models global stall by
re-queuing the entire event at _cnm_stall_until rather than calling inject().
The compare=False path below is retained as a lower-level primitive for future
use (e.g. per-neuron refractory or partial-array stall), but is not used by the
current stall model.
"""

from __future__ import annotations

import numpy as np

from .overshoot_lut import build_psi_lut, quantize_overshoot


class LIFArray:
    """
    CNM-side LIF neuron array.

    State is per-neuron float64 (representing the analog membrane).
    The psi LUT maps overshoot bins to surrogate-derivative values.
    """

    def __init__(
        self,
        n_rec: int,
        thr: float,
        v_reset: float,
        tau_m: float,
        overshoot_n_bits: int,
        overshoot_max: float,
        psi_lut: np.ndarray,
    ) -> None:
        self.n_rec = n_rec
        self.thr = thr
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.overshoot_n_bits = overshoot_n_bits
        self.overshoot_max = overshoot_max
        self.psi_lut = psi_lut

        self.v = np.zeros(n_rec, dtype=np.float64)
        self.t_last = np.zeros(n_rec, dtype=np.float64)

    # ------------------------------------------------------------------
    # internal helpers

    def _leak_to(self, t: float) -> None:
        """Decay all membrane voltages to current time t."""
        dt = t - self.t_last
        self.v *= np.exp(-dt / self.tau_m)
        self.t_last[:] = t

    # ------------------------------------------------------------------
    # public API

    def inject(
        self, t: float, w_col: np.ndarray, compare: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process one spike event fanning out to all n_rec postsynaptic neurons.

        w_col  : weight column for this presynaptic neuron, shape (n_rec,).
        compare: if False, skip threshold comparison and reset. Membrane still
                 leaks and integrates. Not used by the current global-stall model
                 (which re-queues events instead); retained for future per-neuron
                 refractory or partial-array stall experiments.

        Returns
        -------
        spike_mask : bool array (n_rec,) - True where neuron fired
        overshoot  : float array (n_rec,) - pre-reset v - thr (0 for non-spiking)
        psi_bins   : int array (n_rec,)   - quantized bin indices
        """
        self._leak_to(t)
        self.v += w_col

        if not compare:
            zeros = np.zeros(self.n_rec, dtype=np.int32)
            return np.zeros(self.n_rec, dtype=bool), np.zeros(self.n_rec), zeros

        spike_mask = self.v > self.thr
        overshoot = np.where(spike_mask, self.v - self.thr, 0.0)
        psi_bins = quantize_overshoot(overshoot, self.overshoot_n_bits, self.overshoot_max)

        self.v[spike_mask] = self.v_reset
        return spike_mask, overshoot, psi_bins

    def reset(self) -> None:
        self.v[:] = 0.0
        self.t_last[:] = 0.0
