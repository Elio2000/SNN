"""
FPGA-side presynaptic trace maintenance and gradient accumulation.

Trace update rule (TRACEABILITY.md):
    p_i(t_event) = p_i(t_last) * exp(-(t_event - t_last) / tau_trace) + spike_i

On post-spike from neuron j, the eligibility is computed from a snapshot of
traces taken at spike-time (not FPGA_UPDATE time). This is the correct causal
order: under SH_BACKGROUND stall, quantization is delayed but the learning
timestamp is the post-spike time.

    snapshot at spike_t:  p_in_snap = p_in decayed to spike_t
    gradient:             dW_ji += eta * L_j * psi_j * p_in_snap[i]

One p_i is kept per presynaptic neuron. The full n_rec x n_in eligibility
tensor is never materialized.

Causal note: p_in[src] is updated at the same timestamp as the INPUT_SPIKE that
causes a post-spike, so the current input event IS included in the eligibility.
This is the hardware-natural ordering: trace update and spike export occur in the
same clock cycle. p_rec[j] is NOT included in j's own FPGA_UPDATE because
REC_SPIKE(j) is scheduled one dt_hw step later.

Weight quantization (D-0007 scaffold):
    Default: float weights (WeightQuant.n_bits == 0, no quantization).
    With WeightQuant(n_bits > 0): fixed-point over [-weight_max, +weight_max)
    using 2^n_bits bins, round-to-nearest (half-to-even). The floor/truncation
    bias is deliberately avoided because systematic negative bias in 4-5 bit
    training causes divergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightQuant:
    """
    Weight quantization config for the low-bit FPGA update rule (D-0007).

    n_bits == 0  : disabled (float scaffold mode).
    n_bits > 0   : fixed-point over [-weight_max, +weight_max) with 2^n_bits
                   bins of width step = 2*weight_max / 2^n_bits.
                   Rounding: round-to-nearest, ties to even (np.round semantics).
    stochastic_rounding: if True, stochastic rounding replaces round-nearest.
    """
    n_bits: int = 0
    weight_max: float = 1.0
    stochastic_rounding: bool = False


def _quantize_weights(
    w: np.ndarray, quant: WeightQuant, rng: np.random.Generator | None = None
) -> np.ndarray:
    # 2^n_bits bins over [-wm, +wm); step = 2*wm/n_levels (power-of-2 denominator
    # ensures all representable values are exactly on the grid).
    # Max representable = wm - step (values never exceed wm).
    n_levels = 1 << quant.n_bits
    wm = quant.weight_max
    step = 2.0 * wm / n_levels
    # Shift to [0, 2*wm), clip slightly below 2*wm to keep in range.
    shifted = np.clip(w + wm, 0.0, 2.0 * wm * (1.0 - 1e-15))
    if quant.stochastic_rounding:
        assert rng is not None, "rng required for stochastic rounding"
        floored = np.floor(shifted / step)
        frac = shifted / step - floored
        stoch = (rng.random(w.shape) < frac).astype(np.float64)
        bins = np.clip((floored + stoch).astype(np.int64), 0, n_levels - 1)
    else:
        # Round-to-nearest (half-to-even): unbiased, avoids systematic drift.
        bins = np.clip(np.round(shifted / step).astype(np.int64), 0, n_levels - 1)
    return -wm + bins.astype(np.float64) * step


class FPGATrace:
    def __init__(
        self,
        n_in: int,
        n_rec: int,
        tau_trace: float,
        w_in: np.ndarray,
        w_rec: np.ndarray,
    ) -> None:
        self.n_in = n_in
        self.n_rec = n_rec
        self.tau_trace = tau_trace

        self.w_in = w_in.copy().astype(np.float64)
        self.w_rec = w_rec.copy().astype(np.float64)

        self.p_in = np.zeros(n_in, dtype=np.float64)
        self.p_rec = np.zeros(n_rec, dtype=np.float64)
        self.t_last_in = np.zeros(n_in, dtype=np.float64)
        self.t_last_rec = np.zeros(n_rec, dtype=np.float64)

        self.dw_in = np.zeros_like(self.w_in)
        self.dw_rec = np.zeros_like(self.w_rec)

    # ------------------------------------------------------------------
    # trace updates (called when source fires)

    def on_input_spike(self, t: float, src: int) -> None:
        dt = t - self.t_last_in[src]
        self.p_in[src] = self.p_in[src] * np.exp(-dt / self.tau_trace) + 1.0
        self.t_last_in[src] = t

    def on_rec_spike(self, t: float, src: int) -> None:
        dt = t - self.t_last_rec[src]
        self.p_rec[src] = self.p_rec[src] * np.exp(-dt / self.tau_trace) + 1.0
        self.t_last_rec[src] = t

    # ------------------------------------------------------------------
    # snapshot (called at post-spike time, before FPGA_UPDATE is dispatched)

    def _p_in_at(self, t: float) -> np.ndarray:
        return self.p_in * np.exp(-(t - self.t_last_in) / self.tau_trace)

    def _p_rec_at(self, t: float) -> np.ndarray:
        return self.p_rec * np.exp(-(t - self.t_last_rec) / self.tau_trace)

    def snapshot_traces(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return copies of p_in and p_rec decayed to time t.

        Must be called at post-spike time. The copies are stored in the
        FPGA_UPDATE payload so that future trace mutations (from events
        between spike_t and fpga_update_t) do not affect this gradient.
        """
        return self._p_in_at(t).copy(), self._p_rec_at(t).copy()

    # ------------------------------------------------------------------
    # gradient accumulation (called at FPGA_UPDATE time using snapshots)

    def on_post_spike(
        self,
        j: int,
        psi_j: float,
        L_j: float,
        eta: float,
        p_in_snap: np.ndarray,
        p_rec_snap: np.ndarray,
    ) -> None:
        """
        Accumulate eligibility gradient for postsynaptic neuron j.

        p_in_snap / p_rec_snap must be trace values snapshotted at spike-time,
        not at fpga_update_time. This maintains causal correctness under delayed
        FPGA_UPDATE (SH_BACKGROUND or GLOBAL_STALL modes).

        e_ji = psi_j * p_i(spike_t)
        dW_ji += eta * L_j * e_ji
        """
        scale = eta * L_j * psi_j
        if scale == 0.0:
            return
        self.dw_in[j, :]  += scale * p_in_snap
        self.dw_rec[j, :] += scale * p_rec_snap

    # ------------------------------------------------------------------
    # weight application

    def apply_gradients(
        self,
        quant: WeightQuant | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Subtract accumulated gradients from weights and zero them.

        quant: optional fixed-point quantization applied after update.
               quant.n_bits == 0 (default) keeps float weights.
        """
        self.w_in  -= self.dw_in
        self.w_rec -= self.dw_rec
        self.dw_in[:]  = 0.0
        self.dw_rec[:] = 0.0

        if quant is not None and quant.n_bits > 0:
            self.w_in  = _quantize_weights(self.w_in,  quant, rng)
            self.w_rec = _quantize_weights(self.w_rec, quant, rng)

    def reset(self) -> None:
        self.p_in[:] = 0.0
        self.p_rec[:] = 0.0
        self.t_last_in[:] = 0.0
        self.t_last_rec[:] = 0.0
        self.dw_in[:] = 0.0
        self.dw_rec[:] = 0.0
