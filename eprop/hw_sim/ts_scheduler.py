"""
Software-timestep barrier scheduler for hardware-accurate SNN simulation.

Key differences from HWSim (event-driven, continuous time):
  - Operations quantized to software timesteps; no intra-ts leak or trace decay.
  - input_pending: bitmap, at most one spike per input channel per ts.
  - rec_pending: configurable buffer (ideal_fifo / bounded_fifo / per_neuron_counter).
  - Seeded randomized arbiter: uniform pick from all active input+rec request lines.
  - End-of-ts bulk decay for membrane voltages and eligibility traces.
  - Per-ts statistics: overflow, saturation, drain steps, event_order.

Gradient note
-------------
Within a ts, eligibility trace snapshots at spike time depend on which events were
serviced before them — i.e., on arbiter order. This is correct: simultaneous events
within a ts have no physical ordering. A fixed seed produces identical results across
runs; gradients are reproducible but arbiter-order-dependent.

Reproducibility
---------------
reset_state() does NOT re-seed the RNG. To reproduce the same event sequence from the
same arbiter_seed, construct a fresh TSScheduler instance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import exp
from typing import Optional

import numpy as np

from .overshoot_lut import build_psi_lut, quantize_overshoot
from .rec_pending import RecMode, RecPendingBuffer, make_rec_pending


# ---------------------------------------------------------------------------
# Configuration

@dataclass
class TSConfig:
    n_in:             int
    n_rec:            int
    n_out:            int
    thr:              float   = 1.0
    v_reset:          float   = 0.0
    tau_m:            float   = 20.0
    tau_trace:        float   = 20.0
    dt_ts:            float   = 1.0
    gamma:            float   = 0.3
    overshoot_n_bits: int     = 3
    overshoot_max:    float   = 0.5
    eta:              float   = 1e-3
    rec_mode:         RecMode = RecMode.IDEAL_FIFO
    rec_mode_param:   int     = 0
    arbiter_seed:     int     = 0
    max_drain_steps:  int     = 10_000


# ---------------------------------------------------------------------------
# Per-ts statistics

@dataclass
class TSStats:
    """Statistics collected during one software timestep."""
    max_rec_pending_total:      int       = 0
    max_rec_pending_per_neuron: int       = 0
    rec_events_generated:       int       = 0   # total push() attempts
    rec_events_serviced:        int       = 0   # total service() calls (accepted pulls)
    overflow_count:             int       = 0   # rejected push() calls
    saturated_counter_count:    int       = 0   # push() rejections due to counter max
    drain_steps:                int       = 0   # arbiter iterations this ts
    drain_exhausted:            bool      = False  # True if max_drain_steps was hit
    drain_exhausted_pending:    int       = 0   # events dropped when drain was cut short
    event_order: list = field(default_factory=list)   # [("in"|"rec", src), ...]
    post_spikes: list = field(default_factory=list)   # list[int] neuron indices that fired
    psi_bins:    list = field(default_factory=list)   # list[int] overshoot bin per spike (parallel to post_spikes)
    psi_vals:    list = field(default_factory=list)   # list[float] psi surrogate value per spike


# ---------------------------------------------------------------------------
# Run result

@dataclass
class TSRunResult:
    stats_per_ts: list[TSStats]
    output_trace: Optional[np.ndarray]  # (T, n_out) or None
    dw_in:        np.ndarray            # (n_rec, n_in)  accumulated gradients
    dw_rec:       np.ndarray            # (n_rec, n_rec) accumulated gradients


# ---------------------------------------------------------------------------
# Scheduler

class TSScheduler:
    """
    Software-timestep barrier SNN scheduler.

    Usage
    -----
    sched = TSScheduler(cfg, w_in, w_rec)
    result = sched.run(raster, L_signal=L, w_out=w_out)

    Per-sample, call reset_state() to clear membrane + trace + gradient state.
    The RNG is NOT reset; construct a new TSScheduler to replay from the same seed.
    """

    def __init__(
        self,
        cfg:   TSConfig,
        w_in:  np.ndarray,   # (n_rec, n_in)
        w_rec: np.ndarray,   # (n_rec, n_rec)
    ) -> None:
        self.cfg  = cfg
        self.w_in  = np.array(w_in,  dtype=np.float64)
        self.w_rec = np.array(w_rec, dtype=np.float64)

        self._rng = np.random.default_rng(cfg.arbiter_seed)

        # Barrier-mode state: no intra-ts decay; bulk decay at end of each ts
        self._v     = np.zeros(cfg.n_rec, dtype=np.float64)
        self._p_in  = np.zeros(cfg.n_in,  dtype=np.float64)
        self._p_rec = np.zeros(cfg.n_rec, dtype=np.float64)

        # Gradient accumulators
        self._dw_in  = np.zeros((cfg.n_rec, cfg.n_in),  dtype=np.float64)
        self._dw_rec = np.zeros((cfg.n_rec, cfg.n_rec), dtype=np.float64)

        self._rec_pending: RecPendingBuffer = make_rec_pending(
            cfg.rec_mode, cfg.n_rec, cfg.rec_mode_param
        )

        self._psi_lut = build_psi_lut(
            cfg.overshoot_n_bits, cfg.gamma, cfg.thr, cfg.overshoot_max
        )

        # Pre-computed end-of-ts decay factors
        self._decay_m     = exp(-cfg.dt_ts / cfg.tau_m)
        self._decay_trace = exp(-cfg.dt_ts / cfg.tau_trace)

    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """
        Reset membrane voltages, traces, and gradient accumulators.

        Does NOT re-seed the RNG. To reproduce the same event sequence, construct
        a new TSScheduler with the same arbiter_seed.
        """
        self._v[:]      = 0.0
        self._p_in[:]   = 0.0
        self._p_rec[:]  = 0.0
        self._dw_in[:]  = 0.0
        self._dw_rec[:] = 0.0

    # ------------------------------------------------------------------

    def step(
        self,
        input_row: np.ndarray,           # (n_in,) bool — active input channels this ts
        L: Optional[np.ndarray] = None,  # (n_rec,) learning signal; None = no gradient
    ) -> TSStats:
        """
        Process one software timestep.

        Drain loop
        ----------
        1. Load active input channels into input_pending (bitmap; one spike per channel).
        2. Build candidates: input_pending ∪ rec_pending.active_neurons().
        3. Uniform-random pick one candidate.
           - Input (i):   p_in[i]  += 1; inject w_in[:,i]  into membrane; compare+reset.
           - Rec    (j):  service(j); p_rec[j] += 1; inject w_rec[:,j]; compare+reset.
        4. Each spiking neuron k: push rec_pending(k); if L: accumulate gradient.
        5. Repeat until candidates empty or max_drain_steps exhausted.

        End-of-ts barrier
        -----------------
        v *= decay_m; p_in *= decay_trace; p_rec *= decay_trace.

        Returns
        -------
        TSStats for this timestep (including event_order and post_spikes).
        """
        cfg = self.cfg
        rec = self._rec_pending
        rec.clear()

        input_pending: set[int] = {int(i) for i in np.where(input_row)[0]}

        stats    = TSStats()
        rec_gen  = 0
        rec_svc  = 0

        for _ in range(cfg.max_drain_steps):
            active_in  = list(input_pending)
            active_rec = rec.active_neurons()
            if not active_in and not active_rec:
                break

            candidates = [("in",  i) for i in active_in] + \
                         [("rec", j) for j in active_rec]

            idx       = int(self._rng.integers(len(candidates)))
            kind, src = candidates[idx]
            stats.event_order.append((kind, src))
            stats.drain_steps += 1

            if kind == "in":
                input_pending.discard(src)
                self._p_in[src] += 1.0
                w_col = self.w_in[:, src]
            else:
                rec.service(src)
                rec_svc += 1
                self._p_rec[src] += 1.0
                w_col = self.w_rec[:, src]

            # LIF: inject + compare + reset (no intra-ts decay)
            self._v += w_col
            spiking_idx = np.where(self._v > cfg.thr)[0]

            if len(spiking_idx) > 0:
                overshoot_arr = self._v[spiking_idx] - cfg.thr
                overshoot_bins = quantize_overshoot(
                    overshoot_arr, cfg.overshoot_n_bits, cfg.overshoot_max
                )
                self._v[spiking_idx] = cfg.v_reset  # reset all spiking neurons at once

                for k in range(len(spiking_idx)):
                    j       = int(spiking_idx[k])
                    psi_bin = int(overshoot_bins[k])
                    psi_val = float(self._psi_lut[psi_bin])

                    accepted, saturated = rec.push(j)
                    rec_gen += 1

                    # Track peak occupancy after each push
                    rt = rec.total()
                    if rt > stats.max_rec_pending_total:
                        stats.max_rec_pending_total = rt
                    rn = rec.per_neuron_max()
                    if rn > stats.max_rec_pending_per_neuron:
                        stats.max_rec_pending_per_neuron = rn

                    stats.post_spikes.append(j)
                    stats.psi_bins.append(psi_bin)
                    stats.psi_vals.append(psi_val)

                    if L is not None:
                        scale = cfg.eta * float(L[j]) * psi_val
                        if scale != 0.0:
                            self._dw_in[j, :]  += scale * self._p_in
                            self._dw_rec[j, :] += scale * self._p_rec

        stats.rec_events_generated    = rec_gen
        stats.rec_events_serviced     = rec_svc
        stats.overflow_count          = rec.overflow_count
        stats.saturated_counter_count = rec.saturated_count

        # Detect drain exhaustion: events still pending when max_drain_steps ran out.
        # rec.clear() at the start of the next ts will silently discard them, so we
        # must surface this here before the barrier decay overwrites any context.
        remaining = rec.total() + len(input_pending)
        if remaining > 0:
            stats.drain_exhausted         = True
            stats.drain_exhausted_pending = remaining
            warnings.warn(
                f"TSScheduler.step: max_drain_steps={cfg.max_drain_steps} exhausted; "
                f"{remaining} pending event(s) will be dropped at ts boundary. "
                "Increase max_drain_steps or reduce recurrent cascade depth.",
                RuntimeWarning,
                stacklevel=2,
            )

        # End-of-ts barrier: bulk decay
        self._v     *= self._decay_m
        self._p_in  *= self._decay_trace
        self._p_rec *= self._decay_trace

        return stats

    # ------------------------------------------------------------------

    def run(
        self,
        raster:   np.ndarray,                    # (T, n_in) bool
        L_signal: Optional[np.ndarray] = None,   # (n_rec,) static learning signal
        w_out:    Optional[np.ndarray] = None,   # (n_out, n_rec) for LI output
        li_leak:  float = 1.0,
    ) -> TSRunResult:
        """
        Run all timesteps over a raster.

        L_signal: (n_rec,) static learning signal applied every ts. None = no gradient.
        w_out:    (n_out, n_rec); if provided, accumulates LI output_trace (n_ts, n_out).
        li_leak:  inter-ts LI output decay (1.0 = no decay).

        Returns TSRunResult with per-ts statistics, output_trace, and dw_in/dw_rec.
        """
        T     = raster.shape[0]
        n_out = w_out.shape[0] if w_out is not None else 0

        stats_per_ts: list[TSStats]       = []
        output_v                           = np.zeros(n_out, dtype=np.float64)
        output_trace: Optional[np.ndarray] = (
            np.zeros((T, n_out), dtype=np.float64) if w_out is not None else None
        )

        for ts in range(T):
            ts_stats = self.step(raster[ts], L_signal)
            stats_per_ts.append(ts_stats)

            if w_out is not None:
                for j in ts_stats.post_spikes:
                    output_v += w_out[:, j]
                output_trace[ts] = output_v.copy()  # type: ignore[index]
                output_v *= li_leak

        return TSRunResult(
            stats_per_ts=stats_per_ts,
            output_trace=output_trace,
            dw_in=self._dw_in.copy(),
            dw_rec=self._dw_rec.copy(),
        )
