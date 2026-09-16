"""
Hardware-matched event-driven SNN simulator (D-0006, D-0007).

Separate from eprop/model/models.py. Does not share state or imports with the
synchronous training path.

Event kinds and their handlers
-------------------------------
INPUT_SPIKE(src)  -> FPGA input-trace update for src
                  -> CNM fan-out via w_in[:, src]
                  -> for each spiking neuron j:
                       snapshot traces at spike_t  (causal: before future mutations)
                       emit REC_SPIKE(j) at t + dt_hw     (causal: 1 micro-step delay)
                       emit FPGA_UPDATE(j) at stall.fpga_update_time(t)

REC_SPIKE(src)    -> FPGA rec-trace update for src
                  -> CNM fan-out via w_rec[:, src]
                  -> same post-spike / snapshot handling as INPUT_SPIKE

FPGA_UPDATE(j)    -> gradient accumulation using psi_j, L_j, and trace snapshots
                     taken at spike-time (not fpga_update_time)

Recurrent causality
-------------------
REC_SPIKE is always scheduled at t + dt_hw. This prevents same-timestamp
recurrent feedback from creating an infinite loop and matches the hardware
requirement that recurrent fan-out takes at least one micro-step.

Stall modes
-----------
DISABLED     : no stall; FPGA_UPDATE at spike_t; compare always on.
SH_BACKGROUND: FPGA_UPDATE delayed by quant_delay; CNM and compare unaffected.
               Models S&H + background quantization (preferred HW direction).
GLOBAL_STALL : FPGA_UPDATE delayed; any event arriving while t < _cnm_stall_until
               is re-queued at _cnm_stall_until without injecting into the CNM.
               Models a global ADC stall that freezes all CNM event processing.

Trace causality under SH_BACKGROUND
-------------------------------------
Trace snapshots (p_in_snap, p_rec_snap) are taken at spike_t, immediately after
the trace update for the triggering presynaptic spike. They are stored in the
FPGA_UPDATE payload so that future spike events occurring between spike_t and
fpga_update_t cannot retroactively mutate the gradient computation.

Pending FPGA_UPDATE flush
--------------------------
run(t_max) flushes FPGA_UPDATE events with t <= t_max + quant_delay. Non-FPGA
events (REC_SPIKE, INPUT_SPIKE) in that range are re-queued for the next window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .event_queue import Event, EventKind, EventQueue
from .fpga_trace import FPGATrace, WeightQuant
from .lif_cnm import LIFArray
from .overshoot_lut import build_psi_lut
from .stall_model import StallMode, StallModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration

@dataclass
class SimConfig:
    n_in:             int
    n_rec:            int
    n_out:            int
    thr:              float     = 1.0
    v_reset:          float     = 0.0
    tau_m:            float     = 20.0    # ms
    tau_trace:        float     = 20.0    # ms
    dt_hw:            float     = 1.0     # causal recurrent-spike delay (micro-steps)
    gamma:            float     = 0.3     # surrogate gradient scale
    overshoot_n_bits: int       = 5       # 2^5 = 32 bins; hardware counter width
    overshoot_max:    float     = 0.5     # ADC full-scale overshoot range
    eta:              float     = 1e-3
    quant_delay:      float     = 0.0     # stall / S&H delay in micro-steps
    stall_mode:       StallMode = StallMode.DISABLED


# ---------------------------------------------------------------------------
# Per-event record

@dataclass
class EventRecord:
    t:         float
    kind:      str
    src:       int
    overshoot: Optional[float] = None
    psi_bin:   Optional[int]   = None
    psi:       Optional[float] = None
    L_j:       Optional[float] = None


# ---------------------------------------------------------------------------
# Top-level simulator

class HWSim:
    """
    Hardware-matched event-driven SNN simulator.

    Usage
    -----
    sim = HWSim(cfg, w_in, w_rec, w_out)
    sim.load_input_events([(t0, src0), (t1, src1), ...])
    records = sim.run(t_max, learning=True)
    """

    def __init__(
        self,
        cfg: SimConfig,
        w_in:  np.ndarray,   # (n_rec, n_in)
        w_rec: np.ndarray,   # (n_rec, n_rec)
        w_out: np.ndarray,   # (n_out, n_rec)
        learning_signal_fn: Optional[Callable[[int, float], float]] = None,
    ) -> None:
        self.cfg = cfg
        self.w_out = w_out.copy().astype(np.float64)

        psi_lut = build_psi_lut(
            cfg.overshoot_n_bits, cfg.gamma, cfg.thr, cfg.overshoot_max
        )
        self.lif = LIFArray(
            n_rec=cfg.n_rec,
            thr=cfg.thr,
            v_reset=cfg.v_reset,
            tau_m=cfg.tau_m,
            overshoot_n_bits=cfg.overshoot_n_bits,
            overshoot_max=cfg.overshoot_max,
            psi_lut=psi_lut,
        )
        self.fpga = FPGATrace(
            n_in=cfg.n_in,
            n_rec=cfg.n_rec,
            tau_trace=cfg.tau_trace,
            w_in=w_in,
            w_rec=w_rec,
        )
        self.stall = StallModel(cfg.quant_delay, cfg.stall_mode)
        self.queue = EventQueue()
        self.records: list[EventRecord] = []

        self._L_fn: Callable[[int, float], float] = (
            learning_signal_fn if learning_signal_fn is not None
            else lambda j, t: 0.0
        )

        # Tracks when global stall expires (GLOBAL_STALL mode only).
        self._cnm_stall_until: float = 0.0

    # ------------------------------------------------------------------
    # public API

    def load_input_events(self, spike_times: list[tuple[float, int]]) -> None:
        """Schedule input spike events. spike_times: list of (t, input_neuron_index)."""
        for t, src in spike_times:
            self.queue.push(t, EventKind.INPUT_SPIKE, src)

    def run(self, t_max: float, learning: bool = False) -> list[EventRecord]:
        """
        Process queued events up to t_max.

        After the main loop, FPGA_UPDATE events with t <= t_max + quant_delay are
        flushed. Non-FPGA events in that range are re-queued for the next window.
        """
        self.records = []

        while self.queue:
            if self.queue.peek_t() > t_max:  # type: ignore[operator]
                break
            ev = self.queue.pop()
            if ev.kind == EventKind.INPUT_SPIKE:
                self._handle_spike(
                    ev,
                    w_col_fn=lambda s: self.fpga.w_in[:, s],
                    trace_fn=self.fpga.on_input_spike,
                    kind_label="input_spike",
                    learning=learning,
                )
            elif ev.kind == EventKind.REC_SPIKE:
                self._handle_spike(
                    ev,
                    w_col_fn=lambda s: self.fpga.w_rec[:, s],
                    trace_fn=self.fpga.on_rec_spike,
                    kind_label="rec_spike",
                    learning=learning,
                )
            elif ev.kind == EventKind.FPGA_UPDATE:
                self._handle_fpga_update(ev, learning)

        # Flush FPGA_UPDATE events delayed past t_max by quant_delay.
        # Non-FPGA events in the flush window are re-queued for the next run() call.
        flush_until = t_max + self.cfg.quant_delay
        deferred: list[Event] = []
        while self.queue:
            if self.queue.peek_t() > flush_until:  # type: ignore[operator]
                break
            ev = self.queue.pop()
            if ev.kind == EventKind.FPGA_UPDATE:
                self._handle_fpga_update(ev, learning)
            else:
                deferred.append(ev)

        for ev in deferred:
            self.queue.push(ev.t, ev.kind, ev.src, ev.payload)

        return self.records

    # ------------------------------------------------------------------
    # internal handlers

    def _handle_spike(
        self,
        ev: Event,
        w_col_fn: Callable[[int], np.ndarray],
        trace_fn: Callable[[float, int], None],
        kind_label: str,
        learning: bool,
    ) -> None:
        t = ev.t
        src = ev.src

        # GLOBAL_STALL: freeze CNM by rescheduling the event to after the stall.
        # The event is not consumed — it is re-queued at the stall expiry time.
        # This preserves neuron dynamics (no injection or leak during stall).
        if self.stall.mode == StallMode.GLOBAL_STALL and t < self._cnm_stall_until:
            self.queue.push(self._cnm_stall_until, ev.kind, ev.src, ev.payload)
            return

        if learning:
            trace_fn(t, src)

        w_col = w_col_fn(src)
        spike_mask, overshoot, psi_bins = self.lif.inject(t, w_col)

        self.records.append(EventRecord(t=t, kind=kind_label, src=src))

        # Snapshot traces immediately after the triggering spike's trace update,
        # before any future event can mutate p_in / p_rec.
        if learning:
            p_in_snap, p_rec_snap = self.fpga.snapshot_traces(t)
        else:
            p_in_snap = p_rec_snap = None

        spiking = np.where(spike_mask)[0]
        for j in spiking:
            psi_val = float(self.lif.psi_lut[psi_bins[j]])
            L_j = self._L_fn(int(j), t) if learning else 0.0
            fpga_t = self.stall.fpga_update_time(t)

            self.records.append(EventRecord(
                t=t, kind="post_spike", src=int(j),
                overshoot=float(overshoot[j]),
                psi_bin=int(psi_bins[j]),
                psi=psi_val,
                L_j=L_j,
            ))

            # Recurrent fan-out: causal 1-micro-step delay prevents same-t loops.
            self.queue.push(t + self.cfg.dt_hw, EventKind.REC_SPIKE, int(j))

            # FPGA_UPDATE carries trace snapshots taken at spike_t.
            self.queue.push(fpga_t, EventKind.FPGA_UPDATE, int(j), payload={
                "psi": psi_val, "L_j": L_j,
                "p_in_snap": p_in_snap, "p_rec_snap": p_rec_snap,
            })

            if self.stall.mode == StallMode.GLOBAL_STALL:
                self._cnm_stall_until = max(
                    self._cnm_stall_until, t + self.cfg.quant_delay
                )

    def _handle_fpga_update(self, ev: Event, learning: bool) -> None:
        if not learning:
            return
        j = ev.src
        t = ev.t
        payload = ev.payload
        self.fpga.on_post_spike(
            j, payload["psi"], payload["L_j"], self.cfg.eta,
            payload["p_in_snap"], payload["p_rec_snap"],
        )
        self.records.append(EventRecord(
            t=t, kind="fpga_update", src=j,
            psi=payload["psi"], L_j=payload["L_j"],
        ))

    # ------------------------------------------------------------------
    # utilities

    def reset(self) -> None:
        self.lif.reset()
        self.fpga.reset()
        self.queue = EventQueue()
        self._cnm_stall_until = 0.0
        self.records = []

    def spike_records(self) -> list[EventRecord]:
        return [r for r in self.records if r.kind == "post_spike"]

    def input_spike_count(self) -> int:
        return sum(1 for r in self.records if r.kind == "input_spike")
