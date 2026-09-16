"""
Training/inference API wrappers over the two hardware simulators.

Two backends
------------
EventHWNet  (also exported as HWNet for backward compat):
    Event-driven control path backed by HWSim. Per-event continuous-time leak
    and trace decay. NOT hardware-golden in the software-ts barrier sense.
    Use for algorithm comparison against TSHWNet.

TSHWNet (hardware-golden):
    Software-timestep barrier path backed by TSScheduler. Bulk end-of-ts
    leak/decay; seeded randomized AER arbitration; configurable rec_pending
    mode. This is the primary hardware-accurate training API.

Both classes expose the same public API: infer(), train_sample().

Two-pass training (shared design)
----------------------------------
Pass 1 (forward, no gradient):
    Run simulation to get output_trace. Compute average-trial logits.
Pass 2 (backward, learning=True):
    Reset and re-run with identical weights and same arbiter seed (TSHWNet)
    or fresh simulator (EventHWNet), so spiking is identical to pass 1.
    ASSUMPTION: weights unchanged between passes guarantees identical spikes.

L_j computation (FPGA side)
----------------------------
    output_error = softmax(logits) - one_hot(target_class)
    L_j = (w_out.T @ output_error)[j]

Initial weight quantization
-----------------------------
If weight_quant is provided with n_bits > 0, weights are quantized to the
hardware grid at construction (float_init=False, the default). Pass
float_init=True to defer quantization for float-baseline comparison.

w_out handling
--------------
w_out is quantized at construction along with w_in/w_rec (when float_init=False).
w_out is NOT updated by train_sample() — e-prop gradient is only for w_in/w_rec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fpga_trace import FPGATrace, WeightQuant, _quantize_weights
from .input_encoder import raster_to_events
from .li_readout import compute_output_trace
from .rec_pending import RecMode
from .simulator import HWSim, SimConfig, EventRecord
from .stall_model import StallMode
from .ts_scheduler import TSConfig, TSRunResult, TSScheduler


# ---------------------------------------------------------------------------
# Result types

@dataclass
class InferenceResult:
    logits: np.ndarray        # (n_out,) mean over output_trace
    pred: int                 # argmax(logits)
    output_trace: np.ndarray  # (n_ts, n_out) per-software-timestep LI readout
    spike_records: list       # list[EventRecord] (kind == "post_spike")


@dataclass
class TrainResult:
    loss: float               # softmax cross-entropy
    logits: np.ndarray        # (n_out,) from forward pass
    pred: int                 # argmax(logits)
    dw_in_norm: float         # Frobenius norm of dw_in BEFORE apply_gradients
    dw_rec_norm: float        # Frobenius norm of dw_rec BEFORE apply_gradients
    spike_records: list       # list[EventRecord] (kind == "post_spike")
    drain_exhausted: bool = False  # True if any ts hit max_drain_steps (TSHWNet only)


# ---------------------------------------------------------------------------
# Loss utility (shared by both backends)

def _softmax_cross_entropy(
    logits: np.ndarray, target: int
) -> tuple[np.ndarray, float]:
    """Return (log_softmax, cross_entropy_loss). Numerically stable via max-shift."""
    shifted = logits - logits.max()
    log_sum = np.log(np.exp(shifted).sum())
    log_sm = shifted - log_sum
    return log_sm, -float(log_sm[target])


# ---------------------------------------------------------------------------
# Helper: convert TSRunResult post_spikes to EventRecord list

def _ts_spike_records(stats_per_ts: list) -> list[EventRecord]:
    """
    Convert TSRunResult.stats_per_ts[*].post_spikes into EventRecord-compatible list.
    t is set to the software timestep index (float).
    """
    records: list[EventRecord] = []
    for ts, s in enumerate(stats_per_ts):
        for j in s.post_spikes:
            records.append(EventRecord(t=float(ts), kind="post_spike", src=j))
    return records


# ---------------------------------------------------------------------------
# EventHWNet — event-driven control path (HWSim backend)

class EventHWNet:
    """
    Event-driven training/inference wrapper backed by HWSim.

    Continuous-time per-event leak and trace decay; static event queue ordering.
    This is the CONTROL / LEGACY path, NOT hardware-golden in the barrier sense.
    For the barrier-accurate hardware path, use TSHWNet.

    Parameters
    ----------
    cfg : SimConfig
    w_in  : (n_rec, n_in)  initial weights
    w_rec : (n_rec, n_rec) initial weights (diagonal must be 0)
    w_out : (n_out, n_rec) initial weights (not updated during training)
    weight_quant : WeightQuant or None — applied after each gradient step
    float_init : if False (default), quantize w_in/w_rec to hardware grid at
                 construction. Pass True for a float-precision baseline.
    li_leak : inter-timestep LI decay (1.0 = no leak, default)
    rng : numpy Generator for stochastic weight rounding
    """

    def __init__(
        self,
        cfg: SimConfig,
        w_in: np.ndarray,
        w_rec: np.ndarray,
        w_out: np.ndarray,
        weight_quant: Optional[WeightQuant] = None,
        li_leak: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        float_init: bool = False,
    ) -> None:
        self.cfg = cfg
        self.w_in  = np.array(w_in,  dtype=np.float64)
        self.w_rec = np.array(w_rec, dtype=np.float64)
        self.w_out = np.array(w_out, dtype=np.float64)
        self.weight_quant = weight_quant
        self.li_leak = li_leak
        self.rng = rng

        if not float_init and weight_quant is not None and weight_quant.n_bits > 0:
            self.w_in  = _quantize_weights(self.w_in,  weight_quant, rng)
            self.w_rec = _quantize_weights(self.w_rec, weight_quant, rng)
            self.w_out = _quantize_weights(self.w_out, weight_quant, rng)

    # ------------------------------------------------------------------

    def _make_sim(self, learning_signal: Optional[np.ndarray] = None) -> HWSim:
        L_fn = None
        if learning_signal is not None:
            L = learning_signal
            L_fn = lambda j, t: float(L[j])
        return HWSim(self.cfg, self.w_in, self.w_rec, self.w_out, learning_signal_fn=L_fn)

    def _run_forward(
        self, raster: np.ndarray, dt_per_ts: float
    ) -> tuple[list, np.ndarray]:
        T = raster.shape[0]
        t_max = float(T) * dt_per_ts
        sim = self._make_sim()
        sim.load_input_events(raster_to_events(raster, dt_per_ts))
        sim.run(t_max, learning=False)
        output_trace = compute_output_trace(
            sim.records, self.w_out, T, dt_per_ts, self.li_leak
        )
        return sim.spike_records(), output_trace

    def _run_learning(
        self,
        raster: np.ndarray,
        dt_per_ts: float,
        learning_signal: np.ndarray,
    ) -> tuple[list, float, float]:
        T = raster.shape[0]
        t_max = float(T) * dt_per_ts
        sim = self._make_sim(learning_signal)
        sim.load_input_events(raster_to_events(raster, dt_per_ts))
        sim.run(t_max, learning=True)

        dw_in_norm  = float(np.linalg.norm(sim.fpga.dw_in))
        dw_rec_norm = float(np.linalg.norm(sim.fpga.dw_rec))

        sim.fpga.apply_gradients(quant=self.weight_quant, rng=self.rng)
        self.w_in  = sim.fpga.w_in.copy()
        self.w_rec = sim.fpga.w_rec.copy()

        return sim.spike_records(), dw_in_norm, dw_rec_norm

    # ------------------------------------------------------------------

    def infer(self, raster: np.ndarray, dt_per_ts: float = 1.0) -> InferenceResult:
        raster = np.asarray(raster, dtype=bool)
        if raster.ndim != 2:
            raise ValueError(f"raster must be [T, n_in], got shape {raster.shape}")
        spike_records, output_trace = self._run_forward(raster, dt_per_ts)
        logits = output_trace.mean(axis=0)
        return InferenceResult(
            logits=logits,
            pred=int(np.argmax(logits)),
            output_trace=output_trace,
            spike_records=spike_records,
        )

    def train_sample(
        self,
        raster: np.ndarray,
        target_class: int,
        dt_per_ts: float = 1.0,
    ) -> TrainResult:
        raster = np.asarray(raster, dtype=bool)
        if raster.ndim != 2:
            raise ValueError(f"raster must be [T, n_in], got shape {raster.shape}")

        _spike_records_fwd, output_trace = self._run_forward(raster, dt_per_ts)
        logits = output_trace.mean(axis=0)

        log_sm, loss = _softmax_cross_entropy(logits, target_class)
        output_error = np.exp(log_sm)
        output_error[target_class] -= 1.0
        L = self.w_out.T @ output_error

        spike_records_bwd, dw_in_norm, dw_rec_norm = self._run_learning(
            raster, dt_per_ts, L
        )

        return TrainResult(
            loss=float(loss),
            logits=logits,
            pred=int(np.argmax(logits)),
            dw_in_norm=dw_in_norm,
            dw_rec_norm=dw_rec_norm,
            spike_records=spike_records_bwd,
        )


# Backward-compat alias. New code should use EventHWNet (control) or TSHWNet (hardware-golden).
HWNet = EventHWNet


# ---------------------------------------------------------------------------
# EventHWNet config/factory

def default_config(
    n_in: int = 200,
    n_rec: int = 128,
    n_out: int = 10,
    weight_bits: int = 4,
) -> tuple[SimConfig, WeightQuant]:
    """
    Return (SimConfig, WeightQuant) for the EventHWNet (HWSim) control path.

    For the TSHWNet hardware-golden path, use default_ts_config().
    """
    cfg = SimConfig(
        n_in=n_in,
        n_rec=n_rec,
        n_out=n_out,
        thr=1.0,
        v_reset=0.0,
        tau_m=20.0,
        tau_trace=20.0,
        dt_hw=1.0,
        gamma=0.3,
        overshoot_n_bits=3,
        overshoot_max=0.5,
        eta=1e-3,
        quant_delay=3.0,
        stall_mode=StallMode.SH_BACKGROUND,
    )
    quant = WeightQuant(n_bits=weight_bits, weight_max=1.0, stochastic_rounding=False)
    return cfg, quant


def make_default_hwnet(
    n_in: int = 200,
    n_rec: int = 128,
    n_out: int = 10,
    weight_bits: int = 4,
    seed: int = 42,
    float_init: bool = False,
) -> EventHWNet:
    """EventHWNet with default HWSim config and Xavier-scaled random weights."""
    cfg, quant = default_config(n_in=n_in, n_rec=n_rec, n_out=n_out, weight_bits=weight_bits)
    rng = np.random.default_rng(seed)
    w_in  = rng.normal(0, 2.0 / np.sqrt(n_in),  (n_rec, n_in))
    w_rec = rng.normal(0, 0.5 / np.sqrt(n_rec), (n_rec, n_rec))
    np.fill_diagonal(w_rec, 0.0)
    w_out = rng.normal(0, 1.0 / np.sqrt(n_rec), (n_out, n_rec))
    return EventHWNet(cfg, w_in, w_rec, w_out, weight_quant=quant, rng=rng,
                      float_init=float_init)


# ---------------------------------------------------------------------------
# TSHWNet — hardware-golden path (TSScheduler backend)

class TSHWNet:
    """
    Hardware-golden training/inference wrapper backed by TSScheduler.

    Software-timestep barrier; seeded randomized AER arbitration; configurable
    rec_pending mode; end-of-ts bulk leak/decay. This is the primary hardware-
    accurate path. Use EventHWNet for the continuous-time event-driven baseline.

    Two-pass training
    -----------------
    Both passes construct a fresh TSScheduler from the same cfg.arbiter_seed,
    so the RNG sequence is identical and spiking is identical across passes
    (given unchanged weights). Gradient correctness requires this.

    Parameters
    ----------
    cfg : TSConfig — includes rec_mode, arbiter_seed, dt_ts, etc.
    w_in  : (n_rec, n_in)
    w_rec : (n_rec, n_rec)  diagonal must be 0
    w_out : (n_out, n_rec)  not updated during training
    weight_quant : WeightQuant or None
    float_init : if False (default), quantize w_in/w_rec/w_out at construction.
    li_leak : inter-ts LI output decay (1.0 = no decay)
    rng : numpy Generator for stochastic weight rounding
    """

    def __init__(
        self,
        cfg: TSConfig,
        w_in: np.ndarray,
        w_rec: np.ndarray,
        w_out: np.ndarray,
        weight_quant: Optional[WeightQuant] = None,
        li_leak: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        float_init: bool = False,
    ) -> None:
        self.cfg = cfg
        self.w_in  = np.array(w_in,  dtype=np.float64)
        self.w_rec = np.array(w_rec, dtype=np.float64)
        self.w_out = np.array(w_out, dtype=np.float64)
        self.weight_quant = weight_quant
        self.li_leak = li_leak
        self.rng = rng

        if not float_init and weight_quant is not None and weight_quant.n_bits > 0:
            self.w_in  = _quantize_weights(self.w_in,  weight_quant, rng)
            self.w_rec = _quantize_weights(self.w_rec, weight_quant, rng)
            self.w_out = _quantize_weights(self.w_out, weight_quant, rng)

    # ------------------------------------------------------------------

    def _make_sched(self) -> TSScheduler:
        """Fresh TSScheduler from the same seed — ensures reproducible arbitration."""
        return TSScheduler(self.cfg, self.w_in, self.w_rec)

    def _apply_gradients(self, dw_in: np.ndarray, dw_rec: np.ndarray) -> None:
        self.w_in  -= dw_in
        self.w_rec -= dw_rec
        if self.weight_quant is not None and self.weight_quant.n_bits > 0:
            self.w_in  = _quantize_weights(self.w_in,  self.weight_quant, self.rng)
            self.w_rec = _quantize_weights(self.w_rec, self.weight_quant, self.rng)

    # ------------------------------------------------------------------

    def infer(self, raster: np.ndarray) -> InferenceResult:
        """
        Forward pass on raster [T, n_in].

        Returns InferenceResult with logits, pred, output_trace, spike_records.
        """
        raster = np.asarray(raster, dtype=bool)
        if raster.ndim != 2:
            raise ValueError(f"raster must be [T, n_in], got shape {raster.shape}")

        result = self._make_sched().run(raster, w_out=self.w_out, li_leak=self.li_leak)
        logits = result.output_trace.mean(axis=0)
        return InferenceResult(
            logits=logits,
            pred=int(np.argmax(logits)),
            output_trace=result.output_trace,
            spike_records=_ts_spike_records(result.stats_per_ts),
        )

    def train_sample(
        self,
        raster: np.ndarray,
        target_class: int,
    ) -> TrainResult:
        """
        Two-pass e-prop weight update for one sample.

        Pass 1 (forward, no gradient): compute average-trial logits and output error.
        Pass 2 (backward, L_signal set): accumulate dw_in, dw_rec; apply + quantize.

        Both passes use fresh TSScheduler(cfg) instances with the same arbiter_seed,
        guaranteeing identical spiking patterns.
        """
        raster = np.asarray(raster, dtype=bool)
        if raster.ndim != 2:
            raise ValueError(f"raster must be [T, n_in], got shape {raster.shape}")

        # Pass 1: forward
        fwd = self._make_sched().run(raster, w_out=self.w_out, li_leak=self.li_leak)
        logits = fwd.output_trace.mean(axis=0)

        log_sm, loss = _softmax_cross_entropy(logits, target_class)
        output_error = np.exp(log_sm)
        output_error[target_class] -= 1.0
        L = self.w_out.T @ output_error

        # Pass 2: backward — same seed → same spikes, same traces → correct gradient
        bwd = self._make_sched().run(
            raster, L_signal=L, w_out=self.w_out, li_leak=self.li_leak
        )

        dw_in_norm  = float(np.linalg.norm(bwd.dw_in))
        dw_rec_norm = float(np.linalg.norm(bwd.dw_rec))

        drain_exhausted = any(s.drain_exhausted for s in fwd.stats_per_ts) or \
                          any(s.drain_exhausted for s in bwd.stats_per_ts)

        self._apply_gradients(bwd.dw_in, bwd.dw_rec)

        return TrainResult(
            loss=float(loss),
            logits=logits,
            pred=int(np.argmax(logits)),
            dw_in_norm=dw_in_norm,
            dw_rec_norm=dw_rec_norm,
            spike_records=_ts_spike_records(bwd.stats_per_ts),
            drain_exhausted=drain_exhausted,
        )


# ---------------------------------------------------------------------------
# TSHWNet config/factory

def default_ts_config(
    n_in: int = 200,
    n_rec: int = 128,
    n_out: int = 10,
    weight_bits: int = 4,
    arbiter_seed: int = 0,
    rec_mode: RecMode = RecMode.IDEAL_MULTISET,
    rec_mode_param: int = 0,
) -> tuple[TSConfig, WeightQuant]:
    """
    Return (TSConfig, WeightQuant) for the TSHWNet hardware-golden path.

    Hardware parameters match the EventHWNet default:
        overshoot_n_bits=3, dt_ts=1.0, tau_m=20.0, tau_trace=20.0
    TSScheduler-specific:
        rec_mode=IDEAL_MULTISET (algorithm baseline; swap to PER_NEURON_COUNTER for HW)
        arbiter_seed=0
        max_drain_steps=50_000
    """
    cfg = TSConfig(
        n_in=n_in,
        n_rec=n_rec,
        n_out=n_out,
        thr=1.0,
        v_reset=0.0,
        tau_m=20.0,
        tau_trace=20.0,
        dt_ts=1.0,
        gamma=0.3,
        overshoot_n_bits=3,
        overshoot_max=0.5,
        eta=1e-3,
        rec_mode=rec_mode,
        rec_mode_param=rec_mode_param,
        arbiter_seed=arbiter_seed,
        max_drain_steps=50_000,
    )
    quant = WeightQuant(n_bits=weight_bits, weight_max=1.0, stochastic_rounding=False)
    return cfg, quant


def make_default_ts_hwnet(
    n_in: int = 200,
    n_rec: int = 128,
    n_out: int = 10,
    weight_bits: int = 4,
    seed: int = 42,
    arbiter_seed: int = 0,
    float_init: bool = False,
) -> TSHWNet:
    """
    TSHWNet with default hardware config and Xavier-scaled random weights.

    Weight scales match make_default_hwnet. Initial weights are quantized to
    the hardware grid unless float_init=True.
    """
    cfg, quant = default_ts_config(
        n_in=n_in, n_rec=n_rec, n_out=n_out, weight_bits=weight_bits,
        arbiter_seed=arbiter_seed,
    )
    rng = np.random.default_rng(seed)
    w_in  = rng.normal(0, 2.0 / np.sqrt(n_in),  (n_rec, n_in))
    w_rec = rng.normal(0, 0.5 / np.sqrt(n_rec), (n_rec, n_rec))
    np.fill_diagonal(w_rec, 0.0)
    w_out = rng.normal(0, 1.0 / np.sqrt(n_rec), (n_out, n_rec))
    return TSHWNet(cfg, w_in, w_rec, w_out, weight_quant=quant, rng=rng,
                   float_init=float_init)
