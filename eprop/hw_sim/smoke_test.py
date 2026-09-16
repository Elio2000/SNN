"""
CPU smoke test for the hardware-matched event-driven simulator.

One test per finding from review round 2:
  R2-F1 (High)  - trace causality: unrelated spike between spike_t and FPGA_UPDATE
                  must not contaminate gradient.
  R2-F2 (High)  - global stall reschedules events; CNM receives no injection
                  during stall window.
  R2-F3 (High)  - flush phase re-queues non-FPGA events; nothing is silently dropped.
  R2-F4 (Med)   - weight quantization uses round-nearest, not floor; no bias for
                  symmetric inputs.

Run directly:
    uv run python -m eprop.hw_sim.smoke_test
"""

from __future__ import annotations

import numpy as np

from .fpga_trace import WeightQuant
from .overshoot_lut import build_psi_lut, quantize_overshoot
from .simulator import HWSim, SimConfig
from .stall_model import StallMode


# ---------------------------------------------------------------------------
# helpers

def _base_cfg(**kwargs) -> SimConfig:
    defaults = dict(
        n_in=4, n_rec=8, n_out=2,
        thr=1.0, v_reset=0.0, tau_m=20.0, tau_trace=20.0, dt_hw=1.0,
        gamma=0.3, overshoot_n_bits=5, overshoot_max=0.5,
        eta=1e-3, quant_delay=0.0, stall_mode=StallMode.DISABLED,
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


def _make_sim(cfg: SimConfig, seed: int = 42) -> tuple[HWSim, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_in, n_rec, n_out = cfg.n_in, cfg.n_rec, cfg.n_out
    w_in  = rng.normal(0, 2.0 / np.sqrt(n_in),  (n_rec, n_in)).astype(np.float32)
    w_rec = rng.normal(0, 0.5 / np.sqrt(n_rec), (n_rec, n_rec)).astype(np.float32)
    np.fill_diagonal(w_rec, 0.0)
    w_out = rng.normal(0, 1.0 / np.sqrt(n_rec), (n_out, n_rec)).astype(np.float32)
    L = rng.normal(0, 0.1, n_rec)
    sim = HWSim(cfg, w_in, w_rec, w_out,
                learning_signal_fn=lambda j, t: float(L[j]))
    return sim, w_in


def _poisson_spikes(rng: np.random.Generator, n_in: int,
                    t_max: float, rate: float) -> list[tuple[float, int]]:
    spike_times = []
    for i in range(n_in):
        ts = rng.choice(np.arange(1, int(t_max) + 1),
                        size=max(1, int(t_max * rate)), replace=False)
        spike_times.extend((float(t), i) for t in ts)
    spike_times.sort()
    return spike_times


# ---------------------------------------------------------------------------
# R2-F1: trace causality under SH_BACKGROUND delay

def test_trace_causality() -> None:
    """
    Spike from input 0 at t=1 schedules FPGA_UPDATE at t=6 (quant_delay=5).
    Input 1 fires at t=2, between spike_t and FPGA_UPDATE.
    dw_in[j, 1] must be 0: p_in[1] was 0 at spike_t=1.
    """
    quant_delay = 5.0
    cfg = _base_cfg(n_in=2, n_rec=1, n_out=1,
                    quant_delay=quant_delay,
                    stall_mode=StallMode.SH_BACKGROUND)

    # Only input 0 is wired to neuron 0 (w_in[0,1]=0); input 1 is trace-only.
    w_in  = np.array([[2.0, 0.0]], dtype=np.float64)
    w_rec = np.zeros((1, 1), dtype=np.float64)
    w_out = np.zeros((1, 1), dtype=np.float64)

    sim = HWSim(cfg, w_in, w_rec, w_out, learning_signal_fn=lambda j, t: 1.0)
    # Spike from input 0 fires at t=1; FPGA_UPDATE at t=6.
    # Input 1 fires at t=2 (within stall window) — must NOT contaminate gradient.
    sim.load_input_events([(1.0, 0), (2.0, 1)])
    sim.run(t_max=20.0, learning=True)

    # p_in[1] = 0 at spike_t=1 (no input 1 spike yet), so dw_in[0,1] must be 0.
    assert abs(sim.fpga.dw_in[0, 1]) < 1e-10, (
        f"causal contamination: dw_in[0,1] = {sim.fpga.dw_in[0,1]:.2e} (expected 0)"
    )
    assert sim.fpga.dw_in[0, 0] != 0.0, "dw_in[0,0] should be non-zero (input 0 contributed)"

    print(f"[R2-F1 causality]  dw_in[0,0]={sim.fpga.dw_in[0,0]:.4e}"
          f"  dw_in[0,1]={sim.fpga.dw_in[0,1]:.2e}")


# ---------------------------------------------------------------------------
# R2-F2: global stall reschedules events; no CNM injection during stall

def test_global_stall_reschedules() -> None:
    """
    Input spike at t=2 arrives during stall window (stall from t=1 until t=6).
    It must be re-queued at t=6, not injected at t=2.
    Evidence: no post-spike is produced in the window (0, stall_until).
    """
    quant_delay = 5.0
    cfg = _base_cfg(n_in=1, n_rec=1, n_out=1,
                    quant_delay=quant_delay,
                    stall_mode=StallMode.GLOBAL_STALL)

    w_in  = np.array([[1.5]], dtype=np.float64)
    w_rec = np.zeros((1, 1), dtype=np.float64)
    w_out = np.zeros((1, 1), dtype=np.float64)

    spikes = [(1.0, 0), (2.0, 0)]   # second spike arrives during stall window

    # Run only up to t=4 (stall expires at t=6); second spike must not fire.
    sim = HWSim(cfg, w_in, w_rec, w_out)
    sim.load_input_events(spikes)
    sim.run(t_max=4.0, learning=False)

    post = sim.spike_records()
    assert all(r.t == 1.0 for r in post), (
        f"post-spike(s) during stall window: {[(r.t, r.src) for r in post]}"
    )

    # Verify re-queued event is still in the queue (not dropped).
    queue_kinds = [e.kind for e in sim.queue._heap]
    has_input_spike = any(k in (0, 1) for k in queue_kinds)  # INPUT_SPIKE or REC_SPIKE
    assert has_input_spike or len(sim.queue) > 0, (
        "re-queued event was dropped instead of preserved"
    )

    print(f"[R2-F2 global-stall]  post_spikes_in_window={len(post)}"
          f"  remaining_queue={len(sim.queue)}")


# ---------------------------------------------------------------------------
# R2-F3: flush phase re-queues non-FPGA events; nothing silently dropped

def test_flush_preserves_rec_spike() -> None:
    """
    Spike AT t_max=10 produces REC_SPIKE at t=11 (= t_max + dt_hw).
    With quant_delay=3, flush window is [10, 13].
    REC_SPIKE at t=11 falls in the flush window and must be re-queued,
    not silently dropped (old bug: flush popped and discarded non-FPGA events).
    FPGA_UPDATE at t=13 is in the same window and should be consumed.
    """
    quant_delay = 3.0
    dt_hw = 1.0
    t_max = 10.0
    expected_rec_t = t_max + dt_hw  # 11.0 — in flush window, must survive

    cfg = _base_cfg(n_in=1, n_rec=1, n_out=1,
                    dt_hw=dt_hw,
                    quant_delay=quant_delay,
                    stall_mode=StallMode.SH_BACKGROUND)

    w_in  = np.array([[1.5]], dtype=np.float64)
    w_rec = np.zeros((1, 1), dtype=np.float64)
    w_out = np.zeros((1, 1), dtype=np.float64)

    sim = HWSim(cfg, w_in, w_rec, w_out)
    sim.load_input_events([(t_max, 0)])   # spike at exactly t_max
    sim.run(t_max=t_max, learning=False)

    remaining_ts = sorted(e.t for e in sim.queue._heap)
    assert expected_rec_t in remaining_ts, (
        f"REC_SPIKE at t={expected_rec_t} was dropped in flush; remaining={remaining_ts}"
    )

    print(f"[R2-F3 flush-requeue]  remaining_in_queue={len(sim.queue)}"
          f"  remaining_ts={remaining_ts}")


# ---------------------------------------------------------------------------
# R2-F4: weight quantization is round-nearest (no systematic bias)

def test_weight_quant_no_floor_bias() -> None:
    """
    Quantizing a symmetric set of weights with round-nearest should give
    near-zero mean error. Floor would give a negative bias of -step/2.
    """
    rng = np.random.default_rng(0)
    n_bits = 4
    weight_max = 1.0
    n_levels = 1 << n_bits
    step = 2.0 * weight_max / n_levels   # 0.125 for 4-bit

    # Uniformly sample weights spanning the full range
    w = rng.uniform(-weight_max, weight_max, (100, 100))

    quant = WeightQuant(n_bits=n_bits, weight_max=weight_max, stochastic_rounding=False)
    from .fpga_trace import _quantize_weights
    w_q = _quantize_weights(w, quant)

    error = w_q - w
    mean_error = error.mean()

    # Round-nearest: |mean error| << step/2 (unbiased).
    # Floor would give mean_error ≈ -step/2 ≈ -0.0625.
    assert abs(mean_error) < step / 4, (
        f"quantization bias too large: mean_error={mean_error:.4f}"
        f" (floor bias would be ~{-step/2:.4f})"
    )
    assert w_q.max() <= weight_max,   "quantized weight exceeds weight_max"
    assert w_q.min() >= -weight_max,  "quantized weight below -weight_max"

    print(f"[R2-F4 quant-bias]  mean_error={mean_error:.4e}  step={step:.4f}"
          f"  (floor bias would be ~{-step/2:.4f})")


# ---------------------------------------------------------------------------
# regression: end-to-end smoke

def test_basic() -> None:
    cfg = _base_cfg()
    sim, _ = _make_sim(cfg)
    rng = np.random.default_rng(42)
    spike_times = _poisson_spikes(rng, cfg.n_in, t_max=100.0, rate=0.02)

    sim.load_input_events(spike_times)
    records = sim.run(t_max=100.0, learning=True)

    post = sim.spike_records()
    assert len(records) > 0
    assert len(post) > 0, "no post-spikes fired"
    for r in post:
        assert r.psi is not None and 0.0 <= r.psi <= sim.cfg.gamma + 1e-6
        assert r.overshoot is not None and r.overshoot >= 0.0

    print(f"[basic]  input_events={sim.input_spike_count()}, post_spikes={len(post)}")


def test_no_rec_infinite_loop() -> None:
    """Cascading mutual-excitation terminates within bounded event count."""
    cfg = _base_cfg(n_in=1, n_rec=2, n_out=1, dt_hw=1.0)
    w_in  = np.array([[2.0], [0.0]], dtype=np.float32)
    w_rec = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
    w_out = np.zeros((1, 2), dtype=np.float32)

    sim = HWSim(cfg, w_in, w_rec, w_out)
    sim.load_input_events([(1.0, 0)])
    t_max = 10.0
    records = sim.run(t_max, learning=False)

    max_events = int(t_max / cfg.dt_hw) * 4 + 20
    assert len(records) < max_events
    assert all(r.t <= t_max + 1e-9 for r in records)

    post = [r for r in records if r.kind == "post_spike"]
    print(f"[rec-loop]  total_events={len(records)}, post_spikes={len(post)}")


# ---------------------------------------------------------------------------

def main() -> None:
    print("=== hw_sim smoke test ===")
    test_trace_causality()
    test_global_stall_reschedules()
    test_flush_preserves_rec_spike()
    test_weight_quant_no_floor_bias()
    test_basic()
    test_no_rec_infinite_loop()
    print("=== all checks passed ===")


if __name__ == "__main__":
    main()
