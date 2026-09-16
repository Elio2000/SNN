"""
Smoke tests for TSHWNet — hardware-golden training/inference API (TSScheduler backend).

Verifies infer(), train_sample(), initial quantization, and the two-pass
identical-spikes property. No dataset dependency; all inputs are synthetic.

Run directly:
    uv run python -m eprop.hw_sim.ts_train_smoke_test
"""

from __future__ import annotations

import numpy as np

import warnings

from .fpga_trace import WeightQuant
from .network import TSHWNet, InferenceResult, TrainResult, default_ts_config, make_default_ts_hwnet
from .rec_pending import RecMode
from .ts_scheduler import TSConfig, TSScheduler


# ---------------------------------------------------------------------------
# helpers

def _toy_ts_net(
    n_in: int = 4,
    n_rec: int = 4,
    n_out: int = 2,
    arbiter_seed: int = 0,
) -> TSHWNet:
    """
    Minimal deterministic TSHWNet that guarantees spikes on the toy raster.

    w_in = 1.5/n_in so all n_in simultaneous inputs push each neuron to 1.5*thr.
    w_rec = 0 (no recurrent cascades; keeps test sensitivity low).
    w_out drawn randomly; ensures non-trivial learning signal.
    float_init=True: w_in=0.375 is already on the 4-bit grid so behaviour is
    identical to float_init=False, but avoids RNG dependency in quantization path.
    """
    cfg = TSConfig(
        n_in=n_in, n_rec=n_rec, n_out=n_out,
        thr=1.0, v_reset=0.0, tau_m=20.0, tau_trace=20.0,
        dt_ts=1.0, gamma=0.3, overshoot_n_bits=3, overshoot_max=0.5,
        eta=1e-3, rec_mode=RecMode.IDEAL_MULTISET, rec_mode_param=0,
        arbiter_seed=arbiter_seed, max_drain_steps=10_000,
    )
    quant = WeightQuant(n_bits=4, weight_max=1.0, stochastic_rounding=False)
    rng   = np.random.default_rng(7)
    w_in  = np.full((n_rec, n_in), 1.5 / n_in, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)
    w_out = rng.normal(0, 0.5, (n_out, n_rec))
    return TSHWNet(cfg, w_in, w_rec, w_out, weight_quant=quant, rng=rng, float_init=True)


def _toy_raster(T: int, n_in: int) -> np.ndarray:
    """All inputs fire at ts=0 and ts=2 to guarantee membrane crossings."""
    raster = np.zeros((T, n_in), dtype=bool)
    raster[0, :] = True
    if T > 2:
        raster[2, :] = True
    return raster


# ---------------------------------------------------------------------------
# T1: infer returns correct shapes and types

def test_ts_infer_shapes() -> None:
    n_in, n_rec, n_out, T = 4, 4, 2, 5
    net    = _toy_ts_net(n_in, n_rec, n_out)
    raster = _toy_raster(T, n_in)

    result = net.infer(raster)
    assert isinstance(result, InferenceResult)
    assert result.logits.shape == (n_out,), f"logits shape {result.logits.shape}"
    assert result.output_trace.shape == (T, n_out), f"output_trace {result.output_trace.shape}"
    assert isinstance(result.pred, int) and 0 <= result.pred < n_out
    assert isinstance(result.spike_records, list)
    assert np.isfinite(result.logits).all(), "logits contain non-finite values"

    n_post = sum(1 for r in result.spike_records if r.kind == "post_spike")
    print(f"[T1 ts-infer-shapes]  logits={result.logits}  pred={result.pred}"
          f"  post_spikes={n_post}")


# ---------------------------------------------------------------------------
# T2: infer does not modify weights

def test_ts_infer_no_weight_mutation() -> None:
    net = _toy_ts_net()
    w_in_before  = net.w_in.copy()
    w_rec_before = net.w_rec.copy()
    raster = _toy_raster(5, net.cfg.n_in)

    net.infer(raster)

    assert np.array_equal(net.w_in,  w_in_before),  "infer mutated w_in"
    assert np.array_equal(net.w_rec, w_rec_before), "infer mutated w_rec"
    print("[T2 ts-infer-no-mutation]  pass")


# ---------------------------------------------------------------------------
# T3: train_sample returns finite loss and non-zero gradients

def test_ts_train_sample_nonzero_gradients() -> None:
    n_in, n_rec, n_out, T = 4, 4, 2, 5
    net    = _toy_ts_net(n_in, n_rec, n_out)
    raster = _toy_raster(T, n_in)

    result = net.train_sample(raster, target_class=0)
    assert isinstance(result, TrainResult)
    assert np.isfinite(result.loss), f"loss is non-finite: {result.loss}"
    assert result.loss > 0, f"loss unexpectedly zero: {result.loss}"
    assert result.dw_in_norm > 0 or result.dw_rec_norm > 0, (
        f"both gradient norms are zero: dw_in={result.dw_in_norm:.3e}"
        f" dw_rec={result.dw_rec_norm:.3e}"
    )

    print(f"[T3 ts-train-gradients]  loss={result.loss:.4f}"
          f"  dw_in_norm={result.dw_in_norm:.4e}"
          f"  dw_rec_norm={result.dw_rec_norm:.4e}")


# ---------------------------------------------------------------------------
# T4: initial weights are on 4-bit quantization grid (float_init=False default)

def test_ts_initial_weights_on_grid() -> None:
    """
    make_default_ts_hwnet uses float_init=False by default, so Xavier-scaled
    random weights are snapped to the hardware grid at construction.
    """
    quant = WeightQuant(n_bits=4, weight_max=1.0, stochastic_rounding=False)
    net = make_default_ts_hwnet(n_in=8, n_rec=8, n_out=4, seed=3, float_init=False)

    def on_grid(w: np.ndarray, q: WeightQuant) -> bool:
        step = 2.0 * q.weight_max / (1 << q.n_bits)
        bins = np.round((w + q.weight_max) / step)
        return bool(np.allclose(w, -q.weight_max + bins * step, atol=1e-10))

    assert on_grid(net.w_in,  quant), "w_in not on 4-bit grid at construction"
    assert on_grid(net.w_rec, quant), "w_rec not on 4-bit grid at construction"

    step = 2.0 * quant.weight_max / (1 << quant.n_bits)
    print(f"[T4 ts-initial-grid]  step={step:.4f}"
          f"  w_in range=[{net.w_in.min():.3f}, {net.w_in.max():.3f}]")


# ---------------------------------------------------------------------------
# T5: two fresh TSSchedulers with the same seed produce identical event_order

def test_ts_two_pass_identical_spikes() -> None:
    """
    TSHWNet.train_sample() relies on both passes using fresh TSScheduler instances
    with the same cfg.arbiter_seed, so spikes are identical across passes.
    This test verifies that property directly on the scheduler level.
    """
    n_in, n_rec = 4, 4
    w_in  = np.full((n_rec, n_in), 1.5 / n_in, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)
    cfg   = TSConfig(
        n_in=n_in, n_rec=n_rec, n_out=2,
        thr=1.0, v_reset=0.0, tau_m=20.0, tau_trace=20.0,
        dt_ts=1.0, gamma=0.3, overshoot_n_bits=3, overshoot_max=0.5,
        eta=1e-3, rec_mode=RecMode.IDEAL_MULTISET, rec_mode_param=0,
        arbiter_seed=42, max_drain_steps=10_000,
    )
    raster = _toy_raster(T=4, n_in=n_in)

    # Simulate two passes: both fresh from same seed, no L_signal
    sched1 = TSScheduler(cfg, w_in, w_rec)
    sched2 = TSScheduler(cfg, w_in, w_rec)
    r1 = sched1.run(raster)
    r2 = sched2.run(raster)

    for ts, (s1, s2) in enumerate(zip(r1.stats_per_ts, r2.stats_per_ts)):
        assert s1.event_order == s2.event_order, (
            f"ts={ts}: event_order differs between same-seed passes — "
            "TSHWNet two-pass gradient correctness is broken"
        )
        assert s1.post_spikes == s2.post_spikes, (
            f"ts={ts}: post_spikes differ between same-seed passes"
        )

    total_spikes = sum(len(s.post_spikes) for s in r1.stats_per_ts)
    print(f"[T5 ts-two-pass]  total_spikes={total_spikes}"
          f"  ts=0 event_order[:3]={r1.stats_per_ts[0].event_order[:3]}")


# ---------------------------------------------------------------------------
# T6: 10-class default TSHWNet config (larger network, quick sanity)

def test_ts_10class_default_config() -> None:
    net = make_default_ts_hwnet(n_in=8, n_rec=8, n_out=10, seed=0, float_init=False)
    rng = np.random.default_rng(1)
    raster = (rng.random((4, 8)) < 0.4).astype(bool)
    raster[0, :3] = True

    result = net.train_sample(raster, target_class=3)
    assert np.isfinite(result.loss)
    assert result.logits.shape == (10,)

    print(f"[T6 ts-10class]  loss={result.loss:.4f}  pred={result.pred}")


# ---------------------------------------------------------------------------
# T7: w_out is quantized to the hardware grid at construction (float_init=False)

def test_ts_w_out_on_grid() -> None:
    """
    make_default_ts_hwnet float_init=False quantizes w_in, w_rec, AND w_out.
    """
    quant = WeightQuant(n_bits=4, weight_max=1.0, stochastic_rounding=False)
    net = make_default_ts_hwnet(n_in=8, n_rec=8, n_out=4, seed=3, float_init=False)

    def on_grid(w: np.ndarray, q: WeightQuant) -> bool:
        step = 2.0 * q.weight_max / (1 << q.n_bits)
        bins = np.round((w + q.weight_max) / step)
        return bool(np.allclose(w, -q.weight_max + bins * step, atol=1e-10))

    assert on_grid(net.w_in,  quant), "w_in not on 4-bit grid"
    assert on_grid(net.w_rec, quant), "w_rec not on 4-bit grid"
    assert on_grid(net.w_out, quant), "w_out not on 4-bit grid"

    step = 2.0 * quant.weight_max / (1 << quant.n_bits)
    print(f"[T7 ts-w-out-grid]  step={step:.4f}"
          f"  w_out range=[{net.w_out.min():.3f}, {net.w_out.max():.3f}]")


# ---------------------------------------------------------------------------
# T8: drain_exhausted is surfaced in TrainResult

def test_ts_drain_exhausted_in_result() -> None:
    """
    TrainResult.drain_exhausted=True when max_drain_steps is too small.
    Uses a self-exciting neuron (w_rec[0,0]=2.0) with max_drain_steps=3.
    """
    n_in, n_rec, n_out = 1, 1, 2
    cfg = TSConfig(
        n_in=n_in, n_rec=n_rec, n_out=n_out,
        thr=1.0, v_reset=0.0, tau_m=20.0, tau_trace=20.0,
        dt_ts=1.0, gamma=0.3, overshoot_n_bits=3, overshoot_max=0.5,
        eta=1e-3, rec_mode=RecMode.IDEAL_MULTISET, rec_mode_param=0,
        arbiter_seed=0, max_drain_steps=3,
    )
    quant = WeightQuant(n_bits=4, weight_max=1.0, stochastic_rounding=False)
    rng = np.random.default_rng(0)
    w_in  = np.array([[2.0]], dtype=np.float64)
    w_rec = np.array([[2.0]], dtype=np.float64)  # self-exciting
    w_out = np.array([[0.5], [-0.5]], dtype=np.float64)
    net = TSHWNet(cfg, w_in, w_rec, w_out, weight_quant=quant, rng=rng, float_init=True)
    raster = np.ones((1, n_in), dtype=bool)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = net.train_sample(raster, target_class=0)

    assert result.drain_exhausted, "TrainResult.drain_exhausted should be True"
    print(f"[T8 ts-drain-in-result]  drain_exhausted={result.drain_exhausted}"
          f"  loss={result.loss:.4f}")


# ---------------------------------------------------------------------------
# T9: psi_bins and psi_vals are recorded per spike in TSStats

def test_ts_psi_recorded() -> None:
    """
    TSStats.psi_bins and TSStats.psi_vals are parallel to post_spikes.
    Each has the same length as post_spikes; psi_vals are non-negative floats.
    """
    net = _toy_ts_net()
    raster = _toy_raster(T=3, n_in=net.cfg.n_in)

    sched = TSScheduler(net.cfg, net.w_in, net.w_rec)
    result = sched.run(raster)

    total_spikes   = sum(len(s.post_spikes) for s in result.stats_per_ts)
    total_psi_bins = sum(len(s.psi_bins)    for s in result.stats_per_ts)
    total_psi_vals = sum(len(s.psi_vals)    for s in result.stats_per_ts)

    assert total_spikes > 0, "no spikes produced — test is vacuous"
    assert total_psi_bins == total_spikes, (
        f"psi_bins length {total_psi_bins} != post_spikes {total_spikes}"
    )
    assert total_psi_vals == total_spikes, (
        f"psi_vals length {total_psi_vals} != post_spikes {total_spikes}"
    )
    for s in result.stats_per_ts:
        assert all(isinstance(b, int)   for b in s.psi_bins), "psi_bins must be int"
        assert all(isinstance(v, float) for v in s.psi_vals), "psi_vals must be float"
        assert all(v >= 0.0             for v in s.psi_vals), "psi_vals must be >= 0"

    sample_vals = [v for s in result.stats_per_ts for v in s.psi_vals][:4]
    print(f"[T9 ts-psi-recorded]  total_spikes={total_spikes}"
          f"  psi_vals[:4]={[f'{v:.4f}' for v in sample_vals]}")


# ---------------------------------------------------------------------------

def main() -> None:
    print("=== TSHWNet train/infer smoke tests ===")
    test_ts_infer_shapes()
    test_ts_infer_no_weight_mutation()
    test_ts_train_sample_nonzero_gradients()
    test_ts_initial_weights_on_grid()
    test_ts_two_pass_identical_spikes()
    test_ts_10class_default_config()
    test_ts_w_out_on_grid()
    test_ts_drain_exhausted_in_result()
    test_ts_psi_recorded()
    print("=== all checks passed ===")


if __name__ == "__main__":
    main()
