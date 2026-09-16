"""
Smoke tests for the HWNet training/inference API.

Verifies infer(), train_sample(), and gradient quantization without any dataset
dependency. All inputs are synthetic.

Run directly:
    uv run python -m eprop.hw_sim.train_smoke_test
"""

from __future__ import annotations

import numpy as np

from .fpga_trace import WeightQuant
from .network import HWNet, InferenceResult, TrainResult, default_config, make_default_hwnet
from .stall_model import StallMode


# ---------------------------------------------------------------------------
# helpers

def _toy_net(n_in: int = 4, n_rec: int = 4, n_out: int = 2) -> HWNet:
    """
    Minimal deterministic network that guarantees spikes.

    w_in = 1.5/n_in so n_in simultaneous inputs push membrane to 1.5*thr.
    w_rec = 0 (no recurrent for simplicity; avoids test sensitivity to cascade).
    w_out drawn randomly; at least one neuron has non-zero w_out fan-in.
    """
    cfg, quant = default_config(n_in=n_in, n_rec=n_rec, n_out=n_out, weight_bits=4)
    # Override: minimal config for fast smoke test
    from .simulator import SimConfig
    cfg = SimConfig(
        n_in=n_in, n_rec=n_rec, n_out=n_out,
        thr=1.0, v_reset=0.0, tau_m=20.0, tau_trace=20.0, dt_hw=1.0,
        gamma=0.3, overshoot_n_bits=3, overshoot_max=0.5,
        eta=1e-3, quant_delay=2.0, stall_mode=StallMode.SH_BACKGROUND,
    )
    rng = np.random.default_rng(7)
    w_in  = np.full((n_rec, n_in), 1.5 / n_in, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)
    w_out = rng.normal(0, 0.5, (n_out, n_rec))
    return HWNet(cfg, w_in, w_rec, w_out, weight_quant=quant, rng=rng)


def _toy_raster(T: int, n_in: int) -> np.ndarray:
    """Raster with all inputs spiking at ts=0 and ts=2; guarantees membrane crossings."""
    raster = np.zeros((T, n_in), dtype=bool)
    raster[0, :] = True
    if T > 2:
        raster[2, :] = True
    return raster


# ---------------------------------------------------------------------------
# T1: infer returns correct shapes and types

def test_infer_shapes() -> None:
    n_in, n_rec, n_out, T = 4, 4, 2, 5
    net = _toy_net(n_in, n_rec, n_out)
    raster = _toy_raster(T, n_in)

    result = net.infer(raster)
    assert isinstance(result, InferenceResult)
    assert result.logits.shape == (n_out,), f"logits shape {result.logits.shape}"
    assert result.output_trace.shape == (T, n_out), f"output_trace shape {result.output_trace.shape}"
    assert isinstance(result.pred, int) and 0 <= result.pred < n_out
    assert isinstance(result.spike_records, list)
    assert np.isfinite(result.logits).all(), "logits contain non-finite values"

    print(f"[T1 infer-shapes]  logits={result.logits}  pred={result.pred}"
          f"  post_spikes={sum(1 for r in result.spike_records if r.kind == 'post_spike')}")


# ---------------------------------------------------------------------------
# T2: infer does not modify weights

def test_infer_no_weight_mutation() -> None:
    net = _toy_net()
    w_in_before  = net.w_in.copy()
    w_rec_before = net.w_rec.copy()
    raster = _toy_raster(5, net.cfg.n_in)

    net.infer(raster)

    assert np.array_equal(net.w_in,  w_in_before),  "infer mutated w_in"
    assert np.array_equal(net.w_rec, w_rec_before), "infer mutated w_rec"
    print("[T2 infer-no-mutation]  pass")


# ---------------------------------------------------------------------------
# T3: train_sample returns finite loss and non-zero gradients

def test_train_sample_nonzero_gradients() -> None:
    n_in, n_rec, n_out, T = 4, 4, 2, 5
    net = _toy_net(n_in, n_rec, n_out)
    raster = _toy_raster(T, n_in)

    result = net.train_sample(raster, target_class=0)
    assert isinstance(result, TrainResult)
    assert np.isfinite(result.loss),  f"loss is non-finite: {result.loss}"
    assert result.loss > 0,           f"loss unexpectedly zero: {result.loss}"

    # Gradients should be non-zero if any post-spike fired and L_j != 0.
    # Accept either dw_in OR dw_rec to tolerate toy-network edge cases.
    assert result.dw_in_norm > 0 or result.dw_rec_norm > 0, (
        f"both gradient norms are zero: dw_in={result.dw_in_norm:.3e}"
        f" dw_rec={result.dw_rec_norm:.3e}"
    )

    print(f"[T3 train-gradients]  loss={result.loss:.4f}"
          f"  dw_in_norm={result.dw_in_norm:.4e}"
          f"  dw_rec_norm={result.dw_rec_norm:.4e}")


# ---------------------------------------------------------------------------
# T4: apply_gradients leaves weights on 4-bit quantization grid

def test_quantized_weights_on_grid() -> None:
    n_in, n_rec, n_out, T = 4, 4, 2, 5
    quant = WeightQuant(n_bits=4, weight_max=1.0, stochastic_rounding=False)
    net = _toy_net(n_in, n_rec, n_out)
    net.weight_quant = quant

    raster = _toy_raster(T, n_in)
    net.train_sample(raster, target_class=1)  # triggers apply_gradients internally

    def on_grid(w: np.ndarray, q: WeightQuant) -> bool:
        step = 2.0 * q.weight_max / (1 << q.n_bits)
        bins = np.round((w + q.weight_max) / step)
        w_reconstructed = -q.weight_max + bins * step
        return bool(np.allclose(w, w_reconstructed, atol=1e-10))

    assert on_grid(net.w_in,  quant), "w_in not on quantization grid after training"
    assert on_grid(net.w_rec, quant), "w_rec not on quantization grid after training"
    assert np.all(net.w_in  <= quant.weight_max), "w_in  exceeds weight_max"
    assert np.all(net.w_in  >= -quant.weight_max), "w_in below -weight_max"
    assert np.all(net.w_rec <= quant.weight_max), "w_rec exceeds weight_max"
    assert np.all(net.w_rec >= -quant.weight_max), "w_rec below -weight_max"

    step = 2.0 * quant.weight_max / (1 << quant.n_bits)
    print(f"[T4 quant-grid]  step={step:.4f}  w_in range=[{net.w_in.min():.3f}, {net.w_in.max():.3f}]")


# ---------------------------------------------------------------------------
# T5: train_sample with 10-class default config (larger network, quick sanity)

def test_10class_default_config() -> None:
    net = make_default_hwnet(n_in=8, n_rec=8, n_out=10, seed=0)
    # Sparse raster: 3 active inputs over 4 timesteps
    rng = np.random.default_rng(1)
    raster = (rng.random((4, 8)) < 0.4).astype(bool)
    raster[0, :3] = True  # guarantee at least 3 inputs in ts=0

    result = net.train_sample(raster, target_class=3)
    assert np.isfinite(result.loss)
    assert result.logits.shape == (10,)

    print(f"[T5 10class-config]  loss={result.loss:.4f}  pred={result.pred}")


# ---------------------------------------------------------------------------

def main() -> None:
    print("=== train/infer smoke test ===")
    test_infer_shapes()
    test_infer_no_weight_mutation()
    test_train_sample_nonzero_gradients()
    test_quantized_weights_on_grid()
    test_10class_default_config()
    print("=== all checks passed ===")


if __name__ == "__main__":
    main()
