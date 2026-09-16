"""
Smoke tests for TSScheduler — software-timestep barrier scheduler.

All tests use synthetic networks; no dataset dependency.

Run directly:
    uv run python -m eprop.hw_sim.ts_smoke_test
"""

from __future__ import annotations

import warnings

import numpy as np

from .rec_pending import RecMode
from .ts_scheduler import TSConfig, TSScheduler, TSStats


# ---------------------------------------------------------------------------
# helpers

def _make_cfg(
    n_in: int = 4,
    n_rec: int = 4,
    n_out: int = 2,
    rec_mode: RecMode = RecMode.IDEAL_MULTISET,
    rec_mode_param: int = 0,
    arbiter_seed: int = 0,
    max_drain_steps: int = 10_000,
) -> TSConfig:
    return TSConfig(
        n_in=n_in, n_rec=n_rec, n_out=n_out,
        thr=1.0, v_reset=0.0, tau_m=20.0, tau_trace=20.0,
        dt_ts=1.0, gamma=0.3, overshoot_n_bits=3, overshoot_max=0.5,
        eta=1e-3, rec_mode=rec_mode, rec_mode_param=rec_mode_param,
        arbiter_seed=arbiter_seed, max_drain_steps=max_drain_steps,
    )


def _spiking_net(n_in: int, n_rec: int) -> tuple[np.ndarray, np.ndarray]:
    """All n_rec neurons fire when all n_in inputs are active (sum = 1.5 > thr=1.0)."""
    w_in  = np.full((n_rec, n_in), 1.5 / n_in, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)
    return w_in, w_rec


def _all_on_raster(T: int, n_in: int) -> np.ndarray:
    raster = np.zeros((T, n_in), dtype=bool)
    raster[:] = True
    return raster


# ---------------------------------------------------------------------------
# T1: Fixed seed produces identical event_order on two independent runs

def test_fixed_seed_reproducible() -> None:
    n_in, n_rec = 4, 4
    cfg = _make_cfg(n_in=n_in, n_rec=n_rec, arbiter_seed=42)
    w_in, w_rec = _spiking_net(n_in, n_rec)
    raster = _all_on_raster(T=3, n_in=n_in)

    sched_a = TSScheduler(cfg, w_in, w_rec)
    result_a = sched_a.run(raster)

    sched_b = TSScheduler(cfg, w_in, w_rec)
    result_b = sched_b.run(raster)

    for ts, (sa, sb) in enumerate(zip(result_a.stats_per_ts, result_b.stats_per_ts)):
        assert sa.event_order == sb.event_order, (
            f"ts={ts}: event_order differs between runs with same seed\n"
            f"  run_a: {sa.event_order}\n  run_b: {sb.event_order}"
        )
        assert sa.post_spikes == sb.post_spikes, f"ts={ts}: post_spikes differs with same seed"

    print(f"[T1 fixed-seed]  ts=0 drain_steps={result_a.stats_per_ts[0].drain_steps}"
          f"  event_order[:4]={result_a.stats_per_ts[0].event_order[:4]}")


# ---------------------------------------------------------------------------
# T2: Different seeds produce different event orders (over a multi-ts run)

def test_different_seed_changes_order() -> None:
    n_in, n_rec = 4, 4
    w_in, w_rec = _spiking_net(n_in, n_rec)
    raster = _all_on_raster(T=4, n_in=n_in)

    cfg_a = _make_cfg(n_in=n_in, n_rec=n_rec, arbiter_seed=0)
    cfg_b = _make_cfg(n_in=n_in, n_rec=n_rec, arbiter_seed=999)

    result_a = TSScheduler(cfg_a, w_in, w_rec).run(raster)
    result_b = TSScheduler(cfg_b, w_in, w_rec).run(raster)

    def flatten_orders(result) -> list:
        out = []
        for s in result.stats_per_ts:
            out.extend(s.event_order)
        return out

    orders_a = flatten_orders(result_a)
    orders_b = flatten_orders(result_b)
    assert any(a != b for a, b in zip(orders_a, orders_b)), (
        "Different seeds produced identical event orders across all timesteps."
    )

    print(f"[T2 diff-seed]  seed=0 first 3 events={orders_a[:3]}"
          f"  seed=999 first 3 events={orders_b[:3]}")


# ---------------------------------------------------------------------------
# T3: Bounded capacity — overflow detectable

def test_bounded_capacity_overflow() -> None:
    """
    Network: n_rec=4, single input fires all 4 neurons simultaneously.
    depth=1: only the first push accepted; next 3 overflow.
    """
    n_in, n_rec = 1, 4
    w_in  = np.full((n_rec, n_in), 2.0, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)

    cfg = _make_cfg(
        n_in=n_in, n_rec=n_rec,
        rec_mode=RecMode.BOUNDED_CAPACITY, rec_mode_param=1,  # depth=1
        arbiter_seed=0,
    )
    raster = np.ones((1, n_in), dtype=bool)

    result = TSScheduler(cfg, w_in, w_rec).run(raster)
    ts0 = result.stats_per_ts[0]

    assert ts0.overflow_count > 0, (
        f"Expected overflow with depth=1 and {n_rec} simultaneous fires, "
        f"got overflow_count={ts0.overflow_count}"
    )
    assert ts0.rec_events_generated == ts0.rec_events_serviced + ts0.overflow_count, (
        f"Invariant violated: gen={ts0.rec_events_generated}"
        f" svc={ts0.rec_events_serviced} ovf={ts0.overflow_count}"
    )

    print(f"[T3 bounded-capacity]  gen={ts0.rec_events_generated}"
          f"  svc={ts0.rec_events_serviced}  overflow={ts0.overflow_count}")


# ---------------------------------------------------------------------------
# T4: Per-neuron counter saturation detectable

def test_counter_saturation() -> None:
    """
    Network guarantees saturation regardless of arbiter order:
      n_rec=3, n_in=1. Single input fires all 3 neurons.
      w_rec[0,1] = w_rec[0,2] = 2.0: rec[1] and rec[2] each cause neuron 0 to re-fire.
      c_bits=1 (max_val=1):
        - After input: push(0), push(1), push(2) all accepted.
        - Servicing rec[1] or rec[2] fires neuron 0. push(0) when counter[0]=1 → saturated.
        - Even if rec[0] is drained first, neuron 0 fires twice from rec[1] and rec[2];
          the second push always finds counter[0]=1 → saturated. Guaranteed in all orders.
    """
    n_in, n_rec = 1, 3
    w_in  = np.full((n_rec, n_in), 2.0, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)
    w_rec[0, 1] = 2.0
    w_rec[0, 2] = 2.0

    cfg = _make_cfg(
        n_in=n_in, n_rec=n_rec,
        rec_mode=RecMode.PER_NEURON_COUNTER, rec_mode_param=1,  # c_bits=1, max=1
        arbiter_seed=0,
    )
    raster = np.ones((1, n_in), dtype=bool)

    result = TSScheduler(cfg, w_in, w_rec).run(raster)
    ts0 = result.stats_per_ts[0]

    assert ts0.saturated_counter_count > 0, (
        f"Expected counter saturation with c_bits=1 and cascades, "
        f"got saturated_counter_count={ts0.saturated_counter_count}"
    )
    assert ts0.overflow_count == ts0.saturated_counter_count, (
        "For PER_NEURON_COUNTER mode, every overflow must be a saturation event"
    )

    print(f"[T4 counter-sat]  gen={ts0.rec_events_generated}"
          f"  svc={ts0.rec_events_serviced}"
          f"  saturated={ts0.saturated_counter_count}")


# ---------------------------------------------------------------------------
# T5: Ideal multiset — no spike loss over multiple timesteps

def test_ideal_multiset_no_spike_loss() -> None:
    """
    IDEAL_MULTISET: overflow_count == 0 for every ts.
    After complete drain: rec_events_generated == rec_events_serviced.
    """
    n_in, n_rec = 4, 6
    w_in, w_rec = _spiking_net(n_in, n_rec)
    rng = np.random.default_rng(5)
    w_rec = rng.uniform(0.0, 0.3, (n_rec, n_rec))
    np.fill_diagonal(w_rec, 0.0)

    cfg = _make_cfg(
        n_in=n_in, n_rec=n_rec,
        rec_mode=RecMode.IDEAL_MULTISET,
        arbiter_seed=7,
    )
    raster = _all_on_raster(T=4, n_in=n_in)

    result = TSScheduler(cfg, w_in, w_rec).run(raster)

    for ts, s in enumerate(result.stats_per_ts):
        assert s.overflow_count == 0, (
            f"ts={ts}: ideal_multiset should never overflow, got {s.overflow_count}"
        )
        assert s.saturated_counter_count == 0
        assert s.rec_events_generated == s.rec_events_serviced, (
            f"ts={ts}: spike loss: gen={s.rec_events_generated} svc={s.rec_events_serviced}"
        )
        assert not s.drain_exhausted, f"ts={ts}: drain unexpectedly exhausted"

    total_gen = sum(s.rec_events_generated for s in result.stats_per_ts)
    total_svc = sum(s.rec_events_serviced  for s in result.stats_per_ts)
    print(f"[T5 ideal-multiset]  total_gen={total_gen}  total_svc={total_svc}"
          f"  max_rec_total={max(s.max_rec_pending_total for s in result.stats_per_ts)}")


# ---------------------------------------------------------------------------
# T6: drain_exhausted raises warning and sets flag (not silent drop)

def test_drain_exhausted_warns() -> None:
    """
    Self-exciting neuron (w_rec[0,0]=2.0) with max_drain_steps=3 causes
    drain exhaustion. Verify:
      - RuntimeWarning is raised (not silent)
      - drain_exhausted == True
      - drain_exhausted_pending > 0
    """
    n_in, n_rec = 1, 1
    w_in  = np.array([[2.0]], dtype=np.float64)   # input fires neuron 0
    w_rec = np.array([[2.0]], dtype=np.float64)   # neuron 0 re-excites itself
    # NOTE: self-connection (diagonal) is non-zero here intentionally to stress the
    # drain — in production networks the diagonal is zeroed.

    cfg = _make_cfg(
        n_in=n_in, n_rec=n_rec,
        rec_mode=RecMode.IDEAL_MULTISET,
        arbiter_seed=0,
        max_drain_steps=3,   # too small to drain the cascade
    )
    raster = np.ones((1, n_in), dtype=bool)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = TSScheduler(cfg, w_in, w_rec).run(raster)

    ts0 = result.stats_per_ts[0]

    assert ts0.drain_exhausted, "drain_exhausted should be True"
    assert ts0.drain_exhausted_pending > 0, "drain_exhausted_pending should be > 0"
    assert any(issubclass(x.category, RuntimeWarning) for x in w), (
        "Expected a RuntimeWarning for drain exhaustion, got none"
    )

    print(f"[T6 drain-exhausted]  drain_steps={ts0.drain_steps}"
          f"  exhausted_pending={ts0.drain_exhausted_pending}"
          f"  warnings={[str(x.message) for x in w][:1]}")


# ---------------------------------------------------------------------------
# T7: IDEAL_FIFO — rec events serviced in push (arrival) order

def test_ideal_fifo_arrival_order() -> None:
    """
    IDEAL_FIFO: active_neurons() exposes only the queue head, so the arbiter
    is forced to service rec events strictly in push order.

    n_in=1, n_rec=4, w_in[:,0]=2.0 → all 4 neurons spike on the single input.
    np.where returns spiking_idx in ascending order → push order: 0, 1, 2, 3.
    With w_rec=0, servicing a rec event causes no further spikes, so the queue
    drains deterministically: [('rec', 0), ('rec', 1), ('rec', 2), ('rec', 3)].
    """
    n_in, n_rec = 1, 4
    w_in  = np.full((n_rec, n_in), 2.0, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)

    cfg = _make_cfg(
        n_in=n_in, n_rec=n_rec,
        rec_mode=RecMode.IDEAL_FIFO, rec_mode_param=0,
        arbiter_seed=0,
    )
    raster = np.ones((1, n_in), dtype=bool)

    result = TSScheduler(cfg, w_in, w_rec).run(raster)
    ts0 = result.stats_per_ts[0]

    rec_order = [src for kind, src in ts0.event_order if kind == "rec"]

    assert rec_order == list(range(n_rec)), (
        f"IDEAL_FIFO must service rec events in push order (0→{n_rec-1}), "
        f"got: {rec_order}"
    )
    assert ts0.overflow_count == 0

    print(f"[T7 ideal-fifo-order]  event_order={ts0.event_order}")


# ---------------------------------------------------------------------------
# T8: BOUNDED_FIFO — overflow at capacity; accepted events serviced in order

def test_bounded_fifo_overflow() -> None:
    """
    BOUNDED_FIFO depth=2, 4 simultaneous pushes (neurons 0-3) → 2 overflows.
    Accepted: neurons 0 and 1 (first two in push order).
    Serviced in FIFO order: [0, 1].
    """
    n_in, n_rec = 1, 4
    w_in  = np.full((n_rec, n_in), 2.0, dtype=np.float64)
    w_rec = np.zeros((n_rec, n_rec), dtype=np.float64)

    cfg = _make_cfg(
        n_in=n_in, n_rec=n_rec,
        rec_mode=RecMode.BOUNDED_FIFO, rec_mode_param=2,  # depth=2
        arbiter_seed=0,
    )
    raster = np.ones((1, n_in), dtype=bool)

    result = TSScheduler(cfg, w_in, w_rec).run(raster)
    ts0 = result.stats_per_ts[0]

    assert ts0.overflow_count == 2, (
        f"Expected 2 overflows (depth=2, 4 pushes), got {ts0.overflow_count}"
    )
    rec_order = [src for kind, src in ts0.event_order if kind == "rec"]
    assert rec_order == [0, 1], (
        f"Bounded FIFO should service neurons 0 then 1 in push order, got {rec_order}"
    )
    assert ts0.rec_events_generated == 4  # all 4 push attempts counted
    assert ts0.rec_events_serviced == 2

    print(f"[T8 bounded-fifo]  overflow={ts0.overflow_count}"
          f"  gen={ts0.rec_events_generated}  svc={ts0.rec_events_serviced}"
          f"  order={rec_order}")


# ---------------------------------------------------------------------------

def main() -> None:
    print("=== ts_scheduler smoke tests ===")
    test_fixed_seed_reproducible()
    test_different_seed_changes_order()
    test_bounded_capacity_overflow()
    test_counter_saturation()
    test_ideal_multiset_no_spike_loss()
    test_drain_exhausted_warns()
    test_ideal_fifo_arrival_order()
    test_bounded_fifo_overflow()
    print("=== all checks passed ===")


if __name__ == "__main__":
    main()
