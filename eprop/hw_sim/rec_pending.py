"""
Recurrent-spike pending buffers for the hardware-accurate ts-barrier scheduler.

Five modes corresponding to distinct hardware implementation choices:

  IDEAL_MULTISET          — unbounded per-neuron event counter; no loss.
                            Algorithm baseline. Arrival order not preserved
                            (all events for neuron j are functionally identical).

  IDEAL_FIFO              — unbounded global FIFO queue (deque). Arrival order
                            preserved. Arbiter sees only the queue head, so rec
                            events are serviced strictly in push order. Use for
                            FIFO-vs-multiset sweep.

  BOUNDED_CAPACITY        — global depth-D capacity multiset; tail-drop on
                            overflow. Per-neuron counts tracked for active-neuron
                            queries. NOT a FIFO (no order preservation).

  BOUNDED_FIFO            — global depth-D bounded FIFO queue (deque); tail-drop
                            when len(queue) >= D. Arbiter sees only head; events
                            serviced in push order up to capacity. Use for
                            FIFO-vs-capacity sweep.

  PER_NEURON_COUNTER      — C-bit saturating counter per neuron.

Interface (all modes):
  push(j)   -> (accepted: bool, saturated: bool)
  service(j)             consume one pending event for neuron j
  is_pending(j) -> bool
  active_neurons() -> list[int]   neurons with at least one pending event
  total() -> int                  total pending events across all neurons
  per_neuron_max() -> int         highest per-neuron count
  clear()                         reset all state (called at start of each ts)
  overflow_count: int             rejected push attempts this ts
  saturated_count: int            push attempts rejected due to per-neuron max

Invariant after full drain:
  rec_events_generated == rec_events_serviced + overflow_count
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np


class RecMode(str, Enum):
    IDEAL_MULTISET     = "ideal_multiset"
    IDEAL_FIFO         = "ideal_fifo"
    BOUNDED_CAPACITY   = "bounded_capacity_depth_D"
    BOUNDED_FIFO       = "bounded_fifo_depth_D"
    PER_NEURON_COUNTER = "per_neuron_counter_Cbit"


# ---------------------------------------------------------------------------
# Ideal (unbounded) multiset

class IdealMultisetRecPending:
    """
    Unbounded per-neuron event counter. Every push accepted; no saturation.
    Active-neuron set = all neurons with count > 0 (arbiter picks freely).
    """

    def __init__(self, n_rec: int) -> None:
        self.n_rec = n_rec
        self._counts = np.zeros(n_rec, dtype=np.int64)

    def push(self, j: int) -> tuple[bool, bool]:
        self._counts[j] += 1
        return True, False

    def service(self, j: int) -> None:
        if self._counts[j] > 0:
            self._counts[j] -= 1

    def is_pending(self, j: int) -> bool:
        return bool(self._counts[j] > 0)

    def active_neurons(self) -> list[int]:
        return list(np.where(self._counts > 0)[0].astype(int))

    def total(self) -> int:
        return int(self._counts.sum())

    def per_neuron_max(self) -> int:
        return int(self._counts.max())

    def clear(self) -> None:
        self._counts[:] = 0

    @property
    def overflow_count(self) -> int:
        return 0

    @property
    def saturated_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Ideal (unbounded) FIFO

class IdealFifoRecPending:
    """
    Unbounded FIFO queue (deque). Arrival order is preserved.

    active_neurons() returns only [queue_head], so the arbiter is forced to
    service events strictly in push order. This is the key difference from
    IdealMultisetRecPending, which exposes all pending neurons simultaneously.
    """

    def __init__(self, n_rec: int) -> None:
        self.n_rec = n_rec
        self._queue: deque[int] = deque()

    def push(self, j: int) -> tuple[bool, bool]:
        self._queue.append(j)
        return True, False

    def service(self, j: int) -> None:
        if self._queue and self._queue[0] == j:
            self._queue.popleft()

    def is_pending(self, j: int) -> bool:
        return j in self._queue

    def active_neurons(self) -> list[int]:
        return [self._queue[0]] if self._queue else []

    def total(self) -> int:
        return len(self._queue)

    def per_neuron_max(self) -> int:
        if not self._queue:
            return 0
        counts: dict[int, int] = {}
        for x in self._queue:
            counts[x] = counts.get(x, 0) + 1
        return max(counts.values())

    def clear(self) -> None:
        self._queue.clear()

    @property
    def overflow_count(self) -> int:
        return 0

    @property
    def saturated_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Bounded global-capacity multiset

class BoundedCapacityRecPending:
    """
    Global capacity-D pending buffer with tail-drop on overflow.

    NOT arrival-order-preserving. Active-neuron set = all neurons with count > 0.
    """

    def __init__(self, n_rec: int, depth: int) -> None:
        self.n_rec = n_rec
        self.depth = depth
        self._counts = np.zeros(n_rec, dtype=np.int64)
        self._total = 0
        self._overflow = 0

    def push(self, j: int) -> tuple[bool, bool]:
        if self._total >= self.depth:
            self._overflow += 1
            return False, False
        self._counts[j] += 1
        self._total += 1
        return True, False

    def service(self, j: int) -> None:
        if self._counts[j] > 0:
            self._counts[j] -= 1
            self._total -= 1

    def is_pending(self, j: int) -> bool:
        return bool(self._counts[j] > 0)

    def active_neurons(self) -> list[int]:
        return list(np.where(self._counts > 0)[0].astype(int))

    def total(self) -> int:
        return self._total

    def per_neuron_max(self) -> int:
        return int(self._counts.max())

    def clear(self) -> None:
        self._counts[:] = 0
        self._total = 0
        self._overflow = 0

    @property
    def overflow_count(self) -> int:
        return self._overflow

    @property
    def saturated_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Bounded FIFO queue

class BoundedFifoRecPending:
    """
    Global depth-D FIFO queue (deque) with tail-drop on overflow.

    Arrival order is preserved up to capacity. active_neurons() returns only
    [queue_head], so events are serviced strictly in push order.
    """

    def __init__(self, n_rec: int, depth: int) -> None:
        self.n_rec = n_rec
        self.depth = depth
        self._queue: deque[int] = deque()
        self._overflow = 0

    def push(self, j: int) -> tuple[bool, bool]:
        if len(self._queue) >= self.depth:
            self._overflow += 1
            return False, False
        self._queue.append(j)
        return True, False

    def service(self, j: int) -> None:
        if self._queue and self._queue[0] == j:
            self._queue.popleft()

    def is_pending(self, j: int) -> bool:
        return j in self._queue

    def active_neurons(self) -> list[int]:
        return [self._queue[0]] if self._queue else []

    def total(self) -> int:
        return len(self._queue)

    def per_neuron_max(self) -> int:
        if not self._queue:
            return 0
        counts: dict[int, int] = {}
        for x in self._queue:
            counts[x] = counts.get(x, 0) + 1
        return max(counts.values())

    def clear(self) -> None:
        self._queue.clear()
        self._overflow = 0

    @property
    def overflow_count(self) -> int:
        return self._overflow

    @property
    def saturated_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Per-neuron C-bit saturating counter

class PerNeuronCounterRecPending:
    """
    Per-neuron C-bit saturating counter. max_val = 2^C - 1.

    overflow_count == saturated_count (every overflow is a saturation event).
    """

    def __init__(self, n_rec: int, c_bits: int) -> None:
        self.n_rec = n_rec
        self.c_bits = c_bits
        self.max_val: int = (1 << c_bits) - 1
        self._counters = np.zeros(n_rec, dtype=np.int64)
        self._overflow = 0
        self._saturated = 0

    def push(self, j: int) -> tuple[bool, bool]:
        if self._counters[j] >= self.max_val:
            self._overflow += 1
            self._saturated += 1
            return False, True
        self._counters[j] += 1
        return True, False

    def service(self, j: int) -> None:
        if self._counters[j] > 0:
            self._counters[j] -= 1

    def is_pending(self, j: int) -> bool:
        return bool(self._counters[j] > 0)

    def active_neurons(self) -> list[int]:
        return list(np.where(self._counters > 0)[0].astype(int))

    def total(self) -> int:
        return int(self._counters.sum())

    def per_neuron_max(self) -> int:
        return int(self._counters.max())

    def clear(self) -> None:
        self._counters[:] = 0
        self._overflow = 0
        self._saturated = 0

    @property
    def overflow_count(self) -> int:
        return self._overflow

    @property
    def saturated_count(self) -> int:
        return self._saturated


# ---------------------------------------------------------------------------
# Union type and factory

RecPendingBuffer = (
    IdealMultisetRecPending
    | IdealFifoRecPending
    | BoundedCapacityRecPending
    | BoundedFifoRecPending
    | PerNeuronCounterRecPending
)


def make_rec_pending(
    mode: RecMode,
    n_rec: int,
    param: int = 0,
) -> RecPendingBuffer:
    """
    Factory for rec_pending buffers.

    param meaning per mode:
        IDEAL_MULTISET    ignored
        IDEAL_FIFO        ignored
        BOUNDED_CAPACITY  depth D (must be > 0)
        BOUNDED_FIFO      depth D (must be > 0)
        PER_NEURON_COUNTER counter width C in bits (must be >= 1)
    """
    if mode == RecMode.IDEAL_MULTISET:
        return IdealMultisetRecPending(n_rec)
    if mode == RecMode.IDEAL_FIFO:
        return IdealFifoRecPending(n_rec)
    if mode == RecMode.BOUNDED_CAPACITY:
        if param <= 0:
            raise ValueError(f"BOUNDED_CAPACITY requires param=D > 0, got {param}")
        return BoundedCapacityRecPending(n_rec, param)
    if mode == RecMode.BOUNDED_FIFO:
        if param <= 0:
            raise ValueError(f"BOUNDED_FIFO requires param=D > 0, got {param}")
        return BoundedFifoRecPending(n_rec, param)
    if mode == RecMode.PER_NEURON_COUNTER:
        if param < 1:
            raise ValueError(f"PER_NEURON_COUNTER requires param=C >= 1, got {param}")
        return PerNeuronCounterRecPending(n_rec, param)
    raise ValueError(f"Unknown RecMode: {mode}")
