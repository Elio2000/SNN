"""
Ordered event queue for hardware micro-step simulation.

Each event represents one schedulable hardware action. Events at the same
timestamp are ordered by insertion sequence (deterministic FIFO within a slot).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EventKind(IntEnum):
    INPUT_SPIKE = 0   # input neuron src fires; fans out via w_in
    REC_SPIKE   = 1   # recurrent neuron src fires; fans out via w_rec
    FPGA_UPDATE = 2   # FPGA processes psi_bin for post-spike j (may be delayed by stall)


@dataclass(order=True)
class Event:
    t: float
    seq: int
    kind: EventKind = field(compare=False)
    src: int        = field(compare=False)
    payload: Any    = field(compare=False, default=None)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq: int = 0

    def push(self, t: float, kind: EventKind, src: int, payload: Any = None) -> None:
        heapq.heappush(self._heap, Event(t, self._seq, kind, src, payload))
        self._seq += 1

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek_t(self) -> float | None:
        return self._heap[0].t if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
