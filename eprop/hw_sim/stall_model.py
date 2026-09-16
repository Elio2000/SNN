"""
Stall model for shared overshoot quantization resource.

StallMode.DISABLED (default):
    No stall. FPGA_UPDATE fires at the same timestamp as the spike.

StallMode.SH_BACKGROUND (sample-and-hold + background quantization):
    Preferred hardware direction (TRACEABILITY.md). CNM membrane resets
    immediately; FPGA_UPDATE is delayed by quant_delay steps. Forward
    integration continues unblocked.

StallMode.GLOBAL_STALL (pessimistic baseline):
    After any spike, the shared ADC is busy for quant_delay steps. Any
    INPUT_SPIKE or REC_SPIKE that arrives while t < _cnm_stall_until is
    re-queued at _cnm_stall_until without injecting into the CNM — the
    entire event is deferred, not just the comparator. Used for throughput
    analysis (events pile up and are delayed, not silently absorbed).
"""

from __future__ import annotations

from enum import Enum


class StallMode(str, Enum):
    DISABLED      = "disabled"
    SH_BACKGROUND = "sh_background"
    GLOBAL_STALL  = "global_stall"


class StallModel:
    def __init__(
        self,
        quant_delay: float = 0.0,
        mode: StallMode = StallMode.DISABLED,
    ) -> None:
        self.quant_delay = quant_delay
        self.mode = mode

    def fpga_update_time(self, spike_t: float) -> float:
        """Return FPGA_UPDATE timestamp for a spike that fired at spike_t."""
        if self.mode in (StallMode.SH_BACKGROUND, StallMode.GLOBAL_STALL):
            return spike_t + self.quant_delay
        return spike_t
