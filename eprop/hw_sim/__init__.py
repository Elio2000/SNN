from .simulator import HWSim, SimConfig
from .network import (
    EventHWNet, HWNet,          # HWNet = EventHWNet alias (backward compat)
    TSHWNet,
    InferenceResult, TrainResult,
    default_config, make_default_hwnet,
    default_ts_config, make_default_ts_hwnet,
)
from .input_encoder import raster_to_events
from .li_readout import compute_output_trace
from .rec_pending import RecMode, make_rec_pending
from .ts_scheduler import TSConfig, TSStats, TSRunResult, TSScheduler

__all__ = [
    "HWSim", "SimConfig",
    "EventHWNet", "HWNet",
    "TSHWNet",
    "InferenceResult", "TrainResult",
    "default_config", "make_default_hwnet",
    "default_ts_config", "make_default_ts_hwnet",
    "raster_to_events", "compute_output_trace",
    "RecMode", "make_rec_pending",
    "TSConfig", "TSStats", "TSRunResult", "TSScheduler",
]
