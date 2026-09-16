"""
Overshoot-bin to psi LUT for CNM learning-mode readout.

Hardware flow (TRACEABILITY.md):
    overshoot_j = v_pre_reset_j - Vth
    overshoot_bin_j = saturating_counter(overshoot_j, n_bits, overshoot_max)
    psi_j = LUT[overshoot_bin_j]

n_levels  = 2^n_bits  (e.g. 32 for a 5-bit counter)
overshoot_lsb = overshoot_max / n_levels
bin_k represents overshoot in [k*lsb, (k+1)*lsb); saturates at n_levels-1.

overshoot_max must be supplied explicitly — it is a hardware design parameter
(the ADC full-scale range), not derived from Vth.
"""

from __future__ import annotations

import numpy as np


def build_psi_lut(
    n_bits: int, gamma: float, thr: float, overshoot_max: float
) -> np.ndarray:
    """
    Return float32 LUT of length 2^n_bits.

    psi_k = gamma * max(0, 1 - center_k / thr)
    where center_k = (k + 0.5) * overshoot_max / 2^n_bits.
    """
    n_levels = 1 << n_bits
    overshoot_lsb = overshoot_max / n_levels
    centers = (np.arange(n_levels, dtype=np.float64) + 0.5) * overshoot_lsb
    psi = gamma * np.maximum(0.0, 1.0 - centers / thr)
    return psi.astype(np.float32)


def quantize_overshoot(
    overshoot: np.ndarray, n_bits: int, overshoot_max: float
) -> np.ndarray:
    """
    Map overshoot values -> saturating bin indices in [0, 2^n_bits - 1].

    Values >= overshoot_max saturate at the all-ones bin.
    """
    n_levels = 1 << n_bits
    overshoot_lsb = overshoot_max / n_levels
    bins = np.floor(overshoot / overshoot_lsb).astype(np.int32)
    return np.clip(bins, 0, n_levels - 1)
