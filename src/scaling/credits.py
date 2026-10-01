"""Pitt CRC service units alongside FLOPs as the compute axis.

FLOPs are the axis a paper needs: they are hardware-independent and comparable
to Chinchilla, Kaplan and Porian et al. Service units are the axis an
allocation holder reads, because they are what the sweep actually spends. Both
are wired through the analysis; FLOPs stay the default so nothing about the
publishable numbers changes.

Conversion
----------
A run's cost is its wall clock times a per-resource rate:

    SU = wall_hours * SU_PER_GPU_HOUR              (GPU partitions)
    SU = wall_hours * cpus * SU_PER_CORE_HOUR      (SMP partitions)

Wall clock comes from the calibrated model in `src/scaling/configs.py`, refit
on this sweep's measured elapsed times, so credits inherit that model's
accuracy (roughly +/- 20%) rather than being a second guess.

!! The rates below are PLACEHOLDERS. Pitt CRC publishes its own charging
!! rates per cluster and partition, and they change. Confirm them before any
!! credit figure goes in front of an allocation committee, and set them via
!! --su-per-gpu-hour / SU_PER_GPU_HOUR rather than editing this file. The
!! FLOP axis is unaffected either way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

SU_PER_GPU_HOUR = float(os.environ.get("SU_PER_GPU_HOUR", 1.0))
SU_PER_CORE_HOUR = float(os.environ.get("SU_PER_CORE_HOUR", 1.0))
RATES_CONFIRMED = os.environ.get("SU_RATES_CONFIRMED", "0") == "1"

#: The sweep ran one L40S per task on the gpu cluster.
DEFAULT_GPUS_PER_TASK = 1


@dataclass(frozen=True)
class CreditModel:
    su_per_gpu_hour: float = SU_PER_GPU_HOUR
    su_per_core_hour: float = SU_PER_CORE_HOUR
    gpus: int = DEFAULT_GPUS_PER_TASK
    confirmed: bool = RATES_CONFIRMED

    def from_wall_seconds(self, wall_sec: float) -> float:
        return wall_sec / 3600.0 * self.gpus * self.su_per_gpu_hour

    def from_passes(self, passes: float) -> float:
        """Credits for a run of `passes` examples, via the calibrated wall model."""
        from src.scaling.configs import estimate_wall_sec
        return self.from_wall_seconds(estimate_wall_sec(int(passes)))

    @property
    def caveat(self) -> str:
        if self.confirmed:
            return ""
        return ("service-unit rates are unconfirmed placeholders "
                f"({self.su_per_gpu_hour:g} SU per GPU-hour); "
                "set SU_PER_GPU_HOUR and SU_RATES_CONFIRMED=1 once checked "
                "against the CRC published rate")
