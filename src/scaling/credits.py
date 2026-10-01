"""Pitt CRC service units alongside FLOPs as the compute axis.

FLOPs are the axis a paper needs: they are hardware-independent and comparable
to Chinchilla, Kaplan and Porian et al. Service units are the axis an
allocation holder reads, because they are what the sweep actually spends. Both
are wired through the analysis; FLOPs stay the default so nothing about the
publishable numbers changes.

Conversion
----------
CRC bills the MAXIMUM of the weighted resources, not their sum, times walltime
(CRCD User Manual, "Service Units"):

    SU = max(cores * compute_weight, GB * memory_weight,
             GPUs * compute_weight) * hours

The published weights for the gpu cluster's `l40s` partition are a compute
weight of 8 and a memory weight of 0, so one L40S costs 8 SU per hour. The SMP
partition is 0.8 per core.

Wall clock comes from the wall model in `src/scaling/configs.py`, evaluated at
the MEASURED throughput rather than the conservative rate the planner sizes
with. A charge is linear in elapsed time, so the typical rate gives the
expected cost; the p10 rate is right for "does it fit the wall" and wrong for
"what does it cost". Note the startup and DeltaE eval terms are fixed per run,
so the cost is affine in D, not proportional to it.

!! One weight is still unconfirmed. The manual's table lists 8 "per CPU/GPU"
!! and its GPU example elides the core term. If cores on l40s carry the same
!! weight, a job with --cpus-per-task=8 bills max(8 * 8, 1 * 8) = 64 SU per
!! hour, not 8. Check with `scontrol -M gpu show partition l40s` (read
!! TRESBillingWeights) or `sacct -X -M gpu -j <job> --format=AllocTRES%60`
!! (read billing=) on a finished sweep job, then set SU_RATES_CONFIRMED=1.
!! Override the rate via --su-per-gpu-hour / SU_PER_GPU_HOUR, not by editing
!! this file. The FLOP axis is unaffected either way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

#: l40s compute weight, CRCD User Manual "Service Units" (TRES billing weights).
SU_PER_GPU_HOUR = float(os.environ.get("SU_PER_GPU_HOUR", 8.0))
#: smp compute weight, same table.
SU_PER_CORE_HOUR = float(os.environ.get("SU_PER_CORE_HOUR", 0.8))
RATES_CONFIRMED = os.environ.get("SU_RATES_CONFIRMED", "0") == "1"

#: The sweep ran one L40S per task on the gpu cluster.
DEFAULT_GPUS_PER_TASK = 1


def _billing_rate() -> float:
    """Examples per second used for pricing; SU_EXAMPLES_PER_SEC overrides."""
    from src.scaling.configs import MEASURED_EXAMPLES_PER_SEC
    return float(os.environ.get("SU_EXAMPLES_PER_SEC", MEASURED_EXAMPLES_PER_SEC))


@dataclass(frozen=True)
class CreditModel:
    su_per_gpu_hour: float = SU_PER_GPU_HOUR
    su_per_core_hour: float = SU_PER_CORE_HOUR
    gpus: int = DEFAULT_GPUS_PER_TASK
    confirmed: bool = RATES_CONFIRMED

    def from_wall_seconds(self, wall_sec: float) -> float:
        return wall_sec / 3600.0 * self.gpus * self.su_per_gpu_hour

    def from_passes(self, passes: float) -> float:
        """Credits for a run of `passes` examples, at the measured throughput."""
        from src.scaling.configs import estimate_wall_sec
        return self.from_wall_seconds(
            estimate_wall_sec(int(passes), rate=_billing_rate()))

    @property
    def caveat(self) -> str:
        if self.confirmed:
            return ""
        return (f"service units at {self.su_per_gpu_hour:g} SU per GPU-hour "
                "(the published l40s weight) assume cores are not billed at "
                "the same weight; confirm billing= with sacct on a finished "
                "job, then set SU_RATES_CONFIRMED=1")
