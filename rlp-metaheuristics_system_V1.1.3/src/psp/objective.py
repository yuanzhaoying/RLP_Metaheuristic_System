from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .psplib_io import RCPSPInstance


@dataclass
class ScheduleResult:
    start_times: Optional[np.ndarray]
    makespan: int
    is_feasible: bool
    objective_value: float
    resource_usage: Optional[np.ndarray] = None
    n_violations: int = 0
    violation_details: dict = None

    def __post_init__(self):
        if self.violation_details is None:
            self.violation_details = {}


def compute_usage_profile(
    inst: RCPSPInstance,
    start_times: np.ndarray,
    horizon: int
) -> Optional[np.ndarray]:
    n, R = inst.n_activities, inst.n_resources
    usage = np.zeros((horizon, R), dtype=np.int32)

    for j in range(n):
        s = int(start_times[j])
        d = int(inst.durations[j])
        if s < 0 or d <= 0:
            continue
        e = s + d
        if e > horizon:
            return None
        usage[s:e, :] += inst.demands[j, :]

    return usage


def leveling_variance(usage: np.ndarray, capacities: np.ndarray) -> float:
    mean_usage = usage.mean(axis=0, keepdims=True)
    variance = ((usage - mean_usage) ** 2).sum()
    return float(variance)


def leveling_peak(usage: np.ndarray, capacities: np.ndarray) -> float:
    peaks = usage.max(axis=0)
    return float(peaks.sum())


def leveling_absolute(usage: np.ndarray, capacities: np.ndarray) -> float:
    mean_usage = usage.mean(axis=0, keepdims=True)
    absolute_dev = np.abs(usage - mean_usage).sum()
    return float(absolute_dev)


def weighted_objective(
    usage: np.ndarray,
    capacities: np.ndarray,
    weights: Optional[dict] = None
) -> float:
    if weights is None:
        weights = {"variance": 1.0, "peak": 0.0, "absolute": 0.0}

    value = 0.0
    if weights.get("variance", 0) > 0:
        value += weights["variance"] * leveling_variance(usage, capacities)
    if weights.get("peak", 0) > 0:
        value += weights["peak"] * leveling_peak(usage, capacities)
    if weights.get("absolute", 0) > 0:
        value += weights["absolute"] * leveling_absolute(usage, capacities)

    return value


def evaluate_schedule(
    inst: RCPSPInstance,
    start_times: np.ndarray,
    horizon: int,
    objective_type: str = "variance"
) -> ScheduleResult:
    if start_times is None:
        return ScheduleResult(
            start_times=None,
            makespan=0,
            is_feasible=False,
            objective_value=float('inf')
        )

    usage = compute_usage_profile(inst, start_times, horizon)

    if usage is None:
        return ScheduleResult(
            start_times=start_times,
            makespan=0,
            is_feasible=False,
            objective_value=float('inf'),
            n_violations=1,
            violation_details={"horizon_exceeded": True}
        )

    capacity_2d = inst.capacity.reshape(1, -1)
    resource_violations = np.sum(usage > capacity_2d)
    n_violations = int(resource_violations)

    if n_violations > 0:
        return ScheduleResult(
            start_times=start_times,
            makespan=int((start_times + inst.durations).max()),
            is_feasible=False,
            objective_value=float('inf'),
            resource_usage=usage,
            n_violations=n_violations,
            violation_details={"resource_exceeded": True}
        )

    if objective_type == "variance":
        obj_value = leveling_variance(usage, inst.capacity)
    elif objective_type == "peak":
        obj_value = leveling_peak(usage, inst.capacity)
    elif objective_type == "absolute":
        obj_value = leveling_absolute(usage, inst.capacity)
    else:
        obj_value = leveling_variance(usage, inst.capacity)

    makespan = int((start_times + inst.durations).max())

    return ScheduleResult(
        start_times=start_times,
        makespan=makespan,
        is_feasible=True,
        objective_value=obj_value,
        resource_usage=usage,
        n_violations=0
    )


def compute_makespan(start_times: np.ndarray, durations: np.ndarray) -> int:
    return int((start_times + durations).max())


def compute_resource_utilization(
    usage: np.ndarray,
    capacity: np.ndarray,
    horizon: int
) -> dict:
    utilization = np.zeros(len(capacity))
    for r in range(len(capacity)):
        if capacity[r] > 0:
            utilization[r] = usage[:, r].sum() / (capacity[r] * horizon)

    return {
        "mean": float(utilization.mean()),
        "max": float(utilization.max()),
        "min": float(utilization.min()),
        "per_resource": utilization.tolist()
    }
