from __future__ import annotations
from typing import Optional, List
import numpy as np
from .psplib_io import RCPSPInstance
from .objective import compute_usage_profile


class SSGSDecoder:
    def __init__(self, inst: RCPSPInstance, horizon: int):
        self.inst = inst
        self.horizon = horizon
        self.n = inst.n_activities
        self.R = inst.n_resources
        self._predecessors = inst.predecessors
        self._successors = inst.successors

    def decode(self, chromosome: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError("Subclass must implement decode method")


class SerialSSGS(SSGSDecoder):
    def __init__(self, inst: RCPSPInstance, horizon: int):
        super().__init__(inst, horizon)

    def decode(self, activity_list: List[int]) -> Optional[np.ndarray]:
        start_times = np.full(self.n, -1, dtype=np.int32)
        available = np.tile(self.inst.capacity.reshape(1, -1), (self.horizon, 1))

        for idx, j in enumerate(activity_list):
            if j < 0 or j >= self.n:
                return None

            earliest_start = 0
            for pred in self._predecessors[j]:
                if start_times[pred] < 0:
                    return None
                earliest_start = max(
                    earliest_start,
                    start_times[pred] + self.inst.durations[pred]
                )

            duration = int(self.inst.durations[j])
            if duration == 0:
                start_times[j] = earliest_start
                continue

            demand = self.inst.demands[j, :]

            t = earliest_start
            while t + duration <= self.horizon:
                if np.all(available[t:t+duration, :] >= demand):
                    start_times[j] = t
                    available[t:t+duration, :] -= demand
                    break
                t += 1

            if start_times[j] < 0:
                return None

        return start_times

    def decode_with_repair(self, activity_list: List[int]) -> Optional[np.ndarray]:
        repaired = self._repair_topological(activity_list)
        return self.decode(repaired)

    def _repair_topological(self, perm: List[int]) -> List[int]:
        n = len(perm)
        pos = {a: i for i, a in enumerate(perm)}

        indegree = [0] * n
        for i in range(n):
            for j in self._successors[i]:
                indegree[j] += 1

        ready = [i for i in range(n) if indegree[i] == 0]
        ready.sort(key=lambda x: pos[x])

        result = []
        while ready:
            v = ready.pop(0)
            result.append(v)
            for w in self._successors[v]:
                indegree[w] -= 1
                if indegree[w] == 0:
                    ready.append(w)
            ready.sort(key=lambda x: pos[x])

        if len(result) != n:
            raise ValueError("Cycle detected in precedence graph")

        return result


class ParallelSSGS(SSGSDecoder):
    def __init__(self, inst: RCPSPInstance, horizon: int, max_threads: int = 10):
        super().__init__(inst, horizon)
        self.max_threads = max_threads

    def decode(self, activity_list: List[int]) -> Optional[np.ndarray]:
        start_times = np.full(self.n, -1, dtype=np.int32)
        available = np.tile(self.inst.capacity.reshape(1, -1), (self.horizon, 1))
        completed = np.zeros(self.n, dtype=bool)
        scheduled_count = 0

        while scheduled_count < self.n:
            ready_activities = []

            for j in activity_list:
                if completed[j]:
                    continue

                pred_completed = all(completed[p] for p in self._predecessors[j])
                if pred_completed and start_times[j] < 0:
                    ready_activities.append(j)

            if not ready_activities:
                break

            scheduled_this_step = []
            for j in ready_activities:
                earliest_start = 0
                for pred in self._predecessors[j]:
                    earliest_start = max(
                        earliest_start,
                        start_times[pred] + self.inst.durations[pred]
                    )

                duration = int(self.inst.durations[j])
                if duration == 0:
                    start_times[j] = earliest_start
                    scheduled_this_step.append(j)
                    continue

                demand = self.inst.demands[j, :]

                t = earliest_start
                found = False
                while t + duration <= self.horizon:
                    if np.all(available[t:t+duration, :] >= demand):
                        start_times[j] = t
                        available[t:t+duration, :] -= demand
                        scheduled_this_step.append(j)
                        found = True
                        break
                    t += 1

                if not found:
                    return None

            for j in scheduled_this_step:
                completed[j] = True
                scheduled_count += 1

            if not scheduled_this_step and scheduled_count < self.n:
                return None

        return start_times


def create_decoder(
    inst: RCPSPInstance,
    horizon: int,
    scheme: str = "serial"
) -> SSGSDecoder:
    if scheme == "serial":
        return SerialSSGS(inst, horizon)
    elif scheme == "parallel":
        return ParallelSSGS(inst, horizon)
    else:
        raise ValueError(f"Unknown SSGS scheme: {scheme}")


def validate_schedule(
    inst: RCPSPInstance,
    start_times: np.ndarray,
    horizon: int
) -> tuple[bool, dict]:
    violations = {}

    for j in range(inst.n_activities):
        if start_times[j] < 0:
            violations[f"activity_{j}_not_scheduled"] = True

    for j in range(inst.n_activities):
        for pred in inst.predecessors[j]:
            if start_times[j] < start_times[pred] + inst.durations[pred]:
                violations[f"precedence_violation_{pred}_{j}"] = True

    usage = compute_usage_profile(inst, start_times, horizon)
    if usage is not None:
        for r in range(inst.n_resources):
            if np.any(usage[:, r] > inst.capacity[r]):
                violations[f"resource_{r}_exceeded"] = True

    makespan = int((start_times + inst.durations).max())
    if makespan > horizon:
        violations["horizon_exceeded"] = True

    is_valid = len(violations) == 0
    return is_valid, violations
