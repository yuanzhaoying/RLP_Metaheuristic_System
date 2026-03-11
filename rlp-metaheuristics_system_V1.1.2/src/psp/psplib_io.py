from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import os


@dataclass
class RCPSPInstance:
    name: str
    n_activities: int
    durations: np.ndarray
    demands: np.ndarray
    capacity: np.ndarray
    successors: List[List[int]]
    predecessors: List[List[int]] = field(default_factory=list)
    n_resources: int = 0
    mode: int = 1

    def __post_init__(self):
        if self.n_resources == 0:
            self.n_resources = self.demands.shape[1]
        if not self.predecessors:
            self.predecessors = self._compute_predecessors()

    def _compute_predecessors(self) -> List[List[int]]:
        pred = [[] for _ in range(self.n_activities)]
        for i, succ_list in enumerate(self.successors):
            for j in succ_list:
                if j < len(pred):
                    pred[j].append(i)
        return pred


def load_psplib_sm(file_path: str) -> RCPSPInstance:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instance file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.sm', '.rcp']:
        return _parse_rcp_format(file_path)

    try:
        from psplib import parse
        inst = parse(file_path, instance_format="psplib")
        return _parse_psplib_instance(inst, file_path)
    except ImportError:
        return _parse_rcp_format(file_path)


def _parse_rcp_format(file_path: str) -> RCPSPInstance:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]

    name = os.path.basename(file_path)

    if len(lines) < 2:
        raise ValueError(f"Invalid RCP file: {file_path}")

    first_line_parts = lines[0].split()
    n_activities = int(first_line_parts[0])
    n_resources = int(first_line_parts[1])

    second_line_parts = lines[1].split()
    capacity = np.array([int(x) for x in second_line_parts[:n_resources]], dtype=np.int32)

    durations = np.zeros(n_activities, dtype=np.int32)
    demands = np.zeros((n_activities, n_resources), dtype=np.int32)
    successors = [[] for _ in range(n_activities)]

    for line_idx in range(2, min(len(lines), n_activities + 2)):
        parts = lines[line_idx].split()
        if len(parts) < 2:
            continue

        try:
            activity_idx = line_idx - 2
            if activity_idx < 0 or activity_idx >= n_activities:
                continue

            durations[activity_idx] = int(parts[0])

            for r in range(min(n_resources, len(parts) - 2)):
                demands[activity_idx, r] = int(parts[1 + r])

            n_successors = 0
            if len(parts) > 1 + n_resources:
                try:
                    n_successors = int(parts[1 + n_resources])
                except:
                    n_successors = 0

            successor_start = 2 + n_resources
            if n_successors > 0 and len(parts) > successor_start:
                for i in range(min(n_successors, len(parts) - successor_start)):
                    try:
                        succ_id = int(parts[successor_start + i])
                        if 1 <= succ_id <= n_activities:
                            successors[activity_idx].append(succ_id - 1)
                    except ValueError:
                        pass

        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing line {line_idx}: {e}")
            pass

    return RCPSPInstance(
        name=name,
        n_activities=n_activities,
        durations=durations,
        demands=demands,
        capacity=capacity,
        successors=successors,
    )


def _parse_psplib_instance(inst, file_path: str) -> RCPSPInstance:
    n = inst.num_activities
    R = inst.num_resources

    durations = np.zeros(n, dtype=np.int32)
    demands = np.zeros((n, R), dtype=np.int32)
    successors = [[] for _ in range(n)]

    for j, act in enumerate(inst.activities):
        m = act.modes[0]
        durations[j] = int(m.duration)
        demands[j, :] = np.array(m.demands, dtype=np.int32)
        successors[j] = list(act.successors)

    capacity = np.array([r.capacity for r in inst.resources], dtype=np.int32)
    name = os.path.basename(file_path)

    return RCPSPInstance(
        name=name,
        n_activities=n,
        durations=durations,
        demands=demands,
        capacity=capacity,
        successors=successors,
    )


def _parse_psplib_manual(file_path: str) -> RCPSPInstance:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    name = os.path.basename(file_path)
    n_activities = 0
    n_resources = 0
    durations = None
    demands = None
    capacity = None
    successors = None

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*'):
            continue

        if line.startswith('**'):
            section = line.replace('**', '').strip()
            continue

        if section == 'PROJECT' or section == 'NFO':
            parts = line.split()
            if len(parts) >= 2:
                if 'ACTIVITIES' in line.upper():
                    n_activities = int(parts[0])
                    durations = np.zeros(n_activities, dtype=np.int32)
                    demands = np.zeros((n_activities, 4), dtype=np.int32)
                    successors = [[] for _ in range(n_activities)]
                elif 'RESOURCES' in line.upper():
                    n_resources = int(parts[0])
                    capacity = np.zeros(n_resources, dtype=np.int32)

        elif section == 'PRECEDENCE RELATIONS':
            parts = line.split()
            if len(parts) >= 2:
                try:
                    job_id = int(parts[0]) - 1
                    if job_id >= 0 and job_id < n_activities:
                        if len(parts) >= 2:
                            durations[job_id] = int(parts[1])
                        if len(parts) >= 2 + n_resources:
                            for r in range(n_resources):
                                demands[job_id, r] = int(parts[2 + r])
                        if len(parts) > 2 + n_resources:
                            for i in range(3 + n_resources, len(parts)):
                                succ_id = int(parts[i]) - 1
                                if 0 <= succ_id < n_activities:
                                    successors[job_id].append(succ_id)
                except (ValueError, IndexError):
                    pass

        elif section == 'RESOURCE REQUIREMENTS':
            parts = line.split()
            if len(parts) >= 2:
                try:
                    job_id = int(parts[0]) - 1
                    if job_id >= 0 and job_id < n_activities and n_resources > 0:
                        for r in range(min(n_resources, len(parts) - 1)):
                            demands[job_id, r] = int(parts[1 + r])
                except (ValueError, IndexError):
                    pass

    if capacity is None and n_resources > 0:
        capacity = np.ones(n_resources, dtype=np.int32) * 10

    return RCPSPInstance(
        name=name,
        n_activities=n_activities,
        durations=durations,
        demands=demands,
        capacity=capacity,
        successors=successors,
    )


def load_psplib_directory(directory: str, pattern: str = "*.sm") -> List[RCPSPInstance]:
    import glob
    instances = []
    files = sorted(glob.glob(os.path.join(directory, pattern)))

    for file_path in files:
        try:
            inst = load_psplib_sm(file_path)
            instances.append(inst)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return instances


def get_instance_stats(inst: RCPSPInstance) -> Dict:
    return {
        "name": inst.name,
        "n_activities": inst.n_activities,
        "n_resources": inst.n_resources,
        "total_duration": int(inst.durations.sum()),
        "avg_duration": float(inst.durations.mean()),
        "total_demand": int(inst.demands.sum()),
        "resource_types": list(inst.capacity),
    }
