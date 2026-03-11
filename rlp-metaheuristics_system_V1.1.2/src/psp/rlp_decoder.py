from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
from .psplib_io import RCPSPInstance


class RLPDecoder:
    """
    RLP问题解码器：
    - 项目截止日期 D 是给定的
    - 活动必须满足优先关系约束
    - 活动可以在 ES 和 LS 之间延迟开始
    - LS 由截止日期 D 反向计算
    - 目标是最小化资源使用方差
    """
    
    def __init__(self, inst: RCPSPInstance, deadline: int):
        self.inst = inst
        self.n = inst.n_activities
        self.deadline = deadline
        self._predecessors = inst.predecessors
        self._successors = inst.successors
        self.es, self.ls = self._compute_time_windows()

    def _compute_time_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算每个活动的最早开始时间(ES)和最晚开始时间(LS)
        - ES: 正向计算，考虑优先约束
        - LS: 基于截止日期 D 反向计算
        """
        n = self.n
        durations = self.inst.durations

        es = np.zeros(n, dtype=np.int32)
        for j in range(n):
            for pred in self._predecessors[j]:
                es[j] = max(es[j], es[pred] + durations[pred])

        ls = np.full(n, self.deadline, dtype=np.int32)
        for j in reversed(range(n)):
            if len(self._successors[j]) == 0:
                ls[j] = self.deadline - durations[j]
            else:
                min_succ_ls = self.deadline
                for succ in self._successors[j]:
                    min_succ_ls = min(min_succ_ls, ls[succ])
                ls[j] = min_succ_ls - durations[j]

        return es, ls

    def decode(self, activity_list: List[int], delay_factors: List[float] = None) -> Tuple[np.ndarray, bool]:
        """
        解码活动列表到调度方案（带延迟因子版本）
        
        参数:
            activity_list: 活动优先级列表
            delay_factors: 每个活动的延迟因子（0-1），None 表示全部为 0（ES调度）
        
        返回:
            start_times: 活动开始时间
            is_feasible: 是否可行
        """
        if delay_factors is None:
            delay_factors = [0.0] * self.n

        start_times = np.zeros(self.n, dtype=np.int32)
        scheduled = set()

        for idx, j in enumerate(activity_list):
            if j in scheduled:
                continue

            earliest_start = self.es[j]
            latest_start = self.ls[j]

            delay_factor = delay_factors[idx] if idx < len(delay_factors) else 0.0
            delay_factor = max(0.0, min(1.0, delay_factor))
            
            start_time = int(earliest_start + delay_factor * (latest_start - earliest_start))

            for pred in self._predecessors[j]:
                if pred in scheduled:
                    start_time = max(start_time, start_times[pred] + self.inst.durations[pred])

            start_times[j] = start_time
            scheduled.add(j)

        is_feasible = np.all(start_times + self.inst.durations <= self.deadline)

        return start_times, is_feasible

    def decode_es(self, activity_list: List[int]) -> Tuple[np.ndarray, bool]:
        """
        解码活动列表到调度方案（ES版本，不带延迟因子，不考虑资源约束）
        
        调度逻辑：
        1. 依次遍历活动列表
        2. 对于每个活动，检查其紧前活动是否都已完成
        3. 如果紧前活动都完成了，则调度该活动，开始时间为最早开始时间（ES）
        4. 如果紧前活动未完成，则跳过该活动，继续调度下一个活动
        5. 当遍历完列表后，如果还有活动未调度，则重新从头遍历，直到所有活动都被调度
        
        参数:
            activity_list: 活动优先级列表
        
        返回:
            start_times: 活动开始时间
            is_feasible: 是否可行
        """
        start_times = np.zeros(self.n, dtype=np.int32)
        scheduled = set()
        remaining = list(activity_list)
        
        while len(scheduled) < self.n and remaining:
            progress_made = False
            
            for j in remaining.copy():
                if j in scheduled:
                    remaining.remove(j)
                    continue
                
                all_predecessors_scheduled = all(pred in scheduled for pred in self._predecessors[j])
                
                if all_predecessors_scheduled:
                    earliest_start = 0
                    for pred in self._predecessors[j]:
                        earliest_start = max(earliest_start, start_times[pred] + self.inst.durations[pred])
                    
                    start_times[j] = earliest_start
                    scheduled.add(j)
                    remaining.remove(j)
                    progress_made = True
                    break
            
            if not progress_made and remaining:
                break

        is_feasible = len(scheduled) == self.n and np.all(start_times + self.inst.durations <= self.deadline)

        return start_times, is_feasible

    def _repair_topological(self, perm: List[int]) -> List[int]:
        """
        修复活动列表以确保：
        1. 包含所有活动（0到n-1）
        2. 没有重复的活动
        3. 满足拓扑顺序（优先关系约束）
        """
        n = self.n
        
        seen = set()
        unique_perm = []
        for activity in perm:
            if activity not in seen and 0 <= activity < n:
                unique_perm.append(activity)
                seen.add(activity)
        
        missing = [i for i in range(n) if i not in seen]
        
        pos = {a: i for i, a in enumerate(unique_perm)}
        for activity in missing:
            pos[activity] = n + activity
        
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


def evaluate_rlp_schedule(
    inst: RCPSPInstance,
    start_times: np.ndarray,
    deadline: int,
    objective_type: str = "variance"
) -> Tuple[float, np.ndarray, bool]:
    """
    评估RLP调度方案：
    
    目标函数：最小化资源使用方差
    - u_kt = 时段 t 资源 k 的使用量
    - u̅_k = (1/D) * Σ u_kt，资源 k 的平均使用量
    - 方差 = (1/D) * Σ (u_kt - u̅_k)²
    
    参数:
        inst: 项目实例
        start_times: 活动开始时间
        deadline: 项目截止日期 D
        objective_type: 目标函数类型
    
    返回:
        obj_value: 目标函数值
        usage: 资源使用矩阵 (D × K)
        is_feasible: 是否可行
    """
    if start_times is None:
        return float('inf'), None, False

    usage = np.zeros((deadline, inst.n_resources), dtype=np.float64)

    for j in range(inst.n_activities):
        s = int(start_times[j])
        d = int(inst.durations[j])
        if d <= 0:
            continue
        e = min(s + d, deadline)
        if s < deadline:
            usage[s:e, :] += inst.demands[j, :]

    if objective_type == "variance":
        mean_usage = usage.sum(axis=0) / deadline
        variance = np.sum((usage - mean_usage) ** 2) / deadline
        obj_value = float(variance)
    elif objective_type == "peak":
        obj_value = float(usage.max())
    elif objective_type == "absolute":
        mean_usage = usage.sum(axis=0) / deadline
        obj_value = float(np.abs(usage - mean_usage).sum())
    else:
        mean_usage = usage.sum(axis=0) / deadline
        variance = np.sum((usage - mean_usage) ** 2) / deadline
        obj_value = float(variance)

    is_feasible = np.all(start_times + inst.durations <= deadline)

    return obj_value, usage, is_feasible
