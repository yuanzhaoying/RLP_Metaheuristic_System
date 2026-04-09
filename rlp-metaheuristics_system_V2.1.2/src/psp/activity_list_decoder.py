"""
活动列表编码（Activity List Encoding）的解码器

AL编码特点：
    - 编码是一个活动排列（排列编码）
    - 解码时按列表顺序调度活动，满足优先关系
    - 支持延迟因子，允许活动在ES之后延迟开始

编码格式：
    - activity_list: List[int] - 活动ID的排列，如 [0, 3, 1, 2, 4, ...]
    - delay_factors: List[float] - 延迟因子向量，范围[0, 1]
    - 首尾固定为虚拟活动（0和n-1）

解码过程：
    1. 按列表顺序遍历活动
    2. 对于每个活动，计算ES（最早开始时间）
    3. 应用延迟因子：实际开始时间 = ES + delay_factor * (LS - ES)
    4. 确保满足优先关系
"""
from typing import List, Tuple, Optional
import numpy as np
from .psplib_io import RCPSPInstance


class ActivityListDecoder:
    """活动列表编码解码器"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int):
        self.inst = instance
        self.deadline = deadline
        self.n = instance.n_activities
        
        self.es, self.ls = self._compute_time_windows()
    
    def _compute_time_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算每个活动的ES和LS时间窗口"""
        es = np.zeros(self.n, dtype=np.int32)
        ls = np.zeros(self.n, dtype=np.int32)
        
        for j in range(self.n):
            for pred in self.inst.predecessors[j]:
                es[j] = max(es[j], es[pred] + self.inst.durations[pred])
        
        critical_path_length = max([es[i] + self.inst.durations[i] for i in range(self.n)])
        
        for j in range(self.n - 1, -1, -1):
            min_successor_ls = self.deadline
            for succ in range(self.n):
                if j in self.inst.predecessors[succ]:
                    min_successor_ls = min(min_successor_ls, ls[succ] - self.inst.durations[j])
            ls[j] = min_successor_ls if min_successor_ls < self.deadline else self.deadline
        
        for j in range(self.n):
            ls[j] = min(ls[j], self.deadline - self.inst.durations[j])
        
        return es, ls
    
    def get_max_delay(self, activity: int) -> int:
        """获取活动的最大延迟时间（LS - ES）"""
        return max(0, self.ls[activity] - self.es[activity])
    
    def encode_random(self, rng) -> List[int]:
        """
        生成随机AL编码
        
        参数:
            rng: 随机数生成器
        
        返回:
            activity_list: 活动列表编码
        """
        VIRTUAL_START = 0
        VIRTUAL_END = self.n - 1
        
        middle_acts = [i for i in range(self.n) if i not in [VIRTUAL_START, VIRTUAL_END]]
        middle_acts = rng.shuffle(middle_acts)
        
        return [VIRTUAL_START] + middle_acts + [VIRTUAL_END]
    
    def encode_random_with_delay(self, rng) -> Tuple[List[int], List[float]]:
        """
        生成随机AL编码和延迟因子
        
        参数:
            rng: 随机数生成器
        
        返回:
            activity_list: 活动列表编码
            delay_factors: 延迟因子向量（范围[0, 1]）
        """
        activity_list = self.encode_random(rng)
        delay_factors = [rng.random() for _ in range(self.n)]
        delay_factors[0] = 0.0
        delay_factors[self.n - 1] = 0.0
        
        return activity_list, delay_factors
    
    def encode_forward(self) -> List[int]:
        """
        生成前向编码（按拓扑排序）
        
        返回:
            activity_list: 活动列表编码
        """
        in_degree = [0] * self.n
        adj = [[] for _ in range(self.n)]
        
        for i in range(self.n):
            for j in self.inst.predecessors[i]:
                adj[j].append(i)
                in_degree[i] += 1
        
        from collections import deque
        queue = deque([i for i in range(self.n) if in_degree[i] == 0])
        topo_order = []
        
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        return topo_order
    
    def decode(self, activity_list: List[int], delay_factors: Optional[List[float]] = None) -> Tuple[np.ndarray, bool]:
        """
        解码活动列表编码
        
        参数:
            activity_list: AL编码的活动列表
            delay_factors: 延迟因子向量（可选），范围[0, 1]
                          如果为None，则不使用延迟（所有活动按ES开始）
        
        返回:
            start_times_array: 活动开始时间数组
            is_feasible: 是否可行
        """
        start_times = np.zeros(self.n, dtype=np.int32)
        scheduled = set()
        
        for act in activity_list:
            if act in scheduled:
                continue
            
            if self.inst.predecessors[act]:
                pred_finish = max(
                    start_times[p] + self.inst.durations[p]
                    for p in self.inst.predecessors[act]
                )
            else:
                pred_finish = 0
            
            if delay_factors is not None and 0 <= act < len(delay_factors):
                max_delay = self.get_max_delay(act)
                delay = int(delay_factors[act] * max_delay)
                start_times[act] = max(pred_finish, self.es[act] + delay)
            else:
                start_times[act] = pred_finish
            
            scheduled.add(act)
        
        is_feasible = self._check_feasibility(start_times, activity_list)
        
        return start_times, is_feasible
    
    def _check_feasibility(self, start_times: np.ndarray, activity_list: List[int]) -> bool:
        """检查解是否可行"""
        for j in range(self.n):
            if start_times[j] + self.inst.durations[j] > self.deadline:
                return False
            
            for pred in self.inst.predecessors[j]:
                if start_times[pred] + self.inst.durations[pred] > start_times[j]:
                    return False
        
        if len(set(activity_list)) != self.n:
            return False
        
        return True
    
    def repair(self, activity_list: List[int]) -> List[int]:
        """
        修复活动列表编码，确保满足优先关系
        
        参数:
            activity_list: AL编码的活动列表
        
        返回:
            repaired_list: 修复后的活动列表
        """
        VIRTUAL_START = 0
        VIRTUAL_END = self.n - 1
        
        scheduled = set()
        repaired = []
        
        remaining = set(range(self.n))
        
        while remaining:
            candidates = []
            for act in remaining:
                preds = set(self.inst.predecessors[act])
                if preds.issubset(scheduled):
                    candidates.append(act)
            
            if not candidates:
                for act in remaining:
                    if act not in scheduled:
                        candidates.append(act)
                if not candidates:
                    break
            
            if VIRTUAL_START in candidates and VIRTUAL_START not in scheduled:
                repaired.append(VIRTUAL_START)
                scheduled.add(VIRTUAL_START)
                remaining.discard(VIRTUAL_START)
                continue
            
            if VIRTUAL_END in candidates and len(remaining) == 1:
                repaired.append(VIRTUAL_END)
                scheduled.add(VIRTUAL_END)
                remaining.discard(VIRTUAL_END)
                continue
            
            for act in activity_list:
                if act in candidates and act not in scheduled:
                    if act == VIRTUAL_END and len(remaining) > 1:
                        continue
                    repaired.append(act)
                    scheduled.add(act)
                    remaining.discard(act)
                    break
            else:
                if candidates:
                    act = candidates[0]
                    repaired.append(act)
                    scheduled.add(act)
                    remaining.discard(act)
        
        return repaired
    
    def is_valid_permutation(self, activity_list: List[int]) -> bool:
        """检查是否是有效的排列"""
        if len(activity_list) != self.n:
            return False
        
        if set(activity_list) != set(range(self.n)):
            return False
        
        return True
    
    def respects_precedence(self, activity_list: List[int]) -> bool:
        """检查排列是否满足优先关系（活动在其所有前驱之后）"""
        position = {act: i for i, act in enumerate(activity_list)}
        
        for act in range(self.n):
            for pred in self.inst.predecessors[act]:
                if position[pred] >= position[act]:
                    return False
        
        return True
