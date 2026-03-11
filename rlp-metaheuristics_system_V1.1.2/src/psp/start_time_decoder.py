"""
开始时间编码的解码器
"""
from typing import List, Tuple
import numpy as np
from .psplib_io import RCPSPInstance


class StartTimeDecoder:
    """开始时间编码解码器"""
    
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
    
    def decode(self, start_times: List[int]) -> Tuple[np.ndarray, bool]:
        """
        解码开始时间列表
        
        参数:
            start_times: 活动开始时间列表
        
        返回:
            start_times_array: 活动开始时间数组
            is_feasible: 是否可行
        """
        start_times_array = np.array(start_times, dtype=np.int32)
        
        is_feasible = self._check_feasibility(start_times_array)
        
        return start_times_array, is_feasible
    
    def _check_feasibility(self, start_times: np.ndarray) -> bool:
        """检查开始时间是否可行"""
        for j in range(self.n):
            if start_times[j] < self.es[j]:
                return False
            
            if start_times[j] + self.inst.durations[j] > self.deadline:
                return False
            
            for pred in self.inst.predecessors[j]:
                if start_times[pred] + self.inst.durations[pred] > start_times[j]:
                    return False
        
        return True
    
    def repair(self, start_times: List[int]) -> List[int]:
        """
        修复开始时间列表，确保满足优先关系约束
        
        参数:
            start_times: 活动开始时间列表
        
        返回:
            repaired_start_times: 修复后的开始时间列表
        """
        repaired = start_times.copy()
        
        for j in range(self.n):
            repaired[j] = max(repaired[j], self.es[j])
            repaired[j] = min(repaired[j], self.ls[j])
        
        for j in range(self.n):
            for pred in self.inst.predecessors[j]:
                if repaired[pred] + self.inst.durations[pred] > repaired[j]:
                    repaired[j] = repaired[pred] + self.inst.durations[pred]
        
        for j in range(self.n):
            repaired[j] = min(repaired[j], self.ls[j])
        
        return repaired
