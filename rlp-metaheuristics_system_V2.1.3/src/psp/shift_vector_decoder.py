"""
基于位移的编码（Shift Vector Encoding）的解码器

SV编码特点：
    - 编码是一个位移向量，每个活动有一个位移值
    - 位移值范围: [0, LS-ES]
    - 实际开始时间 = ES + 位移值

编码格式：
    - displacement: List[int] - 每个活动的位移时间
    - 长度等于活动数量

解码过程：
    1. 对于每个活动j，计算开始时间 = ES[j] + displacement[j]
    2. 检查是否满足优先关系
    3. 如果不满足，修复到最早可行时间
"""
from typing import List, Tuple
import numpy as np
from .psplib_io import RCPSPInstance


class ShiftVectorDecoder:
    """基于位移的编码解码器"""
    
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
    
    def encode_random(self, rng) -> List[int]:
        """
        生成随机SV编码
        
        参数:
            rng: 随机数生成器
        
        返回:
            displacement: 位移向量
        """
        displacement = []
        for j in range(self.n):
            max_shift = max(0, self.ls[j] - self.es[j])
            if max_shift > 0:
                shift = rng.integers(0, max_shift + 1)
            else:
                shift = 0
            displacement.append(shift)
        return displacement
    
    def decode(self, displacement: List[int]) -> Tuple[np.ndarray, bool]:
        """
        解码位移向量为开始时间
        
        参数:
            displacement: 位移向量
        
        返回:
            start_times_array: 活动开始时间数组
            is_feasible: 是否可行
        """
        start_times = np.zeros(self.n, dtype=np.int32)
        
        for j in range(self.n):
            start_times[j] = self.es[j] + displacement[j]
        
        is_feasible = self._check_feasibility(start_times)
        
        if not is_feasible:
            start_times = self._repair(start_times)
            is_feasible = self._check_feasibility(start_times)
        
        return start_times, is_feasible
    
    def _check_feasibility(self, start_times: np.ndarray) -> bool:
        """检查解是否可行"""
        for j in range(self.n):
            if start_times[j] + self.inst.durations[j] > self.deadline:
                return False
            
            for pred in self.inst.predecessors[j]:
                if start_times[pred] + self.inst.durations[pred] > start_times[j]:
                    return False
        
        return True
    
    def _repair(self, start_times: np.ndarray) -> np.ndarray:
        """修复不可行解"""
        repaired = start_times.copy()
        
        for j in range(self.n):
            if self.inst.predecessors[j]:
                pred_finish = max(
                    repaired[p] + self.inst.durations[p]
                    for p in self.inst.predecessors[j]
                )
            else:
                pred_finish = 0
            
            repaired[j] = max(repaired[j], pred_finish)
            
            if repaired[j] + self.inst.durations[j] > self.deadline:
                repaired[j] = max(0, self.deadline - self.inst.durations[j])
        
        return repaired
    
    def get_max_shift(self, activity: int) -> int:
        """获取活动的最大位移值"""
        return max(0, self.ls[activity] - self.es[activity])
    
    def is_valid_displacement(self, displacement: List[int]) -> bool:
        """检查位移向量是否有效"""
        if len(displacement) != self.n:
            return False
        
        for j in range(self.n):
            max_shift = self.get_max_shift(j)
            if displacement[j] < 0 or displacement[j] > max_shift:
                return False
        
        return True
