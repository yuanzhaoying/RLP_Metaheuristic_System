"""
开始时间编码的评估器
"""
from typing import List, Tuple, Optional
import numpy as np
from .psplib_io import RCPSPInstance
from .start_time_decoder import StartTimeDecoder
from .rlp_decoder import evaluate_rlp_schedule


class StartTimeEvaluator:
    """开始时间编码评估器"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, max_evaluations: int = 1000):
        self.inst = instance
        self.deadline = deadline
        self.max_evaluations = max_evaluations
        self.n_evaluations = 0
        
        self.decoder = StartTimeDecoder(instance, deadline)
    
    def evaluate(self, start_times: List[int]) -> Tuple[float, bool]:
        """
        评估开始时间列表
        
        参数:
            start_times: 活动开始时间列表
        
        返回:
            objective: 目标函数值
            is_feasible: 是否可行
        """
        self.n_evaluations += 1
        
        repaired = self.decoder.repair(start_times)
        
        start_times_array, is_feasible = self.decoder.decode(repaired)
        
        if not is_feasible:
            return float('inf'), False
        
        obj_value, _, _ = evaluate_rlp_schedule(self.inst, start_times_array, self.deadline)
        
        return obj_value, is_feasible
    
    def reset(self):
        """重置评估次数"""
        self.n_evaluations = 0
