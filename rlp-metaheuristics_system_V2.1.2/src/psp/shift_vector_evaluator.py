"""
基于位移的编码（Shift Vector Encoding）的评估器

评估SV编码解的质量，计算目标函数值
"""
from typing import List, Tuple
import numpy as np
from .psplib_io import RCPSPInstance
from .shift_vector_decoder import ShiftVectorDecoder
from .rlp_decoder import evaluate_rlp_schedule


class ShiftVectorEvaluator:
    """基于位移的编码评估器"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, max_evaluations: int = 1000):
        self.inst = instance
        self.deadline = deadline
        self.max_evaluations = max_evaluations
        self.n_evaluations = 0
        
        self.decoder = ShiftVectorDecoder(instance, deadline)
    
    def evaluate(self, displacement: List[int]) -> Tuple[float, bool]:
        """
        评估位移向量编码
        
        参数:
            displacement: SV编码的位移向量
        
        返回:
            objective: 目标函数值
            is_feasible: 是否可行
        """
        self.n_evaluations += 1
        
        start_times_array, is_feasible = self.decoder.decode(displacement)
        
        if not is_feasible:
            return float('inf'), False
        
        obj_value, _, _ = evaluate_rlp_schedule(self.inst, start_times_array, self.deadline)
        
        return obj_value, is_feasible
    
    def reset(self):
        """重置评估次数"""
        self.n_evaluations = 0
