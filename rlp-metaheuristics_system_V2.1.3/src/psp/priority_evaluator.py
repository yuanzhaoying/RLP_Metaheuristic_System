"""
优先级编码的评估器

评估优先级编码对应的调度方案的目标函数值
"""
from typing import Tuple
import numpy as np
from .psplib_io import RCPSPInstance
from .priority_decoder import PriorityDecoder
from .rlp_decoder import evaluate_rlp_schedule


class PriorityEvaluator:
    """优先级编码评估器"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, max_evaluations: int = 10000):
        self.inst = instance
        self.deadline = deadline
        self.max_evaluations = max_evaluations
        self.n_evaluations = 0
        
        self.decoder = PriorityDecoder(instance, deadline)
    
    def evaluate(self, priority) -> Tuple[float, bool]:
        """
        评估优先级编码
        
        参数:
            priority: 优先级向量（范围[0, 1]）
                      该值同时表示优先级和延迟因子
        
        返回:
            objective: 目标函数值
            is_feasible: 是否可行
        """
        self.n_evaluations += 1
        
        start_times, is_feasible = self.decoder.decode(priority)
        
        if not is_feasible:
            return float('inf'), False
        
        objective, _, _ = evaluate_rlp_schedule(self.inst, start_times, self.deadline)
        
        return objective, True
    
    def reset(self):
        """重置评估计数器"""
        self.n_evaluations = 0
