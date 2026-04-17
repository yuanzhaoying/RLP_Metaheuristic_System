"""
活动列表编码（Activity List Encoding）的评估器

评估AL编码解的质量，计算目标函数值
支持延迟因子，允许活动在ES之后延迟开始
"""
from typing import List, Tuple, Optional
import numpy as np
from .psplib_io import RCPSPInstance
from .activity_list_decoder import ActivityListDecoder
from .rlp_decoder import evaluate_rlp_schedule


class ActivityListEvaluator:
    """活动列表编码评估器"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, max_evaluations: int = 1000):
        self.inst = instance
        self.deadline = deadline
        self.max_evaluations = max_evaluations
        self.n_evaluations = 0
        
        self.decoder = ActivityListDecoder(instance, deadline)
    
    def evaluate(self, activity_list: List[int], delay_factors: Optional[List[float]] = None) -> Tuple[float, bool]:
        """
        评估活动列表编码
        
        参数:
            activity_list: AL编码的活动列表
            delay_factors: 延迟因子向量（可选），范围[0, 1]
                          如果为None，则不使用延迟
        
        返回:
            objective: 目标函数值
            is_feasible: 是否可行
        """
        self.n_evaluations += 1
        
        repaired = self.decoder.repair(activity_list)
        
        start_times_array, is_feasible = self.decoder.decode(repaired, delay_factors)
        
        if not is_feasible:
            return float('inf'), False
        
        obj_value, _, _ = evaluate_rlp_schedule(self.inst, start_times_array, self.deadline)
        
        return obj_value, is_feasible
    
    def reset(self):
        """重置评估次数"""
        self.n_evaluations = 0
