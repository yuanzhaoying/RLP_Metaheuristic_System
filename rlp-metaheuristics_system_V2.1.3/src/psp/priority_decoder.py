"""
优先级编码（Priority Encoding）的解码器

PRI编码特点：
    - 编码是一个[0,1]之间的向量，每个活动有一个值
    - 第一个活动（虚拟活动）值固定为1.0（最高优先级，第一个完成）
    - 最后一个活动（虚拟活动）值固定为0.0（最低优先级，最后一个完成）
    - 该值同时表示两层含义：
      1. 活动优先级：值越大，越优先被调度
      2. 延迟因子：值越大，活动延迟越久开始
"""
from typing import List, Tuple
import numpy as np
from .psplib_io import RCPSPInstance


class PriorityDecoder:
    """优先级编码解码器"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int):
        self.inst = instance
        self.deadline = deadline
        self.n = instance.n_activities
        
        self.es, self.ls = self._compute_es_ls()
    
    def _compute_es_ls(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算每个活动的最早开始时间(ES)和最晚开始时间(LS)"""
        es = np.zeros(self.n, dtype=np.int32)
        ls = np.full(self.n, self.deadline, dtype=np.int32)
        
        for i in range(self.n):
            for pred in self.inst.predecessors[i]:
                es[i] = max(es[i], es[pred] + self.inst.durations[pred])
        
        for j in range(self.n - 1, -1, -1):
            for succ in self.inst.successors[j]:
                ls[j] = min(ls[j], ls[succ] - self.inst.durations[j])
        
        for j in range(self.n):
            ls[j] = min(ls[j], self.deadline - self.inst.durations[j])
        
        return es, ls
    
    def get_max_delay(self, activity: int) -> int:
        """获取活动的最大延迟时间（LS - ES）"""
        return max(0, self.ls[activity] - self.es[activity])
    
    def encode_random(self, rng) -> List[float]:
        """生成随机优先级编码 [0,1]"""
        priority = [rng.random() for _ in range(self.n)]
        priority[0] = 1.0
        priority[self.n - 1] = 0.0
        return priority
    
    def priority_to_activity_list(self, priority: List[float]) -> List[int]:
        """将优先级向量转换为活动列表（按优先级值从大到小排序）"""
        sorted_indices = np.argsort(-np.array(priority))
        return sorted_indices.tolist()
    
    def decode(self, priority: List[float]) -> Tuple[np.ndarray, bool]:
        """
        解码优先级编码
        
        该编码同时表示优先级和延迟因子：
        - 优先级：决定调度顺序（值越大越优先）
        - 延迟因子：决定开始时间（值越大延迟越久）
        
        参数:
            priority: 优先级向量（范围[0, 1]）
        
        返回:
            start_times_array: 活动开始时间数组
            is_feasible: 是否可行
        """
        activity_list = self.priority_to_activity_list(priority)
        
        start_times = np.zeros(self.n, dtype=np.int32)
        
        ready_activities = []
        completed = set()
        scheduled_count = 0
        
        ready_activities.append(0)
        
        while ready_activities and scheduled_count < self.n:
            ready_activities.sort(key=lambda x: priority[x], reverse=True)
            
            act = ready_activities.pop(0)
            
            pred_finish = 0
            for pred in self.inst.predecessors[act]:
                pred_finish = max(pred_finish, start_times[pred] + self.inst.durations[pred])
            
            max_delay = self.get_max_delay(act)
            delay_factor = priority[act]
            delay = int(delay_factor * max_delay)
            
            start_times[act] = max(pred_finish, self.es[act] + delay)
            
            completed.add(act)
            scheduled_count += 1
            
            for succ in self.inst.successors[act]:
                if succ not in completed:
                    all_preds_done = all(p in completed for p in self.inst.predecessors[succ])
                    if all_preds_done and succ not in ready_activities:
                        ready_activities.append(succ)
        
        is_feasible = scheduled_count == self.n and start_times[self.n-1] + self.inst.durations[self.n-1] <= self.deadline
        
        return start_times, is_feasible
    
    def repair(self, priority: List[float]) -> List[float]:
        """修复优先级编码，确保虚拟活动优先级正确"""
        repaired = priority.copy()
        for i in range(len(repaired)):
            repaired[i] = max(0.0, min(1.0, repaired[i]))
        repaired[0] = 1.0
        repaired[self.n - 1] = 0.0
        return repaired
    
    def is_valid_priority(self, priority: List[float]) -> bool:
        """检查是否是有效的优先级编码"""
        if len(priority) != self.n:
            return False
        if abs(priority[0] - 1.0) > 1e-6 or abs(priority[self.n - 1] - 0.0) > 1e-6:
            return False
        for val in priority:
            if val < 0.0 or val > 1.0:
                return False
        return True
