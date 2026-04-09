"""
禁忌搜索算法（Tabu Search）- 基于位移编码版本

算子汇总：
    1. 禁忌表更新策略
       - static：静态禁忌表
       - dynamic：动态禁忌表
"""

import math
from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.shift_vector_evaluator import ShiftVectorEvaluator
from ..psp.shift_vector_decoder import ShiftVectorDecoder
from .operators import RandomGenerator


@dataclass
class TSParamsSV:
    """禁忌搜索参数（位移编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    tabu_tenure: int = 10
    neighborhood_size: int = 20
    time_limit: float = 60.0
    max_iterations: int = 100
    aspiration_criterion: bool = True
    tabu_strategy: str = "static"


class TabuSearchSV:
    """禁忌搜索算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: TSParamsSV):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ShiftVectorEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ShiftVectorDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.tabu_list = []
        self.tabu_set = set()
        self.iteration = 0
        self.no_improvement_count = 0
    
    def _get_dynamic_tenure(self) -> int:
        """获取动态禁忌期限"""
        if self.params.tabu_strategy == "static":
            return self.params.tabu_tenure
        
        base_tenure = self.params.tabu_tenure
        
        oscillation = int(3 * math.sin(self.iteration * 0.3))
        
        improvement_factor = min(self.no_improvement_count // 3, 5)
        
        dynamic_tenure = base_tenure + oscillation + improvement_factor
        
        return max(3, min(dynamic_tenure, base_tenure * 3))
    
    def run(self):
        """运行禁忌搜索算法"""
        start_time = time.time()
        convergence = []
        
        current = self._initialize_solution()
        current_obj, _ = self.evaluator.evaluate(current)
        
        best_displacement = current.copy()
        best_objective = current_obj
        
        iteration = 0
        while (iteration < self.params.max_iterations and 
               self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            neighbors = self._generate_neighbors(current)
            
            best_neighbor = None
            best_neighbor_obj = float('inf')
            best_move = None
            
            for neighbor, move in neighbors:
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                neighbor_obj, _ = self.evaluator.evaluate(neighbor)
                
                is_tabu = move in self.tabu_set
                
                if self.params.aspiration_criterion and neighbor_obj < best_objective:
                    if neighbor_obj < best_neighbor_obj:
                        best_neighbor = neighbor.copy()
                        best_neighbor_obj = neighbor_obj
                        best_move = move
                elif not is_tabu:
                    if neighbor_obj < best_neighbor_obj:
                        best_neighbor = neighbor.copy()
                        best_neighbor_obj = neighbor_obj
                        best_move = move
            
            if best_neighbor is not None:
                current = best_neighbor
                current_obj = best_neighbor_obj
                
                self._add_to_tabu(best_move)
                
                if current_obj < best_objective:
                    best_objective = current_obj
                    best_displacement = current.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            
            convergence.append(best_objective)
            iteration += 1
            self.iteration = iteration
        
        start_times, _ = self.decoder.decode(best_displacement)
        
        runtime = time.time() - start_time
        
        return {
            'best_displacement': best_displacement,
            'best_start_times': start_times.tolist(),
            'best_objective': best_objective,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': self._params_to_dict(self.params)
        }
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
    
    def _initialize_solution(self) -> List[int]:
        """初始化解"""
        return self.decoder.encode_random(self.rng)
    
    def _generate_neighbors(self, solution: List[int]) -> List[Tuple[List[int], Tuple]]:
        """生成邻域解"""
        neighbors = []
        
        for _ in range(self.params.neighborhood_size):
            neighbor = solution.copy()
            
            j = self.rng.integers(0, self.n)
            max_shift = self.decoder.get_max_shift(j)
            new_value = self.rng.integers(0, max_shift + 1)
            
            neighbor[j] = new_value
            move = (j, new_value)
            
            neighbors.append((neighbor, move))
        
        return neighbors
    
    def _add_to_tabu(self, move: Tuple):
        """将移动加入禁忌表"""
        dynamic_tenure = self._get_dynamic_tenure()
        while len(self.tabu_list) >= dynamic_tenure:
            old_move = self.tabu_list.pop(0)
            self.tabu_set.discard(old_move)
        
        self.tabu_list.append(move)
        self.tabu_set.add(move)
