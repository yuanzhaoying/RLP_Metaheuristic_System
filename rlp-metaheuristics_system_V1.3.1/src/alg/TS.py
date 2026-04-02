

"""
禁忌搜索算法（Tabu Search）- 开始时间编码版本

算法简介：
    禁忌搜索是一种基于局部搜索的元启发式算法。
    通过禁忌表记录最近的移动，避免重复搜索，从而跳出局部最优。

算子汇总：
    1. 邻域生成算子（Neighborhood Generation）
       - 单点变异（Single-point Mutation）
         随机选择一个活动，在ES-LS范围内随机选择新的开始时间
    
    2. 禁忌表更新策略（Tabu Strategy）
       - "static"：静态禁忌表
         固定禁忌期限（tabu_tenure），FIFO队列，当禁忌表满时移除最早的移动
       
       - "dynamic"：动态禁忌表
         初始禁忌期限为n（活动数量），动态调整范围从均匀分布(√n, 4√n)中选取
         当连续无改进次数超过阈值时调整禁忌期限
    
    3. 特赦准则（Aspiration Criterion）
       - True：启用特赦准则
         如果邻域解优于全局最优，即使禁忌也接受
       
       - False：不启用特赦准则
         严格遵守禁忌表约束
"""

import math
from typing import List, Tuple
import time
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator


class TabuList:
    """禁忌表管理类（支持两种更新策略）"""
    
    def __init__(self, n_activities: int, strategy: str = "static", 
                 initial_tenure: int = 10, min_factor: int = 1, max_factor: int = 4):
        """
        初始化禁忌表
        
        参数:
            n_activities: 活动数量
            strategy: 策略类型 ("static" 或 "dynamic")
            initial_tenure: 初始禁忌期限（静态策略使用）
            min_factor: 最小因子（动态策略使用）
            max_factor: 最大因子（动态策略使用）
        """
        self.n_activities = n_activities
        self.strategy = strategy
        self.min_factor = min_factor
        self.max_factor = max_factor
        
        if strategy == "static":
            self.tabu_list = []
            self.tabu_set = set()
            self.tenure = initial_tenure
        else:
            self.tabu_dict = {}
            self.min_length = int(min_factor * math.sqrt(n_activities))
            self.max_length = int(max_factor * math.sqrt(n_activities))
            self.length = n_activities
    
    def is_tabu(self, move, current_iteration: int = None) -> bool:
        """检查移动是否在禁忌表中"""
        if self.strategy == "static":
            return move in self.tabu_set
        else:
            return self.tabu_dict.get(move, 0) > current_iteration
    
    def add_move(self, move, current_iteration: int = None):
        """将移动加入禁忌表"""
        if self.strategy == "static":
            if len(self.tabu_list) >= self.tenure:
                old_move = self.tabu_list.pop(0)
                self.tabu_set.discard(old_move)
            self.tabu_list.append(move)
            self.tabu_set.add(move)
        else:
            self.tabu_dict[move] = current_iteration + self.length
    
    def update_length(self):
        """动态更新禁忌表长度（仅动态策略使用）"""
        if self.strategy == "dynamic":
            self.length = np.random.randint(self.min_length, self.max_length + 1)


@dataclass
class TSParamsST:
    """禁忌搜索参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    tabu_tenure: int = 10
    neighborhood_size: int = 20
    time_limit: float = 60.0
    max_iterations: int = 100
    aspiration_criterion: bool = True
    tabu_strategy: str = "static"
    min_factor: int = 1
    max_factor: int = 4
    noimprove_threshold: int = 10


class TabuSearchST:
    """禁忌搜索算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: TSParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.tabu_list = TabuList(
            n_activities=self.n,
            strategy=params.tabu_strategy,
            initial_tenure=params.tabu_tenure,
            min_factor=params.min_factor,
            max_factor=params.max_factor
        )
    
    def run(self):
        """运行禁忌搜索算法"""
        start_time = time.time()
        convergence = []
        
        current = self._initialize_solution()
        current_obj, _ = self.evaluator.evaluate(current)
        
        best_start_times = current.copy()
        best_objective = current_obj
        
        nr_noimprove = 0
        
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
                
                is_tabu = self.tabu_list.is_tabu(move, iteration)
                
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
                
                self.tabu_list.add_move(best_move, iteration)
                
                if current_obj < best_objective:
                    best_objective = current_obj
                    best_start_times = current.copy()
                    nr_noimprove = 0
                else:
                    nr_noimprove += 1
            else:
                nr_noimprove += 1
            
            if self.params.tabu_strategy == "dynamic":
                if nr_noimprove >= self.params.noimprove_threshold:
                    self.tabu_list.update_length()
                    nr_noimprove = 0
            
            convergence.append(best_objective)
            iteration += 1
        
        runtime = time.time() - start_time
        
        from .start_time_algorithms import AlgorithmResult, _params_to_dict
        return AlgorithmResult(
            best_start_times=best_start_times,
            best_objective=best_objective,
            n_evaluations=self.evaluator.n_evaluations,
            runtime=runtime,
            convergence=convergence,
            algorithm_params=_params_to_dict(self.params)
        )
    
    def _initialize_solution(self) -> List[int]:
        """初始化解"""
        solution = []
        for j in range(self.n):
            start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
            solution.append(start_time)
        return solution
    
    def _generate_neighbors(self, solution: List[int]) -> List[Tuple[List[int], Tuple]]:
        """生成邻域解"""
        neighbors = []
        
        for _ in range(self.params.neighborhood_size):
            neighbor = solution.copy()
            j = self.rng.integers(0, self.n)
            new_start = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
            
            if new_start != solution[j]:
                move = (j, solution[j], new_start)
                neighbor[j] = new_start
                neighbors.append((neighbor, move))
        
        return neighbors


import numpy as np
