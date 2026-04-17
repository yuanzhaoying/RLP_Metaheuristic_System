"""
禁忌搜索算法（Tabu Search）- 活动列表编码版本

算法简介：
    禁忌搜索是一种基于局部搜索的元启发式算法。
    通过禁忌表记录最近的移动，避免重复搜索，从而跳出局部最优。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 邻域操作使用排列专用的swap、insertion、inversion

算子汇总：
    1. 邻域生成算子
       - swap：交换两个活动
       - insertion：插入操作
       - inversion：逆序操作
    
    2. 禁忌表更新策略
       - static：静态禁忌表
       - dynamic：动态禁忌表
"""

import math
from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.activity_list_evaluator import ActivityListEvaluator
from ..psp.activity_list_decoder import ActivityListDecoder
from .operators import RandomGenerator


class TabuListAL:
    """禁忌表管理类（AL编码版本）"""
    
    def __init__(self, n_activities: int, strategy: str = "static", 
                 initial_tenure: int = 10, min_factor: int = 1, max_factor: int = 4):
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
        """动态更新禁忌表长度"""
        if self.strategy == "dynamic":
            self.length = np.random.randint(self.min_length, self.max_length + 1)


@dataclass
class TSParamsAL:
    """禁忌搜索参数（活动列表编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    tabu_tenure: int = 10
    neighborhood_size: int = 20
    time_limit: float = 60.0
    max_iterations: int = 100
    aspiration_criterion: bool = True
    tabu_strategy: str = "static"
    neighborhood_type: str = "swap"
    min_factor: int = 1
    max_factor: int = 4
    noimprove_threshold: int = 10
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class TabuSearchAL:
    """禁忌搜索算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: TSParamsAL):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ActivityListEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ActivityListDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.tabu_list = TabuListAL(
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
        
        current, current_delay = self._initialize_solution()
        
        if self.params.use_delay_factors and current_delay:
            current_obj, _ = self.evaluator.evaluate(current, current_delay)
        else:
            current_obj, _ = self.evaluator.evaluate(current)
        
        best_activity_list = current.copy()
        best_delay_factors = current_delay.copy() if current_delay else None
        best_objective = current_obj
        
        nr_noimprove = 0
        
        iteration = 0
        while (iteration < self.params.max_iterations and 
               self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            neighbors = self._generate_neighbors(current, current_delay)
            
            best_neighbor = None
            best_neighbor_obj = float('inf')
            best_move = None
            best_neighbor_delay = None
            
            for neighbor, neighbor_delay, move in neighbors:
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                if self.params.use_delay_factors and neighbor_delay:
                    neighbor_obj, _ = self.evaluator.evaluate(neighbor, neighbor_delay)
                else:
                    neighbor_obj, _ = self.evaluator.evaluate(neighbor)
                
                is_tabu = self.tabu_list.is_tabu(move, iteration)
                
                if self.params.aspiration_criterion and neighbor_obj < best_objective:
                    if neighbor_obj < best_neighbor_obj:
                        best_neighbor = neighbor.copy()
                        best_neighbor_obj = neighbor_obj
                        best_move = move
                        best_neighbor_delay = neighbor_delay.copy() if neighbor_delay else None
                elif not is_tabu:
                    if neighbor_obj < best_neighbor_obj:
                        best_neighbor = neighbor.copy()
                        best_neighbor_obj = neighbor_obj
                        best_move = move
                        best_neighbor_delay = neighbor_delay.copy() if neighbor_delay else None
            
            if best_neighbor is not None:
                current = best_neighbor
                current_obj = best_neighbor_obj
                current_delay = best_neighbor_delay
                
                self.tabu_list.add_move(best_move, iteration)
                
                if current_obj < best_objective:
                    best_objective = current_obj
                    best_activity_list = current.copy()
                    if self.params.use_delay_factors and current_delay:
                        best_delay_factors = current_delay.copy()
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
        
        if self.params.use_delay_factors and best_delay_factors:
            start_times, _ = self.decoder.decode(best_activity_list, best_delay_factors)
        else:
            start_times, _ = self.decoder.decode(best_activity_list)
        
        runtime = time.time() - start_time
        
        return {
            'best_activity_list': best_activity_list,
            'best_delay_factors': best_delay_factors,
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
    
    def _initialize_solution(self) -> tuple:
        """初始化解"""
        if self.params.use_delay_factors:
            al, delays = self.decoder.encode_random_with_delay(self.rng)
            return self.decoder.repair(al), delays
        al = self.decoder.encode_random(self.rng)
        return self.decoder.repair(al), None
    
    def _generate_neighbors(self, solution: List[int], delay_factors: List[float] = None) -> List[Tuple]:
        """生成邻域解"""
        neighbors = []
        
        for _ in range(self.params.neighborhood_size):
            neighbor = solution.copy()
            neighbor_delay = delay_factors.copy() if delay_factors else None
            
            if self.params.neighborhood_type == "swap":
                i, j = self.rng.choice(self.n, size=2, replace=False)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                move = ("swap", i, j)
            
            elif self.params.neighborhood_type == "insertion":
                i = self.rng.integers(0, self.n)
                j = self.rng.integers(0, self.n)
                gene = neighbor.pop(i)
                neighbor.insert(j, gene)
                move = ("insertion", i, j)
            
            elif self.params.neighborhood_type == "inversion":
                i, j = sorted(self.rng.choice(self.n, size=2, replace=False))
                neighbor[i:j+1] = neighbor[i:j+1][::-1]
                move = ("inversion", i, j)
            
            else:
                i, j = self.rng.choice(self.n, size=2, replace=False)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                move = ("swap", i, j)
            
            neighbor = self.decoder.repair(neighbor)
            
            if self.params.use_delay_factors and neighbor_delay:
                for k in range(len(neighbor_delay)):
                    if self.rng.random() < self.params.delay_mutation_rate:
                        neighbor_delay[k] = self.rng.random()
                neighbor_delay[0] = 0.0
                neighbor_delay[self.n - 1] = 0.0
            
            neighbors.append((neighbor, neighbor_delay, move))
        
        return neighbors
