"""
禁忌搜索算法（Tabu Search）- 优先级编码版本

算子汇总（2种组合）：
    1. 禁忌表更新策略 - 2种：static, dynamic
"""

import math
from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass, asdict
from psp.psplib_io import RCPSPInstance
from psp.priority_evaluator import PriorityEvaluator
from psp.priority_decoder import PriorityDecoder
from alg.operators import RandomGenerator


class TabuListPRI:
    def __init__(self, n_activities: int, strategy: str = "static", 
                 initial_tenure: int = 10, min_factor: int = 1, max_factor: int = 4):
        self.n_activities = n_activities
        self.strategy = strategy
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
        if self.strategy == "static":
            return move in self.tabu_set
        else:
            return self.tabu_dict.get(move, 0) > current_iteration
    
    def add_move(self, move, current_iteration: int = None):
        if self.strategy == "static":
            if len(self.tabu_list) >= self.tenure:
                old_move = self.tabu_list.pop(0)
                self.tabu_set.discard(old_move)
            self.tabu_list.append(move)
            self.tabu_set.add(move)
        else:
            self.tabu_dict[move] = current_iteration + self.length
    
    def update_length(self):
        if self.strategy == "dynamic":
            self.length = np.random.randint(self.min_length, self.max_length + 1)


@dataclass
class TSParamsPRI:
    max_evaluations: int = 1000
    seed: int = 0
    tabu_tenure: int = 10
    neighborhood_size: int = 20
    time_limit: float = 60.0
    max_iterations: int = 100
    aspiration_criterion: bool = True
    tabu_strategy: str = "static"
    neighborhood_type: str = "uniform"
    min_factor: int = 1
    max_factor: int = 4
    noimprove_threshold: int = 10


class TabuSearchPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: TSParamsPRI):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        self.evaluator = PriorityEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = PriorityDecoder(instance, deadline)
        self.n = instance.n_activities
        self.tabu_list = TabuListPRI(n_activities=self.n, strategy=params.tabu_strategy,
                                      initial_tenure=params.tabu_tenure,
                                      min_factor=params.min_factor, max_factor=params.max_factor)
    
    def run(self):
        start_time = time.time()
        convergence = []
        
        current = self.decoder.encode_random(self.rng)
        current = self.decoder.repair(current)
        current_obj, _ = self.evaluator.evaluate(current)
        
        best_priority = current.copy()
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
                    best_priority = current.copy()
                    nr_noimprove = 0
                else:
                    nr_noimprove += 1
            else:
                nr_noimprove += 1
            
            if self.params.tabu_strategy == "dynamic" and nr_noimprove >= self.params.noimprove_threshold:
                self.tabu_list.update_length()
                nr_noimprove = 0
            
            convergence.append(best_objective)
            iteration += 1
        
        start_times, _ = self.decoder.decode(best_priority)
        runtime = time.time() - start_time
        
        return {
            'best_priority': best_priority,
            'best_start_times': start_times.tolist(),
            'best_objective': best_objective,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': asdict(self.params)
        }
    
    def _generate_neighbors(self, solution: List[float]) -> List[Tuple]:
        neighbors = []
        for _ in range(self.params.neighborhood_size):
            neighbor = solution.copy()
            i = self.rng.integers(1, self.n - 1)
            neighbor[i] = self.rng.random()
            neighbor = self.decoder.repair(neighbor)
            move = ("uniform", i)
            neighbors.append((neighbor, move))
        return neighbors
