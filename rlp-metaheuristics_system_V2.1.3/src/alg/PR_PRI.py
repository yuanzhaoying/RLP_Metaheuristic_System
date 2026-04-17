"""
路径重连算法（Path Relinking）- 优先级编码版本

算子汇总（16种组合）：
    1. 路径探索策略 - 4种：forward, backward, random, bidirectional
    2. 解选择策略 - 2种：best, random_two
    3. 局部优化 - 2种：True, False
"""

from typing import List
import time
import numpy as np
from dataclasses import dataclass, asdict
from psp.psplib_io import RCPSPInstance
from psp.priority_evaluator import PriorityEvaluator
from psp.priority_decoder import PriorityDecoder
from alg.operators import RandomGenerator


@dataclass
class PRParamsPRI:
    max_evaluations: int = 1000
    seed: int = 0
    time_limit: float = 60.0
    max_iterations: int = 100
    elite_size: int = 10
    path_strategy: str = "forward"
    selection_strategy: str = "best"
    use_local_search: bool = False
    local_search_eval_limit: int = 50


class PathRelinkingPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PRParamsPRI):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        self.evaluator = PriorityEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = PriorityDecoder(instance, deadline)
        self.n = instance.n_activities
        self.elite_pool = []
    
    def run(self):
        start_time = time.time()
        convergence = []
        
        self._initialize_elite_pool()
        best_priority = self.elite_pool[0][0].copy()
        best_objective = self.elite_pool[0][1]
        
        iteration = 0
        while (iteration < self.params.max_iterations and 
               self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            init_idx = self.rng.integers(0, len(self.elite_pool))
            guide_idx = self.rng.integers(0, len(self.elite_pool))
            while guide_idx == init_idx:
                guide_idx = self.rng.integers(0, len(self.elite_pool))
            
            init_solution = self.elite_pool[init_idx][0]
            guide_solution = self.elite_pool[guide_idx][0]
            
            if self.params.path_strategy == "bidirectional":
                path_solutions = self._generate_bidirectional_path(init_solution, guide_solution)
            else:
                path_solutions = self._generate_path(init_solution, guide_solution)
            
            for solution in path_solutions:
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                repaired = self.decoder.repair(solution)
                if self.params.use_local_search:
                    improved = self._local_search(repaired, start_time)
                else:
                    improved = repaired
                obj, _ = self.evaluator.evaluate(improved)
                if obj < best_objective:
                    best_objective = obj
                    best_priority = improved.copy()
                self._update_elite_pool(improved, obj)
            
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
    
    def _initialize_elite_pool(self):
        for _ in range(self.params.elite_size):
            priority = self.decoder.encode_random(self.rng)
            priority = self.decoder.repair(priority)
            obj, _ = self.evaluator.evaluate(priority)
            self.elite_pool.append((priority, obj))
        self.elite_pool.sort(key=lambda x: x[1])
    
    def _generate_path(self, init_solution: List[float], guide_solution: List[float]) -> List[List[float]]:
        path = []
        n_steps = 10
        if self.params.path_strategy == "forward":
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                new_solution = [(1 - alpha) * init_solution[j] + alpha * guide_solution[j] for j in range(self.n)]
                path.append(self.decoder.repair(new_solution))
        elif self.params.path_strategy == "backward":
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                new_solution = [alpha * init_solution[j] + (1 - alpha) * guide_solution[j] for j in range(self.n)]
                path.append(self.decoder.repair(new_solution))
        else:  # random
            alphas = sorted([self.rng.random() for _ in range(n_steps)])
            for alpha in alphas:
                new_solution = [(1 - alpha) * init_solution[j] + alpha * guide_solution[j] for j in range(self.n)]
                path.append(self.decoder.repair(new_solution))
        return path
    
    def _generate_bidirectional_path(self, init_solution: List[float], guide_solution: List[float]) -> List[List[float]]:
        path = []
        n_steps = 5
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            new_solution = [(1 - alpha) * init_solution[j] + alpha * guide_solution[j] for j in range(self.n)]
            path.append(self.decoder.repair(new_solution))
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            new_solution = [alpha * init_solution[j] + (1 - alpha) * guide_solution[j] for j in range(self.n)]
            path.append(self.decoder.repair(new_solution))
        return path
    
    def _local_search(self, solution: List[float], start_time: float) -> List[float]:
        improved = True
        best_solution = solution.copy()
        best_obj, _ = self.evaluator.evaluate(best_solution)
        local_eval_count = 0
        
        while improved:
            improved = False
            for i in range(1, self.n - 1):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                if time.time() - start_time >= self.params.time_limit:
                    break
                if local_eval_count >= self.params.local_search_eval_limit:
                    break
                neighbor = best_solution.copy()
                neighbor[i] = self.rng.random()
                neighbor = self.decoder.repair(neighbor)
                obj, _ = self.evaluator.evaluate(neighbor)
                local_eval_count += 1
                if obj < best_obj:
                    best_solution = neighbor.copy()
                    best_obj = obj
                    improved = True
                    break
        return best_solution
    
    def _update_elite_pool(self, solution: List[float], obj: float):
        if len(self.elite_pool) < self.params.elite_size:
            self.elite_pool.append((solution, obj))
            self.elite_pool.sort(key=lambda x: x[1])
        else:
            if obj < self.elite_pool[-1][1]:
                is_different = all(solution != elite_sol for elite_sol, _ in self.elite_pool)
                if is_different:
                    self.elite_pool[-1] = (solution, obj)
                    self.elite_pool.sort(key=lambda x: x[1])
