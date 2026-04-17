"""
模拟退火算法（Simulated Annealing）- 优先级编码版本

算子汇总（3种组合）：
    1. 邻域生成算子 - 3种：uniform, gaussian, swap
"""

from typing import List
import time
import math
from dataclasses import dataclass, asdict
from psp.psplib_io import RCPSPInstance
from psp.priority_evaluator import PriorityEvaluator
from psp.priority_decoder import PriorityDecoder
from alg.operators import RandomGenerator


@dataclass
class SAParamsPRI:
    max_evaluations: int = 1000
    seed: int = 0
    initial_temperature: float = 10000.0
    cooling_rate: float = 0.995
    iterations_per_temperature: int = 10
    time_limit: float = 60.0
    neighborhood_strategy: str = "uniform"


class SimulatedAnnealingPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: SAParamsPRI):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        self.evaluator = PriorityEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = PriorityDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        start_time = time.time()
        convergence = []
        
        current = self.decoder.encode_random(self.rng)
        current = self.decoder.repair(current)
        current_obj, _ = self.evaluator.evaluate(current)
        
        best_priority = current.copy()
        best_objective = current_obj
        
        temperature = self.params.initial_temperature
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            for _ in range(self.params.iterations_per_temperature):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                neighbor = self._generate_neighbor(current)
                neighbor_obj, _ = self.evaluator.evaluate(neighbor)
                delta = neighbor_obj - current_obj
                
                if delta < 0 or self._acceptance_probability(delta, temperature) > self.rng.random():
                    current = neighbor
                    current_obj = neighbor_obj
                    if current_obj < best_objective:
                        best_objective = current_obj
                        best_priority = current.copy()
            
            convergence.append(best_objective)
            temperature *= self.params.cooling_rate
        
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
    
    def _generate_neighbor(self, solution: List[float]) -> List[float]:
        neighbor = solution.copy()
        
        if self.params.neighborhood_strategy == "uniform":
            i = self.rng.integers(1, self.n - 1)
            neighbor[i] = self.rng.random()
        elif self.params.neighborhood_strategy == "gaussian":
            i = self.rng.integers(1, self.n - 1)
            neighbor[i] += self.rng.rng.normal(0, 0.1)
            neighbor[i] = max(0.0, min(1.0, neighbor[i]))
        else:  # swap
            i, j = self.rng.choice(self.n, size=2, replace=False)
            if i not in [0, self.n-1] and j not in [0, self.n-1]:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        neighbor = self.decoder.repair(neighbor)
        return neighbor
    
    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        if temperature <= 0:
            return 0.0
        return math.exp(-delta / temperature)
