"""
蝙蝠算法（Bat Algorithm）- 优先级编码版本

算子汇总（2种组合）：
    1. 局部搜索 - 2种：none, uniform
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
class BAParamsPRI:
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    f_min: float = 0.0
    f_max: float = 2.0
    A0: float = 1.0
    r0: float = 0.5
    alpha: float = 0.9
    gamma: float = 0.9
    local_search_strategy: str = "none"
    local_search_interval: int = 10


class BatAlgorithmPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: BAParamsPRI):
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
        
        positions = []
        velocities = []
        frequencies = []
        loudness = []
        pulse_rate = []
        fitness = []
        
        for _ in range(self.params.population_size):
            priority = self.decoder.encode_random(self.rng)
            priority = self.decoder.repair(priority)
            velocity = np.zeros(self.n, dtype=np.float64)
            obj, _ = self.evaluator.evaluate(priority)
            
            positions.append(np.array(priority))
            velocities.append(velocity)
            frequencies.append(self.rng.rng.uniform(self.params.f_min, self.params.f_max))
            loudness.append(self.params.A0)
            pulse_rate.append(self.params.r0)
            fitness.append(obj)
        
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        best_objective = fitness[best_idx]
        best_priority = best_position.tolist()
        
        convergence.append(best_objective)
        iteration = 0
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            for i in range(self.params.population_size):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                beta = self.rng.rng.uniform(0, 1)
                frequencies[i] = self.params.f_min + (self.params.f_max - self.params.f_min) * beta
                velocities[i] = velocities[i] + frequencies[i] * (positions[i] - best_position)
                new_position = positions[i] + velocities[i]
                
                if self.rng.rng.random() > pulse_rate[i]:
                    new_position = best_position + self.rng.rng.normal(0, 0.1, self.n)
                
                new_priority = self.decoder.repair(np.clip(new_position, 0.0, 1.0).tolist())
                new_position = np.array(new_priority)
                
                new_obj, _ = self.evaluator.evaluate(new_priority)
                
                if (new_obj < fitness[i]) and (self.rng.rng.random() < loudness[i]):
                    positions[i] = new_position.copy()
                    fitness[i] = new_obj
                    loudness[i] *= self.params.alpha
                    pulse_rate[i] = self.params.r0 * (1 - np.exp(-self.params.gamma * iteration))
                
                if new_obj < best_objective:
                    best_objective = new_obj
                    best_position = new_position.copy()
                    best_priority = new_priority
            
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
