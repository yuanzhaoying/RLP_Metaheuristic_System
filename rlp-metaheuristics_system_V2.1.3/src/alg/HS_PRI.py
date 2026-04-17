"""
和谐搜索算法（Harmony Search）- 优先级编码版本

算子汇总（4种组合）：
    1. 参数策略 - 2种：fixed, adaptive
    2. 初始化策略 - 2种：random, zero
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
class HSParamsPRI:
    max_evaluations: int = 1000
    seed: int = 0
    time_limit: float = 60.0
    hm_size: int = 50
    hmcr: float = 0.9
    par: float = 0.3
    par_min: float = 0.1
    par_max: float = 0.9
    bw: float = 0.1
    parameter_strategy: str = "fixed"
    initialization_strategy: str = "random"


class HarmonySearchPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: HSParamsPRI):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        self.evaluator = PriorityEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = PriorityDecoder(instance, deadline)
        self.n = instance.n_activities
        self.max_iterations = params.max_evaluations
    
    def run(self):
        start_time = time.time()
        convergence = []
        
        harmony_memory = []
        fitness = []
        
        for _ in range(self.params.hm_size):
            if self.params.initialization_strategy == "random":
                harmony = self.decoder.encode_random(self.rng)
            else:
                harmony = [0.0] * self.n
                harmony[0] = 1.0
                harmony[self.n - 1] = 0.0
            harmony = self.decoder.repair(harmony)
            obj, _ = self.evaluator.evaluate(harmony)
            harmony_memory.append(harmony)
            fitness.append(obj)
        
        sorted_indices = np.argsort(fitness)
        harmony_memory = [harmony_memory[i] for i in sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]
        
        best_harmony = harmony_memory[0].copy()
        best_fitness = fitness[0]
        
        convergence.append(best_fitness)
        iteration = 0
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            if self.params.parameter_strategy == "adaptive":
                par = self.params.par_min + (self.params.par_max - self.params.par_min) * (iteration / self.max_iterations)
            else:
                par = self.params.par
            
            new_harmony = []
            for j in range(self.n):
                if self.rng.rng.random() < self.params.hmcr:
                    idx = self.rng.integers(0, len(harmony_memory))
                    val = harmony_memory[idx][j]
                    if self.rng.rng.random() < par:
                        val += self.rng.rng.uniform(-self.params.bw, self.params.bw)
                        val = max(0.0, min(1.0, val))
                    new_harmony.append(val)
                else:
                    new_harmony.append(self.rng.random())
            
            new_harmony = self.decoder.repair(new_harmony)
            new_fitness, _ = self.evaluator.evaluate(new_harmony)
            
            if new_fitness < fitness[-1]:
                harmony_memory[-1] = new_harmony.copy()
                fitness[-1] = new_fitness
                sorted_indices = np.argsort(fitness)
                harmony_memory = [harmony_memory[i] for i in sorted_indices]
                fitness = [fitness[i] for i in sorted_indices]
                
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_harmony = new_harmony.copy()
            
            convergence.append(best_fitness)
            iteration += 1
        
        start_times, _ = self.decoder.decode(best_harmony)
        runtime = time.time() - start_time
        
        return {
            'best_priority': best_harmony,
            'best_start_times': start_times.tolist(),
            'best_objective': best_fitness,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': asdict(self.params)
        }
