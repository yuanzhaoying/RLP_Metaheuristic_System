"""
和谐搜索算法（Harmony Search）- 基于位移编码版本

算子汇总：
    1. 参数策略
       - fixed：固定参数
       - adaptive：自适应参数
    
    2. 初始化策略
       - random：随机初始化
       - zero：零位移初始化
"""

from typing import List
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.shift_vector_evaluator import ShiftVectorEvaluator
from ..psp.shift_vector_decoder import ShiftVectorDecoder
from .operators import RandomGenerator


@dataclass
class HSParamsSV:
    """和谐搜索算法参数（位移编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    time_limit: float = 60.0
    hm_size: int = 50
    hmcr: float = 0.9
    par: float = 0.3
    par_min: float = 0.1
    par_max: float = 0.9
    parameter_strategy: str = "fixed"
    initialization_strategy: str = "random"


class HarmonySearchSV:
    """和谐搜索算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: HSParamsSV):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ShiftVectorEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ShiftVectorDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.max_iterations = params.max_evaluations
    
    def run(self):
        """运行和谐搜索算法"""
        start_time = time.time()
        convergence = []
        
        harmony_memory = []
        fitness = []
        
        for _ in range(self.params.hm_size):
            if self.params.initialization_strategy == "zero":
                harmony = [0] * self.n
            else:
                harmony = self.decoder.encode_random(self.rng)
            
            obj, _ = self.evaluator.evaluate(harmony)
            
            harmony_memory.append(harmony)
            fitness.append(obj)
        
        sorted_indices = np.argsort(fitness)
        harmony_memory = [harmony_memory[i] for i in sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]
        
        best_harmony = harmony_memory[0].copy()
        best_fitness = fitness[0]
        best_displacement = best_harmony
        
        convergence.append(best_fitness)
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            par = self._update_par(iteration)
            
            new_harmony = []
            for j in range(self.n):
                if self.rng.rng.random() < self.params.hmcr:
                    idx = self.rng.integers(0, len(harmony_memory))
                    value = harmony_memory[idx][j]
                    
                    if self.rng.rng.random() < par:
                        max_shift = self.decoder.get_max_shift(j)
                        bandwidth = max_shift * 0.1
                        value = int(value + self.rng.rng.uniform(-bandwidth, bandwidth))
                        value = max(0, min(value, max_shift))
                    
                    new_harmony.append(value)
                else:
                    max_shift = self.decoder.get_max_shift(j)
                    new_harmony.append(self.rng.integers(0, max_shift + 1))
            
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
                    best_displacement = new_harmony
            
            convergence.append(best_fitness)
            iteration += 1
        
        start_times, _ = self.decoder.decode(best_displacement)
        
        runtime = time.time() - start_time
        
        return {
            'best_displacement': best_displacement,
            'best_start_times': start_times.tolist(),
            'best_objective': best_fitness,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': self._params_to_dict(self.params)
        }
    
    def _update_par(self, iteration: int) -> float:
        """更新PAR（自适应）"""
        if self.params.parameter_strategy == "adaptive":
            return self.params.par_min + (self.params.par_max - self.params.par_min) * (iteration / self.max_iterations)
        else:
            return self.params.par
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
