"""
和谐搜索算法（Harmony Search）- 活动列表编码版本

算法简介：
    和谐搜索算法是一种基于音乐即兴创作过程的元启发式算法。
    通过模拟音乐家在演奏中寻找完美和声的过程来搜索最优解。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 使用离散化的和声表示

算子汇总：
    1. 参数策略
       - fixed：固定参数
       - adaptive：自适应参数
    
    2. 初始化策略
       - random：随机初始化
       - forward：前向调度初始化
"""

from typing import List
import time
import math
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.activity_list_evaluator import ActivityListEvaluator
from ..psp.activity_list_decoder import ActivityListDecoder
from .operators import RandomGenerator


@dataclass
class HSParamsAL:
    """和谐搜索算法参数（活动列表编码）"""
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
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class HarmonySearchAL:
    """和谐搜索算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: HSParamsAL):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ActivityListEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ActivityListDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.max_iterations = params.max_evaluations
    
    def _update_par(self, iteration: int) -> float:
        """更新PAR（自适应）"""
        if self.params.parameter_strategy == "adaptive":
            return self.params.par_min + (self.params.par_max - self.params.par_min) * (iteration / self.max_iterations)
        else:
            return self.params.par
    
    def _initialize_harmony_random(self) -> tuple:
        """随机初始化和声"""
        if self.params.use_delay_factors:
            return self.decoder.encode_random_with_delay(self.rng)
        return self.decoder.encode_random(self.rng), None
    
    def _initialize_harmony_forward(self) -> tuple:
        """前向调度初始化"""
        al = self.decoder.encode_forward()
        if self.params.use_delay_factors:
            delays = [self.rng.random() for _ in range(self.n)]
            delays[0] = 0.0
            delays[self.n - 1] = 0.0
            return al, delays
        return al, None
    
    def _improvise_harmony(self, harmony_memory: List[List[int]], delay_memory: List[List[float]], par: float) -> tuple:
        """即兴创作新和声"""
        new_harmony = [None] * self.n
        new_delays = [0.0] * self.n if self.params.use_delay_factors else None
        
        for j in range(self.n):
            if self.rng.rng.random() < self.params.hmcr:
                idx = self.rng.integers(0, len(harmony_memory))
                new_harmony[j] = harmony_memory[idx][j]
                if self.params.use_delay_factors and new_delays and delay_memory[idx]:
                    new_delays[j] = delay_memory[idx][j]
                
                if self.rng.rng.random() < par:
                    shift = self.rng.integers(-2, 3)
                    new_pos = j + shift
                    if 0 <= new_pos < self.n and new_harmony[new_pos] is None:
                        new_harmony[j], new_harmony[new_pos] = new_harmony[new_pos], new_harmony[j]
                        if self.params.use_delay_factors and new_delays:
                            new_delays[j], new_delays[new_pos] = new_delays[new_pos], new_delays[j]
            else:
                remaining = [i for i in range(self.n) if i not in new_harmony or new_harmony[i] is None]
                if remaining:
                    new_harmony[j] = self.rng.choice(remaining)
                    if self.params.use_delay_factors and new_delays:
                        new_delays[j] = self.rng.random()
        
        for j in range(self.n):
            if new_harmony[j] is None:
                remaining = [i for i in range(self.n) if i not in new_harmony]
                if remaining:
                    new_harmony[j] = self.rng.choice(remaining)
                else:
                    new_harmony[j] = j
        
        seen = set()
        for j in range(self.n):
            if new_harmony[j] in seen:
                for i in range(self.n):
                    if i not in new_harmony:
                        new_harmony[j] = i
                        break
            seen.add(new_harmony[j])
        
        if self.params.use_delay_factors and new_delays:
            for k in range(len(new_delays)):
                if self.rng.random() < self.params.delay_mutation_rate:
                    new_delays[k] = self.rng.random()
            new_delays[0] = 0.0
            new_delays[self.n - 1] = 0.0
        
        return new_harmony, new_delays
    
    def run(self):
        """运行和谐搜索算法"""
        start_time = time.time()
        convergence = []
        
        harmony_memory = []
        delay_memory = []
        fitness = []
        
        for _ in range(self.params.hm_size):
            if self.params.initialization_strategy == "forward":
                harmony, delays = self._initialize_harmony_forward()
            else:
                harmony, delays = self._initialize_harmony_random()
            
            harmony = self.decoder.repair(harmony)
            
            if self.params.use_delay_factors and delays:
                obj, _ = self.evaluator.evaluate(harmony, delays)
            else:
                obj, _ = self.evaluator.evaluate(harmony)
            
            harmony_memory.append(harmony)
            delay_memory.append(delays)
            fitness.append(obj)
        
        sorted_indices = np.argsort(fitness)
        harmony_memory = [harmony_memory[i] for i in sorted_indices]
        delay_memory = [delay_memory[i] for i in sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]
        
        best_harmony = harmony_memory[0].copy()
        best_delays = delay_memory[0].copy() if delay_memory[0] else None
        best_fitness = fitness[0]
        best_activity_list = best_harmony
        
        convergence.append(best_fitness)
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            par = self._update_par(iteration)
            
            new_harmony, new_delays = self._improvise_harmony(harmony_memory, delay_memory, par)
            new_harmony = self.decoder.repair(new_harmony)
            
            if self.params.use_delay_factors and new_delays:
                new_fitness, _ = self.evaluator.evaluate(new_harmony, new_delays)
            else:
                new_fitness, _ = self.evaluator.evaluate(new_harmony)
            
            if new_fitness < fitness[-1]:
                harmony_memory[-1] = new_harmony.copy()
                delay_memory[-1] = new_delays.copy() if new_delays else None
                fitness[-1] = new_fitness
                
                sorted_indices = np.argsort(fitness)
                harmony_memory = [harmony_memory[i] for i in sorted_indices]
                delay_memory = [delay_memory[i] for i in sorted_indices]
                fitness = [fitness[i] for i in sorted_indices]
                
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_harmony = new_harmony.copy()
                    best_delays = new_delays.copy() if new_delays else None
                    best_activity_list = new_harmony
            
            convergence.append(best_fitness)
            iteration += 1
        
        if self.params.use_delay_factors and best_delays:
            start_times, _ = self.decoder.decode(best_activity_list, best_delays)
        else:
            start_times, _ = self.decoder.decode(best_activity_list)
        
        runtime = time.time() - start_time
        
        return {
            'best_activity_list': best_activity_list,
            'best_delay_factors': best_delays,
            'best_start_times': start_times.tolist(),
            'best_objective': best_fitness,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': self._params_to_dict(self.params)
        }
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
