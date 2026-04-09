"""
蝙蝠算法（Bat Algorithm）- 基于位移编码版本

算子汇总：
    1. 局部搜索
       - none：不使用局部搜索
       - swap：交换局部搜索
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
class BAParamsSV:
    """蝙蝠算法参数（位移编码）"""
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


class BatAlgorithmSV:
    """蝙蝠算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: BAParamsSV):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ShiftVectorEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ShiftVectorDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def _local_search(self, solution: List[int]) -> tuple:
        """局部搜索"""
        if self.params.local_search_strategy == "none":
            return solution, None
        
        best_sol = solution.copy()
        best_obj, _ = self.evaluator.evaluate(best_sol)
        
        improved = True
        while improved and self.evaluator.n_evaluations < self.params.max_evaluations:
            improved = False
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    neighbor = best_sol.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    obj, _ = self.evaluator.evaluate(neighbor)
                    if obj < best_obj:
                        best_obj = obj
                        best_sol = neighbor
                        improved = True
                        break
                if improved:
                    break
        
        return best_sol, best_obj
    
    def run(self):
        """运行蝙蝠算法"""
        start_time = time.time()
        convergence = []
        
        positions = []
        velocities = []
        frequencies = []
        loudness = []
        pulse_rate = []
        fitness = []
        
        for _ in range(self.params.population_size):
            pos = self.decoder.encode_random(self.rng)
            vel = [0.0] * self.n
            
            obj, _ = self.evaluator.evaluate(pos)
            
            positions.append(pos)
            velocities.append(vel)
            frequencies.append(self.rng.rng.uniform(self.params.f_min, self.params.f_max))
            loudness.append(self.params.A0)
            pulse_rate.append(self.params.r0)
            fitness.append(obj)
        
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        best_objective = fitness[best_idx]
        best_displacement = best_position.copy()
        
        convergence.append(best_objective)
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            for i in range(self.params.population_size):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                beta = self.rng.rng.uniform(0, 1)
                frequencies[i] = self.params.f_min + (self.params.f_max - self.params.f_min) * beta
                
                new_vel = []
                new_pos = []
                for j in range(self.n):
                    v = velocities[i][j] + frequencies[i] * (positions[i][j] - best_position[j])
                    max_shift = self.decoder.get_max_shift(j)
                    v = np.clip(v, -max_shift * 0.5, max_shift * 0.5)
                    new_vel.append(v)
                    
                    p = positions[i][j] + v
                    p = int(np.clip(p, 0, max_shift))
                    new_pos.append(p)
                
                velocities[i] = new_vel
                
                if self.rng.rng.random() > pulse_rate[i]:
                    for j in range(self.n):
                        max_shift = self.decoder.get_max_shift(j)
                        new_pos[j] = int(np.clip(best_position[j] + self.rng.rng.normal(0, 1) * max_shift * 0.1, 0, max_shift))
                
                new_obj, _ = self.evaluator.evaluate(new_pos)
                
                if (new_obj < fitness[i]) and (self.rng.rng.random() < loudness[i]):
                    positions[i] = new_pos
                    fitness[i] = new_obj
                    
                    loudness[i] *= self.params.alpha
                    pulse_rate[i] = self.params.r0 * (1 - np.exp(-self.params.gamma * iteration))
                
                if new_obj < best_objective:
                    best_objective = new_obj
                    best_position = new_pos.copy()
                    best_displacement = new_pos.copy()
            
            convergence.append(best_objective)
            iteration += 1
            
            if self.params.local_search_strategy != "none" and iteration % self.params.local_search_interval == 0:
                if self.evaluator.n_evaluations < self.params.max_evaluations:
                    improved_sol, improved_obj = self._local_search(best_displacement)
                    if improved_obj is not None and improved_obj < best_objective:
                        best_objective = improved_obj
                        best_position = improved_sol.copy()
                        best_displacement = improved_sol.copy()
        
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
