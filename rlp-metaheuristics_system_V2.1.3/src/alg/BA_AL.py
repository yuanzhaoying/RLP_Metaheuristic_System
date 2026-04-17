"""
蝙蝠算法（Bat Algorithm）- 活动列表编码版本

算法简介：
    蝙蝠算法是一种基于蝙蝠回声定位行为的元启发式算法。
    通过模拟蝙蝠发出声波并接收回声来搜索猎物的过程。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 使用离散化的位置表示（每个位置代表活动的优先级）

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
from ..psp.activity_list_evaluator import ActivityListEvaluator
from ..psp.activity_list_decoder import ActivityListDecoder
from .operators import RandomGenerator


@dataclass
class BAParamsAL:
    """蝙蝠算法参数（活动列表编码）"""
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
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class BatAlgorithmAL:
    """蝙蝠算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: BAParamsAL):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ActivityListEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ActivityListDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def _al_to_position(self, activity_list: List[int]) -> np.ndarray:
        """将活动列表转换为位置向量（优先级）"""
        position = np.zeros(self.n, dtype=np.float64)
        for i, act in enumerate(activity_list):
            position[act] = self.n - i
        return position
    
    def _position_to_al(self, position: np.ndarray) -> List[int]:
        """将位置向量转换为活动列表"""
        sorted_indices = np.argsort(-position)
        return sorted_indices.tolist()
    
    def _initialize_position(self) -> np.ndarray:
        """初始化蝙蝠位置"""
        al = self.decoder.encode_random(self.rng)
        return self._al_to_position(al)
    
    def _initialize_velocity(self) -> np.ndarray:
        """初始化蝙蝠速度"""
        return np.zeros(self.n, dtype=np.float64)
    
    def _swap_local_search(self, activity_list: List[int], delay_factors: List[float] = None) -> tuple:
        """交换局部搜索"""
        best = activity_list.copy()
        best_delay = delay_factors.copy() if delay_factors else None
        
        if self.params.use_delay_factors and best_delay:
            best_obj, _ = self.evaluator.evaluate(best, best_delay)
        else:
            best_obj, _ = self.evaluator.evaluate(best)
        
        improved = True
        while improved:
            improved = False
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        return best, best_delay
                    
                    new = best.copy()
                    new[i], new[j] = new[j], new[i]
                    new = self.decoder.repair(new)
                    
                    new_delay = best_delay.copy() if best_delay else None
                    if self.params.use_delay_factors and new_delay:
                        for k in range(len(new_delay)):
                            if self.rng.random() < 0.1:
                                new_delay[k] = self.rng.random()
                        new_delay[0] = 0.0
                        new_delay[self.n - 1] = 0.0
                    
                    if self.params.use_delay_factors and new_delay:
                        obj, _ = self.evaluator.evaluate(new, new_delay)
                    else:
                        obj, _ = self.evaluator.evaluate(new)
                    
                    if obj < best_obj:
                        best = new
                        best_delay = new_delay
                        best_obj = obj
                        improved = True
                        break
                if improved:
                    break
        
        return best, best_delay
    
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
        delay_factors_list = []
        
        for _ in range(self.params.population_size):
            position = self._initialize_position()
            velocity = self._initialize_velocity()
            
            if self.params.use_delay_factors:
                al, delays = self.decoder.encode_random_with_delay(self.rng)
                delay_factors_list.append(delays)
            else:
                al = self.decoder.encode_random(self.rng)
                delay_factors_list.append(None)
            
            al = self.decoder.repair(al)
            position = self._al_to_position(al)
            
            if self.params.use_delay_factors:
                obj, _ = self.evaluator.evaluate(al, delay_factors_list[-1])
            else:
                obj, _ = self.evaluator.evaluate(al)
            
            positions.append(position)
            velocities.append(velocity)
            frequencies.append(self.rng.rng.uniform(self.params.f_min, self.params.f_max))
            loudness.append(self.params.A0)
            pulse_rate.append(self.params.r0)
            fitness.append(obj)
        
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        best_objective = fitness[best_idx]
        best_activity_list = self._position_to_al(best_position)
        best_activity_list = self.decoder.repair(best_activity_list)
        best_delay_factors = delay_factors_list[best_idx].copy() if delay_factors_list[best_idx] else None
        
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
                    new_position = best_position + self.rng.rng.normal(0, 1, self.n)
                
                new_al = self._position_to_al(new_position)
                new_al = self.decoder.repair(new_al)
                new_position = self._al_to_position(new_al)
                
                if self.params.use_delay_factors:
                    for k in range(len(delay_factors_list[i])):
                        if self.rng.random() < self.params.delay_mutation_rate:
                            delay_factors_list[i][k] = self.rng.random()
                    delay_factors_list[i][0] = 0.0
                    delay_factors_list[i][self.n - 1] = 0.0
                    new_obj, _ = self.evaluator.evaluate(new_al, delay_factors_list[i])
                else:
                    new_obj, _ = self.evaluator.evaluate(new_al)
                
                if (new_obj < fitness[i]) and (self.rng.rng.random() < loudness[i]):
                    positions[i] = new_position.copy()
                    fitness[i] = new_obj
                    
                    loudness[i] *= self.params.alpha
                    pulse_rate[i] = self.params.r0 * (1 - np.exp(-self.params.gamma * iteration))
                
                if new_obj < best_objective:
                    best_objective = new_obj
                    best_position = new_position.copy()
                    best_activity_list = new_al
                    if self.params.use_delay_factors:
                        best_delay_factors = delay_factors_list[i].copy()
            
            if (self.params.local_search_strategy == "swap" and 
                (iteration + 1) % self.params.local_search_interval == 0):
                
                top_indices = np.argsort(fitness)[:5]
                for idx in top_indices:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    al = self._position_to_al(positions[idx])
                    improved, improved_delay = self._swap_local_search(al, delay_factors_list[idx])
                    
                    if self.params.use_delay_factors and improved_delay:
                        obj, _ = self.evaluator.evaluate(improved, improved_delay)
                    else:
                        obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < fitness[idx]:
                        positions[idx] = self._al_to_position(improved)
                        fitness[idx] = obj
                        
                        if obj < best_objective:
                            best_objective = obj
                            best_position = positions[idx].copy()
                            best_activity_list = improved
                            if self.params.use_delay_factors:
                                best_delay_factors = improved_delay.copy()
            
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
