"""
粒子群算法（Particle Swarm Optimization）- 活动列表编码版本

算法简介：
    粒子群算法是一种基于群体智能的优化算法。
    通过模拟鸟群觅食行为，利用个体和群体的经验来搜索最优解。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 使用优先级向量表示位置

算子汇总：
    1. 局部搜索
       - none：不使用局部搜索
       - swap：交换局部搜索
    
    2. 重启机制
       - none：不使用重启
       - adaptive：自适应重启
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
class PSOParamsAL:
    """粒子群算法参数（活动列表编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    
    w: float = 0.729
    c1: float = 1.49445
    c2: float = 1.49445
    
    local_search_strategy: str = "none"
    restart_strategy: str = "none"
    restart_threshold: int = 30
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class ParticleSwarmOptimizationAL:
    """粒子群算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PSOParamsAL):
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
        """初始化粒子位置"""
        al = self.decoder.encode_random(self.rng)
        return self._al_to_position(al)
    
    def _initialize_velocity(self) -> np.ndarray:
        """初始化粒子速度"""
        velocity = np.zeros(self.n, dtype=np.float64)
        for j in range(self.n):
            velocity[j] = self.rng.rng.uniform(-self.n * 0.1, self.n * 0.1)
        return velocity
    
    def _swap_local_search(self, activity_list: List[int], delay_factors: List[float] = None) -> tuple:
        """交换局部搜索"""
        best = activity_list.copy()
        best_delay = delay_factors.copy() if delay_factors else None
        
        if self.params.use_delay_factors and best_delay:
            best_obj, _ = self.evaluator.evaluate(best, best_delay)
        else:
            best_obj, _ = self.evaluator.evaluate(best)
        
        for _ in range(10):
            if self.evaluator.n_evaluations >= self.params.max_evaluations:
                break
            
            i, j = self.rng.choice(self.n, size=2, replace=False)
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
        
        return best, best_delay
    
    def run(self):
        """运行粒子群算法"""
        start_time = time.time()
        convergence = []
        
        positions = []
        velocities = []
        pbest_positions = []
        pbest_objectives = []
        delay_factors_list = []
        pbest_delays = []
        
        for _ in range(self.params.population_size):
            position = self._initialize_position()
            velocity = self._initialize_velocity()
            
            if self.params.use_delay_factors:
                al, delays = self.decoder.encode_random_with_delay(self.rng)
                delay_factors_list.append(delays)
                pbest_delays.append(delays.copy())
            else:
                al = self.decoder.encode_random(self.rng)
                delay_factors_list.append(None)
                pbest_delays.append(None)
            
            al = self.decoder.repair(al)
            position = self._al_to_position(al)
            
            if self.params.use_delay_factors:
                obj, _ = self.evaluator.evaluate(al, delay_factors_list[-1])
            else:
                obj, _ = self.evaluator.evaluate(al)
            
            positions.append(position)
            velocities.append(velocity)
            pbest_positions.append(position.copy())
            pbest_objectives.append(obj)
        
        gbest_idx = np.argmin(pbest_objectives)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_objective = pbest_objectives[gbest_idx]
        gbest_activity_list = self._position_to_al(gbest_position)
        gbest_activity_list = self.decoder.repair(gbest_activity_list)
        gbest_delay_factors = pbest_delays[gbest_idx].copy() if pbest_delays[gbest_idx] else None
        
        convergence.append(gbest_objective)
        
        no_improvement_count = 0
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            for i in range(self.params.population_size):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                r1 = self.rng.rng.uniform(0, 1, self.n)
                r2 = self.rng.rng.uniform(0, 1, self.n)
                
                new_velocity = (self.params.w * velocities[i] +
                               self.params.c1 * r1 * (pbest_positions[i] - positions[i]) +
                               self.params.c2 * r2 * (gbest_position - positions[i]))
                
                for j in range(self.n):
                    new_velocity[j] = np.clip(new_velocity[j], -self.n * 0.1, self.n * 0.1)
                
                velocities[i] = new_velocity
                positions[i] = positions[i] + velocities[i]
                
                al = self._position_to_al(positions[i])
                al = self.decoder.repair(al)
                positions[i] = self._al_to_position(al)
                
                if self.params.use_delay_factors:
                    for k in range(len(delay_factors_list[i])):
                        if self.rng.random() < self.params.delay_mutation_rate:
                            delay_factors_list[i][k] = self.rng.random()
                    delay_factors_list[i][0] = 0.0
                    delay_factors_list[i][self.n - 1] = 0.0
                    obj, _ = self.evaluator.evaluate(al, delay_factors_list[i])
                else:
                    obj, _ = self.evaluator.evaluate(al)
                
                if obj < pbest_objectives[i]:
                    pbest_objectives[i] = obj
                    pbest_positions[i] = positions[i].copy()
                    if self.params.use_delay_factors:
                        pbest_delays[i] = delay_factors_list[i].copy()
                
                if obj < gbest_objective:
                    gbest_objective = obj
                    gbest_position = positions[i].copy()
                    gbest_activity_list = al
                    if self.params.use_delay_factors:
                        gbest_delay_factors = delay_factors_list[i].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            if (self.params.restart_strategy == "adaptive" and 
                no_improvement_count > self.params.restart_threshold and
                self.params.local_search_strategy == "swap"):
                
                top_indices = np.argsort(pbest_objectives)[:5]
                idx = self.rng.choice(top_indices)
                
                al = self._position_to_al(pbest_positions[idx])
                improved, improved_delay = self._swap_local_search(al, pbest_delays[idx])
                
                if self.params.use_delay_factors and improved_delay:
                    obj, _ = self.evaluator.evaluate(improved, improved_delay)
                else:
                    obj, _ = self.evaluator.evaluate(improved)
                
                if obj < gbest_objective:
                    gbest_objective = obj
                    gbest_position = pbest_positions[idx].copy()
                    gbest_activity_list = improved
                    if self.params.use_delay_factors:
                        gbest_delay_factors = improved_delay.copy()
                
                no_improvement_count = 0
            
            convergence.append(gbest_objective)
            iteration += 1
        
        if self.params.use_delay_factors and gbest_delay_factors:
            start_times, _ = self.decoder.decode(gbest_activity_list, gbest_delay_factors)
        else:
            start_times, _ = self.decoder.decode(gbest_activity_list)
        
        runtime = time.time() - start_time
        
        return {
            'best_activity_list': gbest_activity_list,
            'best_delay_factors': gbest_delay_factors,
            'best_start_times': start_times.tolist(),
            'best_objective': gbest_objective,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': self._params_to_dict(self.params)
        }
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
