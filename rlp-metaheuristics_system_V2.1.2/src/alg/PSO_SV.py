"""
粒子群算法（Particle Swarm Optimization）- 基于位移编码版本

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
from ..psp.shift_vector_evaluator import ShiftVectorEvaluator
from ..psp.shift_vector_decoder import ShiftVectorDecoder
from .operators import RandomGenerator


@dataclass
class PSOParamsSV:
    """粒子群算法参数（位移编码）"""
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


class ParticleSwarmOptimizationSV:
    """粒子群算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PSOParamsSV):
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
    
    def _restart_population(self, positions, pbest_positions, pbest_objectives, gbest_position):
        """重启机制"""
        if self.params.restart_strategy == "none":
            return positions, pbest_positions, pbest_objectives
        
        for i in range(self.params.population_size):
            if self.rng.rng.random() < 0.3:
                new_pos = self.decoder.encode_random(self.rng)
                obj, _ = self.evaluator.evaluate(new_pos)
                positions[i] = new_pos
                pbest_positions[i] = new_pos.copy()
                pbest_objectives[i] = obj
        
        return positions, pbest_positions, pbest_objectives
    
    def run(self):
        """运行粒子群算法"""
        start_time = time.time()
        convergence = []
        
        positions = []
        velocities = []
        pbest_positions = []
        pbest_objectives = []
        
        for _ in range(self.params.population_size):
            pos = self.decoder.encode_random(self.rng)
            vel = [0] * self.n
            for j in range(self.n):
                max_shift = self.decoder.get_max_shift(j)
                vel[j] = self.rng.rng.uniform(-max_shift * 0.1, max_shift * 0.1)
            
            obj, _ = self.evaluator.evaluate(pos)
            
            positions.append(pos)
            velocities.append(vel)
            pbest_positions.append(pos.copy())
            pbest_objectives.append(obj)
        
        gbest_idx = np.argmin(pbest_objectives)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_objective = pbest_objectives[gbest_idx]
        gbest_displacement = gbest_position.copy()
        
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
                
                new_velocity = []
                for j in range(self.n):
                    v = (self.params.w * velocities[i][j] +
                         self.params.c1 * r1[j] * (pbest_positions[i][j] - positions[i][j]) +
                         self.params.c2 * r2[j] * (gbest_position[j] - positions[i][j]))
                    max_shift = self.decoder.get_max_shift(j)
                    v = np.clip(v, -max_shift * 0.5, max_shift * 0.5)
                    new_velocity.append(v)
                
                velocities[i] = new_velocity
                
                new_pos = []
                for j in range(self.n):
                    p = positions[i][j] + velocities[i][j]
                    max_shift = self.decoder.get_max_shift(j)
                    p = int(np.clip(p, 0, max_shift))
                    new_pos.append(p)
                
                positions[i] = new_pos
                
                obj, _ = self.evaluator.evaluate(positions[i])
                
                if obj < pbest_objectives[i]:
                    pbest_objectives[i] = obj
                    pbest_positions[i] = positions[i].copy()
                
                if obj < gbest_objective:
                    gbest_objective = obj
                    gbest_position = positions[i].copy()
                    gbest_displacement = positions[i].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            convergence.append(gbest_objective)
            iteration += 1
            
            if self.params.local_search_strategy != "none" and iteration % 5 == 0:
                if self.evaluator.n_evaluations < self.params.max_evaluations:
                    improved_sol, improved_obj = self._local_search(gbest_displacement)
                    if improved_obj is not None and improved_obj < gbest_objective:
                        gbest_objective = improved_obj
                        gbest_position = improved_sol.copy()
                        gbest_displacement = improved_sol.copy()
            
            if self.params.restart_strategy != "none" and no_improvement_count >= self.params.restart_threshold:
                positions, pbest_positions, pbest_objectives = self._restart_population(
                    positions, pbest_positions, pbest_objectives, gbest_position
                )
                no_improvement_count = 0
        
        start_times, _ = self.decoder.decode(gbest_displacement)
        
        runtime = time.time() - start_time
        
        return {
            'best_displacement': gbest_displacement,
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
