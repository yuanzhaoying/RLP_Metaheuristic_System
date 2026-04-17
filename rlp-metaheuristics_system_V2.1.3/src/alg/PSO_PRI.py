"""
粒子群算法（Particle Swarm Optimization）- 优先级编码版本

算子汇总（4种组合）：
    1. 局部搜索 - 2种：none, uniform
    2. 重启机制 - 2种：none, adaptive
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
class PSOParamsPRI:
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


class ParticleSwarmOptimizationPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PSOParamsPRI):
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
        pbest_positions = []
        pbest_objectives = []
        
        for _ in range(self.params.population_size):
            priority = self.decoder.encode_random(self.rng)
            priority = self.decoder.repair(priority)
            velocity = np.array([self.rng.rng.uniform(-0.5, 0.5) for _ in range(self.n)])
            obj, _ = self.evaluator.evaluate(priority)
            
            positions.append(np.array(priority))
            velocities.append(velocity)
            pbest_positions.append(np.array(priority))
            pbest_objectives.append(obj)
        
        gbest_idx = np.argmin(pbest_objectives)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_objective = pbest_objectives[gbest_idx]
        gbest_priority = gbest_position.tolist()
        
        convergence.append(gbest_objective)
        no_improvement_count = 0
        
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
                new_velocity = np.clip(new_velocity, -0.5, 0.5)
                
                velocities[i] = new_velocity
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0.0, 1.0)
                
                priority = self.decoder.repair(positions[i].tolist())
                positions[i] = np.array(priority)
                
                obj, _ = self.evaluator.evaluate(priority)
                
                if obj < pbest_objectives[i]:
                    pbest_objectives[i] = obj
                    pbest_positions[i] = positions[i].copy()
                
                if obj < gbest_objective:
                    gbest_objective = obj
                    gbest_position = positions[i].copy()
                    gbest_priority = priority
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            convergence.append(gbest_objective)
        
        start_times, _ = self.decoder.decode(gbest_priority)
        runtime = time.time() - start_time
        
        return {
            'best_priority': gbest_priority,
            'best_start_times': start_times.tolist(),
            'best_objective': gbest_objective,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': asdict(self.params)
        }
