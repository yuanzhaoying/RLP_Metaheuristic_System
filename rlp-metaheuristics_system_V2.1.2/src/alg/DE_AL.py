"""
差分进化算法（Differential Evolution）- 活动列表编码版本

算法简介：
    差分进化算法是一种基于群体差异的优化算法。
    通过变异、交叉和选择操作来进化种群。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 使用优先级向量进行差分操作

算子汇总：
    1. 初始化算子
       - random：随机初始化
       - forward：前向拓扑排序初始化
    
    2. 变异算子
       - rand/1, best/1, rand/2, best/2, adaptive
    
    3. 交叉算子
       - bin：二项交叉
       - exp：指数交叉
    
    4. 局部搜索
       - swap：交换局部搜索
"""

from typing import List
import time
import numpy as np
from dataclasses import dataclass, asdict
from ..psp.psplib_io import RCPSPInstance
from ..psp.activity_list_evaluator import ActivityListEvaluator
from ..psp.activity_list_decoder import ActivityListDecoder
from .operators import RandomGenerator


@dataclass
class DEParamsAL:
    """差分进化算法参数（活动列表编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    max_iterations: int = 100
    F: float = 0.5
    CR: float = 0.9
    use_adaptive_F: bool = False
    use_adaptive_CR: bool = False
    mutation_strategy: str = "rand/1"
    crossover_strategy: str = "bin"
    initialization_strategy: str = "random"
    use_local_search: bool = False
    local_search_top: int = 5
    F_min: float = 0.3
    F_max: float = 1.5
    CR_min: float = 0.3
    CR_max: float = 1.0
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class DifferentialEvolutionAL:
    """差分进化算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: DEParamsAL):
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
    
    def run(self):
        """运行差分进化算法"""
        start_time = time.time()
        convergence = []
        
        population, delay_factors = self._initialize_population()
        positions = [self._al_to_position(al) for al in population]
        
        best_activity_list = None
        best_delay_factors = None
        best_objective = float('inf')
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit and
               iteration < self.params.max_iterations):
            
            objectives = []
            for i, al in enumerate(population):
                if self.params.use_delay_factors:
                    obj, _ = self.evaluator.evaluate(al, delay_factors[i])
                else:
                    obj, _ = self.evaluator.evaluate(al)
                objectives.append(obj)
                
                if obj < best_objective:
                    best_objective = obj
                    best_activity_list = al.copy()
                    if self.params.use_delay_factors:
                        best_delay_factors = delay_factors[i].copy()
            
            convergence.append(best_objective)
            
            new_positions = []
            new_population = []
            new_delay_factors = []
            new_objectives = []
            
            for i in range(self.params.population_size):
                mutant = self._mutation(positions, objectives, i, iteration)
                trial_pos = self._crossover(positions[i], mutant, iteration)
                
                trial_al = self._position_to_al(trial_pos)
                trial_al = self.decoder.repair(trial_al)
                
                if self.params.use_delay_factors:
                    trial_delay = [self.rng.random() for _ in range(self.n)]
                    trial_delay[0] = 0.0
                    trial_delay[self.n - 1] = 0.0
                    trial_obj, _ = self.evaluator.evaluate(trial_al, trial_delay)
                else:
                    trial_delay = None
                    trial_obj, _ = self.evaluator.evaluate(trial_al)
                
                if trial_obj <= objectives[i]:
                    new_positions.append(trial_pos)
                    new_population.append(trial_al)
                    new_delay_factors.append(trial_delay if self.params.use_delay_factors else delay_factors[i])
                    new_objectives.append(trial_obj)
                else:
                    new_positions.append(positions[i])
                    new_population.append(population[i])
                    new_delay_factors.append(delay_factors[i])
                    new_objectives.append(objectives[i])
            
            positions = new_positions
            population = new_population
            delay_factors = new_delay_factors
            
            if self.params.use_local_search:
                population, delay_factors = self._local_search(population, objectives, delay_factors)
            
            iteration += 1
        
        if self.params.use_delay_factors:
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
            'algorithm_params': asdict(self.params)
        }
    
    def _initialize_population(self) -> tuple:
        """初始化种群"""
        population = []
        delay_factors = []
        
        for _ in range(self.params.population_size):
            if self.params.initialization_strategy == "random":
                if self.params.use_delay_factors:
                    al, delays = self.decoder.encode_random_with_delay(self.rng)
                    delay_factors.append(delays)
                else:
                    al = self.decoder.encode_random(self.rng)
            elif self.params.initialization_strategy == "forward":
                al = self.decoder.encode_forward()
                if self.params.use_delay_factors:
                    delays = [self.rng.random() for _ in range(self.n)]
                    delays[0] = 0.0
                    delays[self.n - 1] = 0.0
                    delay_factors.append(delays)
            else:
                if self.params.use_delay_factors:
                    al, delays = self.decoder.encode_random_with_delay(self.rng)
                    delay_factors.append(delays)
                else:
                    al = self.decoder.encode_random(self.rng)
            
            al = self.decoder.repair(al)
            population.append(al)
        
        return population, delay_factors
    
    def _mutation(self, positions: List[np.ndarray], objectives: List[float], 
                  current_idx: int, iteration: int) -> np.ndarray:
        """变异操作"""
        n = self.n
        
        if self.params.use_adaptive_F:
            F = self.params.F_max * np.exp(iteration * np.log(self.params.F_min / self.params.F_max) / self.params.max_iterations)
        else:
            F = self.params.F
        
        mutant = np.zeros(n, dtype=np.float64)
        
        if self.params.mutation_strategy == "rand/1":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
            mutant = positions[r1] + F * (positions[r2] - positions[r3])
        
        elif self.params.mutation_strategy == "best/1":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2 = self.rng.choice(indices, size=2, replace=False)
            mutant = positions[best_idx] + F * (positions[r1] - positions[r2])
        
        elif self.params.mutation_strategy == "rand/2":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3, r4, r5 = self.rng.choice(indices, size=5, replace=False)
            mutant = positions[r1] + F * (positions[r2] - positions[r3]) + F * (positions[r4] - positions[r5])
        
        elif self.params.mutation_strategy == "best/2":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2, r3, r4 = self.rng.choice(indices, size=4, replace=False)
            mutant = positions[best_idx] + F * (positions[r1] - positions[r2]) + F * (positions[r3] - positions[r4])
        
        elif self.params.mutation_strategy == "current-to-best/1":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2 = self.rng.choice(indices, size=2, replace=False)
            mutant = positions[current_idx] + F * (positions[best_idx] - positions[current_idx]) + F * (positions[r1] - positions[r2])
        
        elif self.params.mutation_strategy == "adaptive":
            F = self.params.F_max * np.exp(iteration * np.log(self.params.F_min / self.params.F_max) / self.params.max_iterations)
            L = np.exp(-iteration / self.params.max_iterations)
            
            if self.rng.random() < L:
                indices = [i for i in range(self.params.population_size) if i != current_idx]
                r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
                mutant = positions[r1] + F * (positions[r2] - positions[r3])
            else:
                best_idx = np.argmin(objectives)
                indices = [i for i in range(self.params.population_size) if i != current_idx]
                r1, r2 = self.rng.choice(indices, size=2, replace=False)
                mutant = positions[current_idx] + F * (positions[best_idx] - positions[current_idx]) + F * (positions[r1] - positions[r2])
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray, iteration: int) -> np.ndarray:
        """交叉操作"""
        trial = target.copy()
        
        if self.params.use_adaptive_CR:
            CR = self.params.CR_min * np.exp(iteration * np.log(self.params.CR_max / self.params.CR_min) / self.params.max_iterations)
        else:
            CR = self.params.CR
        
        if self.params.crossover_strategy == "bin":
            for j in range(self.n):
                if self.rng.random() < CR:
                    trial[j] = mutant[j]
        
        elif self.params.crossover_strategy == "exp":
            j_rand = self.rng.integers(0, self.n)
            for j in range(self.n):
                if self.rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
        
        return trial
    
    def _local_search(self, population: List[List[int]], objectives: List[float], delay_factors: List[List[float]]) -> tuple:
        """局部搜索"""
        top_indices = np.argsort(objectives)[:self.params.local_search_top]
        
        for idx in top_indices:
            if self.n >= 2:
                i, j = self.rng.choice(self.n, size=2, replace=False)
                population[idx][i], population[idx][j] = population[idx][j], population[idx][i]
                population[idx] = self.decoder.repair(population[idx])
                
                if self.params.use_delay_factors and delay_factors[idx] is not None:
                    for k in range(len(delay_factors[idx])):
                        if self.rng.random() < 0.1:
                            delay_factors[idx][k] = self.rng.random()
                    delay_factors[idx][0] = 0.0
                    delay_factors[idx][self.n - 1] = 0.0
        
        return population, delay_factors
