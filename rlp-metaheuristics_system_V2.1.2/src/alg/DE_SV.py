"""
差分进化算法（Differential Evolution）- 基于位移编码版本

算子汇总：
    1. 变异算子
       - rand/1, best/1, rand/2, best/2, current-to-best/1, adaptive
    
    2. 交叉算子
       - bin：二项交叉
       - exp：指数交叉
    
    3. 局部搜索
       - True/False
"""

from typing import List
import time
import numpy as np
from dataclasses import dataclass, asdict
from ..psp.psplib_io import RCPSPInstance
from ..psp.shift_vector_evaluator import ShiftVectorEvaluator
from ..psp.shift_vector_decoder import ShiftVectorDecoder
from .operators import RandomGenerator


@dataclass
class DEParamsSV:
    """差分进化算法参数（位移编码）"""
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


class DifferentialEvolutionSV:
    """差分进化算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: DEParamsSV):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ShiftVectorEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ShiftVectorDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        """运行差分进化算法"""
        start_time = time.time()
        convergence = []
        
        population = self._initialize_population()
        
        best_displacement = None
        best_objective = float('inf')
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit and
               iteration < self.params.max_iterations):
            
            objectives = []
            for ind in population:
                obj, _ = self.evaluator.evaluate(ind)
                objectives.append(obj)
                
                if obj < best_objective:
                    best_objective = obj
                    best_displacement = ind.copy()
            
            convergence.append(best_objective)
            
            new_population = []
            new_objectives = []
            
            for i in range(self.params.population_size):
                mutant = self._mutation(population, objectives, i, iteration)
                trial = self._crossover(population[i], mutant, iteration)
                
                trial = self._clip_to_bounds(trial)
                
                trial_obj, _ = self.evaluator.evaluate(trial)
                
                if trial_obj <= objectives[i]:
                    new_population.append(trial)
                    new_objectives.append(trial_obj)
                else:
                    new_population.append(population[i])
                    new_objectives.append(objectives[i])
            
            population = new_population
            
            if self.params.use_local_search:
                population = self._local_search(population, objectives)
            
            iteration += 1
        
        start_times, _ = self.decoder.decode(best_displacement)
        
        runtime = time.time() - start_time
        
        return {
            'best_displacement': best_displacement,
            'best_start_times': start_times.tolist(),
            'best_objective': best_objective,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': asdict(self.params)
        }
    
    def _initialize_population(self) -> List[List[int]]:
        """初始化种群"""
        population = []
        for _ in range(self.params.population_size):
            ind = self.decoder.encode_random(self.rng)
            population.append(ind)
        return population
    
    def _mutation(self, population: List[List[int]], objectives: List[float], 
                  current_idx: int, iteration: int) -> List[int]:
        """变异操作"""
        if self.params.use_adaptive_F:
            F = self.params.F_max * np.exp(iteration * np.log(self.params.F_min / self.params.F_max) / self.params.max_iterations)
        else:
            F = self.params.F
        
        mutant = [0] * self.n
        
        if self.params.mutation_strategy == "rand/1":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
            for j in range(self.n):
                mutant[j] = int(population[r1][j] + F * (population[r2][j] - population[r3][j]))
        
        elif self.params.mutation_strategy == "best/1":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2 = self.rng.choice(indices, size=2, replace=False)
            for j in range(self.n):
                mutant[j] = int(population[best_idx][j] + F * (population[r1][j] - population[r2][j]))
        
        elif self.params.mutation_strategy == "rand/2":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3, r4, r5 = self.rng.choice(indices, size=5, replace=False)
            for j in range(self.n):
                mutant[j] = int(population[r1][j] + F * (population[r2][j] - population[r3][j]) + F * (population[r4][j] - population[r5][j]))
        
        elif self.params.mutation_strategy == "best/2":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2, r3, r4 = self.rng.choice(indices, size=4, replace=False)
            for j in range(self.n):
                mutant[j] = int(population[best_idx][j] + F * (population[r1][j] - population[r2][j]) + F * (population[r3][j] - population[r4][j]))
        
        elif self.params.mutation_strategy == "current-to-best/1":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2 = self.rng.choice(indices, size=2, replace=False)
            for j in range(self.n):
                mutant[j] = int(population[current_idx][j] + F * (population[best_idx][j] - population[current_idx][j]) + F * (population[r1][j] - population[r2][j]))
        
        elif self.params.mutation_strategy == "adaptive":
            F = self.params.F_max * np.exp(iteration * np.log(self.params.F_min / self.params.F_max) / self.params.max_iterations)
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
            for j in range(self.n):
                mutant[j] = int(population[r1][j] + F * (population[r2][j] - population[r3][j]))
        
        return mutant
    
    def _crossover(self, target: List[int], mutant: List[int], iteration: int) -> List[int]:
        """交叉操作"""
        trial = target.copy()
        
        if self.params.use_adaptive_CR:
            CR = self.params.CR_min * np.exp(iteration * np.log(self.params.CR_max / self.params.CR_min) / self.params.max_iterations)
        else:
            CR = self.params.CR
        
        if self.params.crossover_strategy == "bin":
            j_rand = self.rng.integers(0, self.n)
            for j in range(self.n):
                if self.rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
        
        elif self.params.crossover_strategy == "exp":
            j = self.rng.integers(0, self.n)
            while True:
                trial[j] = mutant[j]
                j = (j + 1) % self.n
                if self.rng.random() >= CR or j == self.rng.integers(0, self.n):
                    break
        
        return trial
    
    def _clip_to_bounds(self, individual: List[int]) -> List[int]:
        """将个体裁剪到有效范围"""
        clipped = individual.copy()
        for j in range(self.n):
            max_shift = self.decoder.get_max_shift(j)
            clipped[j] = max(0, min(clipped[j], max_shift))
        return clipped
    
    def _local_search(self, population: List[List[int]], objectives: List[float]) -> List[List[int]]:
        """局部搜索"""
        top_indices = np.argsort(objectives)[:self.params.local_search_top]
        
        for idx in top_indices:
            for _ in range(3):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                j = self.rng.integers(0, self.n)
                max_shift = self.decoder.get_max_shift(j)
                new_ind = population[idx].copy()
                new_ind[j] = self.rng.integers(0, max_shift + 1)
                new_obj, _ = self.evaluator.evaluate(new_ind)
                if new_obj < objectives[idx]:
                    population[idx] = new_ind
                    objectives[idx] = new_obj
        
        return population
