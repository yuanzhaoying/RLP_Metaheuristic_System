"""
差分进化算法（Differential Evolution）- 优先级编码版本

算子汇总（48种组合）：
    1. 变异算子 - 6种：rand/1, best/1, rand/2, best/2, current-to-best/1, adaptive
    2. 交叉算子 - 2种：bin, exp
    3. 初始化策略 - 2种：random, zero
    4. 局部搜索 - 2种：True, False
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
class DEParamsPRI:
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    max_iterations: int = 100
    F: float = 0.5
    CR: float = 0.9
    mutation_strategy: str = "rand/1"
    crossover_strategy: str = "bin"
    initialization_strategy: str = "random"
    use_local_search: bool = False
    local_search_top: int = 5


class DifferentialEvolutionPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: DEParamsPRI):
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
        
        population = self._initialize_population()
        best_priority = population[0].copy()
        best_objective = float('inf')
        
        for individual in population:
            obj, _ = self.evaluator.evaluate(individual)
            if obj < best_objective:
                best_objective = obj
                best_priority = individual.copy()
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit and
               iteration < self.params.max_iterations):
            
            objectives = []
            for individual in population:
                obj, _ = self.evaluator.evaluate(individual)
                objectives.append(obj)
                if obj < best_objective:
                    best_objective = obj
                    best_priority = individual.copy()
            
            convergence.append(best_objective)
            new_population = []
            
            for i in range(self.params.population_size):
                mutant = self._mutation(population, objectives, i)
                trial = self._crossover(population[i], mutant)
                trial = self.decoder.repair(trial)
                trial_obj, _ = self.evaluator.evaluate(trial)
                
                if trial_obj <= objectives[i]:
                    new_population.append(trial)
                else:
                    new_population.append(population[i])
            
            population = new_population
            
            if self.params.use_local_search:
                population = self._local_search(population, objectives)
            
            iteration += 1
        
        start_times, _ = self.decoder.decode(best_priority)
        runtime = time.time() - start_time
        
        return {
            'best_priority': best_priority,
            'best_start_times': start_times.tolist(),
            'best_objective': best_objective,
            'n_evaluations': self.evaluator.n_evaluations,
            'runtime': runtime,
            'convergence': convergence,
            'algorithm_params': asdict(self.params)
        }
    
    def _initialize_population(self) -> List[List[float]]:
        population = []
        for _ in range(self.params.population_size):
            if self.params.initialization_strategy == "random":
                individual = self.decoder.encode_random(self.rng)
            else:
                individual = [0.0] * self.n
                individual[0] = 1.0
                individual[self.n - 1] = 0.0
            individual = self.decoder.repair(individual)
            population.append(individual)
        return population
    
    def _mutation(self, population: List[List[float]], objectives: List[float], 
                  current_idx: int) -> np.ndarray:
        F = self.params.F
        mutant = np.zeros(self.n, dtype=np.float64)
        
        if self.params.mutation_strategy == "rand/1":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
            mutant = np.array(population[r1]) + F * (np.array(population[r2]) - np.array(population[r3]))
        elif self.params.mutation_strategy == "best/1":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2 = self.rng.choice(indices, size=2, replace=False)
            mutant = np.array(population[best_idx]) + F * (np.array(population[r1]) - np.array(population[r2]))
        elif self.params.mutation_strategy == "rand/2":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3, r4, r5 = self.rng.choice(indices, size=5, replace=False)
            mutant = np.array(population[r1]) + F * (np.array(population[r2]) - np.array(population[r3])) + F * (np.array(population[r4]) - np.array(population[r5]))
        elif self.params.mutation_strategy == "best/2":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2, r3, r4 = self.rng.choice(indices, size=4, replace=False)
            mutant = np.array(population[best_idx]) + F * (np.array(population[r1]) - np.array(population[r2])) + F * (np.array(population[r3]) - np.array(population[r4]))
        elif self.params.mutation_strategy == "current-to-best/1":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2 = self.rng.choice(indices, size=2, replace=False)
            mutant = np.array(population[current_idx]) + F * (np.array(population[best_idx]) - np.array(population[current_idx])) + F * (np.array(population[r1]) - np.array(population[r2]))
        else:  # adaptive
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
            mutant = np.array(population[r1]) + F * (np.array(population[r2]) - np.array(population[r3]))
        
        return mutant
    
    def _crossover(self, target: List[float], mutant: np.ndarray) -> List[float]:
        trial = target.copy()
        CR = self.params.CR
        
        if self.params.crossover_strategy == "bin":
            for j in range(self.n):
                if self.rng.random() < CR:
                    trial[j] = float(mutant[j])
        else:  # exp
            j_rand = self.rng.integers(0, self.n)
            for j in range(self.n):
                if self.rng.random() < CR or j == j_rand:
                    trial[j] = float(mutant[j])
        
        return trial
    
    def _local_search(self, population: List[List[float]], objectives: List[float]) -> List[List[float]]:
        top_indices = np.argsort(objectives)[:self.params.local_search_top]
        
        for idx in top_indices:
            for _ in range(3):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                new_individual = population[idx].copy()
                for i in range(1, self.n - 1):
                    if self.rng.random() < 0.1:
                        new_individual[i] = self.rng.random()
                new_individual = self.decoder.repair(new_individual)
                new_obj, _ = self.evaluator.evaluate(new_individual)
                if new_obj < objectives[idx]:
                    population[idx] = new_individual
                    objectives[idx] = new_obj
        
        return population
