"""
遗传算法（Genetic Algorithm）- 优先级编码版本

算法简介：
    遗传算法是一种基于自然选择和遗传机制的优化算法。
    通过选择、交叉、变异等操作，模拟生物进化过程来搜索最优解。

PRI编码特点：
    - 编码是[0,1]之间的向量
    - 该值同时表示优先级和延迟因子
    - priority[0] = 1.0（第一个活动）
    - priority[n-1] = 0.0（最后一个活动）

算子汇总（384种组合）：
    1. 选择算子 - 2种：roulette, tournament
    2. 交叉算子 - 3种：arithmetic, blend, sbx
    3. 变异算子 - 4种：uniform, gaussian, polynomial, swap
    4. 初始化策略 - 2种：random, zero
    5. 修复策略 - 2种：True, False
    6. 精英策略 - 2种：True, False
    7. 局部搜索 - 2种：none, uniform
"""

from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass, asdict
from psp.psplib_io import RCPSPInstance
from psp.priority_evaluator import PriorityEvaluator
from psp.priority_decoder import PriorityDecoder
from alg.operators import RandomGenerator


@dataclass
class GAParamsPRI:
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    time_limit: float = 60.0
    selection_strategy: str = "tournament"
    crossover_strategy: str = "arithmetic"
    mutation_strategy: str = "uniform"
    initialization_strategy: str = "random"
    tournament_size: int = 3
    use_repair: bool = True
    elitism: bool = True
    local_search_strategy: str = "none"
    local_search_interval: int = 5


class GeneticAlgorithmPRI:
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: GAParamsPRI):
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
        
        generation = 0
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            fitness_scores = []
            objectives = []
            for individual in population:
                obj, _ = self.evaluator.evaluate(individual)
                objectives.append(obj)
                fitness = 1.0 / (obj + 1e-6)
                fitness_scores.append(fitness)
                
                if obj < best_objective:
                    best_objective = obj
                    best_priority = individual.copy()
            
            convergence.append(best_objective)
            
            if self.params.selection_strategy == "roulette":
                total_fitness = sum(fitness_scores)
                if total_fitness > 0:
                    selection_probs = [f / total_fitness for f in fitness_scores]
                else:
                    selection_probs = [1.0 / len(population)] * len(population)
                
                new_population = []
                for _ in range(self.params.population_size):
                    parent1_idx = self._roulette_wheel_selection(selection_probs)
                    parent2_idx = self._roulette_wheel_selection(selection_probs)
                    
                    child = self._crossover(population[parent1_idx], population[parent2_idx])
                    child = self._mutate(child)
                    new_population.append(child)
                
                population = new_population
            
            elif self.params.selection_strategy == "tournament":
                parents = self._tournament_selection(population, fitness_scores)
                
                new_population = []
                for i in range(0, len(parents), 2):
                    if i + 1 >= len(parents):
                        new_population.append(parents[i])
                        break
                    
                    child1, child2 = self._crossover_two_children(parents[i], parents[i+1])
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    new_population.extend([child1, child2])
                
                new_population = new_population[:self.params.population_size]
                
                if self.params.elitism:
                    combined = population + new_population
                    combined_obj = []
                    for ind in combined:
                        obj, _ = self.evaluator.evaluate(ind)
                        combined_obj.append(obj)
                    
                    sorted_indices = np.argsort(combined_obj)
                    population = [combined[i] for i in sorted_indices[:self.params.population_size]]
                else:
                    population = new_population
            
            if self.params.local_search_strategy != "none" and generation % self.params.local_search_interval == 0:
                population = self._local_search(population, objectives)
            
            generation += 1
        
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
            
            elif self.params.initialization_strategy == "zero":
                individual = [0.0] * self.n
                individual[0] = 1.0
                individual[self.n - 1] = 0.0
            
            else:
                individual = self.decoder.encode_random(self.rng)
            
            if self.params.use_repair:
                individual = self.decoder.repair(individual)
            
            population.append(individual)
        
        return population
    
    def _roulette_wheel_selection(self, selection_probs: List[float]) -> int:
        r = self.rng.random()
        cumsum = 0.0
        for i, prob in enumerate(selection_probs):
            cumsum += prob
            if r <= cumsum:
                return i
        return len(selection_probs) - 1
    
    def _tournament_selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        selected = []
        for _ in range(self.params.population_size):
            indices = self.rng.choice(len(population), size=self.params.tournament_size, replace=False)
            best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
            selected.append(population[best_idx].copy())
        return selected
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        if self.rng.random() >= self.params.crossover_rate:
            return parent1.copy()
        
        child = []
        for i in range(len(parent1)):
            if self.params.crossover_strategy == "arithmetic":
                alpha = self.rng.random()
                child.append(alpha * parent1[i] + (1 - alpha) * parent2[i])
            
            elif self.params.crossover_strategy == "blend":
                alpha = 0.5
                min_val = min(parent1[i], parent2[i])
                max_val = max(parent1[i], parent2[i])
                range_val = max_val - min_val
                val = self.rng.rng.uniform(min_val - alpha * range_val, max_val + alpha * range_val)
                child.append(max(0.0, min(1.0, val)))
            
            elif self.params.crossover_strategy == "sbx":
                eta = 15
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                val = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child.append(max(0.0, min(1.0, val)))
            
            else:
                child.append(parent1[i] if self.rng.random() < 0.5 else parent2[i])
        
        if self.params.use_repair:
            child = self.decoder.repair(child)
        
        return child
    
    def _crossover_two_children(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        if self.rng.random() >= self.params.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = [], []
        for i in range(len(parent1)):
            if self.params.crossover_strategy == "arithmetic":
                alpha = self.rng.random()
                child1.append(alpha * parent1[i] + (1 - alpha) * parent2[i])
                child2.append(alpha * parent2[i] + (1 - alpha) * parent1[i])
            
            elif self.params.crossover_strategy == "blend":
                alpha = 0.5
                min_val = min(parent1[i], parent2[i])
                max_val = max(parent1[i], parent2[i])
                range_val = max_val - min_val
                val1 = self.rng.rng.uniform(min_val - alpha * range_val, max_val + alpha * range_val)
                val2 = self.rng.rng.uniform(min_val - alpha * range_val, max_val + alpha * range_val)
                child1.append(max(0.0, min(1.0, val1)))
                child2.append(max(0.0, min(1.0, val2)))
            
            elif self.params.crossover_strategy == "sbx":
                eta = 15
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                val1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                val2 = 0.5 * ((1 + beta) * parent2[i] + (1 - beta) * parent1[i])
                child1.append(max(0.0, min(1.0, val1)))
                child2.append(max(0.0, min(1.0, val2)))
            
            else:
                child1.append(parent1[i] if self.rng.random() < 0.5 else parent2[i])
                child2.append(parent2[i] if self.rng.random() < 0.5 else parent1[i])
        
        if self.params.use_repair:
            child1 = self.decoder.repair(child1)
            child2 = self.decoder.repair(child2)
        
        return child1, child2
    
    def _mutate(self, individual: List[float]) -> List[float]:
        mutated = individual.copy()
        
        if self.params.mutation_strategy == "uniform":
            for i in range(1, self.n - 1):
                if self.rng.random() < self.params.mutation_rate:
                    mutated[i] = self.rng.random()
        
        elif self.params.mutation_strategy == "gaussian":
            for i in range(1, self.n - 1):
                if self.rng.random() < self.params.mutation_rate:
                    mutated[i] += self.rng.rng.normal(0, 0.1)
                    mutated[i] = max(0.0, min(1.0, mutated[i]))
        
        elif self.params.mutation_strategy == "polynomial":
            for i in range(1, self.n - 1):
                if self.rng.random() < self.params.mutation_rate:
                    eta = 20
                    u = self.rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    mutated[i] = max(0.0, min(1.0, mutated[i] + delta))
        
        elif self.params.mutation_strategy == "swap":
            if self.rng.random() < self.params.mutation_rate:
                i, j = self.rng.choice(self.n, size=2, replace=False)
                if i not in [0, self.n-1] and j not in [0, self.n-1]:
                    mutated[i], mutated[j] = mutated[j], mutated[i]
        
        if self.params.use_repair:
            mutated = self.decoder.repair(mutated)
        
        return mutated
    
    def _local_search(self, population: List[List[float]], objectives: List[float]) -> List[List[float]]:
        top_indices = np.argsort(objectives)[:5]
        
        for idx in top_indices:
            if self.params.local_search_strategy == "uniform":
                for _ in range(3):
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    new_individual = population[idx].copy()
                    for i in range(1, self.n - 1):
                        if self.rng.random() < 0.1:
                            new_individual[i] = self.rng.random()
                    
                    if self.params.use_repair:
                        new_individual = self.decoder.repair(new_individual)
                    
                    new_obj, _ = self.evaluator.evaluate(new_individual)
                    
                    if new_obj < objectives[idx]:
                        population[idx] = new_individual
                        objectives[idx] = new_obj
        
        return population
