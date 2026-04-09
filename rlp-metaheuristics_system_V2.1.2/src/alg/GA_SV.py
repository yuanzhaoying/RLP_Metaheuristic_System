"""
遗传算法（Genetic Algorithm）- 基于位移编码版本

算法简介：
    遗传算法是一种基于自然选择和遗传机制的优化算法。
    通过选择、交叉、变异等操作，模拟生物进化过程来搜索最优解。

SV编码特点：
    - 编码是位移向量（连续值编码）
    - 每个活动的位移值在 [0, LS-ES] 范围内

算子汇总：
    1. 选择算子
       - roulette：轮盘赌选择
       - tournament：锦标赛选择
    
    2. 交叉算子
       - arithmetic：算术交叉
       - blend：混合交叉
       - sbx：模拟二进制交叉
    
    3. 变异算子
       - uniform：均匀变异
       - gaussian：高斯变异
       - polynomial：多项式变异
    
    4. 局部搜索
       - none：不使用局部搜索
       - swap：交换局部搜索
"""

from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.shift_vector_evaluator import ShiftVectorEvaluator
from ..psp.shift_vector_decoder import ShiftVectorDecoder
from .operators import RandomGenerator


@dataclass
class GAParamsSV:
    """遗传算法参数（位移编码）"""
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


class GeneticAlgorithmSV:
    """遗传算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: GAParamsSV):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ShiftVectorEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ShiftVectorDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        """运行遗传算法"""
        start_time = time.time()
        convergence = []
        
        population = self._initialize_population()
        
        best_displacement = None
        best_objective = float('inf')
        
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
                    best_displacement = individual.copy()
            
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
    
    def _initialize_population(self) -> List[List[int]]:
        """初始化种群"""
        population = []
        
        for _ in range(self.params.population_size):
            if self.params.initialization_strategy == "random":
                individual = self.decoder.encode_random(self.rng)
            else:
                individual = self.decoder.encode_random(self.rng)
            
            population.append(individual)
        
        return population
    
    def _roulette_wheel_selection(self, probs: List[float]) -> int:
        """轮盘赌选择"""
        r = self.rng.random()
        cumsum = 0.0
        for i, prob in enumerate(probs):
            cumsum += prob
            if r <= cumsum:
                return i
        return len(probs) - 1
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """锦标赛选择"""
        selected = []
        for _ in range(self.params.population_size):
            indices = self.rng.choice(len(population), size=self.params.tournament_size, replace=False)
            best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
            selected.append(population[best_idx].copy())
        return selected
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """交叉操作（返回一个子代）"""
        if self.rng.random() >= self.params.crossover_rate:
            return parent1.copy()
        
        child = parent1.copy()
        
        if self.params.crossover_strategy == "arithmetic":
            alpha = self.rng.rng.uniform(0, 1)
            for j in range(self.n):
                child[j] = int(alpha * parent1[j] + (1 - alpha) * parent2[j])
        
        elif self.params.crossover_strategy == "blend":
            for j in range(self.n):
                min_val = min(parent1[j], parent2[j])
                max_val = max(parent1[j], parent2[j])
                range_val = max_val - min_val
                child[j] = int(self.rng.rng.uniform(min_val - 0.1 * range_val, max_val + 0.1 * range_val))
        
        elif self.params.crossover_strategy == "sbx":
            eta = 20
            for j in range(self.n):
                u = self.rng.rng.uniform(0, 1)
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                child[j] = int(0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j]))
        
        for j in range(self.n):
            max_shift = self.decoder.get_max_shift(j)
            child[j] = max(0, min(child[j], max_shift))
        
        return child
    
    def _crossover_two_children(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """交叉操作（返回两个子代）"""
        if self.rng.random() >= self.params.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = self._crossover(parent1, parent2)
        child2 = self._crossover(parent2, parent1)
        
        return child1, child2
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """变异操作"""
        mutated = individual.copy()
        
        for j in range(self.n):
            if self.rng.random() < self.params.mutation_rate:
                max_shift = self.decoder.get_max_shift(j)
                
                if self.params.mutation_strategy == "uniform":
                    mutated[j] = self.rng.integers(0, max_shift + 1)
                
                elif self.params.mutation_strategy == "gaussian":
                    mutated[j] = int(mutated[j] + self.rng.rng.normal(0, max_shift * 0.1))
                    mutated[j] = max(0, min(mutated[j], max_shift))
                
                elif self.params.mutation_strategy == "polynomial":
                    eta = 20
                    u = self.rng.rng.uniform(0, 1)
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    mutated[j] = int(mutated[j] + delta * max_shift)
                    mutated[j] = max(0, min(mutated[j], max_shift))
        
        return mutated
    
    def _local_search(self, population: List[List[int]], objectives: List[float]) -> List[List[int]]:
        """局部搜索"""
        top_indices = np.argsort(objectives)[:5]
        
        for idx in top_indices:
            if self.params.local_search_strategy == "swap":
                for _ in range(3):
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    j = self.rng.integers(0, self.n)
                    max_shift = self.decoder.get_max_shift(j)
                    new_individual = population[idx].copy()
                    new_individual[j] = self.rng.integers(0, max_shift + 1)
                    new_obj, _ = self.evaluator.evaluate(new_individual)
                    if new_obj < objectives[idx]:
                        population[idx] = new_individual
                        objectives[idx] = new_obj
        
        return population
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
