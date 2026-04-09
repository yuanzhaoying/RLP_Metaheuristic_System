"""
遗传算法（Genetic Algorithm）- 活动列表编码版本

算法简介：
    遗传算法是一种基于自然选择和遗传机制的优化算法。
    通过选择、交叉、变异等操作，模拟生物进化过程来搜索最优解。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 使用排列编码专用的交叉和变异算子

算子汇总：
    1. 选择算子（Selection）
       - roulette：轮盘赌选择
       - tournament：锦标赛选择
    
    2. 交叉算子
       - ox1：顺序交叉（Order Crossover）
       - pmx：部分映射交叉（Partially Mapped Crossover）
       - order：保序交叉
    
    3. 变异算子
       - swap：交换两个活动
       - insertion：插入操作
       - inversion：逆序操作
       - scramble：打乱操作
    
    4. 修复算子
       - 前置约束修复：确保排列满足优先关系
"""

from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.activity_list_evaluator import ActivityListEvaluator
from ..psp.activity_list_decoder import ActivityListDecoder
from .operators import RandomGenerator, crossover_ox1, crossover_pmx, crossover_order


@dataclass
class GAParamsAL:
    """遗传算法参数（活动列表编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    time_limit: float = 60.0
    selection_strategy: str = "tournament"
    crossover_strategy: str = "ox1"
    mutation_strategy: str = "swap"
    initialization_strategy: str = "random"
    tournament_size: int = 3
    use_repair: bool = True
    elitism: bool = True
    local_search_strategy: str = "none"
    local_search_interval: int = 5
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class GeneticAlgorithmAL:
    """遗传算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: GAParamsAL):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ActivityListEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ActivityListDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        """运行遗传算法"""
        start_time = time.time()
        convergence = []
        
        population, delay_factors = self._initialize_population()
        
        best_activity_list = None
        best_delay_factors = None
        best_objective = float('inf')
        
        generation = 0
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            fitness_scores = []
            objectives = []
            for i, individual in enumerate(population):
                if self.params.use_delay_factors:
                    obj, _ = self.evaluator.evaluate(individual, delay_factors[i])
                else:
                    obj, _ = self.evaluator.evaluate(individual)
                objectives.append(obj)
                fitness = 1.0 / (obj + 1e-6)
                fitness_scores.append(fitness)
                
                if obj < best_objective:
                    best_objective = obj
                    best_activity_list = individual.copy()
                    if self.params.use_delay_factors:
                        best_delay_factors = delay_factors[i].copy()
            
            convergence.append(best_objective)
            
            if self.params.selection_strategy == "roulette":
                total_fitness = sum(fitness_scores)
                if total_fitness > 0:
                    selection_probs = [f / total_fitness for f in fitness_scores]
                else:
                    selection_probs = [1.0 / len(population)] * len(population)
                
                new_population = []
                new_delay_factors = []
                for _ in range(self.params.population_size):
                    parent1_idx = self._roulette_wheel_selection(selection_probs)
                    parent2_idx = self._roulette_wheel_selection(selection_probs)
                    
                    child, child_delay = self._crossover(
                        population[parent1_idx], population[parent2_idx],
                        delay_factors[parent1_idx] if self.params.use_delay_factors else None,
                        delay_factors[parent2_idx] if self.params.use_delay_factors else None
                    )
                    child, child_delay = self._mutate(child, child_delay)
                    new_population.append(child)
                    if self.params.use_delay_factors:
                        new_delay_factors.append(child_delay)
                
                population = new_population
                if self.params.use_delay_factors:
                    delay_factors = new_delay_factors
            
            elif self.params.selection_strategy == "tournament":
                parents, parent_delays = self._tournament_selection(population, fitness_scores, delay_factors)
                
                new_population = []
                new_delay_factors = []
                for i in range(0, len(parents), 2):
                    if i + 1 >= len(parents):
                        new_population.append(parents[i])
                        if self.params.use_delay_factors:
                            new_delay_factors.append(parent_delays[i])
                        break
                    
                    child1, child2, delay1, delay2 = self._crossover_two_children(
                        parents[i], parents[i+1],
                        parent_delays[i] if self.params.use_delay_factors else None,
                        parent_delays[i+1] if self.params.use_delay_factors else None
                    )
                    child1, delay1 = self._mutate(child1, delay1)
                    child2, delay2 = self._mutate(child2, delay2)
                    new_population.extend([child1, child2])
                    if self.params.use_delay_factors:
                        new_delay_factors.extend([delay1, delay2])
                
                new_population = new_population[:self.params.population_size]
                if self.params.use_delay_factors:
                    new_delay_factors = new_delay_factors[:self.params.population_size]
                
                if self.params.elitism:
                    combined = population + new_population
                    combined_obj = []
                    for idx, ind in enumerate(combined):
                        if self.params.use_delay_factors:
                            if idx < len(population):
                                obj, _ = self.evaluator.evaluate(ind, delay_factors[idx])
                            else:
                                obj, _ = self.evaluator.evaluate(ind, new_delay_factors[idx - len(population)])
                        else:
                            obj, _ = self.evaluator.evaluate(ind)
                        combined_obj.append(obj)
                    
                    sorted_indices = np.argsort(combined_obj)
                    population = [combined[i] for i in sorted_indices[:self.params.population_size]]
                    if self.params.use_delay_factors:
                        combined_delays = delay_factors + new_delay_factors
                        delay_factors = [combined_delays[i] for i in sorted_indices[:self.params.population_size]]
                else:
                    population = new_population
                    if self.params.use_delay_factors:
                        delay_factors = new_delay_factors
            
            if self.params.local_search_strategy != "none" and generation % self.params.local_search_interval == 0:
                population, delay_factors = self._local_search(population, objectives, delay_factors)
            
            generation += 1
        
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
            'algorithm_params': self._params_to_dict(self.params)
        }
    
    def _initialize_population(self) -> Tuple[List[List[int]], List[List[float]]]:
        """初始化种群"""
        population = []
        delay_factors = []
        
        for _ in range(self.params.population_size):
            if self.params.initialization_strategy == "random":
                if self.params.use_delay_factors:
                    individual, delays = self.decoder.encode_random_with_delay(self.rng)
                    delay_factors.append(delays)
                else:
                    individual = self.decoder.encode_random(self.rng)
            
            elif self.params.initialization_strategy == "forward":
                individual = self.decoder.encode_forward()
                if self.params.use_delay_factors:
                    delays = [self.rng.random() for _ in range(self.n)]
                    delays[0] = 0.0
                    delays[self.n - 1] = 0.0
                    delay_factors.append(delays)
            
            else:
                if self.params.use_delay_factors:
                    individual, delays = self.decoder.encode_random_with_delay(self.rng)
                    delay_factors.append(delays)
                else:
                    individual = self.decoder.encode_random(self.rng)
            
            if self.params.use_repair:
                individual = self.decoder.repair(individual)
            
            population.append(individual)
        
        return population, delay_factors
    
    def _roulette_wheel_selection(self, probs: List[float]) -> int:
        """轮盘赌选择"""
        r = self.rng.random()
        cumsum = 0.0
        for i, prob in enumerate(probs):
            cumsum += prob
            if r <= cumsum:
                return i
        return len(probs) - 1
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float], delay_factors: List[List[float]]) -> Tuple[List[List[int]], List[List[float]]]:
        """锦标赛选择"""
        selected = []
        selected_delays = []
        for _ in range(self.params.population_size):
            indices = self.rng.choice(len(population), size=self.params.tournament_size, replace=False)
            best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
            selected.append(population[best_idx].copy())
            if self.params.use_delay_factors:
                selected_delays.append(delay_factors[best_idx].copy())
        return selected, selected_delays
    
    def _crossover(self, parent1: List[int], parent2: List[int], delay1: List[float] = None, delay2: List[float] = None) -> Tuple[List[int], List[float]]:
        """交叉操作（返回一个子代）"""
        child_delay = None
        if self.rng.random() >= self.params.crossover_rate:
            child_delay = delay1.copy() if delay1 is not None else None
            return parent1.copy(), child_delay
        
        if self.params.crossover_strategy == "ox1":
            child1, _ = crossover_ox1(parent1, parent2, self.rng)
            child = child1
        elif self.params.crossover_strategy == "pmx":
            child1, _ = crossover_pmx(parent1, parent2, self.rng)
            child = child1
        elif self.params.crossover_strategy == "order":
            child1, _ = crossover_order(parent1, parent2, self.rng)
            child = child1
        else:
            child = parent1.copy()
        
        if self.params.use_repair:
            child = self.decoder.repair(child)
        
        if self.params.use_delay_factors and delay1 is not None and delay2 is not None:
            child_delay = []
            for i in range(len(child)):
                if self.rng.random() < 0.5:
                    child_delay.append(delay1[i] if i < len(delay1) else 0.0)
                else:
                    child_delay.append(delay2[i] if i < len(delay2) else 0.0)
        
        return child, child_delay
    
    def _crossover_two_children(self, parent1: List[int], parent2: List[int], delay1: List[float] = None, delay2: List[float] = None) -> Tuple[List[int], List[int], List[float], List[float]]:
        """交叉操作（返回两个子代）"""
        delay_c1, delay_c2 = None, None
        if self.rng.random() >= self.params.crossover_rate:
            delay_c1 = delay1.copy() if delay1 is not None else None
            delay_c2 = delay2.copy() if delay2 is not None else None
            return parent1.copy(), parent2.copy(), delay_c1, delay_c2
        
        if self.params.crossover_strategy == "ox1":
            child1, child2 = crossover_ox1(parent1, parent2, self.rng)
        elif self.params.crossover_strategy == "pmx":
            child1, child2 = crossover_pmx(parent1, parent2, self.rng)
        elif self.params.crossover_strategy == "order":
            child1, child2 = crossover_order(parent1, parent2, self.rng)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        if self.params.use_repair:
            child1 = self.decoder.repair(child1)
            child2 = self.decoder.repair(child2)
        
        if self.params.use_delay_factors and delay1 is not None and delay2 is not None:
            delay_c1, delay_c2 = [], []
            for i in range(len(child1)):
                if self.rng.random() < 0.5:
                    delay_c1.append(delay1[i] if i < len(delay1) else 0.0)
                    delay_c2.append(delay2[i] if i < len(delay2) else 0.0)
                else:
                    delay_c1.append(delay2[i] if i < len(delay2) else 0.0)
                    delay_c2.append(delay1[i] if i < len(delay1) else 0.0)
        
        return child1, child2, delay_c1, delay_c2
    
    def _mutate(self, individual: List[int], delay_factors: List[float] = None) -> Tuple[List[int], List[float]]:
        """变异操作"""
        mutated = individual.copy()
        mutated_delay = delay_factors.copy() if delay_factors is not None else None
        n = len(mutated)
        
        if n < 2:
            return mutated, mutated_delay
        
        if self.params.mutation_strategy == "swap":
            if self.rng.random() < self.params.mutation_rate:
                i, j = self.rng.choice(n, size=2, replace=False)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        
        elif self.params.mutation_strategy == "insertion":
            if self.rng.random() < self.params.mutation_rate:
                i = self.rng.integers(0, n)
                j = self.rng.integers(0, n)
                gene = mutated.pop(i)
                mutated.insert(j, gene)
        
        elif self.params.mutation_strategy == "inversion":
            if self.rng.random() < self.params.mutation_rate:
                i, j = sorted(self.rng.choice(n, size=2, replace=False))
                mutated[i:j+1] = mutated[i:j+1][::-1]
        
        elif self.params.mutation_strategy == "scramble":
            if self.rng.random() < self.params.mutation_rate:
                i, j = sorted(self.rng.choice(n, size=2, replace=False))
                subset = mutated[i:j+1]
                subset = self.rng.shuffle(subset)
                mutated[i:j+1] = subset
        
        elif self.params.mutation_strategy == "swap_based":
            n_swaps = 2
            for _ in range(n_swaps):
                if self.rng.random() < self.params.mutation_rate:
                    i, j = self.rng.choice(n, size=2, replace=False)
                    mutated[i], mutated[j] = mutated[j], mutated[i]
        
        if self.params.use_repair:
            mutated = self.decoder.repair(mutated)
        
        if self.params.use_delay_factors and mutated_delay is not None:
            for i in range(len(mutated_delay)):
                if self.rng.random() < self.params.delay_mutation_rate:
                    mutated_delay[i] = self.rng.random()
            mutated_delay[0] = 0.0
            mutated_delay[self.n - 1] = 0.0
        
        return mutated, mutated_delay
    
    def _local_search(self, population: List[List[int]], objectives: List[float], delay_factors: List[List[float]]) -> Tuple[List[List[int]], List[List[float]]]:
        """局部搜索"""
        top_indices = np.argsort(objectives)[:5]
        
        for idx in top_indices:
            if self.params.local_search_strategy == "swap":
                for _ in range(3):
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    n = len(population[idx])
                    if n < 2:
                        continue
                    i, j = self.rng.choice(n, size=2, replace=False)
                    new_individual = population[idx].copy()
                    new_individual[i], new_individual[j] = new_individual[j], new_individual[i]
                    if self.params.use_repair:
                        new_individual = self.decoder.repair(new_individual)
                    
                    new_delay = delay_factors[idx].copy() if self.params.use_delay_factors else None
                    if self.params.use_delay_factors:
                        for k in range(len(new_delay)):
                            if self.rng.random() < 0.1:
                                new_delay[k] = self.rng.random()
                        new_delay[0] = 0.0
                        new_delay[self.n - 1] = 0.0
                    
                    if self.params.use_delay_factors:
                        new_obj, _ = self.evaluator.evaluate(new_individual, new_delay)
                    else:
                        new_obj, _ = self.evaluator.evaluate(new_individual)
                    
                    if new_obj < objectives[idx]:
                        population[idx] = new_individual
                        objectives[idx] = new_obj
                        if self.params.use_delay_factors:
                            delay_factors[idx] = new_delay
        
        return population, delay_factors
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
