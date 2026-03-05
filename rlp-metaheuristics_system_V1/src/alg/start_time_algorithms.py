"""
开始时间编码的元启发式算法
"""
from typing import List, Tuple, Optional
import numpy as np
import time
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator


@dataclass
class AlgorithmResult:
    """算法结果"""
    best_start_times: List[int]
    best_objective: float
    n_evaluations: int
    runtime: float
    convergence: List[float]
    algorithm_params: dict = None


def _params_to_dict(params) -> dict:
    """将参数对象转换为字典"""
    from dataclasses import asdict
    return asdict(params)


@dataclass
class GAParamsST:
    """遗传算法参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 20
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    time_limit: float = 60.0


@dataclass
class SAParamsST:
    """模拟退火参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    initial_temperature: float = 10000.0
    cooling_rate: float = 0.995
    iterations_per_temperature: int = 10
    time_limit: float = 60.0


@dataclass
class ILSParamsST:
    """迭代局部搜索参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    perturbation_strength: int = 5
    time_limit: float = 60.0
    max_iterations: int = 100


class GeneticAlgorithmST:
    """遗传算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: GAParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self) -> AlgorithmResult:
        """运行遗传算法"""
        start_time = time.time()
        convergence = []
        
        population = self._initialize_population()
        
        best_start_times = None
        best_objective = float('inf')
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            fitness_scores = []
            for individual in population:
                obj, _ = self.evaluator.evaluate(individual)
                fitness = 1.0 / (obj + 1e-6)
                fitness_scores.append(fitness)
                
                if obj < best_objective:
                    best_objective = obj
                    best_start_times = individual.copy()
            
            convergence.append(best_objective)
            
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
        
        runtime = time.time() - start_time
        
        return AlgorithmResult(
            best_start_times=best_start_times,
            best_objective=best_objective,
            n_evaluations=self.evaluator.n_evaluations,
            runtime=runtime,
            convergence=convergence,
            algorithm_params=_params_to_dict(self.params)
        )
    
    def _initialize_population(self) -> List[List[int]]:
        """初始化种群"""
        population = []
        for _ in range(self.params.population_size):
            individual = []
            for j in range(self.n):
                start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
                individual.append(start_time)
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
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """交叉操作"""
        if self.rng.random() < self.params.crossover_rate:
            cross_point = self.rng.integers(1, self.n)
            child = parent1[:cross_point] + parent2[cross_point:]
        else:
            child = parent1.copy()
        return child
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """变异操作"""
        mutated = individual.copy()
        for j in range(self.n):
            if self.rng.random() < self.params.mutation_rate:
                mutated[j] = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
        return mutated


class SimulatedAnnealingST:
    """模拟退火算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: SAParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self) -> AlgorithmResult:
        """运行模拟退火算法"""
        start_time = time.time()
        convergence = []
        
        current = self._initialize_solution()
        current_obj, _ = self.evaluator.evaluate(current)
        
        best_start_times = current.copy()
        best_objective = current_obj
        
        temperature = self.params.initial_temperature
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            for _ in range(self.params.iterations_per_temperature):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                neighbor = self._generate_neighbor(current)
                neighbor_obj, _ = self.evaluator.evaluate(neighbor)
                
                delta = neighbor_obj - current_obj
                
                if delta < 0 or self._acceptance_probability(delta, temperature) > self.rng.random():
                    current = neighbor
                    current_obj = neighbor_obj
                    
                    if current_obj < best_objective:
                        best_objective = current_obj
                        best_start_times = current.copy()
            
            convergence.append(best_objective)
            temperature *= self.params.cooling_rate
        
        runtime = time.time() - start_time
        
        return AlgorithmResult(
            best_start_times=best_start_times,
            best_objective=best_objective,
            n_evaluations=self.evaluator.n_evaluations,
            runtime=runtime,
            convergence=convergence,
            algorithm_params=_params_to_dict(self.params)
        )
    
    def _initialize_solution(self) -> List[int]:
        """初始化解"""
        solution = []
        for j in range(self.n):
            start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
            solution.append(start_time)
        return solution
    
    def _generate_neighbor(self, solution: List[int]) -> List[int]:
        """生成邻居解"""
        neighbor = solution.copy()
        j = self.rng.integers(0, self.n)
        neighbor[j] = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
        return neighbor
    
    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        """计算接受概率"""
        if temperature <= 0:
            return 0.0
        return np.exp(-delta / temperature)


class IteratedLocalSearchST:
    """迭代局部搜索算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: ILSParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self) -> AlgorithmResult:
        """运行迭代局部搜索算法"""
        start_time = time.time()
        convergence = []
        
        current = self._initialize_solution()
        current_obj, _ = self.evaluator.evaluate(current)
        
        current, current_obj = self._local_search(current, current_obj, start_time)
        
        best_start_times = current.copy()
        best_objective = current_obj
        
        iteration = 0
        while (iteration < self.params.max_iterations and 
               self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            perturbed = self._perturb(current)
            perturbed_obj, _ = self.evaluator.evaluate(perturbed)
            
            improved, improved_obj = self._local_search(perturbed, perturbed_obj, start_time)
            
            if improved_obj < current_obj:
                current = improved
                current_obj = improved_obj
                
                if current_obj < best_objective:
                    best_objective = current_obj
                    best_start_times = current.copy()
            
            convergence.append(best_objective)
            iteration += 1
        
        runtime = time.time() - start_time
        
        return AlgorithmResult(
            best_start_times=best_start_times,
            best_objective=best_objective,
            n_evaluations=self.evaluator.n_evaluations,
            runtime=runtime,
            convergence=convergence,
            algorithm_params=_params_to_dict(self.params)
        )
    
    def _initialize_solution(self) -> List[int]:
        """初始化解"""
        solution = []
        for j in range(self.n):
            start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
            solution.append(start_time)
        return solution
    
    def _local_search(self, solution: List[int], obj: float, start_time: float) -> Tuple[List[int], float]:
        """局部搜索"""
        improved = True
        best_solution = solution.copy()
        best_obj = obj
        
        while improved:
            improved = False
            
            for j in range(self.n):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                if time.time() - start_time >= self.params.time_limit:
                    break
                
                for new_start in range(self.decoder.es[j], self.decoder.ls[j] + 1):
                    if new_start == solution[j]:
                        continue
                    
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    neighbor = solution.copy()
                    neighbor[j] = new_start
                    neighbor_obj, _ = self.evaluator.evaluate(neighbor)
                    
                    if neighbor_obj < best_obj:
                        best_solution = neighbor.copy()
                        best_obj = neighbor_obj
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                solution = best_solution.copy()
                obj = best_obj
        
        return best_solution, best_obj
    
    def _perturb(self, solution: List[int]) -> List[int]:
        """扰动操作"""
        perturbed = solution.copy()
        for _ in range(self.params.perturbation_strength):
            j = self.rng.integers(0, self.n)
            perturbed[j] = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
        return perturbed


def create_algorithm_st(algo_name: str, instance: RCPSPInstance, deadline: int, params):
    """创建算法实例（开始时间编码）"""
    if algo_name == "ga":
        return GeneticAlgorithmST(instance, deadline, params), "start_time"
    elif algo_name == "sa":
        return SimulatedAnnealingST(instance, deadline, params), "start_time"
    elif algo_name == "ils":
        return IteratedLocalSearchST(instance, deadline, params), "start_time"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
