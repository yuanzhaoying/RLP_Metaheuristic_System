"""
差分进化算法（Differential Evolution）- 开始时间编码版本

算法简介：
    差分进化算法是一种基于群体差异的优化算法。
    通过变异、交叉和选择操作来进化种群。

算子汇总：
    1. 初始化算子（Initialization）
       - random：随机初始化
       - heuristic：启发式初始化
    
    2. 变异算子（Mutation）
       - rand/1: v = x_r1 + F * (x_r2 - x_r3)
       - best/1: v = x_best + F * (x_r1 - x_r2)
       - rand/2: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
       - best/2: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
       - adaptive: 自适应混合变异策略
    
    3. 交叉算子（Crossover）
       - bin：二项交叉
       - exp：指数交叉
    
    4. 选择算子（Selection）
       - 贪婪选择：保留更优的个体
    
    5. 局部搜索算子（Local Search）
       - swap：交换局部搜索
    
    6. 修复算子（Repair）
       - 前置约束修复
       - 可行范围修复
"""

from typing import List
import time
import numpy as np
from dataclasses import dataclass, asdict
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
    return asdict(params)


@dataclass
class DEParamsST:
    """差分进化算法参数（开始时间编码）
    
    注意：
        - 修复算子是必选项，始终启用
        - 自适应F和CR可以通过use_adaptive_F和use_adaptive_CR选择是否启用
    """
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    max_iterations: int = 100
    F: float = 0.1  # 默认固定值0.1
    CR: float = 0.9  # 默认固定值0.9
    use_adaptive_F: bool = False  # 是否使用自适应F
    use_adaptive_CR: bool = False  # 是否使用自适应CR
    mutation_strategy: str = "rand/1"
    crossover_strategy: str = "bin"
    initialization_strategy: str = "random"
    use_local_search: bool = False
    local_search_top: int = 5
    F_min: float = 0.3
    F_max: float = 1.5
    CR_min: float = 0.3
    CR_max: float = 1.0
    K0: float = 1.0


class DifferentialEvolutionST:
    """差分进化算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: DEParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        """运行差分进化算法"""
        start_time = time.time()
        convergence = []
        
        population = self._initialize_population()
        
        best_start_times = None
        best_objective = float('inf')
        
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
                    best_start_times = individual.copy()
            
            convergence.append(best_objective)
            
            new_population = []
            new_objectives = []
            
            for i in range(self.params.population_size):
                # 变异操作（内部自适应调整F或K）
                mutant = self._mutation(population, objectives, i, iteration)
                # 交叉操作（内部自适应调整CR）
                trial = self._crossover(population[i], mutant, iteration)
                # 修复操作（必选项）
                trial = self._repair(trial)
                
                trial_obj, _ = self.evaluator.evaluate(trial)
                
                if trial_obj <= objectives[i]:
                    new_population.append(trial)
                    new_objectives.append(trial_obj)
                else:
                    new_population.append(population[i])
                    new_objectives.append(objectives[i])
            
            population = new_population
            objectives = new_objectives
            
            # 局部搜索
            if self.params.use_local_search:
                population = self._local_search(population, objectives)
            
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
    
    def _initialize_population(self) -> List[List[int]]:
        """初始化种群"""
        population = []
        
        for _ in range(self.params.population_size):
            if self.params.initialization_strategy == "random":
                individual = []
                for j in range(self.n):
                    start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
                    individual.append(start_time)
                
                # 修复是必选项
                individual = self._repair(individual)
            
            elif self.params.initialization_strategy == "heuristic":
                individual = self._heuristic_initialization()
            
            else:
                individual = []
                for j in range(self.n):
                    start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
                    individual.append(start_time)
                
                # 修复是必选项
                individual = self._repair(individual)
            
            population.append(individual)
        
        return population
    
    def _heuristic_initialization(self) -> List[int]:
        """启发式初始化"""
        individual = [0] * self.n
        
        in_degree = [0] * self.n
        adj = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in self.inst.predecessors[i]:
                adj[j].append(i)
                in_degree[i] += 1
        
        from collections import deque
        queue = deque([i for i in range(self.n) if in_degree[i] == 0])
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        for i in topo_order:
            if self.inst.predecessors[i]:
                pf = max(individual[j] + self.inst.durations[j] for j in self.inst.predecessors[i])
            else:
                pf = 0
            
            s_min = max(self.decoder.es[i], pf)
            s_max = self.decoder.ls[i]
            
            if s_min <= s_max:
                individual[i] = self.rng.integers(s_min, s_max + 1)
            else:
                individual[i] = s_min
        
        return individual
    
    def _adaptive_parameters(self, iteration: int) -> tuple:
        """自适应参数调整"""
        # F从Fmax指数衰减到Fmin
        F = self.params.F_max * np.exp(iteration * np.log(self.params.F_min / self.params.F_max) / self.params.max_iterations)
        
        # CR从CRmin指数增长到CRmax
        CR = self.params.CR_min * np.exp(iteration * np.log(self.params.CR_max / self.params.CR_min) / self.params.max_iterations)
        
        return F, CR
    
    def _local_search(self, population: List[List[int]], objectives: List[float]) -> List[List[int]]:
        """局部搜索（对前local_search_top个最优个体执行交换操作）"""
        # 找到前local_search_top个最优个体的索引
        top_indices = np.argsort(objectives)[:self.params.local_search_top]
        
        for idx in top_indices:
            # 随机选择两个不同的位置进行交换
            if self.n >= 2:
                i, j = self.rng.choice(self.n, size=2, replace=False)
                # 执行交换
                population[idx][i], population[idx][j] = population[idx][j], population[idx][i]
                # 修复（必选项）
                population[idx] = self._repair(population[idx])
        
        return population
    
    def _mutation(self, population: List[List[int]], objectives: List[float], current_idx: int, iteration: int) -> List[int]:
        """变异操作（内部自适应调整F或K）"""
        mutant = [0] * self.n
        
        # 根据use_adaptive_F决定使用自适应F还是固定F
        if self.params.use_adaptive_F:
            F = self.params.F_max * np.exp(iteration * np.log(self.params.F_min / self.params.F_max) / self.params.max_iterations)
        else:
            F = self.params.F  # 使用固定值（默认0.1）
        
        if self.params.mutation_strategy == "rand/1":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
            
            for j in range(self.n):
                value = population[r1][j] + F * (population[r2][j] - population[r3][j])
                mutant[j] = int(value)
        
        elif self.params.mutation_strategy == "best/1":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2 = self.rng.choice(indices, size=2, replace=False)
            
            for j in range(self.n):
                value = population[best_idx][j] + F * (population[r1][j] - population[r2][j])
                mutant[j] = int(value)
        
        elif self.params.mutation_strategy == "rand/2":
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3, r4, r5 = self.rng.choice(indices, size=5, replace=False)
            
            for j in range(self.n):
                value = population[r1][j] + F * (population[r2][j] - population[r3][j]) + \
                        F * (population[r4][j] - population[r5][j])
                mutant[j] = int(value)
        
        elif self.params.mutation_strategy == "best/2":
            best_idx = np.argmin(objectives)
            indices = [i for i in range(self.params.population_size) if i != current_idx and i != best_idx]
            r1, r2, r3, r4 = self.rng.choice(indices, size=4, replace=False)
            
            for j in range(self.n):
                value = population[best_idx][j] + F * (population[r1][j] - population[r2][j]) + \
                        F * (population[r3][j] - population[r4][j])
                mutant[j] = int(value)
        
        elif self.params.mutation_strategy == "adaptive":
            # 自适应混合变异策略（强制使用自适应F）
            F = self.params.F_max * np.exp(iteration * np.log(self.params.F_min / self.params.F_max) / self.params.max_iterations)
            
            L = np.exp(-iteration / self.params.max_iterations)
            
            if self.rng.random() < L:
                # rand/1策略
                indices = [i for i in range(self.params.population_size) if i != current_idx]
                r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
                
                for j in range(self.n):
                    value = population[r1][j] + F * (population[r2][j] - population[r3][j])
                    mutant[j] = int(value)
            else:
                # best/1 + 差分向量策略
                best_idx = np.argmin(objectives)
                top_10_percent = max(1, self.params.population_size // 10)
                top_indices = np.argsort(objectives)[:top_10_percent]
                
                a_idx = self.rng.choice(top_indices)
                indices = [i for i in range(self.params.population_size) if i != current_idx and i != a_idx]
                b_idx = self.rng.choice(indices)
                
                for j in range(self.n):
                    value = population[current_idx][j] + F * (population[best_idx][j] - population[current_idx][j]) + \
                            F * (population[a_idx][j] - population[b_idx][j])
                    mutant[j] = int(value)
        
        elif self.params.mutation_strategy == "current-to-rand/2":
            # current-to-rand/2策略（使用自适应K）
            K = self.params.K0 * (2 ** np.exp(1 - self.params.max_iterations / (self.params.max_iterations + 1 - iteration)))
            
            indices = [i for i in range(self.params.population_size) if i != current_idx]
            r1, r2, r3, r4 = self.rng.choice(indices, size=4, replace=False)
            
            for j in range(self.n):
                value = population[current_idx][j] + K * (population[r1][j] - population[r2][j] + population[r3][j] - population[r4][j])
                mutant[j] = int(value)
        
        # 边界处理
        for j in range(self.n):
            if mutant[j] < self.decoder.es[j]:
                mutant[j] = self.decoder.es[j]
            elif mutant[j] > self.decoder.ls[j]:
                mutant[j] = self.decoder.ls[j]
        
        return mutant
    
    def _crossover(self, target: List[int], mutant: List[int], iteration: int) -> List[int]:
        """交叉操作（内部自适应调整CR）"""
        trial = target.copy()
        
        # 根据use_adaptive_CR决定使用自适应CR还是固定CR
        if self.params.use_adaptive_CR:
            CR = self.params.CR_min * np.exp(iteration * np.log(self.params.CR_max / self.params.CR_min) / self.params.max_iterations)
        else:
            CR = self.params.CR  # 使用固定值（默认0.9）
        
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
    
    def _repair(self, solution: List[int]) -> List[int]:
        """修复解的约束"""
        repaired = solution.copy()
        
        for i in range(self.n):
            if self.inst.predecessors[i]:
                max_pred_finish = max(
                    repaired[j] + self.inst.durations[j]
                    for j in self.inst.predecessors[i]
                )
                if repaired[i] < max_pred_finish:
                    repaired[i] = max_pred_finish
        
        for i in range(self.n):
            if repaired[i] < self.decoder.es[i]:
                repaired[i] = self.decoder.es[i]
            elif repaired[i] > self.decoder.ls[i]:
                repaired[i] = self.decoder.ls[i]
        
        return repaired
