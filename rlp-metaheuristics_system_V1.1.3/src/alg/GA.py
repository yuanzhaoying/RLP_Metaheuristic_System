"""
遗传算法（Genetic Algorithm）- 开始时间编码版本

算法简介：
    遗传算法是一种基于自然选择和遗传机制的优化算法。
    通过选择、交叉、变异等操作，模拟生物进化过程来搜索最优解。

算子汇总：
    1. 选择算子（Selection）
       - roulette：轮盘赌选择，根据适应度比例选择个体
       - tournament：锦标赛选择，随机选择k个个体，选择最优的
    
    2. 交叉算子
       - single_point：单点交叉，随机选择一个交叉点
       - two_point：双点交叉，随机选择两个交叉点
       - rcx：资源约束交叉，寻找资源使用成本最低的时间窗口
       - hybrid：混合交叉，前α*POP个执行RCX，后(1-α)*POP个执行两点交叉
    
    3. 变异算子（Mutation）
       - random：随机变异，在ES-LS范围内随机选择新的开始时间
       - adaptive：自适应变异，根据可行解比例动态调整变异概率
    
    4. 修复算子（Repair）
       - 前置约束修复：确保每个活动的开始时间满足前置约束
       - 可行范围修复：确保开始时间在ES-LS范围内
"""

from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator
from .start_time_algorithms import AlgorithmResult, _params_to_dict


@dataclass
class GAParamsST:
    """遗传算法参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    time_limit: float = 60.0
    selection_strategy: str = "roulette"
    crossover_strategy: str = "single_point"
    mutation_strategy: str = "random"
    initialization_strategy: str = "random"
    tournament_size: int = 3
    use_repair: bool = True
    elitism: bool = False
    alpha: float = 0.5
    feasible_lower_rate: float = 0.1
    feasible_upper_rate: float = 0.35
    local_search_strategy: str = "none"  # "none", "activity", "shift"
    local_search_interval: int = 5
    neighborhood_size: int = 2
    use_sa_acceptance: bool = False  # 是否使用SA接受准则
    sa_initial_temp: float = 10.0  # SA初始温度
    sa_cooling_rate: float = 0.95  # SA冷却速率


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
    
    def run(self):
        """运行遗传算法"""
        start_time = time.time()
        convergence = []
        
        population = self._initialize_population()
        
        best_start_times = None
        best_objective = float('inf')
        
        # SA相关参数
        temperature = self.params.sa_initial_temp
        
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
                    best_start_times = individual.copy()
            
            convergence.append(best_objective)
            
            # 周期性局部搜索
            if (self.params.local_search_strategy != "none" and 
                (generation + 1) % self.params.local_search_interval == 0):
                # 只对前5个最优个体执行局部搜索，避免消耗过多评估次数
                top_k = min(5, len(population))
                sorted_indices = np.argsort(objectives)
                
                for idx in range(top_k):
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    i = sorted_indices[idx]
                    
                    # 选择局部搜索策略
                    if self.params.local_search_strategy == "shift":
                        population[i] = self._shift_local_search(population[i])
                    elif self.params.local_search_strategy == "activity":
                        population[i] = self._double_pass_local_search(population[i])
            
            feasible_ratio = 1.0
            
            if self.params.selection_strategy == "roulette":
                total_fitness = sum(fitness_scores)
                if total_fitness > 0:
                    selection_probs = [f / total_fitness for f in fitness_scores]
                else:
                    selection_probs = [1.0 / len(population)] * len(population)
                
                new_population = []
                for idx in range(self.params.population_size):
                    parent1_idx = self._roulette_wheel_selection(selection_probs)
                    parent2_idx = self._roulette_wheel_selection(selection_probs)
                    
                    child = self._crossover(population[parent1_idx], population[parent2_idx])
                    child = self._mutate(child, feasible_ratio)
                    
                    # 应用SA接受准则
                    if self.params.use_sa_acceptance and temperature > 0.01:
                        parent_obj = objectives[parent1_idx]
                        child_obj, _ = self.evaluator.evaluate(child)
                        
                        if self._sa_accept(parent_obj, child_obj, temperature):
                            new_population.append(child)
                        else:
                            new_population.append(population[parent1_idx])
                    else:
                        new_population.append(child)
                
                population = new_population
            
            elif self.params.selection_strategy == "tournament":
                parents = self._tournament_selection(population, fitness_scores)
                
                new_population = []
                for i in range(0, len(parents), 2):
                    if i + 1 >= len(parents):
                        new_population.append(parents[i])
                        break
                    
                    child1, child2 = self._crossover_two_children(parents[i], parents[i+1], i)
                    child1 = self._mutate(child1, feasible_ratio)
                    child2 = self._mutate(child2, feasible_ratio)
                    
                    # 应用SA接受准则
                    if self.params.use_sa_acceptance and temperature > 0.01:
                        parent1_obj = objectives[i] if i < len(objectives) else float('inf')
                        parent2_obj = objectives[i+1] if i+1 < len(objectives) else float('inf')
                        
                        child1_obj, _ = self.evaluator.evaluate(child1)
                        child2_obj, _ = self.evaluator.evaluate(child2)
                        
                        if self._sa_accept(parent1_obj, child1_obj, temperature):
                            new_population.append(child1)
                        else:
                            new_population.append(parents[i])
                        
                        if self._sa_accept(parent2_obj, child2_obj, temperature):
                            new_population.append(child2)
                        else:
                            new_population.append(parents[i+1])
                    else:
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
            
            # 更新温度
            if self.params.use_sa_acceptance:
                temperature *= self.params.sa_cooling_rate
            
            generation += 1
        
        # 最后对最优解执行局部搜索
        if self.params.local_search_strategy != "none" and best_start_times is not None:
            if self.evaluator.n_evaluations < self.params.max_evaluations:
                if self.params.local_search_strategy == "shift":
                    improved = self._shift_local_search(best_start_times)
                elif self.params.local_search_strategy == "activity":
                    improved = self._double_pass_local_search(best_start_times)
                obj, _ = self.evaluator.evaluate(improved)
                if obj < best_objective:
                    best_start_times = improved
                    best_objective = obj
        
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
                
                if self.params.use_repair:
                    individual = self._repair(individual)
            
            elif self.params.initialization_strategy == "heuristic":
                individual = self._heuristic_initialization()
            
            else:
                individual = []
                for j in range(self.n):
                    start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
                    individual.append(start_time)
                
                if self.params.use_repair:
                    individual = self._repair(individual)
            
            population.append(individual)
        
        return population
    
    def _heuristic_initialization(self) -> List[int]:
        """启发式初始化（按拓扑排序顺序生成）"""
        individual = [0] * self.n
        
        # 计算拓扑排序
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
        
        # 按拓扑排序顺序生成开始时间
        for i in topo_order:
            # 计算前置活动的最晚完成时间
            if self.inst.predecessors[i]:
                pf = max(individual[j] + self.inst.durations[j] for j in self.inst.predecessors[i])
            else:
                pf = 0
            
            # 在可行范围内随机选择开始时间
            s_min = max(self.decoder.es[i], pf)
            s_max = self.decoder.ls[i]
            
            if s_min <= s_max:
                individual[i] = self.rng.integers(s_min, s_max + 1)
            else:
                individual[i] = s_min
        
        return individual
    
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
        
        if self.params.crossover_strategy == "single_point":
            cross_point = self.rng.integers(1, self.n)
            child = parent1[:cross_point] + parent2[cross_point:]
        
        elif self.params.crossover_strategy == "two_point":
            a, b = sorted(self.rng.choice(self.n, size=2, replace=False))
            child = parent1[:a] + parent2[a:b] + parent1[b:]
        
        elif self.params.crossover_strategy == "rcx":
            child1, _ = self._rcx_crossover(parent1, parent2)
            child = child1
        
        else:
            child = parent1.copy()
        
        if self.params.use_repair:
            child = self._repair(child)
        
        return child
    
    def _crossover_two_children(self, parent1: List[int], parent2: List[int], parent_index: int = 0) -> Tuple[List[int], List[int]]:
        """交叉操作（返回两个子代）"""
        if self.rng.random() >= self.params.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if self.params.crossover_strategy == "single_point":
            cross_point = self.rng.integers(1, self.n)
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]
        
        elif self.params.crossover_strategy == "two_point":
            a, b = sorted(self.rng.choice(self.n, size=2, replace=False))
            child1 = parent1[:a] + parent2[a:b] + parent1[b:]
            child2 = parent2[:a] + parent1[a:b] + parent2[b:]
        
        elif self.params.crossover_strategy == "rcx":
            child1, child2 = self._rcx_crossover(parent1, parent2)
        
        elif self.params.crossover_strategy == "hybrid":
            split_point = int(self.params.alpha * self.params.population_size)
            if parent_index < split_point:
                child1, child2 = self._rcx_crossover(parent1, parent2)
            else:
                a, b = sorted(self.rng.choice(self.n, size=2, replace=False))
                child1 = parent1[:a] + parent2[a:b] + parent1[b:]
                child2 = parent2[:a] + parent1[a:b] + parent2[b:]
        
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        if self.params.use_repair:
            child1 = self._repair(child1)
            child2 = self._repair(child2)
        
        return child1, child2
    
    def _rcx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """资源约束交叉"""
        l = self.rng.integers(self.deadline // 4, 3 * self.deadline // 4 + 1)
        best_t0 = 1
        min_cost = float('inf')
        
        for t0 in range(1, self.deadline - l + 1):
            t1 = t0 + l
            cost = self._calculate_window_cost([parent1, parent2], t0, t1)
            if cost < min_cost:
                min_cost = cost
                best_t0 = t0
        
        aw = set()
        for i in range(self.n):
            s1, e1 = parent1[i], parent1[i] + self.inst.durations[i] - 1
            s2, e2 = parent2[i], parent2[i] + self.inst.durations[i] - 1
            
            if (best_t0 <= s1 <= best_t0 + l or best_t0 <= e1 <= best_t0 + l or
                best_t0 <= s2 <= best_t0 + l or best_t0 <= e2 <= best_t0 + l):
                aw.add(i)
        
        child1, child2 = parent1.copy(), parent2.copy()
        for i in aw:
            child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    def _calculate_window_cost(self, solutions: List[List[int]], t0: int, t1: int) -> float:
        """计算时间窗口内的资源使用成本"""
        total_cost = 0.0
        
        for sol in solutions:
            for i in range(self.n):
                s, e = sol[i], sol[i] + self.inst.durations[i] - 1
                if s > t1 or e < t0:
                    continue
                
                cs, ce = max(s, t0), min(e, t1)
                for t in range(cs, ce + 1):
                    for k in range(self.inst.n_resources):
                        if t <= self.deadline:
                            usage = self.inst.demands[i][k]
                            capacity = self.inst.capacity[k]
                            if usage > capacity:
                                total_cost += usage - capacity
        
        return total_cost
    
    def _mutate(self, individual: List[int], feasible_ratio: float = 1.0) -> List[int]:
        """变异操作"""
        mutated = individual.copy()
        
        if self.params.mutation_strategy == "random":
            for j in range(self.n):
                if self.rng.random() < self.params.mutation_rate:
                    mutated[j] = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
        
        elif self.params.mutation_strategy == "adaptive":
            pm = self.params.mutation_rate
            if feasible_ratio < self.params.feasible_lower_rate:
                pm *= 0.5
            elif feasible_ratio > self.params.feasible_upper_rate:
                pm *= 2.0
            
            for j in range(self.n):
                if self.rng.random() < pm:
                    mutated[j] = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
        
        elif self.params.mutation_strategy == "hybrid":
            m = self.rng.integers(0, 3)
            nd = [i for i in range(1, self.n - 1)]
            
            if m == 0 and len(nd) >= 2:
                i, j = self.rng.choice(nd, size=2, replace=False)
                mutated[i], mutated[j] = mutated[j], mutated[i]
            
            elif m == 1 and len(nd) >= 2:
                a, b = sorted(self.rng.choice(nd, size=2, replace=False))
                mutated[a:b+1] = mutated[a:b+1][::-1]
            
            elif m == 2:
                for i in nd:
                    if self.rng.random() < self.params.mutation_rate:
                        mutated[i] = self.rng.integers(self.decoder.es[i], self.decoder.ls[i] + 1)
        
        elif self.params.mutation_strategy == "neighborhood":
            for j in range(self.n):
                if self.rng.random() < self.params.mutation_rate:
                    s_min = max(self.decoder.es[j], mutated[j] - self.params.neighborhood_size)
                    s_max = min(self.decoder.ls[j], mutated[j] + self.params.neighborhood_size)
                    
                    if s_min <= s_max:
                        mutated[j] = self.rng.integers(s_min, s_max + 1)
                    else:
                        mutated[j] = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
        
        if self.params.use_repair:
            mutated = self._repair(mutated)
        
        return mutated
    
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
    
    def _double_pass_local_search(self, solution: List[int]) -> List[int]:
        """双遍局部搜索"""
        best_solution = solution.copy()
        best_obj, _ = self.evaluator.evaluate(best_solution)
        
        # 第一遍：按开始时间升序遍历
        order1 = sorted(range(self.n), key=lambda x: best_solution[x])
        # 第二遍：按开始时间降序遍历
        order2 = sorted(range(self.n), key=lambda x: best_solution[x], reverse=True)
        
        for order in [order1, order2]:
            for i in order:
                if i == 0 or i == self.n - 1:
                    continue
                
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                # 计算可行范围
                if self.inst.predecessors[i]:
                    max_pred_finish = max(
                        best_solution[j] + self.inst.durations[j]
                        for j in self.inst.predecessors[i]
                    )
                else:
                    max_pred_finish = 0
                
                s_min = max(self.decoder.es[i], max_pred_finish)
                s_max = self.decoder.ls[i]
                
                # 在可行范围内搜索最优开始时间
                for s_new in range(s_min, s_max + 1):
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    if s_new == best_solution[i]:
                        continue
                    
                    temp_solution = best_solution.copy()
                    temp_solution[i] = s_new
                    
                    obj, _ = self.evaluator.evaluate(temp_solution)
                    
                    if obj < best_obj:
                        best_solution = temp_solution.copy()
                        best_obj = obj
        
        return best_solution
    
    def _shift_local_search(self, solution: List[int]) -> List[int]:
        """移位局部搜索（对每个活动进行小步长移位，迭代改进）"""
        best = solution.copy()
        best_obj, _ = self.evaluator.evaluate(best)
        
        improved = True
        iteration = 0
        max_iterations = 3  # 最多迭代3次
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 随机打乱活动顺序
            activity_order = list(range(self.n))
            self.rng.shuffle(activity_order)
            
            for i in activity_order:
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                # 尝试多个移位步长
                for shift in [-3, -2, -1, 1, 2, 3]:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    new = best.copy()
                    new[i] = max(0, new[i] + shift)
                    
                    # 修复约束
                    new = self._repair(new)
                    
                    obj, _ = self.evaluator.evaluate(new)
                    
                    if obj < best_obj - 1e-6:  # 确保有实质性改进
                        best = new.copy()
                        best_obj = obj
                        improved = True
                        break  # first-improvement
        
        return best
    
    def _sa_accept(self, old_obj: float, new_obj: float, temperature: float) -> bool:
        """模拟退火接受准则"""
        if new_obj < old_obj:
            return True
        
        delta = new_obj - old_obj
        prob = np.exp(-delta / temperature)
        
        return self.rng.random() < prob
