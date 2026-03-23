"""
引力搜索算法（Gravitational Search Algorithm）- 开始时间编码版本

算法简介：
    引力搜索算法是一种基于万有引力定律的优化算法。
    每个解被视为一个质点，质点之间通过万有引力相互吸引。
    质量越大的质点（适应度越好的解）对其他质点的吸引力越大。

算子汇总：
    1. 初始化算子（Initialization）
       - random：随机初始化，在ES-LS范围内随机选择开始时间
       - heuristic：启发式初始化，按拓扑排序顺序生成
    
    2. 质量计算（Mass Calculation）
       - 根据适应度计算每个粒子的质量
    
    3. 引力计算（Force Calculation）
       - 计算粒子之间的引力
    
    4. 位置更新（Position Update）
       - 根据加速度更新粒子的速度和位置
    
    5. 修复算子（Repair）
       - 前置约束修复：确保每个活动的开始时间满足前置约束
       - 可行范围修复：确保开始时间在ES-LS范围内
"""

from typing import List
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator
from .start_time_algorithms import AlgorithmResult, _params_to_dict


@dataclass
class GSAParamsST:
    """引力搜索算法参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    max_iterations: int = 100
    G0: float = 100.0
    alpha: float = 20.0
    initialization_strategy: str = "random"
    use_repair: bool = True


class GravitationalSearchAlgorithmST:
    """引力搜索算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: GSAParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        """运行引力搜索算法"""
        start_time = time.time()
        convergence = []
        
        # 初始化粒子群
        positions = self._initialize_population()
        velocities = [[0.0] * self.n for _ in range(self.params.population_size)]
        
        best_start_times = None
        best_objective = float('inf')
        
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit and
               iteration < self.params.max_iterations):
            
            # 评估所有粒子
            objectives = []
            for pos in positions:
                obj, _ = self.evaluator.evaluate(pos)
                objectives.append(obj)
                
                if obj < best_objective:
                    best_objective = obj
                    best_start_times = pos.copy()
            
            convergence.append(best_objective)
            
            # 计算质量
            masses = self._calculate_masses(objectives)
            
            # 计算引力常数G
            G = self._calculate_G(iteration)
            
            # 计算加速度
            accelerations = self._calculate_accelerations(positions, masses, G)
            
            # 更新速度和位置
            for i in range(self.params.population_size):
                for j in range(self.n):
                    # 更新速度
                    velocities[i][j] = self.rng.random() * velocities[i][j] + accelerations[i][j]
                    
                    # 更新位置
                    positions[i][j] = int(positions[i][j] + velocities[i][j])
                    
                    # 边界处理
                    if positions[i][j] < self.decoder.es[j]:
                        positions[i][j] = self.decoder.es[j]
                        velocities[i][j] = 0.0
                    elif positions[i][j] > self.decoder.ls[j]:
                        positions[i][j] = self.decoder.ls[j]
                        velocities[i][j] = 0.0
                
                # 修复
                if self.params.use_repair:
                    positions[i] = self._repair(positions[i])
            
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
        """初始化粒子群"""
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
    
    def _calculate_masses(self, objectives: List[float]) -> List[float]:
        """计算每个粒子的质量"""
        best = min(objectives)
        worst = max(objectives)
        
        if worst == best:
            return [1.0] * len(objectives)
        
        masses = []
        for obj in objectives:
            m = (obj - worst) / (best - worst + 1e-10)
            masses.append(m)
        
        total_mass = sum(masses)
        if total_mass > 0:
            masses = [m / total_mass for m in masses]
        else:
            masses = [1.0 / len(objectives)] * len(objectives)
        
        return masses
    
    def _calculate_G(self, iteration: int) -> float:
        """计算引力常数G"""
        G = self.params.G0 * np.exp(-self.params.alpha * iteration / self.params.max_iterations)
        return G
    
    def _calculate_accelerations(self, positions: List[List[int]], masses: List[float], G: float) -> List[List[float]]:
        """计算每个粒子的加速度"""
        accelerations = [[0.0] * self.n for _ in range(self.params.population_size)]
        
        for i in range(self.params.population_size):
            for j in range(self.params.population_size):
                if i == j:
                    continue
                
                # 计算距离
                distance = 0.0
                for k in range(self.n):
                    distance += (positions[i][k] - positions[j][k]) ** 2
                distance = np.sqrt(distance) + 1e-10
                
                # 计算引力
                force = G * masses[i] * masses[j] / (distance ** 2)
                
                # 计算加速度
                for k in range(self.n):
                    accelerations[i][k] += force * (positions[j][k] - positions[i][k]) / distance
        
        return accelerations
    
    def _repair(self, solution: List[int]) -> List[int]:
        """修复解的约束"""
        repaired = solution.copy()
        
        # 修复前置约束
        for i in range(self.n):
            if self.inst.predecessors[i]:
                max_pred_finish = max(
                    repaired[j] + self.inst.durations[j]
                    for j in self.inst.predecessors[i]
                )
                if repaired[i] < max_pred_finish:
                    repaired[i] = max_pred_finish
        
        # 修复ES-LS约束
        for i in range(self.n):
            if repaired[i] < self.decoder.es[i]:
                repaired[i] = self.decoder.es[i]
            elif repaired[i] > self.decoder.ls[i]:
                repaired[i] = self.decoder.ls[i]
        
        return repaired
