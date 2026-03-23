

"""
迭代局部搜索算法（Iterated Local Search）- 开始时间编码版本

算法简介：
    迭代局部搜索是一种结合局部搜索和扰动的元启发式算法。
    通过局部搜索找到局部最优，然后通过扰动跳出局部最优，再进行局部搜索。

算子汇总：
    1. 局部搜索算子（Local Search）
       - 逐活动优化（Activity-by-activity Optimization）
         对每个活动，在ES-LS范围内搜索最优开始时间，使用first-improvement策略
    
    2. 扰动算子（Perturbation）
       - 多点变异（Multi-point Mutation）
         随机选择多个活动（perturbation_strength个），在ES-LS范围内随机选择新的开始时间
"""

from typing import List, Tuple
import time
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator


@dataclass
class ILSParamsST:
    """迭代局部搜索参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    perturbation_strength: int = 5
    time_limit: float = 60.0
    max_iterations: int = 100


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
    
    def run(self):
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
        
        from .start_time_algorithms import AlgorithmResult, _params_to_dict
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
