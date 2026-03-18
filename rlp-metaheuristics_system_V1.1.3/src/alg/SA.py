

from typing import List
import time
import math
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator


@dataclass
class SAParamsST:
    """模拟退火参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    initial_temperature: float = 10000.0
    cooling_rate: float = 0.995
    iterations_per_temperature: int = 10
    time_limit: float = 60.0


class SimulatedAnnealingST:
    """
模拟退火算法（Simulated Annealing）- 开始时间编码版本

算法简介：
    模拟退火算法是一种基于物理退火过程的随机搜索算法。
    通过逐渐降低"温度"，从高能态（随机搜索）逐渐过渡到低能态（局部搜索）。

算子汇总：
    1. 邻域生成算子（Neighborhood Generation）
       - 单点变异（Single-point Mutation）
         随机选择一个活动，在ES-LS范围内随机选择新的开始时间
    
    2. 接受准则（Acceptance Criterion）
       - Metropolis准则
         如果新解更优，总是接受；如果新解更差，以概率 exp(-Δ/T) 接受
    
    3. 冷却策略（Cooling Schedule）
       - 指数冷却（Exponential Cooling）
         T_new = T_old * cooling_rate，简单易实现，收敛稳定
    """
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: SAParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
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
        return math.exp(-delta / temperature)
