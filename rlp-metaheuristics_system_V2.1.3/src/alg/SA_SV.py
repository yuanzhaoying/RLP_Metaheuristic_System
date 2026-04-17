"""
模拟退火算法（Simulated Annealing）- 基于位移编码版本

算子汇总：
    1. 邻域生成算子
       - uniform：均匀邻域
       - gaussian：高斯邻域
       - swap：交换邻域
"""

from typing import List
import time
import math
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.shift_vector_evaluator import ShiftVectorEvaluator
from ..psp.shift_vector_decoder import ShiftVectorDecoder
from .operators import RandomGenerator


@dataclass
class SAParamsSV:
    """模拟退火参数（位移编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    initial_temperature: float = 10000.0
    cooling_rate: float = 0.995
    iterations_per_temperature: int = 10
    time_limit: float = 60.0
    neighborhood_strategy: str = "uniform"


class SimulatedAnnealingSV:
    """模拟退火算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: SAParamsSV):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ShiftVectorEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ShiftVectorDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        """运行模拟退火算法"""
        start_time = time.time()
        convergence = []
        
        current = self._initialize_solution()
        current_obj, _ = self.evaluator.evaluate(current)
        
        best_displacement = current.copy()
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
                        best_displacement = current.copy()
            
            convergence.append(best_objective)
            temperature *= self.params.cooling_rate
        
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
    
    def _initialize_solution(self) -> List[int]:
        """初始化解"""
        return self.decoder.encode_random(self.rng)
    
    def _generate_neighbor(self, solution: List[int]) -> List[int]:
        """生成邻居解"""
        neighbor = solution.copy()
        
        j = self.rng.integers(0, self.n)
        max_shift = self.decoder.get_max_shift(j)
        
        if self.params.neighborhood_strategy == "uniform":
            neighbor[j] = self.rng.integers(0, max_shift + 1)
        
        elif self.params.neighborhood_strategy == "gaussian":
            neighbor[j] = int(neighbor[j] + self.rng.rng.normal(0, max_shift * 0.2))
            neighbor[j] = max(0, min(neighbor[j], max_shift))
        
        elif self.params.neighborhood_strategy == "swap":
            j2 = self.rng.integers(0, self.n)
            max_shift2 = self.decoder.get_max_shift(j2)
            neighbor[j], neighbor[j2] = self.rng.integers(0, max_shift + 1), self.rng.integers(0, max_shift2 + 1)
        
        return neighbor
    
    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        """计算接受概率"""
        if temperature <= 0:
            return 0.0
        return math.exp(-delta / temperature)
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
