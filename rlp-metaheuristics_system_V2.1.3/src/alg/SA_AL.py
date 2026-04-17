"""
模拟退火算法（Simulated Annealing）- 活动列表编码版本

算法简介：
    模拟退火算法是一种基于物理退火过程的随机搜索算法。
    通过逐渐降低"温度"，从高能态（随机搜索）逐渐过渡到低能态（局部搜索）。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 使用排列编码专用的邻域操作

算子汇总：
    1. 邻域生成算子
       - swap：交换两个活动
       - insertion：插入操作
       - inversion：逆序操作
    
    2. 接受准则
       - Metropolis准则
    
    3. 冷却策略
       - 指数冷却
"""

from typing import List
import time
import math
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.activity_list_evaluator import ActivityListEvaluator
from ..psp.activity_list_decoder import ActivityListDecoder
from .operators import RandomGenerator


@dataclass
class SAParamsAL:
    """模拟退火参数（活动列表编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    initial_temperature: float = 10000.0
    cooling_rate: float = 0.995
    iterations_per_temperature: int = 10
    time_limit: float = 60.0
    neighborhood_strategy: str = "swap"
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class SimulatedAnnealingAL:
    """模拟退火算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: SAParamsAL):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ActivityListEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ActivityListDecoder(instance, deadline)
        self.n = instance.n_activities
    
    def run(self):
        """运行模拟退火算法"""
        start_time = time.time()
        convergence = []
        
        if self.params.use_delay_factors:
            current, current_delay = self.decoder.encode_random_with_delay(self.rng)
        else:
            current = self._initialize_solution()
            current_delay = None
        
        if self.params.use_delay_factors:
            current_obj, _ = self.evaluator.evaluate(current, current_delay)
        else:
            current_obj, _ = self.evaluator.evaluate(current)
        
        best_activity_list = current.copy()
        best_delay_factors = current_delay.copy() if current_delay else None
        best_objective = current_obj
        
        temperature = self.params.initial_temperature
        
        while (self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            for _ in range(self.params.iterations_per_temperature):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                neighbor, neighbor_delay = self._generate_neighbor(current, current_delay)
                
                if self.params.use_delay_factors:
                    neighbor_obj, _ = self.evaluator.evaluate(neighbor, neighbor_delay)
                else:
                    neighbor_obj, _ = self.evaluator.evaluate(neighbor)
                
                delta = neighbor_obj - current_obj
                
                if delta < 0 or self._acceptance_probability(delta, temperature) > self.rng.random():
                    current = neighbor
                    current_obj = neighbor_obj
                    current_delay = neighbor_delay
                    
                    if current_obj < best_objective:
                        best_objective = current_obj
                        best_activity_list = current.copy()
                        if self.params.use_delay_factors:
                            best_delay_factors = current_delay.copy()
            
            convergence.append(best_objective)
            temperature *= self.params.cooling_rate
        
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
    
    def _initialize_solution(self) -> List[int]:
        """初始化解"""
        return self.decoder.encode_random(self.rng)
    
    def _generate_neighbor(self, solution: List[int], delay_factors: List[float] = None) -> tuple:
        """生成邻居解"""
        neighbor = solution.copy()
        neighbor_delay = delay_factors.copy() if delay_factors else None
        
        if self.params.neighborhood_strategy == "swap":
            i, j = self.rng.choice(self.n, size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif self.params.neighborhood_strategy == "insertion":
            i = self.rng.integers(0, self.n)
            j = self.rng.integers(0, self.n)
            gene = neighbor.pop(i)
            neighbor.insert(j, gene)
        
        elif self.params.neighborhood_strategy == "inversion":
            i, j = sorted(self.rng.choice(self.n, size=2, replace=False))
            neighbor[i:j+1] = neighbor[i:j+1][::-1]
        
        else:
            i, j = self.rng.choice(self.n, size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        neighbor = self.decoder.repair(neighbor)
        
        if self.params.use_delay_factors and neighbor_delay is not None:
            for k in range(len(neighbor_delay)):
                if self.rng.random() < self.params.delay_mutation_rate:
                    neighbor_delay[k] = self.rng.random()
            neighbor_delay[0] = 0.0
            neighbor_delay[self.n - 1] = 0.0
        
        return neighbor, neighbor_delay
    
    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        """计算接受概率"""
        if temperature <= 0:
            return 0.0
        return math.exp(-delta / temperature)
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
