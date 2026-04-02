"""
和谐搜索算法（Harmony Search）- 开始时间编码版本

算法简介：
    和谐搜索算法是一种基于音乐即兴创作过程的元启发式算法。
    通过模拟音乐家在演奏中寻找完美和声的过程来搜索最优解。

算法原理：
    算法维护一个和声记忆库，存储一组优秀的解。
    通过三种方式生成新解：
    1. 从和声记忆库中选择
    2. 音调调整（局部调整）
    3. 随机生成
    
    参数：
    - HMCR（和声记忆库选择率）：从记忆库中选择的概率
    - PAR（音调调整率）：对选中的解进行调整的概率
    - BW（音调调整带宽）：调整的范围

算子汇总：
    1. 参数策略（Parameter Strategy）
       - fixed：固定参数
       - adaptive：自适应参数（PAR线性增加，BW指数衰减）
    
    2. 初始化策略（Initialization Strategy）
       - random：随机初始化
       - forward：前向调度初始化（ES/LS混合）
"""

from typing import List
import time
import math
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator
from .start_time_algorithms import AlgorithmResult, _params_to_dict


@dataclass
class HSParamsST:
    """和谐搜索算法参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    time_limit: float = 60.0
    
    # HS参数
    hm_size: int = 50  # 和声记忆库大小
    hmcr: float = 0.9  # 和声记忆库选择率
    
    # 固定参数
    par: float = 0.3   # 音调调整率
    bw: float = 2.0    # 音调调整带宽
    
    # 自适应参数范围
    par_min: float = 0.1  # 最小音调调整率
    par_max: float = 0.9  # 最大音调调整率
    bw_min: float = 1.0   # 最小带宽
    bw_max: float = 5.0   # 最大带宽
    
    # 算子选择
    parameter_strategy: str = "fixed"  # "fixed" 或 "adaptive"
    initialization_strategy: str = "random"  # "random" 或 "forward"


class HarmonySearchST:
    """和谐搜索算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: HSParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
        
        # 估算最大迭代次数
        self.max_iterations = params.max_evaluations
    
    def _update_par(self, iteration: int) -> float:
        """更新PAR（自适应）- 参考用户代码"""
        if self.params.parameter_strategy == "adaptive":
            # PAR线性增加：从par_min到par_max
            return self.params.par_min + (self.params.par_max - self.params.par_min) * (iteration / self.max_iterations)
        else:
            return self.params.par
    
    def _update_bw(self, iteration: int) -> float:
        """更新BW（自适应）- 参考用户代码"""
        if self.params.parameter_strategy == "adaptive":
            # BW指数衰减：从bw_max到bw_min
            return self.params.bw_max * math.exp(
                (iteration / self.max_iterations) * math.log(self.params.bw_min / self.params.bw_max)
            )
        else:
            return self.params.bw
    
    def _initialize_harmony_random(self) -> np.ndarray:
        """随机初始化和声"""
        harmony = np.zeros(self.n, dtype=np.float64)
        
        for j in range(self.n):
            es = self.decoder.es[j]
            ls = self.decoder.ls[j]
            harmony[j] = self.rng.rng.uniform(es, ls)
        
        return harmony
    
    def _initialize_harmony_forward(self) -> np.ndarray:
        """前向调度初始化 - 参考用户代码"""
        harmony = np.zeros(self.n, dtype=np.float64)
        
        # 前向调度：每个活动以0.5概率选择ES或LS
        for j in range(self.n):
            if self.rng.rng.random() < 0.5:
                harmony[j] = self.decoder.es[j]
            else:
                harmony[j] = self.decoder.ls[j]
        
        return harmony
    
    def _repair_position(self, position: np.ndarray) -> List[int]:
        """修复位置（确保满足约束）- 标准修复算子"""
        # 转换为整数开始时间
        start_times = np.round(position).astype(int)
        
        # 确保在ES-LS范围内
        for j in range(self.n):
            start_times[j] = max(self.decoder.es[j], min(self.decoder.ls[j], start_times[j]))
        
        # 修复前置约束
        for j in range(self.n):
            if self.inst.predecessors[j]:
                max_pred_finish = max(
                    start_times[p] + self.inst.durations[p]
                    for p in self.inst.predecessors[j]
                )
                start_times[j] = max(start_times[j], max_pred_finish)
        
        return start_times.tolist()
    
    def _improvise_harmony(self, harmony_memory: List[np.ndarray], par: float, bw: float) -> np.ndarray:
        """即兴创作新和声"""
        new_harmony = np.zeros(self.n, dtype=np.float64)
        
        for j in range(self.n):
            if self.rng.rng.random() < self.params.hmcr:
                # 从和声记忆库中选择
                idx = self.rng.integers(0, len(harmony_memory))
                new_harmony[j] = harmony_memory[idx][j]
                
                # 音调调整 - 参考用户代码
                if self.rng.rng.random() < par:
                    # 使用整数调整
                    adjustment = self.rng.integers(-int(bw), int(bw) + 1)
                    new_harmony[j] += adjustment
            else:
                # 随机生成
                es = self.decoder.es[j]
                ls = self.decoder.ls[j]
                new_harmony[j] = self.rng.rng.uniform(es, ls)
        
        return new_harmony
    
    def run(self):
        """运行和谐搜索算法"""
        start_time = time.time()
        convergence = []
        
        # 初始化和声记忆库
        harmony_memory = []
        fitness = []
        
        for _ in range(self.params.hm_size):
            # 根据初始化策略选择初始化方式
            if self.params.initialization_strategy == "forward":
                harmony = self._initialize_harmony_forward()
            else:
                harmony = self._initialize_harmony_random()
            
            # 评估
            start_times = self._repair_position(harmony)
            obj, _ = self.evaluator.evaluate(start_times)
            
            harmony_memory.append(harmony)
            fitness.append(obj)
        
        # 排序和声记忆库
        sorted_indices = np.argsort(fitness)
        harmony_memory = [harmony_memory[i] for i in sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]
        
        # 记录最优解
        best_harmony = harmony_memory[0].copy()
        best_fitness = fitness[0]
        best_start_times = self._repair_position(best_harmony)
        
        convergence.append(best_fitness)
        
        # 主循环
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            # 更新自适应参数
            par = self._update_par(iteration)
            bw = self._update_bw(iteration)
            
            # 即兴创作新和声
            new_harmony = self._improvise_harmony(harmony_memory, par, bw)
            
            # 修复并评估
            new_start_times = self._repair_position(new_harmony)
            new_fitness, _ = self.evaluator.evaluate(new_start_times)
            
            # 更新和声记忆库
            if new_fitness < fitness[-1]:
                # 替换最差的和声
                harmony_memory[-1] = new_harmony.copy()
                fitness[-1] = new_fitness
                
                # 重新排序
                sorted_indices = np.argsort(fitness)
                harmony_memory = [harmony_memory[i] for i in sorted_indices]
                fitness = [fitness[i] for i in sorted_indices]
                
                # 更新最优解
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_harmony = new_harmony.copy()
                    best_start_times = new_start_times
            
            convergence.append(best_fitness)
            iteration += 1
        
        runtime = time.time() - start_time
        
        return AlgorithmResult(
            best_start_times=best_start_times,
            best_objective=best_fitness,
            n_evaluations=self.evaluator.n_evaluations,
            runtime=runtime,
            convergence=convergence,
            algorithm_params=_params_to_dict(self.params)
        )
