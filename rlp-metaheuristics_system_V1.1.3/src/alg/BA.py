"""
蝙蝠算法（Bat Algorithm）- 开始时间编码版本

算法简介：
    蝙蝠算法是一种基于蝙蝠回声定位行为的元启发式算法。
    通过模拟蝙蝠发出声波并接收回声来搜索猎物的过程。

算法原理：
    每只蝙蝠代表一个解，具有位置、速度和频率三个属性。
    蝙蝠通过调整频率、速度和位置来搜索最优解。
    
    频率更新公式：
        f = f_min + (f_max - f_min) * beta
    
    速度更新公式：
        v(t+1) = v(t) + f * (x(t) - x_best)
    
    位置更新公式：
        x(t+1) = x(t) + v(t+1)
    
    其中：
        f_min, f_max: 频率范围
        beta: [0,1]范围内的随机数
        x_best: 当前全局最优位置

算子汇总：
    1. 局部搜索（Local Search）
       - none：不使用局部搜索
       - tlim：TLIM局部搜索，两阶段优化（Forward + Backward）
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
class BAParamsST:
    """蝙蝠算法参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    
    # BA参数
    f_min: float = 0.0  # 最小频率
    f_max: float = 2.0  # 最大频率
    A0: float = 1.0  # 初始响度
    r0: float = 0.5  # 初始脉冲率
    alpha: float = 0.9  # 响度衰减系数
    gamma: float = 0.9  # 脉冲率增加系数
    
    # 算子选择
    local_search_strategy: str = "none"  # "none" 或 "tlim"
    
    # 局部搜索参数
    local_search_interval: int = 10  # 局部搜索执行间隔


class BatAlgorithmST:
    """蝙蝠算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: BAParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
        
        # 构造后继列表
        self._build_successors()
    
    def _build_successors(self):
        """构造后继列表"""
        self.successors = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in self.inst.predecessors[i]:
                self.successors[j].append(i)
    
    def _compute_es(self, i: int, start_times: List[int]) -> int:
        """计算活动的最早开始时间"""
        if not self.inst.predecessors[i]:
            return 0
        return max(
            start_times[j] + self.inst.durations[j]
            for j in self.inst.predecessors[i]
        )
    
    def _compute_ls(self, i: int, start_times: List[int]) -> int:
        """计算活动的最晚开始时间"""
        if not self.successors[i]:
            return self.deadline - self.inst.durations[i]
        return min(
            start_times[j] - self.inst.durations[i]
            for j in self.successors[i]
        )
    
    def _initialize_position(self) -> np.ndarray:
        """初始化蝙蝠位置"""
        position = np.zeros(self.n, dtype=np.float64)
        
        for j in range(self.n):
            es = self.decoder.es[j]
            ls = self.decoder.ls[j]
            position[j] = self.rng.rng.uniform(es, ls)
        
        return position
    
    def _initialize_velocity(self) -> np.ndarray:
        """初始化蝙蝠速度"""
        velocity = np.zeros(self.n, dtype=np.float64)
        
        for j in range(self.n):
            range_size = self.decoder.ls[j] - self.decoder.es[j]
            velocity[j] = self.rng.rng.uniform(-range_size * 0.1, range_size * 0.1)
        
        return velocity
    
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
    
    def _tlim_local_search(self, start_times: List[int]) -> List[int]:
        """TLIM局部搜索 - 参考用户代码的两阶段优化"""
        best = start_times.copy()
        
        # ========================
        # 第一阶段：Forward
        # ========================
        for i in range(self.n):
            if self.evaluator.n_evaluations >= self.params.max_evaluations:
                break
            
            # 计算ES和LS
            ES = self._compute_es(i, best)
            LS = self._compute_ls(i, best)
            
            if ES > LS:
                continue
            
            best_s = best[i]
            best_obj, _ = self.evaluator.evaluate(best)
            
            # 在ES-LS范围内搜索最优开始时间
            for s in range(ES, LS + 1):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                if s == best[i]:
                    continue
                
                new = best.copy()
                new[i] = s
                
                obj, _ = self.evaluator.evaluate(new)
                
                if obj < best_obj:
                    best_obj = obj
                    best_s = s
            
            best[i] = best_s
        
        # ========================
        # 第二阶段：Backward
        # ========================
        for i in range(self.n - 1, -1, -1):
            if self.evaluator.n_evaluations >= self.params.max_evaluations:
                break
            
            # 计算ES和LS
            ES = self._compute_es(i, best)
            LS = self._compute_ls(i, best)
            
            if ES > LS:
                continue
            
            best_s = best[i]
            best_obj, _ = self.evaluator.evaluate(best)
            
            # 在ES-LS范围内搜索最优开始时间
            for s in range(ES, LS + 1):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                if s == best[i]:
                    continue
                
                new = best.copy()
                new[i] = s
                
                obj, _ = self.evaluator.evaluate(new)
                
                if obj < best_obj:
                    best_obj = obj
                    best_s = s
            
            best[i] = best_s
        
        return best
    
    def run(self):
        """运行蝙蝠算法"""
        start_time = time.time()
        convergence = []
        
        # 初始化蝙蝠群
        positions = []
        velocities = []
        frequencies = []
        loudness = []
        pulse_rate = []
        fitness = []
        
        for _ in range(self.params.population_size):
            position = self._initialize_position()
            velocity = self._initialize_velocity()
            
            # 评估初始位置
            start_times = self._repair_position(position)
            obj, _ = self.evaluator.evaluate(start_times)
            
            positions.append(position)
            velocities.append(velocity)
            frequencies.append(self.rng.rng.uniform(self.params.f_min, self.params.f_max))
            loudness.append(self.params.A0)
            pulse_rate.append(self.params.r0)
            fitness.append(obj)
        
        # 初始化全局最优
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        best_objective = fitness[best_idx]
        best_start_times = self._repair_position(best_position)
        
        convergence.append(best_objective)
        
        # 主循环
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            # 更新每个蝙蝠
            for i in range(self.params.population_size):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                # 更新频率
                beta = self.rng.rng.uniform(0, 1)
                frequencies[i] = self.params.f_min + (self.params.f_max - self.params.f_min) * beta
                
                # 更新速度
                velocities[i] = velocities[i] + frequencies[i] * (positions[i] - best_position)
                
                # 更新位置
                new_position = positions[i] + velocities[i]
                
                # 限制位置范围
                for j in range(self.n):
                    new_position[j] = np.clip(
                        new_position[j],
                        self.decoder.es[j],
                        self.decoder.ls[j]
                    )
                
                # 局部搜索触发
                if self.rng.rng.random() > pulse_rate[i]:
                    # 在最优解附近生成新解
                    new_position = best_position + self.rng.rng.normal(0, 1, self.n)
                    
                    # 限制位置范围
                    for j in range(self.n):
                        new_position[j] = np.clip(
                            new_position[j],
                            self.decoder.es[j],
                            self.decoder.ls[j]
                        )
                
                # 修复并评估
                new_start_times = self._repair_position(new_position)
                new_obj, _ = self.evaluator.evaluate(new_start_times)
                
                # 接受准则
                if (new_obj < fitness[i]) and (self.rng.rng.random() < loudness[i]):
                    positions[i] = new_position.copy()
                    fitness[i] = new_obj
                    
                    # 更新响度和脉冲率
                    loudness[i] *= self.params.alpha
                    pulse_rate[i] = self.params.r0 * (1 - np.exp(-self.params.gamma * iteration))
                
                # 更新全局最优
                if new_obj < best_objective:
                    best_objective = new_obj
                    best_position = new_position.copy()
                    best_start_times = new_start_times
            
            # 周期性局部搜索
            if (self.params.local_search_strategy == "tlim" and 
                (iteration + 1) % self.params.local_search_interval == 0):
                
                # 对前5个最优个体执行局部搜索
                top_indices = np.argsort(fitness)[:5]
                for idx in top_indices:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    improved = self._tlim_local_search(self._repair_position(positions[idx]))
                    obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < fitness[idx]:
                        positions[idx] = np.array(improved, dtype=np.float64)
                        fitness[idx] = obj
                        
                        if obj < best_objective:
                            best_objective = obj
                            best_position = positions[idx].copy()
                            best_start_times = improved
            
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
