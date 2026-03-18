"""
粒子群算法（Particle Swarm Optimization）- 开始时间编码版本

算法简介：
    粒子群算法是一种基于群体智能的优化算法。
    通过模拟鸟群觅食行为，利用个体和群体的经验来搜索最优解。

算法原理：
    每个粒子代表一个解，具有位置和速度两个属性。
    粒子根据个体最优位置（pbest）和全局最优位置（gbest）更新自己的速度和位置。
    
    速度更新公式：
        v(t+1) = w * v(t) + c1 * r1 * (pbest - x(t)) + c2 * r2 * (gbest - x(t))
    
    位置更新公式：
        x(t+1) = x(t) + v(t+1)
    
    其中：
        w: 惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
        r1, r2: [0,1]范围内的随机数

算子汇总：
    1. 局部搜索（Local Search）
       - none：不使用局部搜索
       - sa：模拟退火局部搜索，当算法停滞时触发
    
    2. 重启机制
       - use_restart：是否使用重启机制
       - restart_threshold：停滞阈值（多少次迭代无改进时触发重启）
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
class PSOParamsST:
    """粒子群算法参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    population_size: int = 50
    time_limit: float = 60.0
    
    # PSO参数
    w: float = 0.729  # 惯性权重
    c1: float = 1.49445  # 个体学习因子
    c2: float = 1.49445  # 社会学习因子
    
    # 算子选择
    local_search_strategy: str = "none"  # "none" 或 "sa"
    restart_strategy: str = "none"  # "none" 或 "adaptive"
    
    # 重启参数
    restart_threshold: int = 30  # 停滞阈值
    
    # SA参数
    sa_initial_temp: float = 80.0  # SA初始温度
    sa_final_temp: float = 0.01  # SA终止温度
    sa_cooling_rate: float = 0.9  # SA冷却速率


class ParticleSwarmOptimizationST:
    """粒子群算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PSOParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
        
        # 计算关键路径和总时差
        self._compute_critical_path()
    
    def _compute_critical_path(self):
        """计算关键路径和总时差"""
        # 计算ES和EF
        self.es = [0] * self.n
        self.ef = [0] * self.n
        for j in range(self.n):
            if self.inst.predecessors[j]:
                self.es[j] = max(
                    self.ef[p] for p in self.inst.predecessors[j]
                )
            self.ef[j] = self.es[j] + self.inst.durations[j]
        
        # 项目工期
        self.project_duration = max(self.ef)
        
        # 计算LF和LS
        self.lf = [self.project_duration] * self.n
        self.ls = [self.project_duration - self.inst.durations[j] for j in range(self.n)]
        
        # 反向计算
        for j in range(self.n - 1, -1, -1):
            successors = [i for i in range(self.n) if j in self.inst.predecessors[i]]
            if successors:
                self.lf[j] = min(self.ls[i] for i in successors)
            self.ls[j] = self.lf[j] - self.inst.durations[j]
        
        # 计算总时差
        self.tf = [self.ls[j] - self.es[j] for j in range(self.n)]
        
        # 识别关键活动
        self.critical_activities = [j for j in range(self.n) if self.tf[j] == 0]
        self.non_critical_activities = [j for j in range(self.n) if self.tf[j] > 0]
    
    def _initialize_particle(self) -> np.ndarray:
        """初始化粒子位置"""
        position = np.zeros(self.n, dtype=np.float64)
        
        for j in range(self.n):
            es = self.decoder.es[j]
            ls = self.decoder.ls[j]
            position[j] = self.rng.rng.uniform(es, ls)
        
        return position
    
    def _initialize_velocity(self) -> np.ndarray:
        """初始化粒子速度"""
        velocity = np.zeros(self.n, dtype=np.float64)
        
        for j in range(self.n):
            range_size = self.decoder.ls[j] - self.decoder.es[j]
            velocity[j] = self.rng.rng.uniform(-range_size * 0.1, range_size * 0.1)
        
        return velocity
    
    def _repair_position(self, position: np.ndarray) -> List[int]:
        """修复位置（确保满足约束）- 参考其他算法的标准修复"""
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
    
    def _sa_local_search(self, start_times: List[int]) -> List[int]:
        """SA局部搜索 - 参考用户提供的代码"""
        current = start_times.copy()
        current_obj, _ = self.evaluator.evaluate(current)
        
        T = self.params.sa_initial_temp
        
        while T > self.params.sa_final_temp:
            # 随机选择一个非关键活动
            idx = self.rng.choice(self.non_critical_activities)
            
            # 生成新解 - 使用TF//2作为移位范围
            new = current.copy()
            shift_range = max(1, self.tf[idx] // 2)
            shift = self.rng.integers(-shift_range, shift_range + 1)
            new[idx] = max(0, new[idx] + shift)
            
            # 修复
            new = self._repair_position(np.array(new))
            
            # 评估
            new_obj, _ = self.evaluator.evaluate(new)
            
            # 接受准则
            if new_obj < current_obj:
                current = new
                current_obj = new_obj
            elif self.rng.rng.random() < np.exp((current_obj - new_obj) / T):
                current = new
                current_obj = new_obj
            
            # 降温
            T *= self.params.sa_cooling_rate
        
        return current
    
    def _update_velocity(self, velocity: np.ndarray, position: np.ndarray,
                        pbest: np.ndarray, gbest: np.ndarray) -> np.ndarray:
        """更新粒子速度"""
        # 生成随机数
        r1 = self.rng.rng.uniform(0, 1, self.n)
        r2 = self.rng.rng.uniform(0, 1, self.n)
        
        # 速度更新公式
        new_velocity = (self.params.w * velocity +
                       self.params.c1 * r1 * (pbest - position) +
                       self.params.c2 * r2 * (gbest - position))
        
        # 限制速度范围
        for j in range(self.n):
            range_size = self.decoder.ls[j] - self.decoder.es[j]
            v_min = -range_size * 0.1
            v_max = range_size * 0.1
            new_velocity[j] = np.clip(new_velocity[j], v_min, v_max)
        
        return new_velocity
    
    def _update_position(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """更新位置"""
        new_position = position + velocity
        
        # 限制位置范围
        for j in range(self.n):
            new_position[j] = np.clip(
                new_position[j],
                self.decoder.es[j],
                self.decoder.ls[j]
            )
        
        return new_position
    
    def run(self):
        """运行粒子群算法"""
        start_time = time.time()
        convergence = []
        
        # 初始化粒子群
        positions = []
        velocities = []
        pbest_positions = []
        pbest_objectives = []
        
        for _ in range(self.params.population_size):
            position = self._initialize_particle()
            velocity = self._initialize_velocity()
            
            # 评估初始位置
            start_times = self._repair_position(position)
            obj, _ = self.evaluator.evaluate(start_times)
            
            positions.append(position)
            velocities.append(velocity)
            pbest_positions.append(position.copy())
            pbest_objectives.append(obj)
        
        # 初始化全局最优
        gbest_idx = np.argmin(pbest_objectives)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_objective = pbest_objectives[gbest_idx]
        gbest_start_times = self._repair_position(gbest_position)
        
        convergence.append(gbest_objective)
        
        # 停滞计数
        no_improvement_count = 0
        
        # 主循环
        iteration = 0
        while (self.evaluator.n_evaluations < self.params.max_evaluations and
               time.time() - start_time < self.params.time_limit):
            
            # 更新每个粒子
            for i in range(self.params.population_size):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                
                # 更新速度
                velocities[i] = self._update_velocity(
                    velocities[i], positions[i],
                    pbest_positions[i], gbest_position
                )
                
                # 更新位置
                positions[i] = self._update_position(positions[i], velocities[i])
                
                # 评估新位置
                start_times = self._repair_position(positions[i])
                obj, _ = self.evaluator.evaluate(start_times)
                
                # 更新个体最优
                if obj < pbest_objectives[i]:
                    pbest_objectives[i] = obj
                    pbest_positions[i] = positions[i].copy()
                
                # 更新全局最优
                if obj < gbest_objective:
                    gbest_objective = obj
                    gbest_position = positions[i].copy()
                    gbest_start_times = start_times
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            # 重启机制 - 参考用户代码：当停滞次数超过阈值时触发SA
            if (self.params.restart_strategy == "adaptive" and 
                no_improvement_count > self.params.restart_threshold and
                self.params.local_search_strategy == "sa"):
                
                # 对前5个最优个体执行SA局部搜索
                top_indices = np.argsort(pbest_objectives)[:5]
                idx = self.rng.choice(top_indices)
                
                improved = self._sa_local_search(self._repair_position(pbest_positions[idx]))
                obj, _ = self.evaluator.evaluate(improved)
                
                if obj < gbest_objective:
                    gbest_objective = obj
                    gbest_position = pbest_positions[idx].copy()
                    gbest_start_times = improved
                
                no_improvement_count = 0
            
            convergence.append(gbest_objective)
            iteration += 1
        
        runtime = time.time() - start_time
        
        return AlgorithmResult(
            best_start_times=gbest_start_times,
            best_objective=gbest_objective,
            n_evaluations=self.evaluator.n_evaluations,
            runtime=runtime,
            convergence=convergence,
            algorithm_params=_params_to_dict(self.params)
        )
