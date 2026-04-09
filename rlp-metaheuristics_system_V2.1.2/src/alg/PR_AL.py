"""
路径重连算法（Path Relinking）- 活动列表编码版本

算法简介：
    路径重连是一种通过连接两个解之间的路径来探索搜索空间的元启发式算法。
    从初始解出发，通过一系列中间解到达目标解，在路径上探索新的解。

AL编码特点：
    - 编码是活动排列（排列编码）
    - 路径操作使用排列专用的交叉操作

算子汇总：
    1. 路径探索策略
       - forward：正向探索
       - backward：反向探索
       - random：随机探索
       - bidirectional：双向探索
    
    2. 解选择策略
       - best：选择路径上的最优解
       - random_two：随机选择两个解
    
    3. 局部优化
       - swap：交换局部搜索
"""

from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.activity_list_evaluator import ActivityListEvaluator
from ..psp.activity_list_decoder import ActivityListDecoder
from .operators import RandomGenerator


@dataclass
class PRParamsAL:
    """路径重连参数（活动列表编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    time_limit: float = 60.0
    max_iterations: int = 100
    elite_size: int = 10
    path_strategy: str = "forward"
    selection_strategy: str = "best"
    use_local_search: bool = False
    local_search_eval_limit: int = 50
    use_delay_factors: bool = True
    delay_mutation_rate: float = 0.1


class PathRelinkingAL:
    """路径重连算法（活动列表编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PRParamsAL):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ActivityListEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ActivityListDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.elite_pool = []
    
    def run(self):
        """运行路径重连算法"""
        start_time = time.time()
        convergence = []
        
        self._initialize_elite_pool()
        
        best_activity_list = None
        best_delay_factors = None
        best_objective = float('inf')
        
        iteration = 0
        while (iteration < self.params.max_iterations and 
               self.evaluator.n_evaluations < self.params.max_evaluations and 
               time.time() - start_time < self.params.time_limit):
            
            init_idx = self.rng.integers(0, len(self.elite_pool))
            guide_idx = self.rng.integers(0, len(self.elite_pool))
            
            while guide_idx == init_idx:
                guide_idx = self.rng.integers(0, len(self.elite_pool))
            
            init_solution = self.elite_pool[init_idx][0]
            init_delay = self.elite_pool[init_idx][2]
            guide_solution = self.elite_pool[guide_idx][0]
            guide_delay = self.elite_pool[guide_idx][2]
            
            if self.params.path_strategy == "bidirectional":
                path_solutions = self._generate_bidirectional_path(init_solution, guide_solution, init_delay, guide_delay)
            else:
                path_solutions = self._generate_path(init_solution, guide_solution, init_delay, guide_delay)
            
            if self.params.selection_strategy == "best":
                for solution, delays in path_solutions:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    repaired = self.decoder.repair(solution)
                    
                    if self.params.use_local_search:
                        improved, improved_delay = self._local_search(repaired, delays, start_time)
                    else:
                        improved = repaired
                        improved_delay = delays
                    
                    if self.params.use_delay_factors and improved_delay:
                        obj, _ = self.evaluator.evaluate(improved, improved_delay)
                    else:
                        obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < best_objective:
                        best_objective = obj
                        best_activity_list = improved.copy()
                        if self.params.use_delay_factors and improved_delay:
                            best_delay_factors = improved_delay.copy()
                    
                    self._update_elite_pool(improved, obj, improved_delay)
            
            elif self.params.selection_strategy == "random_two":
                if len(path_solutions) >= 2:
                    selected_indices = self.rng.choice(len(path_solutions), size=2, replace=False)
                    selected_solutions = [path_solutions[i] for i in selected_indices]
                else:
                    selected_solutions = path_solutions
                
                for solution, delays in selected_solutions:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    repaired = self.decoder.repair(solution)
                    
                    if self.params.use_local_search:
                        improved, improved_delay = self._local_search(repaired, delays, start_time)
                    else:
                        improved = repaired
                        improved_delay = delays
                    
                    if self.params.use_delay_factors and improved_delay:
                        obj, _ = self.evaluator.evaluate(improved, improved_delay)
                    else:
                        obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < best_objective:
                        best_objective = obj
                        best_activity_list = improved.copy()
                        if self.params.use_delay_factors and improved_delay:
                            best_delay_factors = improved_delay.copy()
                    
                    self._update_elite_pool(improved, obj, improved_delay)
            
            convergence.append(best_objective)
            iteration += 1
        
        if self.params.use_delay_factors and best_delay_factors:
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
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
    
    def _initialize_elite_pool(self):
        """初始化精英池"""
        for _ in range(self.params.elite_size):
            if self.params.use_delay_factors:
                al, delays = self.decoder.encode_random_with_delay(self.rng)
            else:
                al = self.decoder.encode_random(self.rng)
                delays = None
            al = self.decoder.repair(al)
            
            if self.params.use_delay_factors and delays:
                obj, _ = self.evaluator.evaluate(al, delays)
            else:
                obj, _ = self.evaluator.evaluate(al)
            
            self.elite_pool.append((al, obj, delays))
        
        self.elite_pool.sort(key=lambda x: x[1])
    
    def _generate_path(self, init_solution: List[int], guide_solution: List[int], 
                       init_delay: List[float] = None, guide_delay: List[float] = None) -> List[Tuple]:
        """生成从初始解到目标解的路径"""
        path = []
        current = init_solution.copy()
        current_delay = init_delay.copy() if init_delay else None
        
        if self.params.path_strategy == "forward":
            for j in range(self.n):
                if current[j] != guide_solution[j]:
                    current, current_delay = self._move_towards(current, guide_solution, j, current_delay, guide_delay)
                    path.append((current.copy(), current_delay.copy() if current_delay else None))
        
        elif self.params.path_strategy == "backward":
            for j in range(self.n - 1, -1, -1):
                if current[j] != guide_solution[j]:
                    current, current_delay = self._move_towards(current, guide_solution, j, current_delay, guide_delay)
                    path.append((current.copy(), current_delay.copy() if current_delay else None))
        
        elif self.params.path_strategy == "random":
            indices = list(range(self.n))
            self.rng.shuffle(indices)
            for j in indices:
                if current[j] != guide_solution[j]:
                    current, current_delay = self._move_towards(current, guide_solution, j, current_delay, guide_delay)
                    path.append((current.copy(), current_delay.copy() if current_delay else None))
        
        return path
    
    def _generate_bidirectional_path(self, init_solution: List[int], guide_solution: List[int],
                                      init_delay: List[float] = None, guide_delay: List[float] = None) -> List[Tuple]:
        """生成双向路径"""
        path = []
        
        current = init_solution.copy()
        current_delay = init_delay.copy() if init_delay else None
        for j in range(self.n):
            if current[j] != guide_solution[j]:
                current, current_delay = self._move_towards(current, guide_solution, j, current_delay, guide_delay)
                path.append((current.copy(), current_delay.copy() if current_delay else None))
        
        current = guide_solution.copy()
        current_delay = guide_delay.copy() if guide_delay else None
        for j in range(self.n - 1, -1, -1):
            if current[j] != init_solution[j]:
                current, current_delay = self._move_towards(current, init_solution, j, current_delay, init_delay)
                path.append((current.copy(), current_delay.copy() if current_delay else None))
        
        return path
    
    def _move_towards(self, current: List[int], guide: List[int], position: int,
                      current_delay: List[float] = None, guide_delay: List[float] = None) -> tuple:
        """将当前位置的元素移动到目标位置"""
        result = current.copy()
        result_delay = current_delay.copy() if current_delay else None
        target_value = guide[position]
        
        if target_value in result:
            current_pos = result.index(target_value)
            result[current_pos], result[position] = result[position], result[current_pos]
            
            if self.params.use_delay_factors and result_delay and guide_delay:
                result_delay[current_pos], result_delay[position] = result_delay[position], result_delay[current_pos]
        
        if self.params.use_delay_factors and result_delay:
            for k in range(len(result_delay)):
                if self.rng.random() < self.params.delay_mutation_rate:
                    result_delay[k] = self.rng.random()
            result_delay[0] = 0.0
            result_delay[self.n - 1] = 0.0
        
        return result, result_delay
    
    def _local_search(self, solution: List[int], delay_factors: List[float], start_time: float) -> tuple:
        """局部搜索优化"""
        improved = True
        best_solution = solution.copy()
        best_delay = delay_factors.copy() if delay_factors else None
        
        if self.params.use_delay_factors and best_delay:
            best_obj, _ = self.evaluator.evaluate(best_solution, best_delay)
        else:
            best_obj, _ = self.evaluator.evaluate(best_solution)
        
        local_eval_count = 0
        
        while improved:
            improved = False
            
            for i in range(self.n):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break
                if time.time() - start_time >= self.params.time_limit:
                    break
                if local_eval_count >= self.params.local_search_eval_limit:
                    break
                
                for j in range(i + 1, self.n):
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    if local_eval_count >= self.params.local_search_eval_limit:
                        break
                    
                    neighbor = best_solution.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbor = self.decoder.repair(neighbor)
                    
                    neighbor_delay = best_delay.copy() if best_delay else None
                    if self.params.use_delay_factors and neighbor_delay:
                        for k in range(len(neighbor_delay)):
                            if self.rng.random() < 0.1:
                                neighbor_delay[k] = self.rng.random()
                        neighbor_delay[0] = 0.0
                        neighbor_delay[self.n - 1] = 0.0
                    
                    if self.params.use_delay_factors and neighbor_delay:
                        obj, _ = self.evaluator.evaluate(neighbor, neighbor_delay)
                    else:
                        obj, _ = self.evaluator.evaluate(neighbor)
                    local_eval_count += 1
                    
                    if obj < best_obj:
                        best_solution = neighbor.copy()
                        best_delay = neighbor_delay
                        best_obj = obj
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_solution, best_delay
    
    def _update_elite_pool(self, solution: List[int], obj: float, delays: List[float] = None):
        """更新精英池"""
        if len(self.elite_pool) < self.params.elite_size:
            self.elite_pool.append((solution, obj, delays))
            self.elite_pool.sort(key=lambda x: x[1])
        else:
            if obj < self.elite_pool[-1][1]:
                is_different = True
                for elite_sol, _, _ in self.elite_pool:
                    if solution == elite_sol:
                        is_different = False
                        break
                
                if is_different:
                    self.elite_pool[-1] = (solution, obj, delays)
                    self.elite_pool.sort(key=lambda x: x[1])
