"""
路径重连算法（Path Relinking）- 开始时间编码版本

算法简介：
    路径重连是一种通过连接两个解之间的路径来探索搜索空间的元启发式算法。
    从初始解出发，通过一系列中间解到达目标解，在路径上探索新的解。

算子汇总：
    1. 路径生成算子（Path Generation）
       - 逐活动调整
         从初始解到目标解，逐个活动调整开始时间
    
    2. 路径探索策略
       - forward：正向探索，从初始解到目标解
       - backward：反向探索，从目标解到初始解
       - random：随机探索，随机顺序调整活动
       - bidirectional：双向探索，先正向再反向
    
    3. 解选择策略
       - best：选择路径上的最优解
       - random_two：从路径上随机选择两个解，然后选优
    
    4. 约束修复算子
       - 前置约束修复：确保每个活动的开始时间满足前置约束
       - 可行范围修复：确保开始时间在ES-LS范围内
    
    5. 局部优化算子
       - 逐活动优化：对每个活动，在可行范围内搜索最优开始时间
       - first-improvement策略：找到第一个改进就移动
"""

from typing import List, Tuple
import time
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.start_time_evaluator import StartTimeEvaluator
from ..psp.start_time_decoder import StartTimeDecoder
from .operators import RandomGenerator


@dataclass
class PRParamsST:
    """路径重连参数（开始时间编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    time_limit: float = 60.0
    max_iterations: int = 100
    elite_size: int = 10
    path_strategy: str = "forward"
    selection_strategy: str = "best"
    use_local_search: bool = False
    local_search_eval_limit: int = 50


class PathRelinkingST:
    """路径重连算法（开始时间编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PRParamsST):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = StartTimeEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = StartTimeDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.elite_pool = []
    
    def run(self):
        """运行路径重连算法"""
        start_time = time.time()
        convergence = []
        
        self._initialize_elite_pool()
        
        best_start_times = None
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
            guide_solution = self.elite_pool[guide_idx][0]
            
            if self.params.path_strategy == "bidirectional":
                path_solutions = self._generate_bidirectional_path(init_solution, guide_solution)
            else:
                path_solutions = self._generate_path(init_solution, guide_solution)
            
            if self.params.selection_strategy == "best":
                for solution in path_solutions:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    repaired = self._repair(solution)
                    
                    if self.params.use_local_search:
                        improved = self._local_search(repaired, start_time)
                    else:
                        improved = repaired
                    
                    obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < best_objective:
                        best_objective = obj
                        best_start_times = improved.copy()
                    
                    self._update_elite_pool(improved, obj)
            
            elif self.params.selection_strategy == "random_two":
                if len(path_solutions) >= 2:
                    selected_indices = self.rng.choice(len(path_solutions), size=2, replace=False)
                    selected_solutions = [path_solutions[i] for i in selected_indices]
                else:
                    selected_solutions = path_solutions
                
                for solution in selected_solutions:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    repaired = self._repair(solution)
                    
                    if self.params.use_local_search:
                        improved = self._local_search(repaired, start_time)
                    else:
                        improved = repaired
                    
                    obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < best_objective:
                        best_objective = obj
                        best_start_times = improved.copy()
                    
                    self._update_elite_pool(improved, obj)
            
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
    
    def _initialize_elite_pool(self):
        """初始化精英池"""
        for _ in range(self.params.elite_size):
            solution = []
            for j in range(self.n):
                start_time = self.rng.integers(self.decoder.es[j], self.decoder.ls[j] + 1)
                solution.append(start_time)
            
            obj, _ = self.evaluator.evaluate(solution)
            self.elite_pool.append((solution, obj))
        
        self.elite_pool.sort(key=lambda x: x[1])
    
    def _generate_path(self, init_solution: List[int], guide_solution: List[int]) -> List[List[int]]:
        """生成从初始解到目标解的路径"""
        path = []
        current = init_solution.copy()
        
        if self.params.path_strategy == "forward":
            for j in range(self.n):
                if current[j] != guide_solution[j]:
                    current[j] = guide_solution[j]
                    path.append(current.copy())
        
        elif self.params.path_strategy == "backward":
            for j in range(self.n - 1, -1, -1):
                if current[j] != guide_solution[j]:
                    current[j] = guide_solution[j]
                    path.append(current.copy())
        
        elif self.params.path_strategy == "random":
            indices = list(range(self.n))
            self.rng.shuffle(indices)
            for j in indices:
                if current[j] != guide_solution[j]:
                    current[j] = guide_solution[j]
                    path.append(current.copy())
        
        return path
    
    def _generate_bidirectional_path(self, init_solution: List[int], guide_solution: List[int]) -> List[List[int]]:
        """生成双向路径：先正向，再反向"""
        path = []
        
        # 正向路径：init -> guide
        current = init_solution.copy()
        for j in range(self.n):
            if current[j] != guide_solution[j]:
                current[j] = guide_solution[j]
                path.append(current.copy())
        
        # 反向路径：guide -> init
        current = guide_solution.copy()
        for j in range(self.n - 1, -1, -1):
            if current[j] != init_solution[j]:
                current[j] = init_solution[j]
                path.append(current.copy())
        
        return path
    
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
    
    def _local_search(self, solution: List[int], start_time: float) -> List[int]:
        """局部搜索优化"""
        improved = True
        best_solution = solution.copy()
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
                
                for t in range(self.decoder.es[i], self.decoder.ls[i] + 1):
                    if t == best_solution[i]:
                        continue
                    
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    if local_eval_count >= self.params.local_search_eval_limit:
                        break
                    
                    neighbor = best_solution.copy()
                    neighbor[i] = t
                    
                    neighbor = self._repair(neighbor)
                    
                    obj, _ = self.evaluator.evaluate(neighbor)
                    local_eval_count += 1
                    
                    if obj < best_obj:
                        best_solution = neighbor.copy()
                        best_obj = obj
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_solution
    
    def _update_elite_pool(self, solution: List[int], obj: float):
        """更新精英池"""
        if len(self.elite_pool) < self.params.elite_size:
            self.elite_pool.append((solution, obj))
            self.elite_pool.sort(key=lambda x: x[1])
        else:
            if obj < self.elite_pool[-1][1]:
                is_different = True
                for elite_sol, _ in self.elite_pool:
                    if solution == elite_sol:
                        is_different = False
                        break
                
                if is_different:
                    self.elite_pool[-1] = (solution, obj)
                    self.elite_pool.sort(key=lambda x: x[1])
