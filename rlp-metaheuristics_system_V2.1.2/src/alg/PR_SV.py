"""
路径重连算法（Path Relinking）- 基于位移编码版本

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
       - True/False
"""

from typing import List, Tuple
import time
import numpy as np
from dataclasses import dataclass
from ..psp.psplib_io import RCPSPInstance
from ..psp.shift_vector_evaluator import ShiftVectorEvaluator
from ..psp.shift_vector_decoder import ShiftVectorDecoder
from .operators import RandomGenerator


@dataclass
class PRParamsSV:
    """路径重连参数（位移编码）"""
    max_evaluations: int = 1000
    seed: int = 0
    time_limit: float = 60.0
    max_iterations: int = 100
    elite_size: int = 10
    path_strategy: str = "forward"
    selection_strategy: str = "best"
    use_local_search: bool = False
    local_search_eval_limit: int = 50


class PathRelinkingSV:
    """路径重连算法（位移编码）"""
    
    def __init__(self, instance: RCPSPInstance, deadline: int, params: PRParamsSV):
        self.inst = instance
        self.deadline = deadline
        self.params = params
        self.rng = RandomGenerator(params.seed)
        
        self.evaluator = ShiftVectorEvaluator(instance, deadline, params.max_evaluations)
        self.decoder = ShiftVectorDecoder(instance, deadline)
        self.n = instance.n_activities
        
        self.elite_pool = []
    
    def run(self):
        """运行路径重连算法"""
        start_time = time.time()
        convergence = []
        
        self._initialize_elite_pool()
        
        best_displacement = None
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
            
            path_solutions = self._generate_path(init_solution, guide_solution)
            
            if self.params.selection_strategy == "best":
                for solution in path_solutions:
                    if self.evaluator.n_evaluations >= self.params.max_evaluations:
                        break
                    
                    if self.params.use_local_search:
                        improved = self._local_search(solution, start_time)
                    else:
                        improved = solution
                    
                    obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < best_objective:
                        best_objective = obj
                        best_displacement = improved.copy()
                    
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
                    
                    if self.params.use_local_search:
                        improved = self._local_search(solution, start_time)
                    else:
                        improved = solution
                    
                    obj, _ = self.evaluator.evaluate(improved)
                    
                    if obj < best_objective:
                        best_objective = obj
                        best_displacement = improved.copy()
                    
                    self._update_elite_pool(improved, obj)
            
            convergence.append(best_objective)
            iteration += 1
        
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
    
    def _params_to_dict(self, params) -> dict:
        """将参数对象转换为字典"""
        from dataclasses import asdict
        return asdict(params)
    
    def _initialize_elite_pool(self):
        """初始化精英池"""
        for _ in range(self.params.elite_size):
            solution = self.decoder.encode_random(self.rng)
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
        
        elif self.params.path_strategy == "bidirectional":
            for j in range(self.n):
                if current[j] != guide_solution[j]:
                    current[j] = guide_solution[j]
                    path.append(current.copy())
            
            current = guide_solution.copy()
            for j in range(self.n - 1, -1, -1):
                if current[j] != init_solution[j]:
                    current[j] = init_solution[j]
                    path.append(current.copy())
        
        return path
    
    def _local_search(self, solution: List[int], start_time: float) -> List[int]:
        """局部搜索优化"""
        best_solution = solution.copy()
        best_obj, _ = self.evaluator.evaluate(best_solution)
        
        local_eval_count = 0
        
        for _ in range(10):
            if self.evaluator.n_evaluations >= self.params.max_evaluations:
                break
            if time.time() - start_time >= self.params.time_limit:
                break
            if local_eval_count >= self.params.local_search_eval_limit:
                break
            
            j = self.rng.integers(0, self.n)
            max_shift = self.decoder.get_max_shift(j)
            
            neighbor = best_solution.copy()
            neighbor[j] = self.rng.integers(0, max_shift + 1)
            
            obj, _ = self.evaluator.evaluate(neighbor)
            local_eval_count += 1
            
            if obj < best_obj:
                best_solution = neighbor.copy()
                best_obj = obj
        
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
