"""
活动列表编码的元启发式算法 - 统一接口

这个文件提供了统一的接口来访问所有AL编码算法。

编码说明:
    - 活动列表编码（Activity List Encoding）是一种排列编码
    - 编码是一个活动排列，如 [0, 3, 1, 2, 4, ...]
    - 解码时按列表顺序调度活动，满足优先关系
"""

from typing import List
from dataclasses import dataclass, asdict
from ..psp.psplib_io import RCPSPInstance


@dataclass
class AlgorithmResultAL:
    """算法结果（活动列表编码）"""
    best_activity_list: List[int]
    best_start_times: List[int]
    best_objective: float
    n_evaluations: int
    runtime: float
    convergence: List[float]
    algorithm_params: dict = None


def _params_to_dict(params) -> dict:
    """将参数对象转换为字典"""
    return asdict(params)


def create_algorithm_al(algo_name: str, instance: RCPSPInstance, deadline: int, params):
    """
    创建算法实例（活动列表编码）
    
    参数:
        algo_name: 算法名称 ("ga", "sa", "pso", "ba", "hs", "de", "ts", "pr")
        instance: 问题实例
        deadline: 截止日期
        params: 算法参数
    
    返回:
        algorithm: 算法实例
        encoding_type: 编码类型 ("activity_list")
    """
    algo_name = algo_name.lower()
    
    if algo_name == "ga":
        from .GA_AL import GeneticAlgorithmAL
        return GeneticAlgorithmAL(instance, deadline, params), "activity_list"
    elif algo_name == "sa":
        from .SA_AL import SimulatedAnnealingAL
        return SimulatedAnnealingAL(instance, deadline, params), "activity_list"
    elif algo_name == "pso":
        from .PSO_AL import ParticleSwarmOptimizationAL
        return ParticleSwarmOptimizationAL(instance, deadline, params), "activity_list"
    elif algo_name == "ba":
        from .BA_AL import BatAlgorithmAL
        return BatAlgorithmAL(instance, deadline, params), "activity_list"
    elif algo_name == "hs":
        from .HS_AL import HarmonySearchAL
        return HarmonySearchAL(instance, deadline, params), "activity_list"
    elif algo_name == "de":
        from .DE_AL import DifferentialEvolutionAL
        return DifferentialEvolutionAL(instance, deadline, params), "activity_list"
    elif algo_name == "ts":
        from .TS_AL import TabuSearchAL
        return TabuSearchAL(instance, deadline, params), "activity_list"
    elif algo_name == "pr":
        from .PR_AL import PathRelinkingAL
        return PathRelinkingAL(instance, deadline, params), "activity_list"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: ga, sa, pso, ba, hs, de, ts, pr")


from .GA_AL import GeneticAlgorithmAL, GAParamsAL
from .SA_AL import SimulatedAnnealingAL, SAParamsAL
from .PSO_AL import ParticleSwarmOptimizationAL, PSOParamsAL
from .BA_AL import BatAlgorithmAL, BAParamsAL
from .HS_AL import HarmonySearchAL, HSParamsAL
from .DE_AL import DifferentialEvolutionAL, DEParamsAL
from .TS_AL import TabuSearchAL, TSParamsAL
from .PR_AL import PathRelinkingAL, PRParamsAL

__all__ = [
    'AlgorithmResultAL',
    'create_algorithm_al',
    '_params_to_dict',
    
    'GeneticAlgorithmAL',
    'GAParamsAL',
    'SimulatedAnnealingAL',
    'SAParamsAL',
    'ParticleSwarmOptimizationAL',
    'PSOParamsAL',
    'BatAlgorithmAL',
    'BAParamsAL',
    'HarmonySearchAL',
    'HSParamsAL',
    'DifferentialEvolutionAL',
    'DEParamsAL',
    'TabuSearchAL',
    'TSParamsAL',
    'PathRelinkingAL',
    'PRParamsAL',
]
