"""
基于位移的编码（Shift Vector Encoding）的元启发式算法 - 统一接口

这个文件提供了统一的接口来访问所有SV编码算法。

编码说明:
    - 基于位移的编码（Shift Vector Encoding）是一种连续值编码
    - 编码是一个位移向量，每个活动有一个位移值
    - 位移值范围: [0, LS-ES]
    - 实际开始时间 = ES + 位移值
"""

from typing import List
from dataclasses import dataclass, asdict
from ..psp.psplib_io import RCPSPInstance


@dataclass
class AlgorithmResultSV:
    """算法结果（位移编码）"""
    best_displacement: List[int]
    best_start_times: List[int]
    best_objective: float
    n_evaluations: int
    runtime: float
    convergence: List[float]
    algorithm_params: dict = None


def _params_to_dict(params) -> dict:
    """将参数对象转换为字典"""
    return asdict(params)


def create_algorithm_sv(algo_name: str, instance: RCPSPInstance, deadline: int, params):
    """
    创建算法实例（位移编码）
    
    参数:
        algo_name: 算法名称 ("ga", "sa", "pso", "ba", "hs", "de", "ts", "pr")
        instance: 问题实例
        deadline: 截止日期
        params: 算法参数
    
    返回:
        algorithm: 算法实例
        encoding_type: 编码类型 ("shift_vector")
    """
    algo_name = algo_name.lower()
    
    if algo_name == "ga":
        from .GA_SV import GeneticAlgorithmSV
        return GeneticAlgorithmSV(instance, deadline, params), "shift_vector"
    elif algo_name == "sa":
        from .SA_SV import SimulatedAnnealingSV
        return SimulatedAnnealingSV(instance, deadline, params), "shift_vector"
    elif algo_name == "pso":
        from .PSO_SV import ParticleSwarmOptimizationSV
        return ParticleSwarmOptimizationSV(instance, deadline, params), "shift_vector"
    elif algo_name == "ba":
        from .BA_SV import BatAlgorithmSV
        return BatAlgorithmSV(instance, deadline, params), "shift_vector"
    elif algo_name == "hs":
        from .HS_SV import HarmonySearchSV
        return HarmonySearchSV(instance, deadline, params), "shift_vector"
    elif algo_name == "de":
        from .DE_SV import DifferentialEvolutionSV
        return DifferentialEvolutionSV(instance, deadline, params), "shift_vector"
    elif algo_name == "ts":
        from .TS_SV import TabuSearchSV
        return TabuSearchSV(instance, deadline, params), "shift_vector"
    elif algo_name == "pr":
        from .PR_SV import PathRelinkingSV
        return PathRelinkingSV(instance, deadline, params), "shift_vector"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: ga, sa, pso, ba, hs, de, ts, pr")


from .GA_SV import GeneticAlgorithmSV, GAParamsSV
from .SA_SV import SimulatedAnnealingSV, SAParamsSV
from .PSO_SV import ParticleSwarmOptimizationSV, PSOParamsSV
from .BA_SV import BatAlgorithmSV, BAParamsSV
from .HS_SV import HarmonySearchSV, HSParamsSV
from .DE_SV import DifferentialEvolutionSV, DEParamsSV
from .TS_SV import TabuSearchSV, TSParamsSV
from .PR_SV import PathRelinkingSV, PRParamsSV

__all__ = [
    'AlgorithmResultSV',
    'create_algorithm_sv',
    '_params_to_dict',
    
    'GeneticAlgorithmSV',
    'GAParamsSV',
    'SimulatedAnnealingSV',
    'SAParamsSV',
    'ParticleSwarmOptimizationSV',
    'PSOParamsSV',
    'BatAlgorithmSV',
    'BAParamsSV',
    'HarmonySearchSV',
    'HSParamsSV',
    'DifferentialEvolutionSV',
    'DEParamsSV',
    'TabuSearchSV',
    'TSParamsSV',
    'PathRelinkingSV',
    'PRParamsSV',
]
