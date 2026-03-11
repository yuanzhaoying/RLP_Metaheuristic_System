"""
开始时间编码的元启发式算法 - 统一接口

这个文件提供了统一的接口来访问所有算法。

算法文件说明:
    - GA.py: 遗传算法
    - SA.py: 模拟退火算法
    - ILS.py: 迭代局部搜索算法
    - TS.py: 禁忌搜索算法
    - PR.py: 路径重连算法
    - GSA.py: 引力搜索算法
    - DE.py: 差分进化算法

每个算法文件都包含详细的注释和算子说明
请查看对应的文件了解更多信息
"""

from typing import List
from dataclasses import dataclass, asdict
from ..psp.psplib_io import RCPSPInstance


@dataclass
class AlgorithmResult:
    """算法结果"""
    best_start_times: List[int]
    best_objective: float
    n_evaluations: int
    runtime: float
    convergence: List[float]
    algorithm_params: dict = None


def _params_to_dict(params) -> dict:
    """将参数对象转换为字典"""
    return asdict(params)


def create_algorithm_st(algo_name: str, instance: RCPSPInstance, deadline: int, params):
    """
    创建算法实例（开始时间编码)
    
    参数:
        algo_name: 算法名称 ("ga", "sa", "ils", "ts", "pr", "gsa", "de")
        instance: 问题实例
        deadline: 截止日期
        params: 算法参数
    
    返回:
        algorithm: 算法实例
        encoding_type: 编码类型 ("start_time")
    """
    if algo_name == "ga":
        from .GA import GeneticAlgorithmST
        return GeneticAlgorithmST(instance, deadline, params), "start_time"
    elif algo_name == "sa":
        from .SA import SimulatedAnnealingST
        return SimulatedAnnealingST(instance, deadline, params), "start_time"
    elif algo_name == "ils":
        from .ILS import IteratedLocalSearchST
        return IteratedLocalSearchST(instance, deadline, params), "start_time"
    elif algo_name == "ts":
        from .TS import TabuSearchST
        return TabuSearchST(instance, deadline, params), "start_time"
    elif algo_name == "pr":
        from .PR import PathRelinkingST
        return PathRelinkingST(instance, deadline, params), "start_time"
    elif algo_name == "gsa":
        from .GSA import GravitationalSearchAlgorithmST
        return GravitationalSearchAlgorithmST(instance, deadline, params), "start_time"
    elif algo_name == "de":
        from .DE import DifferentialEvolutionST
        return DifferentialEvolutionST(instance, deadline, params), "start_time"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: ga, sa, ils, ts, pr, gsa, de")


# 从各个算法文件导入参数类
from .GA import GeneticAlgorithmST, GAParamsST
from .SA import SimulatedAnnealingST, SAParamsST
from .ILS import IteratedLocalSearchST, ILSParamsST
from .TS import TabuSearchST, TSParamsST
from .PR import PathRelinkingST, PRParamsST
from .GSA import GravitationalSearchAlgorithmST, GSAParamsST
# DE的导入在create_algorithm_st函数中延迟导入，避免循环导入

# 导出所有公共接口
__all__ = [
    'AlgorithmResult',
    'create_algorithm_st',
    '_params_to_dict',
    
    'GeneticAlgorithmST',
    'GAParamsST',
    'SimulatedAnnealingST',
    'SAParamsST',
    'IteratedLocalSearchST',
    'ILSParamsST',
    'TabuSearchST',
    'TSParamsST',
    'PathRelinkingST',
    'PRParamsST',
    'GravitationalSearchAlgorithmST',
    'GSAParamsST'
]
