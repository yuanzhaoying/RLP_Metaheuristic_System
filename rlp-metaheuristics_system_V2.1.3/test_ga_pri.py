"""
测试GA_PRI优先级编码算法
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.psp.psplib_io import load_psplib_sm
from src.alg.GA_PRI import GeneticAlgorithmPRI, GAParamsPRI
from src.psp.priority_decoder import PriorityDecoder


def get_deadline(instance):
    n = instance.n_activities
    es = [0] * n
    for j in range(n):
        for pred in instance.predecessors[j]:
            es[j] = max(es[j], es[pred] + instance.durations[pred])
    critical_path_length = max([es[i] + instance.durations[i] for i in range(n)])
    return int(critical_path_length)


def test_ga_pri():
    print("="*60)
    print("测试GA_PRI优先级编码算法 - J30_1实例")
    print("="*60)
    
    instance_path = "data/psplib_raw/j30/J30_1.RCP"
    instance = load_psplib_sm(instance_path)
    deadline = get_deadline(instance)
    
    print(f"实例: {instance.name}")
    print(f"活动数: {instance.n_activities}")
    print(f"截止日期: {deadline}")
    
    max_evaluations = 500
    time_limit = 10.0
    seed = 42
    
    print("\n测试1: 不使用延迟因子")
    params_no_delay = GAParamsPRI(
        max_evaluations=max_evaluations,
        seed=seed,
        population_size=30,
        time_limit=time_limit,
        use_delay_factors=False
    )
    
    ga_no_delay = GeneticAlgorithmPRI(instance, deadline, params_no_delay)
    result_no_delay = ga_no_delay.run()
    
    print(f"  目标值: {result_no_delay['best_objective']:.4f}")
    print(f"  评估次数: {result_no_delay['n_evaluations']}")
    
    print("\n测试2: 使用延迟因子")
    params_with_delay = GAParamsPRI(
        max_evaluations=max_evaluations,
        seed=seed,
        population_size=30,
        time_limit=time_limit,
        use_delay_factors=True
    )
    
    ga_with_delay = GeneticAlgorithmPRI(instance, deadline, params_with_delay)
    result_with_delay = ga_with_delay.run()
    
    print(f"  目标值: {result_with_delay['best_objective']:.4f}")
    print(f"  评估次数: {result_with_delay['n_evaluations']}")
    
    print("\n测试3: 不同种子测试")
    objectives = []
    for s in [42, 123, 456]:
        params = GAParamsPRI(
            max_evaluations=max_evaluations,
            seed=s,
            population_size=30,
            time_limit=time_limit,
            use_delay_factors=True
        )
        ga = GeneticAlgorithmPRI(instance, deadline, params)
        result = ga.run()
        objectives.append(result['best_objective'])
        print(f"  seed={s}: {result['best_objective']:.4f}")
    
    print(f"\n目标值: {objectives}")
    print(f"不同目标值数量: {len(set(objectives))}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_ga_pri()
