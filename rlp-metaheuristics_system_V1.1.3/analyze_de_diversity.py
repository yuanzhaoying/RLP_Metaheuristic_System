#!/usr/bin/env python3
"""分析为什么有些组合目标函数值永远相同"""
import sys
sys.path.insert(0, '.')

from src.psp.psplib_io import load_psplib_sm
from src.alg.start_time_algorithms import create_algorithm_st, DEParamsST
import numpy as np

print('分析为什么有些组合目标函数值永远相同:')
print('='*60)

# 加载J30_2实例
instance = load_psplib_sm('data/psplib_raw/j30/J30_2.RCP')
n = instance.n_activities

es = [0] * n
for j in range(n):
    for pred in instance.predecessors[j]:
        es[j] = max(es[j], es[pred] + instance.durations[pred])
critical_path_length = max([es[i] + instance.durations[i] for i in range(n)])
deadline = int(critical_path_length)

print(f'实例: J30_2, Deadline: {deadline}, 活动数: {n}')
print()

# 测试1：比较random和heuristic初始化
print('测试1：比较random和heuristic初始化')
print('-'*60)

for init_strategy in ["random", "heuristic"]:
    print(f'\n初始化策略: {init_strategy}')
    
    params = DEParamsST(
        max_evaluations=100,
        seed=0,
        population_size=20,
        time_limit=5.0,
        max_iterations=20,
        mutation_strategy="rand/1",
        crossover_strategy="bin",
        initialization_strategy=init_strategy,
        use_local_search=False,
        F_min=0.3,
        F_max=1.5,
        CR_min=0.3,
        CR_max=1.0,
        K0=1.0
    )
    
    algo, _ = create_algorithm_st("de", instance, deadline, params)
    result = algo.run()
    
    print(f'  最优目标值: {result.best_objective:.2f}')
    print(f'  评估次数: {result.n_evaluations}')
    print(f'  收敛曲线长度: {len(result.convergence)}')
    print(f'  收敛曲线前5个值: {result.convergence[:5]}')
    print(f'  收敛曲线后5个值: {result.convergence[-5:]}')

# 测试2：增加评估次数
print()
print('='*60)
print('测试2：增加评估次数（从100增加到1000）')
print('-'*60)

for init_strategy in ["random", "heuristic"]:
    print(f'\n初始化策略: {init_strategy}')
    
    params = DEParamsST(
        max_evaluations=1000,
        seed=0,
        population_size=50,
        time_limit=10.0,
        max_iterations=100,
        mutation_strategy="rand/1",
        crossover_strategy="bin",
        initialization_strategy=init_strategy,
        use_local_search=False,
        F_min=0.3,
        F_max=1.5,
        CR_min=0.3,
        CR_max=1.0,
        K0=1.0
    )
    
    algo, _ = create_algorithm_st("de", instance, deadline, params)
    result = algo.run()
    
    print(f'  最优目标值: {result.best_objective:.2f}')
    print(f'  评估次数: {result.n_evaluations}')
    print(f'  收敛曲线长度: {len(result.convergence)}')

# 测试3：检查heuristic初始化的种群多样性
print()
print('='*60)
print('测试3：检查heuristic初始化的种群多样性')
print('-'*60)

# 手动初始化种群并检查多样性
from src.psp.start_time_decoder import StartTimeDecoder
from src.alg.operators import RandomGenerator

decoder = StartTimeDecoder(instance, deadline)
rng = RandomGenerator(0)

# Random初始化
print('\nRandom初始化（20个个体）:')
random_pop = []
for _ in range(20):
    individual = []
    for j in range(n):
        start_time = rng.integers(decoder.es[j], decoder.ls[j] + 1)
        individual.append(start_time)
    random_pop.append(individual)

# 计算种群多样性（个体之间的平均距离）
random_diversity = []
for i in range(len(random_pop)):
    for j in range(i+1, len(random_pop)):
        dist = np.sqrt(sum((random_pop[i][k] - random_pop[j][k])**2 for k in range(n)))
        random_diversity.append(dist)

print(f'  种群平均多样性（平均距离）: {np.mean(random_diversity):.2f}')
print(f'  种群多样性标准差: {np.std(random_diversity):.2f}')

# Heuristic初始化
print('\nHeuristic初始化（20个个体）:')
heuristic_pop = []

# 计算拓扑排序
in_degree = [0] * n
adj = [[] for _ in range(n)]
for i in range(n):
    for j in instance.predecessors[i]:
        adj[j].append(i)
        in_degree[i] += 1

from collections import deque
queue = deque([i for i in range(n) if in_degree[i] == 0])
topo_order = []
while queue:
    u = queue.popleft()
    topo_order.append(u)
    for v in adj[u]:
        in_degree[v] -= 1
        if in_degree[v] == 0:
            queue.append(v)

for _ in range(20):
    individual = [0] * n
    for i in topo_order:
        if instance.predecessors[i]:
            pf = max(individual[j] + instance.durations[j] for j in instance.predecessors[i])
        else:
            pf = 0
        
        s_min = max(decoder.es[i], pf)
        s_max = decoder.ls[i]
        
        if s_min <= s_max:
            individual[i] = rng.integers(s_min, s_max + 1)
        else:
            individual[i] = s_min
    
    heuristic_pop.append(individual)

# 计算种群多样性
heuristic_diversity = []
for i in range(len(heuristic_pop)):
    for j in range(i+1, len(heuristic_pop)):
        dist = np.sqrt(sum((heuristic_pop[i][k] - heuristic_pop[j][k])**2 for k in range(n)))
        heuristic_diversity.append(dist)

print(f'  种群平均多样性（平均距离）: {np.mean(heuristic_diversity):.2f}')
print(f'  种群多样性标准差: {np.std(heuristic_diversity):.2f}')

# 检查是否有重复个体
unique_individuals = len(set(tuple(ind) for ind in heuristic_pop))
print(f'  唯一个体数量: {unique_individuals}/20')

# 测试4：检查关键活动和非关键活动
print()
print('='*60)
print('测试4：检查关键活动和非关键活动')
print('-'*60)

# 计算LS
ls = [deadline - instance.durations[i] for i in range(n)]
for i in range(n-1, -1, -1):
    for j in range(n):
        if i in instance.predecessors[j]:
            ls[i] = min(ls[i], ls[j] - instance.durations[i])

critical_activities = [i for i in range(n) if ls[i] - es[i] == 0]
non_critical_activities = [i for i in range(n) if ls[i] - es[i] > 0]

print(f'关键活动数量: {len(critical_activities)}')
print(f'非关键活动数量: {len(non_critical_activities)}')
print(f'非关键活动占比: {len(non_critical_activities)/n*100:.1f}%')

# 检查非关键活动的松弛时间
slack_times = [ls[i] - es[i] for i in non_critical_activities]
print(f'非关键活动平均松弛时间: {np.mean(slack_times):.2f}')
print(f'非关键活动最大松弛时间: {max(slack_times)}')
print(f'非关键活动最小松弛时间: {min(slack_times)}')

print()
print('='*60)
print('分析完成！')
print('='*60)
print()
print('结论:')
print('1. 如果heuristic初始化的种群多样性很低，可能导致算法快速收敛到相同的解')
print('2. 如果非关键活动数量很少或松弛时间很小，搜索空间有限')
print('3. 增加评估次数可能有助于找到更好的解')
