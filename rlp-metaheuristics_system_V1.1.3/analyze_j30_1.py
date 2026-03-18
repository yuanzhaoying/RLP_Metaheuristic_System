#!/usr/bin/env python3
"""分析J30_1实例的网络结构"""
import sys
sys.path.insert(0, '.')

from src.psp.psplib_io import load_psplib_sm
import numpy as np

print('分析J30_1实例的网络结构:')
print('='*60)

# 加载实例
instance = load_psplib_sm('data/psplib_raw/j30/J30_1.RCP')
n = instance.n_activities

print(f'\n【基本信息】')
print(f'活动数量: {n}')
print(f'资源类型数: {instance.n_resources}')
print(f'资源限量: {instance.capacity}')

print(f'\n【活动信息】')
print(f'活动编号: 0 到 {n-1}')
print(f'活动0: 虚拟开始活动 (持续时间=0, 资源需求=0)')
print(f'活动{n-1}: 虚拟结束活动 (持续时间=0, 资源需求=0)')

print(f'\n【活动详细信息】')
print('-'*60)
print(f'{"活动":<4} {"持续时间":<8} {"前置活动":<20} {"资源需求"}')
print('-'*60)

for i in range(n):
    duration = instance.durations[i]
    predecessors = instance.predecessors[i]
    resources = instance.resource_demands[i]
    
    pred_str = str(predecessors) if predecessors else "[]"
    res_str = str(resources)
    
    print(f'{i:<4} {duration:<8} {pred_str:<20} {res_str}')

# 计算ES/LS
es = [0] * n
for j in range(n):
    for pred in instance.predecessors[j]:
        es[j] = max(es[j], es[pred] + instance.durations[pred])

critical_path_length = max([es[i] + instance.durations[i] for i in range(n)])
deadline = int(critical_path_length)

ls = [deadline - instance.durations[i] for i in range(n)]
for i in range(n-1, -1, -1):
    for j in range(n):
        if i in instance.predecessors[j]:
            ls[i] = min(ls[i], ls[j] - instance.durations[i])

print(f'\n【时间参数】')
print(f'关键路径长度: {critical_path_length}')
print(f'项目截止日期: {deadline}')

print(f'\n【ES/LS时间】')
print('-'*60)
print(f'{"活动":<4} {"ES":<8} {"LS":<8} {"TF":<8} {"是否关键活动"}')
print('-'*60)

for i in range(n):
    tf = ls[i] - es[i]
    is_critical = "是" if tf == 0 else "否"
    print(f'{i:<4} {es[i]:<8} {ls[i]:<8} {tf:<8} {is_critical}')

# 统计关键活动
critical_activities = [i for i in range(n) if ls[i] - es[i] == 0]
non_critical_activities = [i for i in range(n) if ls[i] - es[i] > 0]

print(f'\n【关键活动统计】')
print(f'关键活动数量: {len(critical_activities)}')
print(f'关键活动: {critical_activities}')
print(f'非关键活动数量: {len(non_critical_activities)}')
print(f'非关键活动: {non_critical_activities}')

# 分析网络层次
print(f'\n【网络层次结构】')
in_degree = [0] * n
adj = [[] for _ in range(n)]
for i in range(n):
    for j in instance.predecessors[i]:
        adj[j].append(i)
        in_degree[i] += 1

from collections import deque
queue = deque([i for i in range(n) if in_degree[i] == 0])
levels = {}
level = 0

while queue:
    level_size = len(queue)
    levels[level] = list(queue)
    
    for _ in range(level_size):
        u = queue.popleft()
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    level += 1

for level_num, activities in sorted(levels.items()):
    print(f'第{level_num}层: {activities}')

print(f'\n【网络复杂度】')
print(f'网络层数: {len(levels)}')
print(f'平均每层活动数: {n / len(levels):.2f}')

# 统计前置关系
print(f'\n【前置关系统计】')
predecessor_counts = {}
for i in range(n):
    count = len(instance.predecessors[i])
    if count not in predecessor_counts:
        predecessor_counts[count] = 0
    predecessor_counts[count] += 1

for count, freq in sorted(predecessor_counts.items()):
    print(f'{count}个前置活动: {freq}个活动')

# 统计后继关系
successor_counts = {}
for i in range(n):
    count = len(adj[i])
    if count not in successor_counts:
        successor_counts[count] = 0
    successor_counts[count] += 1

print(f'\n【后继关系统计】')
for count, freq in sorted(successor_counts.items()):
    print(f'{count}个后继活动: {freq}个活动')

print('\n' + '='*60)
print('分析完成！')
