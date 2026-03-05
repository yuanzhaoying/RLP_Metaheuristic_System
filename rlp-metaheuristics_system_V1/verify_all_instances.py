#!/usr/bin/env python3
import os
import glob
import re

print('验证运行所有2040个实例的配置:')
print('='*60)

def extract_number(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r'_(\d+)\.', basename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

total_instances = 0
all_files = []

for subset in ['j30', 'j60', 'j90', 'j120']:
    subset_path = f'data/psplib_raw/{subset}'
    if os.path.exists(subset_path):
        files = []
        for ext in ['*.sm', '*.RCP', '*.rcp']:
            files.extend(glob.glob(os.path.join(subset_path, ext)))
        files = list(set(files))
        files = sorted(files, key=extract_number)
        all_files.extend(files)
        print(f'{subset}: {len(files)} 个实例')
        total_instances += len(files)

print('='*60)
print(f'总计: {total_instances} 个实例')
print()

# 计算总运行次数
algorithms = ['ga', 'sa', 'ils']
seeds = [0, 1]
total_runs = total_instances * len(algorithms) * len(seeds)
time_per_run = 60  # 秒
total_time = total_runs * time_per_run / 3600  # 小时

print('实验配置:')
print('-'*60)
print(f'实例数: {total_instances}')
print(f'算法数: {len(algorithms)}')
print(f'种子数: {len(seeds)}')
print(f'总运行次数: {total_runs}')
print(f'每次运行时间限制: {time_per_run}秒')
print(f'预计总时间: {total_time:.1f}小时')
print()

print('运行命令:')
print('-'*60)
print('python main.py --subset all')
