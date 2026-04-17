"""验证BA、PSO、HS算法配置"""
from src.eval.runner import generate_all_algorithm_configs

configs = generate_all_algorithm_configs()

# 过滤出BA、PSO、HS算法的配置
ba_configs = [c for c in configs if c[1] == 'BA']
pso_configs = [c for c in configs if c[1] == 'PSO']
hs_configs = [c for c in configs if c[1] == 'HS']

print("=" * 80)
print("BA、PSO、HS算法配置统计")
print("=" * 80)

print('\nBA算法配置 ({} 种):'.format(len(ba_configs)))
for name, _, _ in ba_configs:
    print(f'  - {name}')

print('\nPSO算法配置 ({} 种):'.format(len(pso_configs)))
for name, _, _ in pso_configs:
    print(f'  - {name}')

print('\nHS算法配置 ({} 种):'.format(len(hs_configs)))
for name, _, _ in hs_configs:
    print(f'  - {name}')

total = len(ba_configs) + len(pso_configs) + len(hs_configs)
print(f'\n总计: {total} 种配置')
print("=" * 80)
