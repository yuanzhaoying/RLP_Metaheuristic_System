"""检查过滤逻辑"""
from src.eval.runner import generate_all_algorithm_configs

configs = generate_all_algorithm_configs()

# 模拟runner.py中的过滤逻辑
algorithms = ["BA", "PSO", "HS"]

selected_configs = [
    cfg for cfg in configs
    if any(algo.lower() in cfg[0].lower() for algo in algorithms)
]

print("=" * 80)
print("过滤逻辑测试")
print("=" * 80)

print(f"\n用户指定的算法: {algorithms}")
print(f"\n过滤后的配置数: {len(selected_configs)}")

print("\n过滤后的配置:")
for name, algo_type, params in selected_configs:
    print(f"  - {name} ({algo_type})")

# 按算法类型统计
algo_counts = {}
for name, algo_type, params in selected_configs:
    if algo_type not in algo_counts:
        algo_counts[algo_type] = 0
    algo_counts[algo_type] += 1

print("\n按算法类型统计:")
for algo_type, count in sorted(algo_counts.items()):
    print(f"  {algo_type}: {count} 种")

print("\n" + "=" * 80)
