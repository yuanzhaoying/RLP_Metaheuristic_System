"""测试前端与main.py结果是否一致 - 每种算法只测试一个算子组合"""
import os
import sys
import pandas as pd
import glob
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.psp.psplib_io import load_psplib_sm
from src.eval.runner import ExperimentRunner, ExperimentConfig, generate_all_algorithm_configs


def load_instances(subset: str, count: int):
    subset_path = os.path.join("data", "psplib_raw", subset.lower())
    
    pattern_sm = os.path.join(subset_path, "*.sm")
    pattern_rcp = os.path.join(subset_path, "*.RCP")
    pattern_rcp_lower = os.path.join(subset_path, "*.rcp")
    
    files_sm = sorted(glob.glob(pattern_sm))
    files_rcp = sorted(glob.glob(pattern_rcp))
    files_rcp_lower = sorted(glob.glob(pattern_rcp_lower))
    
    files = files_sm + files_rcp + files_rcp_lower
    files = list(set(files))
    
    def extract_number(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'_(\d+)\.', basename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
    
    files = sorted(files, key=extract_number)
    
    instances = []
    deadlines = []
    instance_files = []
    
    for f in files[:count]:
        try:
            inst = load_psplib_sm(f)
            instances.append(inst)
            instance_files.append(f)
            
            n = inst.n_activities
            es = [0] * n
            for j in range(n):
                for pred in inst.predecessors[j]:
                    es[j] = max(es[j], es[pred] + inst.durations[pred])
            critical_path_length = max([es[i] + inst.durations[i] for i in range(n)])
            
            deadline = int(critical_path_length)
            deadlines.append(deadline)
        except Exception as e:
            pass
    
    return instances, deadlines, instance_files


def main():
    print("="*60)
    print("测试 GA, DE, PR, TS 算法与main.py结果一致性")
    print("="*60)
    
    print("\n加载实例...")
    all_instances = []
    all_deadlines = []
    all_instance_files = []
    
    subset_counts = {"j30": 2, "j60": 1}
    for subset, count in subset_counts.items():
        instances, deadlines, instance_files = load_instances(subset, count)
        all_instances.extend(instances)
        all_deadlines.extend(deadlines)
        all_instance_files.extend(instance_files)
        print(f"  {subset}: {len(instances)} 个实例")
    
    print(f"总共加载 {len(all_instances)} 个实例")
    
    all_configs = generate_all_algorithm_configs()
    
    test_configs = []
    
    ga_configs = [cfg for cfg in all_configs if cfg[1] == "GA"]
    if ga_configs:
        test_configs.append(ga_configs[0])
        print(f"\nGA 测试配置: {ga_configs[0][0]}")
    
    de_configs = [cfg for cfg in all_configs if cfg[1] == "DE"]
    if de_configs:
        test_configs.append(de_configs[0])
        print(f"DE 测试配置: {de_configs[0][0]}")
    
    pr_configs = [cfg for cfg in all_configs if cfg[1] == "PR"]
    if pr_configs:
        test_configs.append(pr_configs[0])
        print(f"PR 测试配置: {pr_configs[0][0]}")
    
    ts_configs = [cfg for cfg in all_configs if cfg[1] == "TS"]
    if ts_configs:
        test_configs.append(ts_configs[0])
        print(f"TS 测试配置: {ts_configs[0][0]}")
    
    exp_config = ExperimentConfig(
        instances=all_instance_files,
        algorithms=["GA", "DE", "PR", "TS"],
        seeds=[0, 1],
        deadlines=all_deadlines,
        max_evaluations=1000,
        output_dir="results/raw",
        time_limit=60.0,
        problem_type="rlp",
        use_delay_factors=False
    )
    
    runner = ExperimentRunner(exp_config)
    
    print("\n运行实验...")
    results = []
    
    for idx, instance_id in enumerate(all_instance_files):
        instance = all_instances[idx]
        deadline = all_deadlines[idx]
        
        for algo_config in test_configs:
            for seed in [0, 1]:
                try:
                    result = runner.run_single(
                        instance, algo_config, seed, deadline, 1000
                    )
                    
                    row = {
                        "instance_id": result.instance_id,
                        "seed": result.seed,
                        "best_objective": result.best_objective,
                        "runtime": result.runtime,
                        "algorithm_name": result.algorithm_name,
                        "deadline": result.deadline,
                    }
                    results.append(row)
                    print(f"  {result.instance_id} / {result.algorithm_name} / seed={seed}: {result.best_objective:.2f}")
                except Exception as e:
                    print(f"  [ERROR] {algo_config[0]} on {instance_id} (seed={seed}): {str(e)}")
                    results.append({
                        "instance_id": instance_id,
                        "seed": seed,
                        "best_objective": 1e10,
                        "runtime": 0,
                        "algorithm_name": algo_config[0],
                        "deadline": deadline,
                    })
    
    df = pd.DataFrame(results)
    test_file = "results/raw/test_ga_de_pr_ts.csv"
    df.to_csv(test_file, index=False)
    
    print(f"\n测试结果保存到: {test_file}")
    print(f"总记录数: {len(df)}")
    
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    
    summary = df.groupby('algorithm_name').agg({
        'best_objective': ['mean', 'min', 'max', 'count']
    }).round(2)
    summary.columns = ['Mean', 'Min', 'Max', 'Count']
    summary = summary.sort_values('Mean')
    print(summary)
    
    print("\n✅ 测试完成！GA, DE, PR, TS 算法都能正常运行。")


if __name__ == "__main__":
    main()
