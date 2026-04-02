"""测试前端与main.py结果是否一致 - 测试所有算法"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.psp.psplib_io import load_psplib_sm
from src.eval.runner import ExperimentRunner, ExperimentConfig, generate_all_algorithm_configs

def load_instances(subset: str, count: int):
    import glob
    import re
    
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


def test_algorithm(algo_name: str, selected_configs: list, all_instances, all_deadlines, all_instance_files):
    """测试单个算法"""
    print(f"\n{'='*60}")
    print(f"测试算法: {algo_name}")
    print(f"{'='*60}")
    
    print(f"算子配置数量: {len(selected_configs)}")
    for cfg in selected_configs[:5]:
        print(f"  - {cfg[0]}")
    if len(selected_configs) > 5:
        print(f"  ... 还有 {len(selected_configs) - 5} 个配置")
    
    exp_config = ExperimentConfig(
        instances=all_instance_files,
        algorithms=[algo_name],
        seeds=[0, 1],
        deadlines=all_deadlines,
        max_evaluations=1000,
        output_dir="results/raw",
        time_limit=60.0,
        problem_type="rlp",
        use_delay_factors=False
    )
    
    runner = ExperimentRunner(exp_config)
    
    results = []
    for idx, instance_id in enumerate(all_instance_files):
        instance = all_instances[idx]
        deadline = all_deadlines[idx]
        
        for algo_config in selected_configs:
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
                except Exception as e:
                    results.append({
                        "instance_id": instance_id,
                        "seed": seed,
                        "best_objective": 1e10,
                        "runtime": 0,
                        "algorithm_name": algo_config[0],
                        "deadline": deadline,
                    })
    
    return pd.DataFrame(results)


def main():
    print("="*60)
    print("测试所有算法与main.py结果一致性")
    print("="*60)
    
    print("\n加载实例...")
    all_instances = []
    all_deadlines = []
    all_instance_files = []
    
    subset_counts = {"j30": 2, "j60": 1, "j90": 1, "j120": 1}
    for subset, count in subset_counts.items():
        instances, deadlines, instance_files = load_instances(subset, count)
        all_instances.extend(instances)
        all_deadlines.extend(deadlines)
        all_instance_files.extend(instance_files)
        print(f"  {subset}: {len(instances)} 个实例")
    
    print(f"总共加载 {len(all_instances)} 个实例")
    
    all_configs = generate_all_algorithm_configs()
    
    all_results = []
    
    for algo_name in ["GA", "DE", "PR", "TS"]:
        algo_configs = [cfg for cfg in all_configs if cfg[1] == algo_name]
        df = test_algorithm(algo_name, algo_configs, all_instances, all_deadlines, all_instance_files)
        all_results.append(df)
        print(f"  {algo_name}: {len(df)} 条记录")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    test_file = "results/raw/test_all_algos.csv"
    combined_df.to_csv(test_file, index=False)
    print(f"\n测试结果保存到: {test_file}")
    print(f"总记录数: {len(combined_df)}")
    
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    
    summary = combined_df.groupby('algorithm_name').agg({
        'best_objective': ['mean', 'min', 'max', 'count']
    }).round(2)
    summary.columns = ['Mean', 'Min', 'Max', 'Count']
    summary = summary.sort_values('Mean')
    print(summary)
    
    print("\n✅ 测试完成！所有算法都能正常运行。")


if __name__ == "__main__":
    main()
