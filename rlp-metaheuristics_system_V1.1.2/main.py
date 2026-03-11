# 添加真实数据
# https://www.projectmanagement.ugent.be/research/data

from __future__ import annotations
import os
import sys
import argparse
import time
import datetime
from datetime import datetime as dt_now

import yaml
import glob
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.psp.psplib_io import load_psplib_sm, load_psplib_directory
from src.psp.features import extract_features_batch
from src.eval.runner import ExperimentRunner, ExperimentConfig
from src.eval.statistics import friedman_test, pairwise_wilcoxon, statistical_summary
from src.ml.selector import AlgorithmSelector, nested_cv_evaluation, prepare_ml_data


# ============================================================
# 📊 实验配置区域 - 在这里直接修改参数
# ============================================================

# 实例数量配置 (0 表示运行该子集的所有实例)
J30_COUNT = 4      # j30实例数量 (总共480个)
J60_COUNT = 3      # j60实例数量 (总共480个)
J90_COUNT = 2      # j90实例数量 (总共480个)
J120_COUNT = 1     # j120实例数量 (总共600个)

# 子集选择: "j30", "j60", "j90", "j120", "all", "custom"
# - "j30": 只运行j30的所有实例
# - "j60": 只运行j60的所有实例
# - "j90": 只运行j90的所有实例
# - "j120": 只运行j120的所有实例
# - "all": 运行所有2040个实例
# - "custom": 根据上面的J30_COUNT等参数运行自定义数量
SUBSET = "custom"

# 其他配置
SKIP_EXPERIMENTS = False      # 是否跳过实验运行
SKIP_ML = False               # 是否跳过ML训练
USE_DELAY_FACTORS = False     # 是否使用延迟因子

# ============================================================
# 以下代码无需修改
# ============================================================


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_instances(config: dict, subset: str = "j30") -> tuple:
    subset_path = os.path.join("data", "psplib_raw", subset.lower())

    if not os.path.exists(subset_path):
        print(f"Warning: {subset_path} does not exist")
        print("Please download PSPLIB datasets first")
        return [], [], []

    pattern_sm = os.path.join(subset_path, "*.sm")
    pattern_rcp = os.path.join(subset_path, "*.RCP")
    pattern_rcp_lower = os.path.join(subset_path, "*.rcp")

    files_sm = sorted(glob.glob(pattern_sm))
    files_rcp = sorted(glob.glob(pattern_rcp))
    files_rcp_lower = sorted(glob.glob(pattern_rcp_lower))

    files = files_sm + files_rcp + files_rcp_lower
    files = list(set(files))

    import re
    def extract_number(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'_(\d+)\.', basename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
    
    files = sorted(files, key=extract_number)

    print(f"Found {len(files)} instances in {subset}")

    instances = []
    deadlines = []
    instance_files = []

    for f in tqdm(files, desc="Loading instances"):
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
            print(f"Error loading {f}: {e}")

    return instances, deadlines, instance_files


def run_experiments(config: dict, instances, deadlines, instance_files, use_delay_factors: bool = False):
    exp_config = ExperimentConfig(
        instances=instance_files,
        algorithms=["all"],  # 使用"all"调用所有算法的所有算子组合
        seeds=[0, 1],
        deadlines=deadlines,
        max_evaluations=1000,
        output_dir="results/raw",
        time_limit=60.0,
        problem_type="rlp",
        use_delay_factors=use_delay_factors
    )

    data_dir = config.get('psplib', {}).get('data_dir', 'data/psplib_raw')

    print(f"\nRunning experiments with:")
    print(f"  - Instances: {len(exp_config.instances)}")
    print(f"  - Algorithms: all algorithms with all operator combinations")
    print(f"  - Seeds: {len(exp_config.seeds)}")
    print(f"  - Time limit: {exp_config.time_limit}s")
    print(f"  - Problem type: {exp_config.problem_type}")
    print(f"  - Use delay factors: {exp_config.use_delay_factors}")

    runner = ExperimentRunner(exp_config)
    
    # 显示算子组合统计
    print(f"\n  Algorithm configurations:")
    print(f"    - DE: 80 combinations (10 mutation strategies × 4 crossover strategies × 2 local search)")
    print(f"    - GA: 96 combinations (2 selection × 4 crossover × 2 mutation × 2 initialization × 2 local search × 2 neighborhood)")
    print(f"    - SA: 1 combination")
    print(f"    - ILS: 5 combinations (different perturbation strengths)")
    print(f"    - TS: 2 combinations (static/dynamic tabu strategy)")
    print(f"    - PR: 8 combinations (2 path strategies × 2 selection strategies × 2 local search)")
    print(f"    - GSA: 1 combination")
    print(f"    - Total: 193 algorithm configurations")
    
    df = runner.run_batch(data_dir, verbose=True)

    print(f"\nExperiment results:")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")

    filename = dt_now.now().strftime('%m%d_%H%M') + ".csv"
    output_dir = "results/raw"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)

    print(f"\nResults saved to {output_dir}/{filename}")
    return df


def run_statistics(results_df: pd.DataFrame):
    print("\n" + "="*50)
    print("Statistical Analysis")
    print("="*50)

    print(f"\nData shape: {results_df.shape}")
    print(f"Columns: {list(results_df.columns)}")

    if results_df.empty or 'best_objective' not in results_df.columns:
        print("Warning: No valid results found for statistical analysis!")
        return None, None, None

    friedman_result = friedman_test(results_df)
    if "error" in friedman_result:
        print(f"\nFriedman Test Error: {friedman_result['error']}")
    else:
        print(f"\nFriedman Test:")
        print(f"  Statistic: {friedman_result.get('statistic', 'N/A'):.4f}")
        print(f"  P-value: {friedman_result.get('p_value', 'N/A'):.4f}")
        print(f"  Significant: {friedman_result.get('significant', False)}")

    summary = statistical_summary(results_df)
    print(f"\nAlgorithm Summary:")
    print(summary)

    pairwise = pairwise_wilcoxon(results_df)
    if len(pairwise) > 0:
        print(f"\nPairwise Comparisons (Wilcoxon):")
        print(pairwise.head(10))

    return friedman_result, summary, pairwise


def train_selector(config: dict, results_df: pd.DataFrame, instances, deadlines):
    print("\n" + "="*50)
    print("Training Algorithm Selector")
    print("="*50)

    features_df = extract_features_batch(instances, deadlines)

    X, y = prepare_ml_data(results_df, features_df)

    if X.empty or y.empty:
        print("Failed to prepare ML data. Skipping selector training.")
        return None, None

    if len(X) < 10:
        print("Not enough data for ML training")
        return None, None

    print(f"Training data: {len(X)} instances, {len(X.columns)} features")

    nested_results = nested_cv_evaluation(
        X, y,
        model_type="random_forest",
        outer_folds=5,
        n_estimators=500,
        max_depth=10
    )

    print(f"\nNested CV Results:")
    print(f"  Selector mean performance: {nested_results['selector_mean']:.4f}")
    print(f"  SBS mean performance: {nested_results['sbs_mean']:.4f}")
    print(f"  VBS mean performance: {nested_results['vbs_mean']:.4f}")
    print(f"  Selection Accuracy: {nested_results['selection_accuracy_mean']:.2%} (±{nested_results['selection_accuracy_std']:.2%})")
    print(f"  Improvement over SBS: {nested_results['selector_improvement_over_sbs']:.2%}")
    print(f"  Gap to VBS: {nested_results['gap_to_vbs']:.2%}")

    selector = AlgorithmSelector(
        model_type="random_forest",
        n_estimators=500,
        max_depth=10
    )
    selector.fit(X, y)

    importance = selector.get_feature_importance()
    print(f"\nTop 10 Features:")
    print(importance.head(10))

    os.makedirs("results/ml", exist_ok=True)
    selector.save("results/ml/selector.pkl")
    importance.to_csv("results/ml/feature_importance.csv", index=False)

    return selector, nested_results


def main():
    parser = argparse.ArgumentParser(description="PRLP Metaheuristics Research Framework")
    parser.add_argument("--config", default="config/experiment.yaml", help="Config file path")
    parser.add_argument("--subset", default=SUBSET, choices=["j30", "j60", "j90", "j120", "all", "custom"], help="PSPLIB subset")
    parser.add_argument("--j30-count", type=int, default=J30_COUNT, help="Number of j30 instances to run (0 means all)")
    parser.add_argument("--j60-count", type=int, default=J60_COUNT, help="Number of j60 instances to run (0 means all)")
    parser.add_argument("--j90-count", type=int, default=J90_COUNT, help="Number of j90 instances to run (0 means all)")
    parser.add_argument("--j120-count", type=int, default=J120_COUNT, help="Number of j120 instances to run (0 means all)")
    parser.add_argument("--skip-experiments", action="store_true", default=SKIP_EXPERIMENTS, help="Skip running experiments")
    parser.add_argument("--skip-ml", action="store_true", default=SKIP_ML, help="Skip ML training")
    parser.add_argument("--use-delay-factors", action="store_true", default=USE_DELAY_FACTORS, help="Use delay factors in RLP solving")

    args = parser.parse_args()

    print("="*60)
    print("PRLP Metaheuristics Research Framework")
    print("="*60)
    
    print("\n实验配置:")
    print(f"  - 子集: {args.subset}")
    if args.subset == "custom":
        print(f"  - j30实例数: {args.j30_count if args.j30_count > 0 else '全部'}")
        print(f"  - j60实例数: {args.j60_count if args.j60_count > 0 else '全部'}")
        print(f"  - j90实例数: {args.j90_count if args.j90_count > 0 else '全部'}")
        print(f"  - j120实例数: {args.j120_count if args.j120_count > 0 else '全部'}")
    print(f"  - 跳过实验: {args.skip_experiments}")
    print(f"  - 跳过ML: {args.skip_ml}")
    print(f"  - 使用延迟因子: {args.use_delay_factors}")
    print()

    config = load_config(args.config)

    if args.subset == "all":
        print(f"\nStep 1: Loading all PSPLIB instances...")
        all_instances = []
        all_deadlines = []
        all_instance_files = []
        
        for subset in ["j30", "j60", "j90", "j120"]:
            print(f"  Loading {subset}...")
            instances, deadlines, instance_files = prepare_instances(config, subset)
            all_instances.extend(instances)
            all_deadlines.extend(deadlines)
            all_instance_files.extend(instance_files)
        
        instances = all_instances
        deadlines = all_deadlines
        instance_files = all_instance_files
    elif args.subset == "custom":
        print(f"\nStep 1: Loading custom PSPLIB instances...")
        all_instances = []
        all_deadlines = []
        all_instance_files = []
        
        subset_counts = {
            "j30": args.j30_count,
            "j60": args.j60_count,
            "j90": args.j90_count,
            "j120": args.j120_count
        }
        
        for subset, count in subset_counts.items():
            if count > 0:
                print(f"  Loading {count} {subset} instances...")
                instances, deadlines, instance_files = prepare_instances(config, subset)
                all_instances.extend(instances[:count])
                all_deadlines.extend(deadlines[:count])
                all_instance_files.extend(instance_files[:count])
        
        instances = all_instances
        deadlines = all_deadlines
        instance_files = all_instance_files
    else:
        print(f"\nStep 1: Loading {args.subset} instances...")
        instances, deadlines, instance_files = prepare_instances(config, args.subset)

    if not instances:
        print("No instances loaded. Please check data directory.")
        return

    print(f"Loaded {len(instances)} instances")

    if not args.skip_experiments:
        print(f"\nStep 2: Running experiments...")
        results_df = run_experiments(config, instances, deadlines, instance_files, args.use_delay_factors)
    else:
        csv_path = "results/raw/runs.csv"
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(args.config).replace('config', ''), "results/raw/runs.csv")

        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
            print(f"Loaded existing results from {csv_path}")
        else:
            print("No results found. Run experiments first.")
            return

    print(f"\nStep 3: Statistical analysis...")
    friedman_result, summary, pairwise = run_statistics(results_df)

    if not args.skip_ml:
        print(f"\nStep 4: Training algorithm selector...")
        selector, nested_results = train_selector(config, results_df, instances, deadlines)

    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)


if __name__ == "__main__":
    main()
