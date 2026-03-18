"""从CSV文件执行Step3和Step4"""
import sys
import os
import pandas as pd
sys.path.insert(0, '/Users/yuanzhaoying/Desktop/rlp-metaheuristics')

# 读取CSV文件
csv_path = '/Users/yuanzhaoying/Desktop/rlp-metaheuristics/results/raw/0311_2216.csv'
results_df = pd.read_csv(csv_path, header=None)

# 添加正确的列名
column_names = [
    'instance_id', 'seed', 'best_objective', 'runtime', 'algorithm_name', 'deadline',
    'param_max_evaluations', 'param_seed', 'param_population_size', 'param_time_limit',
    'param_F', 'param_CR', 'param_use_adaptive_F', 'param_use_adaptive_CR',
    'param_mutation_strategy', 'param_crossover_strategy', 'param_initialization_strategy',
    'param_use_local_search', 'param_local_search_top', 'param_F_min', 'param_F_max',
    'param_CR_min', 'param_CR_max', 'param_K0'
]

# 只设置前25列的名称，其余列忽略
results_df = results_df.iloc[:, :len(column_names)]
results_df.columns = column_names

print("=" * 100)
print("Step 3: Statistical Analysis")
print("=" * 100)

print(f"\nData shape: {results_df.shape}")
print(f"Columns: {list(results_df.columns)}")

# 检查是否有inf值
inf_count = (results_df['best_objective'] == float('inf')).sum() + (results_df['best_objective'] >= 1e10).sum()
print(f"\nInf values in data: {inf_count}")

# 过滤掉inf值
results_df_clean = results_df[results_df['best_objective'] < 1e10].copy()
print(f"Clean data shape: {results_df_clean.shape}")

if results_df_clean.empty:
    print("Warning: No valid results found for statistical analysis!")
else:
    # Step 3: 统计分析
    print("\n" + "=" * 100)
    print("Running Friedman Test")
    print("=" * 100)
    
    try:
        from src.eval.statistics import friedman_test, pairwise_wilcoxon, statistical_summary
        
        friedman_result = friedman_test(results_df_clean)
        if "error" in friedman_result:
            print(f"\nFriedman Test Error: {friedman_result['error']}")
        else:
            print(f"\nFriedman Test:")
            print(f"  Statistic: {friedman_result.get('statistic', 'N/A'):.4f}")
            print(f"  P-value: {friedman_result.get('p_value', 'N/A'):.4f}")
            print(f"  Significant: {friedman_result.get('significant', False)}")
        
        print("\n" + "=" * 100)
        print("Algorithm Summary")
        print("=" * 100)
        
        summary = statistical_summary(results_df_clean)
        print(summary)
        
        print("\n" + "=" * 100)
        print("Pairwise Comparisons (Wilcoxon)")
        print("=" * 100)
        
        pairwise = pairwise_wilcoxon(results_df_clean)
        if len(pairwise) > 0:
            print(pairwise.head(10))
        
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
        import traceback
        traceback.print_exc()

# Step 4: 机器学习训练
print("\n" + "=" * 100)
print("Step 4: Training Algorithm Selector")
print("=" * 100)

try:
    from src.psp.psplib_io import load_psplib_sm
    from src.psp.features import extract_features_batch
    from src.ml.selector import AlgorithmSelector, nested_cv_evaluation, prepare_ml_data
    
    # 加载实例
    print("\nLoading instances...")
    inst = load_psplib_sm('/Users/yuanzhaoying/Desktop/rlp-metaheuristics/data/psplib_raw/j30/J30_1.RCP')
    
    # 计算deadline
    n = inst.n_activities
    es = [0] * n
    for j in range(n):
        for pred in inst.predecessors[j]:
            es[j] = max(es[j], es[pred] + inst.durations[pred])
    critical_path_length = max([es[i] + inst.durations[i] for i in range(n)])
    deadline = int(critical_path_length)
    
    instances = [inst]
    deadlines = [deadline]
    
    # 提取特征
    print("\nExtracting features...")
    features_df = extract_features_batch(instances, deadlines)
    
    # 准备ML数据
    print("\nPreparing ML data...")
    X, y = prepare_ml_data(results_df_clean, features_df)
    
    if X.empty or y.empty:
        print("Failed to prepare ML data. Skipping selector training.")
    elif len(X) < 10:
        print("Not enough data for ML training")
    else:
        print(f"\nTraining data: {len(X)} instances, {len(X.columns)} features")
        
        # 嵌套交叉验证
        print("\nRunning nested cross-validation...")
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
        
        # 训练最终模型
        print("\nTraining final model...")
        selector = AlgorithmSelector(
            model_type="random_forest",
            n_estimators=500,
            max_depth=10
        )
        selector.fit(X, y)
        
        # 特征重要性
        importance = selector.get_feature_importance()
        print(f"\nTop 10 Features:")
        print(importance.head(10))
        
        # 保存模型
        os.makedirs("results/ml", exist_ok=True)
        selector.save("results/ml/selector.pkl")
        importance.to_csv("results/ml/feature_importance.csv", index=False)
        
        print("\nModel saved to results/ml/")
        
except Exception as e:
    print(f"Error in ML training: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 100)
print("Pipeline completed!")
print("=" * 100)
