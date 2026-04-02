"""
测试所有ML分析流程
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("测试ML分析流程")
print("=" * 60)

# 1. 加载实验结果
print("\n1. 加载实验结果...")
results_dir = project_root / "results" / "raw"
csv_files = list(results_dir.glob("*.csv"))

if not csv_files:
    print("错误: 没有找到实验结果文件！")
    sys.exit(1)

latest_file = max(csv_files, key=os.path.getctime)
print(f"加载文件: {latest_file.name}")

results_df = pd.read_csv(latest_file)
print(f"数据行数: {len(results_df)}")
print(f"数据列: {list(results_df.columns)[:10]}...")

# 2. Layer A检验
print("\n2. Layer A检验 (Friedman)...")
from scipy.stats import friedmanchisquare, wilcoxon

valid_df = results_df[results_df['best_objective'] < 1e9].copy()
print(f"有效数据行数: {len(valid_df)}")

pivot = valid_df.pivot_table(
    index='instance_id',
    columns='algorithm_name',
    values='best_objective',
    aggfunc='median'
)
pivot = pivot.dropna()

if len(pivot.columns) >= 2:
    algos = list(pivot.columns)
    performance_data = [pivot[a].values for a in algos]
    
    stat, p_value = friedmanchisquare(*performance_data)
    ranks = pivot.rank(axis=1, ascending=True)
    mean_ranks = ranks.mean()
    
    print(f"Friedman统计量: {stat:.4f}")
    print(f"p值: {p_value:.6f}")
    print(f"是否显著: {'是' if p_value < 0.05 else '否'}")
    print(f"算法平均排名:")
    for algo, rank in sorted(mean_ranks.items(), key=lambda x: x[1]):
        print(f"  {algo}: {rank:.2f}")
else:
    print("警告: 算法数量不足，跳过Friedman检验")

# 3. Layer B检验
print("\n3. Layer B检验 (ART ANOVA)...")
try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from scipy.stats import rankdata
    
    layerb_df = valid_df.copy()
    layerb_df['y'] = np.log1p(layerb_df['best_objective'])
    layerb_df['algorithm_name'] = layerb_df['algorithm_name'].astype('category')
    layerb_df['instance_id'] = layerb_df['instance_id'].astype('category')
    
    terms = ["C(algorithm_name)"]
    rows = []
    
    for t in terms:
        red_formula = "y ~ C(instance_id)"
        red = smf.ols(red_formula, layerb_df).fit()
        
        aligned = layerb_df["y"].values - red.fittedvalues.values
        y_rank = rankdata(aligned)
        
        d2 = layerb_df.copy()
        d2["y_rank"] = y_rank
        
        full_formula = "y_rank ~ C(instance_id) + C(algorithm_name)"
        full = smf.ols(full_formula, d2).fit()
        aov = anova_lm(full, typ=2)
        
        key = "C(algorithm_name)"
        if key in aov.index:
            ss_term = float(aov.loc[key, "sum_sq"])
            ss_res = float(aov.loc["Residual", "sum_sq"])
            eta2 = ss_term / (ss_term + ss_res + 1e-12)
            
            rows.append({
                "term": "algorithm_name",
                "F": float(aov.loc[key, "F"]),
                "p_value": float(aov.loc[key, "PR(>F)"]),
                "partial_eta2": float(eta2),
            })
    
    print(f"ART ANOVA结果:")
    for row in rows:
        print(f"  因子: {row['term']}")
        print(f"  F值: {row['F']:.4f}")
        print(f"  p值: {row['p_value']:.4f}")
        print(f"  偏eta²: {row['partial_eta2']:.4f}")
        
except Exception as e:
    print(f"Layer B检验错误: {e}")

# 4. Anytime分析
print("\n4. Anytime分析...")
anytime_df = valid_df.copy()
anytime_df['best_obj'] = anytime_df['best_objective']
anytime_df['algo_id'] = anytime_df['algorithm_name']

best_per_instance = anytime_df.groupby("instance_id")["best_obj"].min()

rows = []
for algo in anytime_df["algo_id"].unique():
    algo_runs = anytime_df[anytime_df["algo_id"] == algo]
    for inst_id, group in algo_runs.groupby("instance_id"):
        best_algo = group["best_obj"].min()
        best_known = best_per_instance.loc[inst_id]
        if best_known > 0:
            ratio = best_algo / best_known
        else:
            ratio = 1.0
        rows.append({"algo_id": algo, "instance_id": inst_id, "ratio": ratio})

df = pd.DataFrame(rows)

algo_stats = []
for algo in df["algo_id"].unique():
    algo_data = df[df["algo_id"] == algo]
    
    mean_ratio = algo_data["ratio"].mean()
    perfect_count = int((algo_data["ratio"] == 1.0).sum())
    total_count = len(algo_data)
    perfect_rate = perfect_count / total_count if total_count > 0 else 0
    
    algo_stats.append({
        "algorithm": algo,
        "mean_ratio": mean_ratio,
        "perfect_rate": perfect_rate,
        "total_count": total_count
    })

algo_stats.sort(key=lambda x: x["mean_ratio"])

print("算法性能统计:")
for stats in algo_stats:
    print(f"  {stats['algorithm']}: 平均比率={stats['mean_ratio']:.4f}, 最优解率={stats['perfect_rate']*100:.1f}%")

# 5. 机器学习算法选择
print("\n5. 机器学习算法选择...")
try:
    from src.analysis import AlgorithmSelector
    from src.psp.features import FeatureExtractor
    from src.psp.psplib_io import load_psplib_sm
    
    n_instances = valid_df['instance_id'].nunique()
    n_algorithms = valid_df['algorithm_name'].nunique()
    print(f"实例数量: {n_instances}, 算法数量: {n_algorithms}")
    
    if n_instances >= 5:
        print("正在提取实例特征...")
        
        instances = valid_df['instance_id'].unique()
        feature_list = []
        
        for instance_id in instances:
            instance_id_clean = instance_id.replace('.RCP', '').replace('.rcp', '').replace('.sm', '')
            
            instance_file = None
            for subset in ['j30', 'j60', 'j90', 'j120']:
                test_path = os.path.join(project_root, "data", "psplib_raw", subset, instance_id_clean + ".RCP")
                if os.path.exists(test_path):
                    instance_file = test_path
                    break
                test_path = os.path.join(project_root, "data", "psplib_raw", subset, instance_id_clean + ".rcp")
                if os.path.exists(test_path):
                    instance_file = test_path
                    break
            
            if instance_file and os.path.exists(instance_file):
                try:
                    inst = load_psplib_sm(instance_file)
                    
                    n = inst.n_activities
                    es = [0] * n
                    for j in range(n):
                        for pred in inst.predecessors[j]:
                            es[j] = max(es[j], es[pred] + inst.durations[pred])
                    critical_path_length = max([es[i] + inst.durations[i] for i in range(n)])
                    horizon = int(critical_path_length * 1.5)
                    
                    extractor = FeatureExtractor(inst, horizon)
                    features = extractor.extract_all()
                    features['instance_id'] = instance_id
                    feature_list.append(features)
                except Exception as e:
                    pass
        
        if len(feature_list) > 0:
            feature_df = pd.DataFrame(feature_list)
            print(f"成功提取 {len(feature_list)} 个实例的特征")
            
            # 使用analyze_selector函数进行完整分析
            from src.analysis.algorithm_selector import analyze_selector
            
            ml_results = analyze_selector(
                perf_df=valid_df,
                feature_df=feature_df,
                model_type='random_forest',
                instance_col='instance_id',
                algo_col='algorithm_name',
                perf_col='best_objective',
                test_size=0.3,
                random_state=42
            )
            
            print(f"\n性能基准对比:")
            print(f"  SBS: {ml_results['SBS']['algorithm']} = {ml_results['SBS']['score']:.2f}")
            print(f"  Selector: {ml_results['Selector']['score']:.2f}")
            print(f"  VBS: {ml_results['VBS']['score']:.2f}")
            print(f"\n选择器性能:")
            print(f"  命中率: {ml_results['Selector']['hit_rate']*100:.1f}%")
            print(f"  相比SBS改进: {ml_results['Selector']['improvement_over_sbs']:.2f}%")
            print(f"  距离VBS差距: {ml_results['Selector']['gap_to_vbs']:.2f}%")
            
            # 特征重要性
            if ml_results['feature_importance'] is not None and len(ml_results['feature_importance']) > 0:
                print(f"\n特征重要性 (Top 5):")
                for i, row in ml_results['feature_importance'].head(5).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        else:
            print("警告: 无法提取实例特征")
    else:
        print("警告: 实例数量太少，跳过机器学习分析")
        
except Exception as e:
    import traceback
    print(f"机器学习分析错误: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
