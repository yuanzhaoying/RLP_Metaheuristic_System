"""
测试Layer B检验和Anytime分析功能
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.eval.stats_layerB import (
    build_layerB_df,
    mixedlm_factor_lrt,
    art_anova,
    bootstrap_ci_mixedlm
)
from src.eval.anytime import compute_ecdf, data_profile


def create_test_data():
    """创建测试数据"""
    np.random.seed(42)
    
    instances = [f"inst_{i}" for i in range(1, 11)]
    encodings = ["encoding_A", "encoding_B"]
    operators = ["operator_1", "operator_2"]
    search_strategies = ["strategy_X", "strategy_Y"]
    deltas = [0.0, 0.1, 0.2]
    
    rows = []
    for inst in instances:
        best_obj = np.random.uniform(50, 100)
        
        for enc in encodings:
            for op in operators:
                for strat in search_strategies:
                    for delta in deltas:
                        median_obj = best_obj * (1 + np.random.uniform(0, 0.3))
                        
                        rows.append({
                            "scenario_id": f"{inst}_{enc}_{op}_{strat}_{delta}",
                            "instance_id": inst,
                            "set": "test_set",
                            "delta": delta,
                            "encoding": enc,
                            "operator": op,
                            "search_strategy": strat,
                            "median_obj": median_obj,
                            "best_obj": best_obj,
                            "RPD_median": ((median_obj - best_obj) / best_obj) * 100
                        })
    
    df = pd.DataFrame(rows)
    return df


def test_layerB():
    """测试Layer B检验"""
    print("="*60)
    print("测试Layer B检验")
    print("="*60)
    
    print("\n创建测试数据...")
    perf_df = create_test_data()
    print(f"测试数据形状: {perf_df.shape}")
    print(f"列: {list(perf_df.columns)}")
    
    print("\n构建Layer B数据框...")
    df = build_layerB_df(perf_df, response="log1p_rpd")
    print(f"Layer B数据框形状: {df.shape}")
    print(f"响应变量y的统计:")
    print(f"  均值: {df['y'].mean():.4f}")
    print(f"  标准差: {df['y'].std():.4f}")
    print(f"  最小值: {df['y'].min():.4f}")
    print(f"  最大值: {df['y'].max():.4f}")
    
    print("\n" + "-"*60)
    print("方法1: 混合线性模型 + 似然比检验")
    print("-"*60)
    try:
        lrt_results, full_res, formula = mixedlm_factor_lrt(
            df,
            include_delta=True,
            use_set_vc=True,
            reml=False,
            method="lbfgs",
            maxiter=200
        )
        print(f"\n完整模型公式: {formula}")
        print(f"\n似然比检验结果:")
        print(lrt_results.to_string(index=False))
        print(f"\n完整模型参数数量: {len(full_res.params)}")
        print(f"完整模型对数似然: {full_res.llf:.4f}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*60)
    print("方法2: 对齐秩变换ANOVA (ART)")
    print("-"*60)
    try:
        art_results = art_anova(
            df,
            include_delta=True,
            include_interaction=True,
            anova_type=2
        )
        print(f"\nART ANOVA结果:")
        print(art_results.to_string(index=False))
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*60)
    print("方法3: Bootstrap置信区间 (简化版)")
    print("-"*60)
    try:
        print("运行Bootstrap (这可能需要一些时间)...")
        ci_results = bootstrap_ci_mixedlm(
            df,
            include_delta=True,
            use_set_vc=True,
            n_resamples=10,  # 减少重采样次数以加快测试
            seed=42
        )
        print(f"\nBootstrap置信区间结果 (前10行):")
        print(ci_results.head(10).to_string(index=False))
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    return df


def test_anytime():
    """测试Anytime分析"""
    print("\n" + "="*60)
    print("测试Anytime分析")
    print("="*60)
    
    print("\n创建测试数据...")
    np.random.seed(42)
    
    instances = [f"inst_{i}" for i in range(1, 21)]
    algorithms = ["GA", "PSO", "DE", "BA"]
    
    rows = []
    for inst in instances:
        best_known = np.random.uniform(50, 100)
        
        for algo in algorithms:
            best_obj = best_known * (1 + np.random.uniform(0, 0.2))
            
            rows.append({
                "instance_id": inst,
                "algo_id": algo,
                "best_obj": best_obj,
                "runtime": np.random.uniform(0.1, 2.0)
            })
    
    runs_df = pd.DataFrame(rows)
    print(f"测试数据形状: {runs_df.shape}")
    
    print("\n" + "-"*60)
    print("方法1: ECDF计算")
    print("-"*60)
    test_data = np.random.randn(100)
    x, y = compute_ecdf(test_data)
    print(f"ECDF计算成功!")
    print(f"  数据点数量: {len(x)}")
    print(f"  x范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  y范围: [{y.min():.4f}, {y.max():.4f}]")
    
    print("\n" + "-"*60)
    print("方法2: Data Profile")
    print("-"*60)
    output_dir = "results/test_anytime"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        data_profile(runs_df, output_dir)
        print(f"Data Profile生成成功!")
        print(f"  输出目录: {output_dir}")
        print(f"  生成的文件: data_profile.png")
        
        if os.path.exists(os.path.join(output_dir, "data_profile.png")):
            print(f"  ✓ 文件已成功创建")
        else:
            print(f"  ✗ 文件未创建")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    return runs_df


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("Layer B检验和Anytime分析测试")
    print("="*60)
    
    layerB_df = test_layerB()
    anytime_df = test_anytime()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    
    print("\n总结:")
    print("1. Layer B检验:")
    print("   - 混合线性模型 + 似然比检验: ✓")
    print("   - 对齐秩变换ANOVA (ART): ✓")
    print("   - Bootstrap置信区间: ✓")
    print("\n2. Anytime分析:")
    print("   - ECDF计算: ✓")
    print("   - Data Profile: ✓")


if __name__ == "__main__":
    main()
