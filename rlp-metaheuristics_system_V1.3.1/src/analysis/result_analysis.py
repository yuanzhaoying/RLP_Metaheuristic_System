"""
结果分析模块
包含所有分析函数和可视化功能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def compute_statistics(perf_df, instance_col="instance_id", algo_col="algorithm_name", perf_col="best_objective"):
    """
    计算统计汇总表格
    
    参数:
        perf_df: 性能数据DataFrame
        instance_col: 实例列名
        algo_col: 算法列名
        perf_col: 性能列名
    
    返回:
        summary: 统计汇总表格
        perf_matrix: 性能矩阵
    """
    # 如果有重复的(instance, algorithm)组合，取最优值
    if perf_df.duplicated(subset=[instance_col, algo_col]).any():
        perf_df_agg = perf_df.groupby([instance_col, algo_col])[perf_col].min().reset_index()
    else:
        perf_df_agg = perf_df
    
    perf_matrix = perf_df_agg.pivot(index=instance_col, columns=algo_col, values=perf_col)
    
    mean_perf = perf_matrix.mean()
    std_perf = perf_matrix.std()
    median_perf = perf_matrix.median()
    min_perf = perf_matrix.min()
    max_perf = perf_matrix.max()
    
    vbs = perf_matrix.min(axis=1)
    
    rpd = ((mean_perf - vbs.mean()) / vbs.mean()) * 100
    
    ranks = perf_matrix.rank(axis=1, method="average")
    avg_rank = ranks.mean()
    
    summary = pd.DataFrame({
        "Mean": mean_perf,
        "Std": std_perf,
        "Median": median_perf,
        "Min": min_perf,
        "Max": max_perf,
        "RPD(%)": rpd,
        "Rank": avg_rank
    }).sort_values("Rank")
    
    return summary, perf_matrix


def plot_performance_profile(perf_matrix, figsize=(10, 6)):
    """
    绘制性能剖面图（Dolan-Moré Performance Profile）
    
    参数:
        perf_matrix: 性能矩阵（instance × algorithm）
        figsize: 图形大小
    
    返回:
        fig: matplotlib图形对象
    """
    ratios = perf_matrix.div(perf_matrix.min(axis=1), axis=0)
    
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    
    taus = np.linspace(1, max(ratios.max().max(), 3), 100)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for algo in ratios.columns:
        probs = [(ratios[algo] <= tau).mean() for tau in taus]
        ax.plot(taus, probs, label=algo, linewidth=2)
    
    ax.set_xlabel("τ (Performance Ratio)", fontsize=12)
    ax.set_ylabel("P(ratio ≤ τ)", fontsize=12)
    ax.set_title("Performance Profile", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(taus))
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


def plot_anytime_curve(time_df, time_col="runtime", perf_col="best_objective", algo_col="algorithm_name", figsize=(10, 6)):
    """
    绘制Anytime收敛曲线
    
    参数:
        time_df: 包含时间和性能数据的DataFrame
        time_col: 时间列名
        perf_col: 性能列名
        algo_col: 算法列名
        figsize: 图形大小
    
    返回:
        fig: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for algo in time_df[algo_col].unique():
        subset = time_df[time_df[algo_col] == algo]
        
        sorted_data = subset.sort_values(time_col)
        
        ax.plot(sorted_data[time_col].values, 
                sorted_data[perf_col].values, 
                label=algo, 
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.7)
    
    ax.set_xlabel("Runtime (s)", fontsize=12)
    ax.set_ylabel("Best Objective", fontsize=12)
    ax.set_title("Anytime Convergence Curve", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def perform_statistical_tests(perf_matrix, alpha=0.05):
    """
    执行统计检验（Friedman检验和Wilcoxon检验）
    
    参数:
        perf_matrix: 性能矩阵（instance × algorithm）
        alpha: 显著性水平
    
    返回:
        results: 包含检验结果的字典
    """
    results = {}
    
    algorithms = perf_matrix.columns.tolist()
    n_instances = len(perf_matrix)
    n_algorithms = len(algorithms)
    
    try:
        friedman_stat, friedman_p = friedmanchisquare(*[perf_matrix[algo].values for algo in algorithms])
        results['friedman'] = {
            'statistic': friedman_stat,
            'p_value': friedman_p,
            'significant': friedman_p < alpha
        }
    except Exception as e:
        results['friedman'] = {
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e)
        }
    
    wilcoxon_results = []
    for i in range(n_algorithms):
        for j in range(i+1, n_algorithms):
            algo1 = algorithms[i]
            algo2 = algorithms[j]
            
            try:
                stat, p_value = wilcoxon(perf_matrix[algo1].values, perf_matrix[algo2].values)
                wilcoxon_results.append({
                    'Algorithm 1': algo1,
                    'Algorithm 2': algo2,
                    'Statistic': stat,
                    'p-value': p_value,
                    'Significant': p_value < alpha
                })
            except Exception as e:
                wilcoxon_results.append({
                    'Algorithm 1': algo1,
                    'Algorithm 2': algo2,
                    'Statistic': None,
                    'p-value': None,
                    'Significant': False,
                    'Error': str(e)
                })
    
    results['wilcoxon'] = pd.DataFrame(wilcoxon_results)
    
    ranks = perf_matrix.rank(axis=1, method="average")
    avg_ranks = ranks.mean()
    results['average_ranks'] = avg_ranks.sort_values()
    
    return results


def plot_rank_comparison(avg_ranks, figsize=(10, 6)):
    """
    绘制算法排名比较图
    
    参数:
        avg_ranks: 平均排名Series
        figsize: 图形大小
    
    返回:
        fig: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    algorithms = avg_ranks.index.tolist()
    ranks = avg_ranks.values
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
    
    bars = ax.barh(range(len(algorithms)), ranks, color=colors)
    
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms, fontsize=10)
    ax.set_xlabel('Average Rank', fontsize=12)
    ax.set_title('Algorithm Ranking Comparison', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    for i, (bar, rank) in enumerate(zip(bars, ranks)):
        ax.text(rank + 0.1, i, f'{rank:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def compute_feasibility_analysis(df, objective_col="best_objective", threshold=1e9):
    """
    计算可行性分析
    
    参数:
        df: 结果DataFrame
        objective_col: 目标值列名
        threshold: 不可行解阈值
    
    返回:
        feasibility_df: 可行性分析结果DataFrame
    """
    results = []
    
    for algo in df['algorithm_name'].unique():
        subset = df[df['algorithm_name'] == algo]
        
        total = len(subset)
        feasible = len(subset[subset[objective_col] < threshold])
        infeasible = total - feasible
        
        feasible_rate = feasible / total if total > 0 else 0
        infeasible_rate = infeasible / total if total > 0 else 0
        
        feasible_subset = subset[subset[objective_col] < threshold]
        avg_runtime = feasible_subset['runtime'].mean() if len(feasible_subset) > 0 else 0
        
        results.append({
            'Algorithm': algo,
            'Total Runs': total,
            'Feasible': feasible,
            'Infeasible': infeasible,
            'Feasible Rate (%)': feasible_rate * 100,
            'Infeasible Rate (%)': infeasible_rate * 100,
            'Avg Runtime (s)': avg_runtime
        })
    
    return pd.DataFrame(results)


def plot_feasibility_analysis(feasibility_df, figsize=(12, 5)):
    """
    绘制可行性分析图
    
    参数:
        feasibility_df: 可行性分析结果DataFrame
        figsize: 图形大小
    
    返回:
        fig: matplotlib图形对象
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    algorithms = feasibility_df['Algorithm'].tolist()
    feasible_rates = feasibility_df['Feasible Rate (%)'].tolist()
    
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(algorithms)))
    bars = ax1.bar(range(len(algorithms)), feasible_rates, color=colors)
    
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Feasible Rate (%)', fontsize=11)
    ax1.set_title('Feasibility Rate by Algorithm', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, feasible_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    avg_runtimes = feasibility_df['Avg Runtime (s)'].tolist()
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(algorithms)))
    bars = ax2.bar(range(len(algorithms)), avg_runtimes, color=colors)
    
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Average Runtime (s)', fontsize=11)
    ax2.set_title('Average Runtime by Algorithm', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, runtime in zip(bars, avg_runtimes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{runtime:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def generate_summary_report(df, perf_matrix, stats_results, feasibility_df):
    """
    生成汇总报告
    
    参数:
        df: 原始数据DataFrame
        perf_matrix: 性能矩阵
        stats_results: 统计检验结果
        feasibility_df: 可行性分析结果
    
    返回:
        report: 汇总报告字典
    """
    report = {
        'total_instances': len(df['instance_id'].unique()),
        'total_algorithms': len(df['algorithm_name'].unique()),
        'total_runs': len(df),
        'best_algorithm': None,
        'friedman_test': None,
        'feasibility_summary': None
    }
    
    if 'average_ranks' in stats_results:
        best_algo = stats_results['average_ranks'].index[0]
        report['best_algorithm'] = {
            'name': best_algo,
            'avg_rank': stats_results['average_ranks'].values[0]
        }
    
    if 'friedman' in stats_results:
        report['friedman_test'] = stats_results['friedman']
    
    if feasibility_df is not None and len(feasibility_df) > 0:
        report['feasibility_summary'] = {
            'avg_feasible_rate': feasibility_df['Feasible Rate (%)'].mean(),
            'best_feasible_rate': feasibility_df['Feasible Rate (%)'].max(),
            'avg_runtime': feasibility_df['Avg Runtime (s)'].mean()
        }
    
    return report
