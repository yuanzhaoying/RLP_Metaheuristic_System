"""RLP元启发式算法研究框架 - Web界面"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os
import glob
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.psp.psplib_io import load_psplib_sm
from src.eval.runner import ExperimentRunner, ExperimentConfig, generate_all_algorithm_configs
from src.analysis import (
    compute_statistics as compute_stats,
    plot_performance_profile as plot_perf_profile,
    plot_anytime_curve as plot_anytime,
    perform_statistical_tests,
    plot_rank_comparison,
    compute_feasibility_analysis as compute_feasibility,
    plot_feasibility_analysis,
    generate_summary_report,
    AlgorithmSelector,
    analyze_selector
)
from src.psp.features import FeatureExtractor, extract_features_batch

st.set_page_config(
    page_title="RLP Metaheuristics Research Framework",
    page_icon="",
    layout="wide"
)

if 'current_results_df' not in st.session_state:
    st.session_state.current_results_df = None
if 'current_results_file' not in st.session_state:
    st.session_state.current_results_file = None
if 'stop_experiment' not in st.session_state:
    st.session_state.stop_experiment = False
if 'is_running' not in st.session_state:
    st.session_state.is_running = False


def load_instances_from_subset(subset: str, count: int):
    subset_path = os.path.join(PROJECT_ROOT, "data", "psplib_raw", subset.lower())
    
    if not os.path.exists(subset_path):
        return [], [], []
    
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


def get_results_dir():
    return os.path.join(PROJECT_ROOT, "results", "raw")


def get_current_results():
    if st.session_state.current_results_df is not None:
        return st.session_state.current_results_file, st.session_state.current_results_df
    
    results_dir = get_results_dir()
    if not os.path.exists(results_dir):
        return None, None
    
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")), reverse=True)
    if not csv_files:
        return None, None
    
    latest_file = csv_files[0]
    try:
        df = pd.read_csv(latest_file)
        return latest_file, df
    except Exception as e:
        return None, None


def calculate_rpd(df):
    if df is None or df.empty:
        return df
    
    valid_df = df[df['best_objective'] < 1e9].copy()
    if valid_df.empty:
        return df
    
    best_per_instance = valid_df.groupby('instance_id')['best_objective'].min()
    
    def get_rpd(row):
        if row['best_objective'] >= 1e9:
            return np.nan
        best = best_per_instance.get(row['instance_id'], row['best_objective'])
        if best == 0:
            return 0
        return ((row['best_objective'] - best) / best) * 100
    
    df['rpd'] = df.apply(get_rpd, axis=1)
    return df


def calculate_statistics(df):
    """计算统计汇总表格"""
    if df is None or df.empty:
        return None
    
    valid_df = df[df['best_objective'] < 1e9].copy()
    if valid_df.empty:
        return None
    
    try:
        summary, perf_matrix = compute_stats(
            valid_df, 
            instance_col="instance_id", 
            algo_col="algorithm_name", 
            perf_col="best_objective"
        )
        return summary
    except Exception as e:
        st.error(f"计算统计数据时出错: {e}")
        return None


def fig_to_svg(fig):
    """将matplotlib图形转换为SVG字符串"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='svg', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    svg_str = img_buffer.read().decode('utf-8')
    plt.close(fig)
    return svg_str


def plot_performance_profile(df):
    """绘制性能剖面图"""
    if df is None or df.empty:
        return None
    
    valid_df = df[df['best_objective'] < 1e9].copy()
    if valid_df.empty:
        return None
    
    try:
        # 如果有重复的(instance, algorithm)组合，取最优值
        if valid_df.duplicated(subset=['instance_id', 'algorithm_name']).any():
            perf_df_agg = valid_df.groupby(['instance_id', 'algorithm_name'])['best_objective'].min().reset_index()
        else:
            perf_df_agg = valid_df[['instance_id', 'algorithm_name', 'best_objective']]
        
        perf_matrix = perf_df_agg.pivot(
            index='instance_id', 
            columns='algorithm_name', 
            values='best_objective'
        )
        
        fig = plot_perf_profile(perf_matrix, figsize=(10, 6))
        return fig
    except Exception as e:
        st.error(f"绘制性能剖面图时出错: {e}")
        return None


def plot_anytime_curve(df):
    """绘制Anytime收敛曲线"""
    if df is None or df.empty:
        return None
    
    valid_df = df[df['best_objective'] < 1e9].copy()
    if valid_df.empty:
        return None
    
    try:
        time_df = valid_df[['algorithm_name', 'runtime', 'best_objective']].copy()
        time_df.columns = ['algo', 'time', 'best_so_far']
        
        fig = plot_anytime(time_df, time_col='time', perf_col='best_so_far', algo_col='algo', figsize=(10, 6))
        return fig
    except Exception as e:
        st.error(f"绘制Anytime曲线时出错: {e}")
        return None


def calculate_feasibility_analysis(df):
    """计算可行性分析"""
    if df is None or df.empty:
        return None
    
    try:
        feasibility_df = compute_feasibility(df, objective_col='best_objective', threshold=1e9)
        
        if feasibility_df is not None and len(feasibility_df) > 0:
            avg_row = {
                'feasible_ratio': feasibility_df['Feasible Rate (%)'].mean(),
                'infeasible_ratio': feasibility_df['Infeasible Rate (%)'].mean(),
                'avg_runtime': feasibility_df['Avg Runtime (s)'].mean()
            }
            return avg_row
        return None
    except Exception as e:
        st.error(f"计算可行性分析时出错: {e}")
        return None


# =========================
# Sidebar (配置区)
# =========================
st.sidebar.title("Configuration")

DATASET_MAX_COUNTS = {
    "j30": 480,
    "j60": 480,
    "j90": 480,
    "j120": 600
}

st.sidebar.subheader("Dataset")
dataset = st.sidebar.selectbox("Dataset", ["j30", "j60", "j90", "j120", "custom"])

if dataset == "custom":
    st.sidebar.write("**Custom Instance Counts:**")
    custom_j30 = st.sidebar.number_input("j30 Count", 0, 480, 0, key="custom_j30")
    custom_j60 = st.sidebar.number_input("j60 Count", 0, 480, 0, key="custom_j60")
    custom_j90 = st.sidebar.number_input("j90 Count", 0, 480, 0, key="custom_j90")
    custom_j120 = st.sidebar.number_input("j120 Count", 0, 600, 0, key="custom_j120")
    instance_count = custom_j30 + custom_j60 + custom_j90 + custom_j120
else:
    max_count = DATASET_MAX_COUNTS[dataset]
    instance_count = st.sidebar.number_input("Instance Count", 1, max_count, min(10, max_count))

st.sidebar.subheader("Budget")
budget_type = st.sidebar.radio("Budget Type", ["evaluations", "time"])
if budget_type == "evaluations":
    budget_value = st.sidebar.number_input("Max Evaluations", 100, 100000, 1000)
else:
    budget_value = st.sidebar.number_input("Time Limit (seconds)", 1, 3600, 60)

st.sidebar.subheader("Seeds")
seeds = st.sidebar.multiselect("Seeds", list(range(100)), default=[0, 1])

st.sidebar.markdown("---")

st.sidebar.subheader("Algorithms")
algo_ba = st.sidebar.checkbox("BA (Bat Algorithm)", False)
algo_pso = st.sidebar.checkbox("PSO (Particle Swarm)", False)
algo_hs = st.sidebar.checkbox("HS (Harmony Search)", False)
algo_ga = st.sidebar.checkbox("GA (Genetic Algorithm)", False)
algo_de = st.sidebar.checkbox("DE (Differential Evolution)", False)
algo_pr = st.sidebar.checkbox("PR (Path Relinking)", False)
algo_ts = st.sidebar.checkbox("TS (Tabu Search)", False)

st.sidebar.markdown("---")

if algo_ba:
    st.sidebar.subheader("BA Operators")
    ba_ls = st.sidebar.selectbox("BA Local Search", ["none", "tlim"], index=0)

if algo_pso:
    st.sidebar.subheader("PSO Operators")
    pso_ls = st.sidebar.selectbox("PSO Local Search", ["none", "sa"], index=0)
    pso_restart = st.sidebar.selectbox("PSO Restart", ["none", "adaptive"], index=0)

if algo_hs:
    st.sidebar.subheader("HS Operators")
    hs_param = st.sidebar.selectbox("HS Parameter Strategy", ["fixed", "adaptive"], index=0)
    hs_init = st.sidebar.selectbox("HS Initialization", ["random", "forward"], index=0)

if algo_ga:
    st.sidebar.subheader("GA Operators")
    ga_selection = st.sidebar.selectbox("GA Selection", ["roulette", "tournament"], index=0)
    ga_crossover = st.sidebar.selectbox("GA Crossover", ["single_point", "two_point", "rcx", "hybrid"], index=0)
    ga_mutation = st.sidebar.selectbox("GA Mutation", ["random", "adaptive"], index=0)
    ga_init = st.sidebar.selectbox("GA Initialization", ["random", "heuristic"], index=0)
    ga_ls = st.sidebar.selectbox("GA Local Search", ["none", "activity", "shift"], index=0)
    ga_neighborhood = st.sidebar.checkbox("GA Neighborhood", value=False)
    ga_elitism = st.sidebar.checkbox("GA Elitism", value=False)
    ga_sa_acceptance = st.sidebar.checkbox("GA SA Acceptance", value=False)

if algo_de:
    st.sidebar.subheader("DE Operators")
    de_mutation = st.sidebar.selectbox("DE Mutation", ["rand/1", "rand/2", "best/1", "best/2", "adaptive", "current-to-rand/2"], index=0)
    de_crossover = st.sidebar.selectbox("DE Crossover", ["bin", "exp"], index=0)
    if de_mutation in ["rand/1", "rand/2", "best/1", "best/2"]:
        de_adaptive_f = st.sidebar.checkbox("DE Adaptive F", value=False)
    else:
        de_adaptive_f = False
    de_adaptive_cr = st.sidebar.checkbox("DE Adaptive CR", value=False)
    de_ls = st.sidebar.checkbox("DE Local Search", value=False)

if algo_pr:
    st.sidebar.subheader("PR Operators")
    pr_path = st.sidebar.selectbox("PR Path Strategy", ["forward", "backward", "random", "bidirectional"], index=0)
    pr_selection = st.sidebar.selectbox("PR Selection Strategy", ["best", "random_two"], index=0)
    pr_ls = st.sidebar.checkbox("PR Local Search", value=False)

if algo_ts:
    st.sidebar.subheader("TS Operators")
    ts_strategy = st.sidebar.selectbox("TS Tabu Strategy", ["static", "dynamic"], index=0)

st.sidebar.markdown("---")

st.sidebar.subheader("Algorithm Selector")
ml_model = st.sidebar.selectbox(
    "Machine Learning Model", 
    ["decision_tree", "random_forest", "gradient_boosting", "svm", "knn"],
    index=0,
    help="选择用于算法选择的机器学习模型"
)
test_size = st.sidebar.slider(
    "Test Set Size (%)", 
    min_value=10, 
    max_value=50, 
    value=30, 
    step=5,
    help="测试集占总数据的比例"
)
use_tsne = st.sidebar.checkbox(
    "Use t-SNE for Instance Space Analysis", 
    value=False,
    help="使用t-SNE代替PCA进行降维（适用于非线性结构）"
)

st.sidebar.markdown("---")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    run_button = st.button("RUN", type="primary", use_container_width=True)
with col_btn2:
    stop_button = st.button("STOP", type="secondary", use_container_width=True)

if stop_button:
    st.session_state.stop_experiment = True
    st.session_state.is_running = False

# =========================
# 主界面 Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["运行监控", "结果分析", "算法选择 (Selector)"])

# =========================
# Tab 1: 运行监控
# =========================
with tab1:
    st.title("运行监控")
    
    if run_button:
        st.session_state.stop_experiment = False
        st.session_state.is_running = True
        
        selected_algos = []
        if algo_ba:
            selected_algos.append("BA")
        if algo_pso:
            selected_algos.append("PSO")
        if algo_hs:
            selected_algos.append("HS")
        if algo_ga:
            selected_algos.append("GA")
        if algo_de:
            selected_algos.append("DE")
        if algo_pr:
            selected_algos.append("PR")
        if algo_ts:
            selected_algos.append("TS")
        
        if not selected_algos:
            st.error("请至少选择一个算法！")
            st.session_state.is_running = False
        elif not seeds:
            st.error("请至少选择一个种子！")
            st.session_state.is_running = False
        elif instance_count == 0:
            st.error("实例数量不能为0！")
            st.session_state.is_running = False
        else:
            stop_placeholder = st.empty()
            stop_placeholder.warning("实验运行中... 点击 STOP 按钮可终止实验")
            
            with st.status("正在运行实验...", expanded=True) as status:
                st.write("正在加载实例...")
                
                all_instances = []
                all_deadlines = []
                all_instance_files = []
                
                if dataset == "custom":
                    subset_counts = {
                        "j30": custom_j30,
                        "j60": custom_j60,
                        "j90": custom_j90,
                        "j120": custom_j120
                    }
                    for subset, count in subset_counts.items():
                        if count > 0:
                            instances, deadlines, instance_files = load_instances_from_subset(subset, count)
                            all_instances.extend(instances)
                            all_deadlines.extend(deadlines)
                            all_instance_files.extend(instance_files)
                else:
                    all_instances, all_deadlines, all_instance_files = load_instances_from_subset(dataset, instance_count)
                
                if not all_instances:
                    st.error("无法加载实例，请检查数据目录！")
                    st.session_state.is_running = False
                else:
                    st.write(f"成功加载 {len(all_instances)} 个实例")
                    
                    if budget_type == "evaluations":
                        max_evaluations = budget_value
                        time_limit = 60.0
                    else:
                        max_evaluations = 100000
                        time_limit = float(budget_value)
                    
                    exp_config = ExperimentConfig(
                        instances=all_instance_files,
                        algorithms=selected_algos,
                        seeds=seeds,
                        deadlines=all_deadlines,
                        max_evaluations=max_evaluations,
                        output_dir=get_results_dir(),
                        time_limit=time_limit,
                        problem_type="rlp",
                        use_delay_factors=False
                    )
                    
                    runner = ExperimentRunner(exp_config)
                    
                    all_configs = generate_all_algorithm_configs()
                    
                    selected_configs = [
                        cfg for cfg in all_configs
                        if cfg[1] in selected_algos
                    ]
                    
                    filtered_configs = []
                    for config_name, algo_type, params in selected_configs:
                        include = True
                        
                        if algo_type == "BA":
                            if params.get("local_search_strategy") != ba_ls:
                                include = False
                        elif algo_type == "PSO":
                            if params.get("local_search_strategy") != pso_ls:
                                include = False
                            if params.get("restart_strategy") != pso_restart:
                                include = False
                        elif algo_type == "HS":
                            if params.get("parameter_strategy") != hs_param:
                                include = False
                            if params.get("initialization_strategy") != hs_init:
                                include = False
                        elif algo_type == "GA":
                            if params.get("selection_strategy") != ga_selection:
                                include = False
                            if params.get("crossover_strategy") != ga_crossover:
                                include = False
                            if params.get("mutation_strategy") != ga_mutation:
                                include = False
                            if params.get("initialization_strategy") != ga_init:
                                include = False
                            if params.get("local_search_strategy") != ga_ls:
                                include = False
                            if (params.get("neighborhood_size", 0) > 0) != ga_neighborhood:
                                include = False
                            if params.get("elitism", False) != ga_elitism:
                                include = False
                            if params.get("use_sa_acceptance", False) != ga_sa_acceptance:
                                include = False
                        elif algo_type == "DE":
                            if params.get("mutation_strategy") != de_mutation:
                                include = False
                            if params.get("crossover_strategy") != de_crossover:
                                include = False
                            if params.get("use_adaptive_F", False) != de_adaptive_f:
                                include = False
                            if params.get("use_adaptive_CR", False) != de_adaptive_cr:
                                include = False
                            if params.get("use_local_search", False) != de_ls:
                                include = False
                        elif algo_type == "PR":
                            if params.get("path_strategy") != pr_path:
                                include = False
                            if params.get("selection_strategy") != pr_selection:
                                include = False
                            if params.get("use_local_search", False) != pr_ls:
                                include = False
                        elif algo_type == "TS":
                            if params.get("tabu_strategy") != ts_strategy:
                                include = False
                        
                        if include:
                            filtered_configs.append((config_name, algo_type, params))
                    
                    selected_configs = filtered_configs
                    
                    total_runs = len(all_instances) * len(selected_configs) * len(seeds)
                    st.write(f"总运行次数: {total_runs}")
                    st.write(f"算法配置数量: {len(selected_configs)}")
                    
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    results = []
                    completed_runs = 0
                    stopped_early = False
                    
                    for idx, instance_id in enumerate(all_instance_files):
                        if st.session_state.stop_experiment:
                            stopped_early = True
                            break
                        
                        instance = all_instances[idx]
                        deadline = all_deadlines[idx]
                        
                        for algo_config in selected_configs:
                            if st.session_state.stop_experiment:
                                stopped_early = True
                                break
                            
                            for seed in seeds:
                                if st.session_state.stop_experiment:
                                    stopped_early = True
                                    break
                                
                                try:
                                    progress_text.write(f"运行: {os.path.basename(instance_id)} / {algo_config[0]} / seed={seed}")
                                    
                                    result = runner.run_single(
                                        instance, algo_config, seed, deadline, max_evaluations
                                    )
                                    
                                    row = {
                                        "instance_id": result.instance_id,
                                        "seed": result.seed,
                                        "best_objective": result.best_objective,
                                        "runtime": result.runtime,
                                        "algorithm_name": result.algorithm_name,
                                        "deadline": result.deadline,
                                    }
                                    for key, value in result.algorithm_params.items():
                                        row[f"param_{key}"] = value
                                    results.append(row)
                                    
                                except Exception as e:
                                    import traceback
                                    traceback.print_exc()
                                    results.append({
                                        "instance_id": instance_id,
                                        "seed": seed,
                                        "best_objective": 1e10,
                                        "runtime": 0,
                                        "algorithm_name": algo_config[0],
                                        "deadline": deadline,
                                    })
                                
                                completed_runs += 1
                                progress_bar.progress(int(completed_runs / total_runs * 100))
                    
                    st.session_state.is_running = False
                    
                    if results:
                        df = pd.DataFrame(results)
                        df = calculate_rpd(df)
                        
                        filename = datetime.now().strftime('%m%d_%H%M') + ".csv"
                        output_dir = get_results_dir()
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, filename)
                        df.to_csv(output_path, index=False)
                        
                        st.session_state.current_results_df = df
                        st.session_state.current_results_file = output_path
                        
                        if stopped_early:
                            stop_placeholder.warning(f"实验已被用户终止！已完成 {completed_runs}/{total_runs} 次运行")
                            status.update(label=f"实验已终止！已完成 {completed_runs} 次运行，结果已保存到 {output_path}", state="error")
                        else:
                            stop_placeholder.empty()
                            status.update(label=f"实验完成！结果已保存到 {output_path}", state="complete")
                            st.success(f"实验完成！共运行 {total_runs} 次")
                        
                        st.info("请切换到 '结果分析' 标签页查看详细结果。")
                    else:
                        if stopped_early:
                            stop_placeholder.warning("实验已被用户终止！未完成任何运行")
                            status.update(label="实验已终止！未完成任何运行", state="error")
    else:
        if st.session_state.is_running:
            st.warning("实验正在运行中... 点击 STOP 按钮可终止实验")
        else:
            st.info("请在左侧配置实验参数，然后点击 'RUN' 开始实验。")
        
        st.subheader("实验进度")
        st.progress(0)
        st.metric("完成度", "0%")
        st.metric("总运行次数", "-")
        
        st.markdown("---")
        
        st.subheader("当前任务状态")
        col1, col2, col3 = st.columns(3)
        col1.info("**实例:** -")
        col2.info("**算法:** -")
        col3.info("**种子:** -")
        
        st.markdown("---")
        
        st.subheader("运行日志")
        st.code("暂无运行日志", language="plaintext")

# =========================
# Tab 2: 结果分析
# =========================
with tab2:
    st.title("结果分析")
    
    latest_file, df = get_current_results()
    
    if latest_file is not None and df is not None:
        df = calculate_rpd(df)
        
        st.subheader("数据概览")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("数据来源", os.path.basename(latest_file))
        col2.metric("总记录数", len(df))
        valid_count = len(df[df['best_objective'] < 1e9])
        col3.metric("有效记录数", valid_count)
        col4.metric("可行解比例", f"{valid_count/len(df)*100:.1f}%")
        
        st.markdown("---")
        
        st.subheader("统计汇总表格")
        stats = calculate_statistics(df)
        if stats is not None:
            st.dataframe(stats, use_container_width=True)
            
            st.markdown("#### 算法性能结论")
            st.info("从不同角度比较算法性能，帮助您选择最优算法")
            
            conclusion_col1, conclusion_col2, conclusion_col3 = st.columns(3)
            
            with conclusion_col1:
                best_mean_algo = stats['Mean'].idxmin()
                best_mean_value = stats['Mean'].min()
                st.metric(
                    "最优均值算法", 
                    best_mean_algo,
                    f"Mean = {best_mean_value:.2f}"
                )
            
            with conclusion_col2:
                best_rank_algo = stats['Rank'].idxmin()
                best_rank_value = stats['Rank'].min()
                st.metric(
                    "最高排名算法", 
                    best_rank_algo,
                    f"Rank = {best_rank_value:.2f}"
                )
            
            with conclusion_col3:
                best_rpd_algo = stats['RPD(%)'].idxmin()
                best_rpd_value = stats['RPD(%)'].min()
                st.metric(
                    "最低RPD算法", 
                    best_rpd_algo,
                    f"RPD = {best_rpd_value:.2f}%"
                )
            
            st.markdown("**综合评价:**")
            
            if best_mean_algo == best_rank_algo == best_rpd_algo:
                st.success(f"🏆 **{best_mean_algo}** 在所有指标上均表现最优，是本次实验的最佳算法！")
            else:
                st.markdown(f"""
                - **均值最优**: {best_mean_algo} (Mean = {best_mean_value:.2f})
                - **排名最高**: {best_rank_algo} (Rank = {best_rank_value:.2f})
                - **RPD最低**: {best_rpd_algo} (RPD = {best_rpd_value:.2f}%)
                
                💡 **建议**: 综合考虑均值、排名和RPD指标，选择最适合您需求的算法。
                """)
            
            st.markdown("**指标说明:**")
            st.markdown("""
            - **Mean**: 平均目标值，越小越好
            - **Rank**: 平均排名，越小越好
            - **RPD(%)**: 相对性能偏差，越小越好
            - **Std**: 标准差，越小表示算法越稳定
            """)
        else:
            st.warning("没有有效的统计数据")
        
        st.markdown("---")
        
        st.subheader("性能剖面图")
        fig_pp = plot_performance_profile(df)
        if fig_pp is not None:
            st.pyplot(fig_pp)
            svg_pp = fig_to_svg(fig_pp)
            st.download_button(
                label="下载 SVG 格式",
                data=svg_pp,
                file_name="performance_profile.svg",
                mime="image/svg+xml"
            )
        else:
            st.info("无法生成性能剖面图")
        
        st.markdown("---")
        
        st.subheader("Anytime 收敛曲线")
        fig_anytime = plot_anytime_curve(df)
        if fig_anytime is not None:
            st.pyplot(fig_anytime)
            svg_anytime = fig_to_svg(fig_anytime)
            st.download_button(
                label="下载 SVG 格式",
                data=svg_anytime,
                file_name="anytime_curve.svg",
                mime="image/svg+xml"
            )
        else:
            st.info("无法生成 Anytime 曲线")
        
        st.markdown("---")
        
        st.subheader("显著性比较")
        st.info("""
        **统计检验说明:**
        - **Friedman检验**: 非参数检验，判断算法间是否存在显著差异
        - **Wilcoxon检验**: 成对比较，判断两个算法之间是否存在显著差异
        - **显著性水平**: α = 0.05
        """)
        
        valid_df_stats = df[df['best_objective'] < 1e9].copy()
        if not valid_df_stats.empty:
            try:
                # 如果有重复的(instance, algorithm)组合，取最优值
                if valid_df_stats.duplicated(subset=['instance_id', 'algorithm_name']).any():
                    perf_df_agg = valid_df_stats.groupby(['instance_id', 'algorithm_name'])['best_objective'].min().reset_index()
                else:
                    perf_df_agg = valid_df_stats[['instance_id', 'algorithm_name', 'best_objective']]
                
                perf_matrix = perf_df_agg.pivot(
                    index='instance_id', 
                    columns='algorithm_name', 
                    values='best_objective'
                )
                
                stats_results = perform_statistical_tests(perf_matrix, alpha=0.05)
                
                if 'friedman' in stats_results:
                    friedman = stats_results['friedman']
                    friedman_col1, friedman_col2, friedman_col3 = st.columns(3)
                    
                    with friedman_col1:
                        st.metric("Friedman统计量", f"{friedman['statistic']:.4f}" if friedman['statistic'] else "N/A")
                    
                    with friedman_col2:
                        st.metric("p值", f"{friedman['p_value']:.6f}" if friedman['p_value'] else "N/A")
                    
                    with friedman_col3:
                        if friedman['significant']:
                            st.metric("结论", "存在显著差异", delta="p < 0.05", delta_color="normal")
                        else:
                            st.metric("结论", "无显著差异", delta="p ≥ 0.05", delta_color="inverse")
                
                if 'average_ranks' in stats_results:
                    st.markdown("**算法平均排名:**")
                    fig_rank = plot_rank_comparison(stats_results['average_ranks'], figsize=(10, 6))
                    if fig_rank is not None:
                        st.pyplot(fig_rank)
                        svg_rank = fig_to_svg(fig_rank)
                        st.download_button(
                            label="下载排名图 SVG 格式",
                            data=svg_rank,
                            file_name="algorithm_ranking.svg",
                            mime="image/svg+xml"
                        )
                
                if 'wilcoxon' in stats_results and len(stats_results['wilcoxon']) > 0:
                    st.markdown("**Wilcoxon成对检验结果:**")
                    wilcoxon_df = stats_results['wilcoxon']
                    st.dataframe(wilcoxon_df, use_container_width=True)
                    
                    significant_pairs = wilcoxon_df[wilcoxon_df['Significant'] == True]
                    if len(significant_pairs) > 0:
                        st.success(f"发现 {len(significant_pairs)} 对算法存在显著差异")
                    else:
                        st.info("未发现显著差异的算法对")
            
            except Exception as e:
                st.error(f"统计检验出错: {e}")
        
        st.markdown("---")
        
        st.subheader("可行性与计算成本分析")
        feasibility = calculate_feasibility_analysis(df)
        if feasibility is not None:
            feas_col1, feas_col2, feas_col3 = st.columns(3)
            feas_col1.metric("可行解比例", f"{feasibility['feasible_ratio']:.1f}%")
            feas_col2.metric("不可行评估比例", f"{feasibility['infeasible_ratio']:.1f}%")
            feas_col3.metric("平均运行时间", f"{feasibility['avg_runtime']:.3f}s")
        
        st.markdown("---")
        
        st.subheader("原始数据预览")
        valid_df = df[df['best_objective'] < 1e9]
        if not valid_df.empty:
            display_cols = ['instance_id', 'seed', 'algorithm_name', 'best_objective', 'runtime', 'rpd']
            display_cols = [c for c in display_cols if c in valid_df.columns]
            st.dataframe(valid_df[display_cols].head(50), use_container_width=True)
    else:
        st.info("没有找到结果文件，请先运行实验")

# =========================
# Tab 3: 算法选择
# =========================
with tab3:
    st.title("算法选择 (Algorithm Selector)")
    
    latest_file, df = get_current_results()
    
    if latest_file is not None and df is not None:
        df = calculate_rpd(df)
        
        st.info("""
        **算法选择模块说明:**
        
        本模块旨在解决"按实例动态选择最优算法"的监督学习问题。通过机器学习方法，根据每个问题实例的特征，预测最适合该实例的算法，从而尽可能接近理论最优的虚拟最佳算法（VBS）。
        
        **四个关键阶段:**
        1. **数据构建**: 基于实验结果构建性能矩阵和特征数据集
        2. **模型训练**: 使用机器学习模型训练算法选择器
        3. **性能评估**: 与SBS和VBS对比，评估选择器性能
        4. **解释性分析**: 特征重要性分析和实例空间可视化
        """)
        
        st.markdown("---")
        
        # 第一阶段：数据构建
        st.subheader("第一阶段：数据构建")
        
        valid_df = df[df['best_objective'] < 1e9].copy()
        
        if valid_df.empty:
            st.warning("没有有效的实验数据，请先运行实验")
        else:
            # 显示数据概览
            data_col1, data_col2, data_col3 = st.columns(3)
            
            with data_col1:
                n_instances = valid_df['instance_id'].nunique()
                st.metric("实例数量", n_instances)
            
            with data_col2:
                n_algorithms = valid_df['algorithm_name'].nunique()
                st.metric("算法数量", n_algorithms)
            
            with data_col3:
                n_records = len(valid_df)
                st.metric("总记录数", n_records)
            
            # 检查是否有足够的实例
            if n_instances < 10:
                st.warning("实例数量较少（< 10），建议运行更多实例以获得可靠的选择器")
            
            st.markdown("---")
            
            # 第二阶段：模型训练
            st.subheader("第二阶段：模型训练与预测")
            
            st.markdown(f"""
            **训练配置:**
            - **机器学习模型**: {ml_model}
            - **测试集比例**: {test_size}%
            - **降维方法**: {'t-SNE' if use_tsne else 'PCA'}
            """)
            
            # 提取特征
            with st.spinner("正在提取实例特征..."):
                try:
                    # 获取实例列表
                    instances = valid_df['instance_id'].unique()
                    
                    # 加载实例并提取特征
                    feature_list = []
                    instance_files = []
                    
                    for instance_id in instances:
                        # 构造实例文件路径
                        instance_id_clean = instance_id.replace('.RCP', '').replace('.rcp', '')  # 移除后缀
                        if instance_id.upper().startswith('J30'):
                            instance_file = f"data/psplib_raw/j30/{instance_id_clean}.RCP"
                        elif instance_id.upper().startswith('J60'):
                            instance_file = f"data/psplib_raw/j60/{instance_id_clean}.RCP"
                        elif instance_id.upper().startswith('J90'):
                            instance_file = f"data/psplib_raw/j90/{instance_id_clean}.RCP"
                        elif instance_id.upper().startswith('J120'):
                            instance_file = f"data/psplib_raw/j120/{instance_id_clean}.RCP"
                        else:
                            continue
                        
                        if os.path.exists(instance_file):
                            try:
                                inst = load_psplib_sm(instance_file)
                                
                                # 计算截止时间
                                n = inst.n_activities
                                es = [0] * n
                                for j in range(n):
                                    for pred in inst.predecessors[j]:
                                        es[j] = max(es[j], es[pred] + inst.durations[pred])
                                critical_path_length = max([es[i] + inst.durations[i] for i in range(n)])
                                horizon = int(critical_path_length * 1.5)
                                
                                # 提取特征
                                extractor = FeatureExtractor(inst, horizon)
                                features = extractor.extract_all()
                                features['instance_id'] = instance_id
                                feature_list.append(features)
                                instance_files.append(instance_id)
                            except Exception as e:
                                st.warning(f"无法提取实例 {instance_id} 的特征: {e}")
                    
                    if len(feature_list) == 0:
                        st.error("无法提取任何实例的特征，请检查实例文件路径")
                    else:
                        feature_df = pd.DataFrame(feature_list)
                        
                        st.success(f"成功提取 {len(feature_list)} 个实例的特征")
                        
                        # 第三阶段：性能评估
                        st.markdown("---")
                        st.subheader("第三阶段：性能评估")
                        
                        with st.spinner("正在训练算法选择器..."):
                            try:
                                # 运行算法选择分析
                                results = analyze_selector(
                                    perf_df=valid_df,
                                    feature_df=feature_df,
                                    model_type=ml_model,
                                    instance_col='instance_id',
                                    algo_col='algorithm_name',
                                    perf_col='best_objective',
                                    test_size=test_size/100.0,
                                    random_state=42,
                                    use_tsne=use_tsne
                                )
                                
                                # 显示性能基准
                                st.markdown("#### 性能基准对比")
                                
                                bench_col1, bench_col2, bench_col3 = st.columns(3)
                                
                                with bench_col1:
                                    st.metric(
                                        "SBS (Single Best Solver)", 
                                        f"{results['SBS']['score']:.2f}",
                                        f"算法: {results['SBS']['algorithm']}"
                                    )
                                
                                with bench_col2:
                                    st.metric(
                                        "Selector (算法选择器)", 
                                        f"{results['Selector']['score']:.2f}",
                                        f"改进: {results['Selector']['improvement_over_sbs']:.2f}%"
                                    )
                                
                                with bench_col3:
                                    st.metric(
                                        "VBS (Virtual Best Solver)", 
                                        f"{results['VBS']['score']:.2f}",
                                        f"Gap: {results['Selector']['gap_to_vbs']:.2f}%"
                                    )
                                
                                # 显示选择器性能指标
                                st.markdown("#### 选择器性能指标")
                                
                                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                
                                with metric_col1:
                                    st.metric("命中率", f"{results['Selector']['hit_rate']*100:.1f}%")
                                
                                with metric_col2:
                                    st.metric("平均Regret", f"{results['Selector']['avg_regret']:.2f}")
                                
                                with metric_col3:
                                    st.metric("P90 Penalty", f"{results['Selector']['p90_penalty']:.2f}")
                                
                                with metric_col4:
                                    improvement = results['Selector']['improvement_over_sbs']
                                    if improvement > 0:
                                        st.metric("相比SBS改进", f"{improvement:.2f}%", delta="↑")
                                    else:
                                        st.metric("相比SBS改进", f"{improvement:.2f}%", delta="↓")
                                
                                # 性能比较图
                                st.markdown("#### 性能比较可视化")
                                fig_pc = results['figures']['performance_comparison']
                                if fig_pc is not None:
                                    st.pyplot(fig_pc)
                                    svg_pc = fig_to_svg(fig_pc)
                                    st.download_button(
                                        label="下载性能比较图 SVG 格式",
                                        data=svg_pc,
                                        file_name="performance_comparison.svg",
                                        mime="image/svg+xml"
                                    )
                                
                                # 第四阶段：解释性分析
                                st.markdown("---")
                                st.subheader("第四阶段：解释性分析与可视化")
                                
                                # 特征重要性
                                st.markdown("#### 特征重要性分析")
                                
                                feature_importance = results['feature_importance']
                                if feature_importance is not None and len(feature_importance) > 0:
                                    st.dataframe(feature_importance.head(20), use_container_width=True)
                                    
                                    fig_fi = results['figures']['feature_importance']
                                    if fig_fi is not None:
                                        st.pyplot(fig_fi)
                                        svg_fi = fig_to_svg(fig_fi)
                                        st.download_button(
                                            label="下载特征重要性图 SVG 格式",
                                            data=svg_fi,
                                            file_name="feature_importance.svg",
                                            mime="image/svg+xml"
                                        )
                                
                                # 实例空间分析
                                st.markdown("#### 实例空间分析")
                                st.info("""
                                **实例空间分析说明:**
                                
                                该图通过降维方法（PCA或t-SNE）将高维特征映射到二维空间，展示：
                                1. **实例分布**: 每个点代表一个问题实例
                                2. **算法优势区域**: 不同颜色表示不同算法在这些实例上表现最优
                                3. **算法选择边界**: 可以直观看出哪些类型的实例适合哪种算法
                                
                                这有助于回答"在什么情况下哪种算法更优"这一关键研究问题。
                                """)
                                
                                fig_isa = results['figures']['instance_space']
                                if fig_isa is not None:
                                    st.pyplot(fig_isa)
                                    svg_isa = fig_to_svg(fig_isa)
                                    st.download_button(
                                        label="下载实例空间分析图 SVG 格式",
                                        data=svg_isa,
                                        file_name="instance_space_analysis.svg",
                                        mime="image/svg+xml"
                                    )
                                
                                # 显示结论
                                st.markdown("---")
                                st.subheader("结论与建议")
                                
                                if results['Selector']['improvement_over_sbs'] > 5:
                                    st.success(f"""
                                    🏆 **算法选择器表现优秀！**
                                    
                                    - 选择器相比SBS改进了 {results['Selector']['improvement_over_sbs']:.2f}%
                                    - 命中率达到 {results['Selector']['hit_rate']*100:.1f}%
                                    - 距离理论最优VBS仅差 {results['Selector']['gap_to_vbs']:.2f}%
                                    
                                    **建议**: 可以在实际应用中使用该选择器进行算法推荐。
                                    """)
                                elif results['Selector']['improvement_over_sbs'] > 0:
                                    st.info(f"""
                                    ✅ **算法选择器表现良好**
                                    
                                    - 选择器相比SBS改进了 {results['Selector']['improvement_over_sbs']:.2f}%
                                    - 命中率为 {results['Selector']['hit_rate']*100:.1f}%
                                    
                                    **建议**: 可以进一步优化模型或增加训练数据以提升性能。
                                    """)
                                else:
                                    st.warning(f"""
                                    ⚠️ **算法选择器未优于SBS**
                                    
                                    - 选择器相比SBS改进了 {results['Selector']['improvement_over_sbs']:.2f}%
                                    - 可能原因：实例数量不足、特征区分度不够、模型选择不当
                                    
                                    **建议**: 
                                    1. 增加实验实例数量
                                    2. 尝试其他机器学习模型
                                    3. 检查特征提取是否合理
                                    """)
                                
                            except Exception as e:
                                st.error(f"训练算法选择器时出错: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                        
                except Exception as e:
                    st.error(f"提取特征时出错: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("没有找到结果文件，请先运行实验")
