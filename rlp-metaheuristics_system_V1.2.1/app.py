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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

from src.psp.psplib_io import load_psplib_sm
from src.eval.runner import ExperimentRunner, ExperimentConfig, generate_all_algorithm_configs

st.set_page_config(
    page_title="RLP Metaheuristics Research Framework",
    page_icon="",
    layout="wide"
)

# 会话状态 - 存储当前运行的结果
if 'current_results_df' not in st.session_state:
    st.session_state.current_results_df = None
if 'current_results_file' not in st.session_state:
    st.session_state.current_results_file = None


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
    """获取当前运行的结果（优先使用会话中存储的结果）"""
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
    pso_ls = st.sidebar.selectbox("PSO Local Search", ["none", "shift"], index=0)
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
    ga_sa_acceptance = st.sidebar.checkbox("GA SA Acceptance", value=False)

if algo_de:
    st.sidebar.subheader("DE Operators")
    de_mutation = st.sidebar.selectbox("DE Mutation", ["rand/1", "rand/2", "best/1", "best/2", "adaptive", "current-to-rand/2"], index=0)
    de_crossover = st.sidebar.selectbox("DE Crossover", ["bin", "exp"], index=0)
    de_adaptive_f = st.sidebar.checkbox("DE Adaptive F", value=False)
    de_adaptive_cr = st.sidebar.checkbox("DE Adaptive CR", value=False)
    de_ls = st.sidebar.checkbox("DE Local Search", value=False)

if algo_pr:
    st.sidebar.subheader("PR Operators")
    pr_path = st.sidebar.selectbox("PR Path Strategy", ["forward", "backward"], index=0)
    pr_selection = st.sidebar.selectbox("PR Selection Strategy", ["best", "random_two"], index=0)
    pr_ls = st.sidebar.checkbox("PR Local Search", value=False)

if algo_ts:
    st.sidebar.subheader("TS Operators")
    ts_strategy = st.sidebar.selectbox("TS Tabu Strategy", ["static", "dynamic"], index=0)

st.sidebar.markdown("---")

run_button = st.sidebar.button("RUN EXPERIMENTS", type="primary")
stop_button = st.sidebar.button("STOP")

# =========================
# 主界面 Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Execution", "Analysis", "Selector", "Results"])

# =========================
# Tab 1: Execution
# =========================
with tab1:
    st.title("Execution Monitor")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.subheader("Current Task")
    task_col1, task_col2, task_col3 = st.columns(3)
    task_col1.write("**Instance:** -")
    task_col2.write("**Algorithm:** -")
    task_col3.write("**Seed:** -")
    
    if run_button:
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
        elif not seeds:
            st.error("请至少选择一个种子！")
        elif instance_count == 0:
            st.error("实例数量不能为0！")
        else:
            if dataset == "custom":
                st.info(f"""
                **实验配置:**
                - 数据集: Custom
                - j30 实例数量: {custom_j30}
                - j60 实例数量: {custom_j60}
                - j90 实例数量: {custom_j90}
                - j120 实例数量: {custom_j120}
                - 总实例数量: {instance_count}
                - 预算类型: {budget_type} = {budget_value}
                - 种子: {seeds}
                - 选择算法: {', '.join(selected_algos)}
                """)
            else:
                st.info(f"""
                **实验配置:**
                - 数据集: {dataset}
                - 实例数量: {instance_count}
                - 预算类型: {budget_type} = {budget_value}
                - 种子: {seeds}
                - 选择算法: {', '.join(selected_algos)}
                """)
            
            st.subheader("算法算子配置")
            if algo_ba:
                st.write(f"**BA:** Local Search = {ba_ls}")
            if algo_pso:
                st.write(f"**PSO:** Local Search = {pso_ls}, Restart = {pso_restart}")
            if algo_hs:
                st.write(f"**HS:** Parameter Strategy = {hs_param}, Initialization = {hs_init}")
            if algo_ga:
                st.write(f"**GA:** Selection = {ga_selection}, Crossover = {ga_crossover}, Mutation = {ga_mutation}, Init = {ga_init}, LS = {ga_ls}, Neighborhood = {ga_neighborhood}, SA Acceptance = {ga_sa_acceptance}")
            if algo_de:
                st.write(f"**DE:** Mutation = {de_mutation}, Crossover = {de_crossover}, Adaptive F = {de_adaptive_f}, Adaptive CR = {de_adaptive_cr}, LS = {de_ls}")
            if algo_pr:
                st.write(f"**PR:** Path Strategy = {pr_path}, Selection = {pr_selection}, LS = {pr_ls}")
            if algo_ts:
                st.write(f"**TS:** Tabu Strategy = {ts_strategy}")
            
            status_text.text("正在加载实例...")
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
            else:
                status_text.text(f"已加载 {len(all_instances)} 个实例")
                
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
                st.write(f"**总运行次数:** {total_runs}")
                st.write(f"**算法配置数量:** {len(selected_configs)}")
                
                results = []
                progress_step = 100 / total_runs if total_runs > 0 else 100
                current_progress = 0
                
                log_messages = []
                last_instance = "-"
                last_algo = "-"
                last_seed = "-"
                
                for idx, instance_id in enumerate(all_instance_files):
                    instance = all_instances[idx]
                    deadline = all_deadlines[idx]
                    
                    for algo_config in selected_configs:
                        for seed in seeds:
                            try:
                                last_instance = instance_id
                                last_algo = algo_config[0]
                                last_seed = str(seed)
                                
                                status_text.text(f"运行: {instance_id} / {algo_config[0]} / seed={seed}")
                                
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
                                
                                log_messages.append(f"[INFO] {algo_config[0]} on {instance_id} (seed={seed}): {result.best_objective:.2f}")
                            except Exception as e:
                                import traceback
                                error_msg = f"[ERROR] {algo_config[0]} on {instance_id} (seed={seed}): {str(e)}"
                                log_messages.append(error_msg)
                                print(f"\n{error_msg}")
                                traceback.print_exc()
                                results.append({
                                    "instance_id": instance_id,
                                    "seed": seed,
                                    "best_objective": 1e10,
                                    "runtime": 0,
                                    "algorithm_name": algo_config[0],
                                    "deadline": deadline,
                                })
                            
                            current_progress += progress_step
                            progress_bar.progress(min(int(current_progress), 100))
                
                df = pd.DataFrame(results)
                filename = datetime.now().strftime('%m%d_%H%M') + ".csv"
                output_dir = get_results_dir()
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                df.to_csv(output_path, index=False)
                
                st.session_state.current_results_df = df
                st.session_state.current_results_file = output_path
                
                st.success(f"Experiment Finished! 结果已保存到 {output_path}")
                
                st.subheader("Results Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Runs", total_runs)
                
                valid_results = df[df['best_objective'] < 1e9]
                if not valid_results.empty:
                    col2.metric("Best Objective", f"{valid_results['best_objective'].min():.2f}")
                    col3.metric("Average Objective", f"{valid_results['best_objective'].mean():.2f}")
                    success_rate = len(valid_results) / len(df) * 100
                    col4.metric("Success Rate", f"{success_rate:.1f}%")
                else:
                    col2.metric("Best Objective", "N/A")
                    col3.metric("Average Objective", "N/A")
                    col4.metric("Success Rate", "0%")
                
                st.subheader("Execution Logs")
                st.code("\n".join(log_messages[-20:]), language="plaintext")
                
                st.subheader("Last Completed Task")
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Instance:** {last_instance}")
                col2.write(f"**Algorithm:** {last_algo}")
                col3.write(f"**Seed:** {last_seed}")

# =========================
# Tab 2: Analysis
# =========================
with tab2:
    st.title("Statistical Analysis")
    
    latest_file, df = get_current_results()
    
    if latest_file is not None and df is not None:
        st.write(f"**数据来源:** {latest_file}")
        st.write(f"**总记录数:** {len(df)}")
        
        st.subheader("Summary Table")
        valid_df = df[df['best_objective'] < 1e9]
        
        if not valid_df.empty:
            st.write(f"**有效记录数:** {len(valid_df)}")
            
            summary = valid_df.groupby('algorithm_name').agg({
                'best_objective': ['mean', 'std', 'min', 'max', 'count']
            }).round(2)
            summary.columns = ['Mean', 'Std', 'Min', 'Max', 'Count']
            summary = summary.sort_values('Mean')
            st.dataframe(summary, use_container_width=True)
            
            st.subheader("Performance Comparison")
            chart_data = summary.reset_index()[['algorithm_name', 'Mean']]
            st.bar_chart(chart_data.set_index('algorithm_name'))
            
            st.subheader("Raw Data Preview")
            st.dataframe(valid_df[['instance_id', 'seed', 'algorithm_name', 'best_objective', 'runtime']].head(20), use_container_width=True)
        else:
            st.warning("没有有效的实验结果（所有目标值都 >= 1e9）")
            st.write("数据预览：")
            st.dataframe(df.head(10))
    else:
        st.info("没有找到结果文件，请先运行实验")
        st.write(f"结果目录: {get_results_dir()}")

# =========================
# Tab 3: Selector
# =========================
with tab3:
    st.title("Algorithm Selector")
    
    latest_file, df = get_current_results()
    
    if latest_file is not None and df is not None:
        st.subheader("数据概览")
        st.write(f"**数据来源:** {latest_file}")
        st.write(f"**总记录数:** {len(df)}")
        
        valid_df = df[df['best_objective'] < 1e9]
        if not valid_df.empty:
            st.write(f"**有效记录数:** {len(valid_df)}")
            
            st.subheader("算法性能排名")
            algo_stats = valid_df.groupby('algorithm_name').agg({
                'best_objective': ['mean', 'min', 'count']
            }).round(2)
            algo_stats.columns = ['平均目标值', '最优目标值', '运行次数']
            algo_stats = algo_stats.sort_values('平均目标值')
            st.dataframe(algo_stats, use_container_width=True)
            
            best_algo = algo_stats.index[0]
            st.success(f"**最佳算法:** {best_algo} (平均目标值: {algo_stats.loc[best_algo, '平均目标值']:.2f})")
            
            st.subheader("训练算法选择器")
            st.info("""
            **训练步骤:**
            1. 确保有足够的实验数据（建议每种算法至少运行 50+ 实例）
            2. 在命令行运行以下命令训练选择器:
               ```
               python main.py --skip-experiments --train-selector
               ```
            3. 训练完成后，选择器会保存在 `results/ml/selector.pkl`
            """)
            
            st.subheader("实例特征")
            st.write("""
            算法选择器使用以下实例特征:
            - 活动数量 (n_activities)
            - 资源种类数 (n_resources)
            - 平均紧前关系数 (avg_predecessors)
            - 最大紧前关系数 (max_predecessors)
            - 关键路径长度 (critical_path_length)
            - 资源需求均值 (avg_resource_demand)
            - 资源需求方差 (std_resource_demand)
            """)
        else:
            st.warning("没有有效的实验结果，无法进行算法选择分析")
    else:
        st.info("没有找到结果文件，请先运行实验")

# =========================
# Tab 4: Results
# =========================
with tab4:
    st.title("Results & Downloads")
    
    st.subheader("Result Files")
    results_dir = get_results_dir()
    
    if os.path.exists(results_dir):
        csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")), reverse=True)
        if csv_files:
            result_files = []
            for f in csv_files:
                size = os.path.getsize(f) / 1024
                created = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
                result_files.append({
                    "File Name": os.path.basename(f),
                    "Size": f"{size:.1f} KB",
                    "Created": created
                })
            st.dataframe(pd.DataFrame(result_files), use_container_width=True)
            
            latest_file = csv_files[0]
            with open(latest_file, 'r') as f:
                csv_data = f.read()
            
            st.download_button(
                label="Download Latest Results",
                data=csv_data,
                file_name=os.path.basename(latest_file),
                mime="text/csv"
            )
            
            st.subheader("Latest Results Preview")
            try:
                preview_df = pd.read_csv(latest_file)
                st.dataframe(preview_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"无法读取文件: {e}")
        else:
            st.info("没有找到结果文件")
    else:
        st.info(f"没有找到结果目录: {results_dir}")
