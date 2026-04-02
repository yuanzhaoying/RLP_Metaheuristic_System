"""
RLP Metaheuristics GUI - Main Entry Point
基于PySide6的桌面应用程序
完全参照prlp-platform-v0.2/gui框架
"""
import sys
import os
import yaml
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

rlp_mate_path = str(project_root)
if rlp_mate_path not in sys.path:
    sys.path.insert(0, rlp_mate_path)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QFormLayout, QListWidget, QListWidgetItem, QMessageBox,
    QSplitter, QFrame, QScrollArea, QGridLayout, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QThread, Signal, QProcess
from PySide6.QtGui import QFont, QIcon

from gui.tabs.tab1_metaheuristics import MetaheuristicsTab
from gui.tabs.tab2_ml import MLTab
from gui.plugin_loader import PluginLoader
from gui.export_utils import ExportUtils


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RLP Metaheuristics Research Framework")
        self.setMinimumSize(1200, 800)
        
        self.plugin_loader = PluginLoader()
        self.export_utils = ExportUtils()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        self._create_header()
        self._create_tabs()
        self._create_control_panel()
        self._create_status_bar()
        
        self._load_default_config()
    
    def _create_header(self):
        """创建头部区域"""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("RLP Metaheuristics Research Framework")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        version_label = QLabel("v1.0")
        version_label.setStyleSheet("color: gray;")
        header_layout.addWidget(version_label)
        
        self.main_layout.addWidget(header)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #cccccc;")
        line.setFixedHeight(2)
        self.main_layout.addWidget(line)
    
    def _create_tabs(self):
        """创建选项卡"""
        self.tab_widget = QTabWidget()
        
        self.tab_meta = MetaheuristicsTab(self.plugin_loader)
        self.tab_widget.addTab(self.tab_meta, "Metaheuristics")
        
        self.tab_ml = MLTab()
        self.tab_widget.addTab(self.tab_ml, "Machine Learning")
        
        self.main_layout.addWidget(self.tab_widget, stretch=1)
    
    def _create_control_panel(self):
        """创建控制面板"""
        control_group = QGroupBox("Control Panel")
        control_layout = QHBoxLayout(control_group)
        
        self.run_btn = QPushButton("Run Experiment")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self._run_experiment)
        control_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_experiment)
        control_layout.addWidget(self.stop_btn)
        
        control_layout.addSpacing(20)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.export_btn.clicked.connect(self._export_results)
        control_layout.addWidget(self.export_btn)
        
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._save_config)
        control_layout.addWidget(self.save_config_btn)
        
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self._load_config)
        control_layout.addWidget(self.load_config_btn)
        
        control_layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        control_layout.addWidget(self.progress_bar)
        
        self.main_layout.addWidget(control_group)
        
        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        self.main_layout.addWidget(log_group)
    
    def _create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def _load_default_config(self):
        """加载默认配置"""
        config_path = project_root / "configs" / "experiment.yaml"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.tab_meta.load_config(config)
                self.log_message(f"Loaded default config from {config_path}")
            except Exception as e:
                self.log_message(f"Warning: Could not load default config: {e}")
    
    def log_message(self, message: str):
        """添加日志消息"""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _run_experiment(self):
        """运行实验"""
        current_tab_index = self.tab_widget.currentIndex()
        
        if current_tab_index == 0:
            self._run_metaheuristics_experiment()
        elif current_tab_index == 1:
            self._run_ml_analysis()
    
    def _run_metaheuristics_experiment(self):
        """运行元启发式算法实验"""
        try:
            config = self.tab_meta.get_config()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to collect configuration: {e}")
            return
        
        temp_config_path = project_root / "configs" / "gui_run.yaml"
        try:
            os.makedirs(temp_config_path.parent, exist_ok=True)
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")
            return
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.runner = ExperimentRunner(temp_config_path)
        self.runner.progress_signal.connect(self._update_progress)
        self.runner.log_signal.connect(self.log_message)
        self.runner.finished_signal.connect(self._experiment_finished)
        self.runner.start()
        
        self.status_bar.showMessage("Running metaheuristics experiment...")
    
    def _run_ml_analysis(self):
        """运行机器学习分析"""
        try:
            config = self.tab_ml.get_config()
            self.log_message(f"ML配置: {config}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to collect ML configuration: {e}")
            return
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.ml_runner = MLAnalysisRunner(config)
        self.ml_runner.progress_signal.connect(self._update_progress)
        self.ml_runner.log_signal.connect(self.log_message)
        self.ml_runner.result_signal.connect(self._display_ml_results)
        self.ml_runner.finished_signal.connect(self._ml_analysis_finished)
        self.ml_runner.start()
        
        self.status_bar.showMessage("Running ML analysis...")
    
    def _display_ml_results(self, results: dict):
        """显示ML分析结果"""
        self.log_message(f"显示ML结果: {list(results.keys())}")
        self.tab_ml.display_results(results)
    
    def _ml_analysis_finished(self, success: bool, message: str):
        """ML分析完成回调"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.status_bar.showMessage("ML analysis completed successfully")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_bar.showMessage("ML analysis failed")
            QMessageBox.critical(self, "Error", message)
    
    def _stop_experiment(self):
        """停止实验"""
        if hasattr(self, 'runner') and self.runner.isRunning():
            self.runner.stop()
            self.log_message("Stopping experiment...")
    
    def _update_progress(self, value: int):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def _experiment_finished(self, success: bool, message: str):
        """实验完成回调"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.status_bar.showMessage("Experiment completed successfully")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_bar.showMessage("Experiment failed")
            QMessageBox.critical(self, "Error", message)
    
    def _export_results(self):
        """导出结果"""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        
        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                export_dir = Path(selected_dirs[0])
                try:
                    self.export_utils.export_all(export_dir)
                    QMessageBox.information(self, "Success", 
                                          f"Results exported to {export_dir}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def _save_config(self):
        """保存配置"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "YAML Files (*.yaml);;All Files (*)"
        )
        if file_path:
            try:
                config = self._collect_config()
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False)
                QMessageBox.information(self, "Success", 
                                      f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def _load_config(self):
        """加载配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "YAML Files (*.yaml);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.tab_meta.load_config(config)
                self.tab_ml.load_config(config)
                QMessageBox.information(self, "Success", 
                                      f"Configuration loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
    
    def _collect_config(self) -> dict:
        """收集所有配置"""
        config = {}
        
        config.update(self.tab_meta.get_config())
        
        config.update(self.tab_ml.get_config())
        
        return config


class ExperimentRunner(QThread):
    """实验运行线程"""
    progress_signal = Signal(int)
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)
    
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.process = None
        self._is_running = False
    
    def run(self):
        """运行实验"""
        self._is_running = True
        
        try:
            import pandas as pd
            import numpy as np
            import glob
            import re
            from src.psp.psplib_io import load_psplib_sm
            from src.eval.runner import ExperimentRunner, ExperimentConfig, generate_all_algorithm_configs
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.log_signal.emit("正在加载实例...")
            
            all_instances = []
            all_deadlines = []
            all_instance_files = []
            
            instance_sets = config.get('instance_sets', [])
            for inst_set in instance_sets:
                dataset = inst_set.get('name', 'j30')
                
                if dataset == "custom":
                    subset_counts = {
                        "j30": inst_set.get('j30_count', 0),
                        "j60": inst_set.get('j60_count', 0),
                        "j90": inst_set.get('j90_count', 0),
                        "j120": inst_set.get('j120_count', 0)
                    }
                    
                    for subset, count in subset_counts.items():
                        if count > 0:
                            instances, deadlines, instance_files = self._load_instances_from_subset(subset, count)
                            all_instances.extend(instances)
                            all_deadlines.extend(deadlines)
                            all_instance_files.extend(instance_files)
                else:
                    count = inst_set.get('limit', 10)
                    instances, deadlines, instance_files = self._load_instances_from_subset(dataset, count)
                    all_instances.extend(instances)
                    all_deadlines.extend(deadlines)
                    all_instance_files.extend(instance_files)
            
            if not all_instances:
                self.finished_signal.emit(False, "无法加载实例，请检查数据目录！")
                return
            
            self.log_signal.emit(f"成功加载 {len(all_instances)} 个实例")
            
            main_budget = config.get('main_budget', {})
            budget_mode = main_budget.get('mode', 'evals')
            max_evaluations = main_budget.get('max_evals', 1000) if budget_mode == 'evals' else 1000
            time_limit = main_budget.get('budget_sec', 60.0) if budget_mode == 'time' else 60.0
            
            seeds = config.get('seeds', list(range(10)))
            
            selected_algos = config.get('selected_algorithms', [])
            if not selected_algos:
                self.finished_signal.emit(False, "请至少选择一个算法！")
                return
            
            output_dir = str(project_root / "results" / "raw")
            
            exp_config = ExperimentConfig(
                instances=all_instance_files,
                algorithms=selected_algos,
                seeds=seeds,
                deadlines=all_deadlines,
                max_evaluations=max_evaluations,
                output_dir=output_dir,
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
            
            operator_config = config.get('operator_config', {})
            filtered_configs = []
            for config_name, algo_type, params in selected_configs:
                include = True
                
                if algo_type == "BA":
                    if params.get("local_search_strategy") != operator_config.get('ba_ls', 'none'):
                        include = False
                elif algo_type == "PSO":
                    if params.get("local_search_strategy") != operator_config.get('pso_ls', 'none'):
                        include = False
                    if params.get("restart_strategy") != operator_config.get('pso_restart', 'none'):
                        include = False
                elif algo_type == "HS":
                    if params.get("parameter_strategy") != operator_config.get('hs_param', 'fixed'):
                        include = False
                    if params.get("initialization_strategy") != operator_config.get('hs_init', 'random'):
                        include = False
                elif algo_type == "GA":
                    if params.get("selection_strategy") != operator_config.get('ga_selection', 'roulette'):
                        include = False
                    if params.get("crossover_strategy") != operator_config.get('ga_crossover', 'single_point'):
                        include = False
                    if params.get("mutation_strategy") != operator_config.get('ga_mutation', 'random'):
                        include = False
                    if params.get("initialization_strategy") != operator_config.get('ga_init', 'random'):
                        include = False
                    if params.get("local_search_strategy") != operator_config.get('ga_ls', 'none'):
                        include = False
                    if (params.get("neighborhood_size", 0) > 0) != operator_config.get('ga_neighborhood', False):
                        include = False
                    if params.get("elitism", False) != operator_config.get('ga_elitism', False):
                        include = False
                    if params.get("use_sa_acceptance", False) != operator_config.get('ga_sa_acceptance', False):
                        include = False
                elif algo_type == "DE":
                    if params.get("mutation_strategy") != operator_config.get('de_mutation', 'rand/1'):
                        include = False
                    if params.get("crossover_strategy") != operator_config.get('de_crossover', 'bin'):
                        include = False
                    if params.get("use_adaptive_F", False) != operator_config.get('de_adaptive_f', False):
                        include = False
                    if params.get("use_adaptive_CR", False) != operator_config.get('de_adaptive_cr', False):
                        include = False
                    if params.get("use_local_search", False) != operator_config.get('de_ls', False):
                        include = False
                elif algo_type == "PR":
                    if params.get("path_strategy") != operator_config.get('pr_path', 'forward'):
                        include = False
                    if params.get("selection_strategy") != operator_config.get('pr_selection', 'best'):
                        include = False
                    if params.get("use_local_search", False) != operator_config.get('pr_ls', False):
                        include = False
                elif algo_type == "TS":
                    if params.get("tabu_strategy") != operator_config.get('ts_strategy', 'static'):
                        include = False
                
                if include:
                    filtered_configs.append((config_name, algo_type, params))
            
            selected_configs = filtered_configs
            
            total_runs = len(all_instances) * len(selected_configs) * len(seeds)
            self.log_signal.emit(f"总运行次数: {total_runs}")
            self.log_signal.emit(f"算法配置数量: {len(selected_configs)}")
            
            results = []
            completed_runs = 0
            
            for idx, instance_file in enumerate(all_instance_files):
                if not self._is_running:
                    break
                
                instance = all_instances[idx]
                deadline = all_deadlines[idx]
                
                for algo_config in selected_configs:
                    if not self._is_running:
                        break
                    
                    for seed in seeds:
                        if not self._is_running:
                            break
                        
                        try:
                            self.log_signal.emit(f"运行: {os.path.basename(instance_file)} / {algo_config[0]} / seed={seed}")
                            
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
                                "instance_id": instance_file,
                                "seed": seed,
                                "best_objective": 1e10,
                                "runtime": 0,
                                "algorithm_name": algo_config[0],
                                "deadline": deadline,
                            })
                        
                        completed_runs += 1
                        self.progress_signal.emit(int(completed_runs / total_runs * 100))
            
            if results:
                df = pd.DataFrame(results)
                df = self._calculate_rpd(df)
                
                filename = datetime.now().strftime('%m%d_%H%M') + ".csv"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                df.to_csv(output_path, index=False)
                
                if not self._is_running:
                    self.finished_signal.emit(False, f"实验已终止！已完成 {completed_runs}/{total_runs} 次运行")
                else:
                    self.finished_signal.emit(True, f"实验完成！共运行 {total_runs} 次，结果已保存到 {output_path}")
            else:
                self.finished_signal.emit(False, "实验未完成任何运行")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"实验出错: {str(e)}")
        finally:
            self._is_running = False
    
    def stop(self):
        """停止实验"""
        self._is_running = False
    
    def _load_instances_from_subset(self, subset: str, count: int):
        """加载指定子集的实例"""
        import glob
        import re
        from src.psp.psplib_io import load_psplib_sm
        
        subset_path = os.path.join(project_root, "data", "psplib_raw", subset.lower())
        
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
    
    def _calculate_rpd(self, df):
        """计算RPD"""
        import numpy as np
        
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


class MLAnalysisRunner(QThread):
    """机器学习分析运行线程"""
    progress_signal = Signal(int)
    log_signal = Signal(str)
    result_signal = Signal(dict)
    finished_signal = Signal(bool, str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_running = False
    
    def run(self):
        """运行ML分析"""
        self._is_running = True
        
        try:
            import pandas as pd
            import numpy as np
            from scipy import stats as scipy_stats
            from scipy.stats import friedmanchisquare, wilcoxon
            
            self.log_signal.emit("正在加载实验结果...")
            self.progress_signal.emit(10)
            
            results_dir = project_root / "results" / "raw"
            csv_files = list(results_dir.glob("*.csv"))
            
            if not csv_files:
                self.finished_signal.emit(False, "没有找到实验结果文件！请先运行元启发式算法实验。")
                return
            
            latest_file = max(csv_files, key=os.path.getctime)
            self.log_signal.emit(f"加载结果文件: {latest_file.name}")
            
            results_df = pd.read_csv(latest_file)
            self.log_signal.emit(f"数据列: {list(results_df.columns)}")
            self.log_signal.emit(f"数据行数: {len(results_df)}")
            self.progress_signal.emit(20)
            
            stats_config = self.config.get('stats', {})
            anytime_config = self.config.get('anytime', {})
            
            results = {}
            
            if stats_config.get('layerA', {}).get('enabled', False):
                self.log_signal.emit("执行Layer A检验 (Friedman)...")
                
                pivot = results_df.pivot_table(
                    index='instance_id',
                    columns='algorithm_name',
                    values='best_objective',
                    aggfunc='median'
                )
                pivot = pivot.dropna()
                
                if len(pivot.columns) < 2:
                    self.log_signal.emit("警告: 需要至少2个算法才能进行Friedman检验")
                else:
                    algos = list(pivot.columns)
                    performance_data = [pivot[a].values for a in algos]
                    
                    stat, p_value = friedmanchisquare(*performance_data)
                    
                    ranks = pivot.rank(axis=1, ascending=True)
                    mean_ranks = ranks.mean()
                    
                    results['friedman'] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "n_instances": len(pivot),
                        "n_algorithms": len(algos),
                        "mean_ranks": mean_ranks.to_dict(),
                        "significant": p_value < 0.05
                    }
                    
                    self.log_signal.emit(f"Friedman检验: 统计量={stat:.4f}, p值={p_value:.4f}")
                    
                    pairwise_results = []
                    n = len(algos)
                    for i in range(n):
                        for j in range(i+1, n):
                            algo1, algo2 = algos[i], algos[j]
                            diff = pivot[algo1].values - pivot[algo2].values
                            
                            if np.all(diff == 0):
                                continue
                            
                            try:
                                w_stat, w_p = wilcoxon(diff, alternative='two-sided')
                                wins = int(np.sum(diff < 0))
                                losses = int(np.sum(diff > 0))
                                
                                pairwise_results.append({
                                    "algorithm_1": algo1,
                                    "algorithm_2": algo2,
                                    "statistic": float(w_stat),
                                    "p_value": float(w_p),
                                    "wins": wins,
                                    "losses": losses,
                                    "better": algo1 if wins > losses else algo2
                                })
                            except Exception:
                                continue
                    
                    if pairwise_results:
                        pw_df = pd.DataFrame(pairwise_results)
                        pw_df = pw_df.sort_values("p_value").reset_index(drop=True)
                        m = len(pw_df)
                        adjusted_p = []
                        for k, row in pw_df.iterrows():
                            adjusted = min(1.0, (m - k) * row["p_value"])
                            adjusted_p.append(adjusted)
                        pw_df["p_adjusted"] = adjusted_p
                        pw_df["significant"] = pw_df["p_adjusted"] < 0.05
                        results['pairwise'] = pw_df.to_dict('records')
                
                self.progress_signal.emit(50)
            
            if stats_config.get('layerB', {}).get('enabled', False):
                self.log_signal.emit("执行Layer B检验...")
                
                layerb_df = results_df.copy()
                
                if 'best_objective' in layerb_df.columns:
                    layerb_df['y'] = np.log1p(layerb_df['best_objective'])
                else:
                    self.log_signal.emit("警告: 没有找到best_objective列，跳过Layer B检验")
                    results['layerB_art'] = []
                    self.progress_signal.emit(70)
                
                if 'algorithm_name' in layerb_df.columns:
                    layerb_df['algorithm_name'] = layerb_df['algorithm_name'].astype('category')
                if 'instance_id' in layerb_df.columns:
                    layerb_df['instance_id'] = layerb_df['instance_id'].astype('category')
                
                method = stats_config.get('layerB', {}).get('method', 'art_anova')
                
                if method == 'art_anova':
                    try:
                        import statsmodels.formula.api as smf
                        from statsmodels.stats.anova import anova_lm
                        from scipy.stats import rankdata
                        
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
                        
                        results['layerB_art'] = rows
                        self.log_signal.emit(f"ART ANOVA完成: {len(rows)}个因子")
                        
                    except Exception as e:
                        self.log_signal.emit(f"ART ANOVA错误: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        results['layerB_art'] = []
                else:
                    self.log_signal.emit(f"Layer B方法 {method} 暂未实现，自动使用ART ANOVA")
                    try:
                        import statsmodels.formula.api as smf
                        from statsmodels.stats.anova import anova_lm
                        from scipy.stats import rankdata
                        
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
                        
                        results['layerB_art'] = rows
                        self.log_signal.emit(f"ART ANOVA完成: {len(rows)}个因子")
                        
                    except Exception as e:
                        self.log_signal.emit(f"ART ANOVA错误: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        results['layerB_art'] = []
                
                self.progress_signal.emit(70)
            
            if anytime_config.get('enabled', False):
                self.log_signal.emit("执行Anytime分析...")
                
                anytime_df = results_df.copy()
                anytime_df['best_obj'] = anytime_df['best_objective']
                anytime_df['algo_id'] = anytime_df['algorithm_name']
                
                file_stem = latest_file.stem
                output_dir = str(project_root / "results" / "anytime" / file_stem)
                os.makedirs(output_dir, exist_ok=True)
                self.log_signal.emit(f"输出目录: {output_dir}")
                
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    from matplotlib import rcParams
                    
                    rcParams['font.family'] = 'serif'
                    rcParams['font.serif'] = ['Times New Roman']
                    rcParams['mathtext.fontset'] = 'stix'
                    
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
                    
                    source_data_path = os.path.join(output_dir, "data_profile_source.csv")
                    df.to_csv(source_data_path, index=False)
                    self.log_signal.emit(f"源数据已保存到: {source_data_path}")
                    
                    ratios = np.linspace(1.0, 2.0, 100)
                    profiles = {}
                    
                    for algo in df["algo_id"].unique():
                        algo_ratios = df[df["algo_id"] == algo]["ratio"].values
                        profile = []
                        for r in ratios:
                            profile.append(np.mean(algo_ratios <= r))
                        profiles[algo] = profile
                    
                    profile_data = {"ratio": ratios}
                    for algo, profile in profiles.items():
                        profile_data[algo] = profile
                    profile_df = pd.DataFrame(profile_data)
                    profile_df.to_csv(os.path.join(output_dir, "data_profile_curve.csv"), index=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for algo, profile in profiles.items():
                        ax.plot(ratios, profile, label=algo, linewidth=2)
                    ax.set_xlabel("Ratio to best known", fontsize=12)
                    ax.set_ylabel("Fraction of instances", fontsize=12)
                    ax.set_title("Data Profile", fontsize=14)
                    ax.legend(loc='lower right', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(1.0, 2.0)
                    ax.set_ylim(0, 1.05)
                    plt.tight_layout()
                    
                    svg_path = os.path.join(output_dir, "data_profile.svg")
                    plt.savefig(svg_path, format='svg', dpi=300)
                    plt.close()
                    
                    self.log_signal.emit(f"Data Profile已保存到: {svg_path}")
                    
                    anytime_analysis = self._analyze_anytime_results(df, profiles, ratios)
                    results['anytime_analysis'] = anytime_analysis
                    results['anytime_output'] = output_dir
                    
                except Exception as e:
                    self.log_signal.emit(f"Anytime分析错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                self.progress_signal.emit(90)
            
            ml_config = self.config.get('ml', {})
            if ml_config.get('enabled', False):
                self.log_signal.emit("执行机器学习算法选择分析...")
                
                try:
                    from src.analysis import AlgorithmSelector, analyze_selector
                    from src.psp.features import FeatureExtractor, extract_features_batch
                    from src.psp.psplib_io import load_psplib_sm
                    
                    valid_df = results_df[results_df['best_objective'] < 1e9].copy()
                    
                    if valid_df.empty:
                        self.log_signal.emit("警告: 没有有效的实验数据")
                    else:
                        n_instances = valid_df['instance_id'].nunique()
                        n_algorithms = valid_df['algorithm_name'].nunique()
                        self.log_signal.emit(f"实例数量: {n_instances}, 算法数量: {n_algorithms}")
                        
                        if n_instances < 5:
                            self.log_signal.emit("警告: 实例数量太少，建议至少5个实例")
                        else:
                            self.log_signal.emit("正在提取实例特征...")
                            
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
                                    test_path = os.path.join(project_root, "data", "psplib_raw", subset, instance_id_clean + ".sm")
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
                                        self.log_signal.emit(f"  无法提取实例 {instance_id} 的特征: {e}")
                            
                            if len(feature_list) == 0:
                                self.log_signal.emit("警告: 无法提取任何实例的特征")
                            else:
                                feature_df = pd.DataFrame(feature_list)
                                self.log_signal.emit(f"成功提取 {len(feature_list)} 个实例的特征")
                                
                                model_type = ml_config.get('model_type', 'random_forest')
                                test_size = ml_config.get('test_size', 0.3)
                                random_state = ml_config.get('random_state', 42)
                                
                                self.log_signal.emit(f"训练模型: {model_type}, 测试集比例: {test_size}")
                                
                                selector_analysis = ml_config.get('selector_analysis', {})
                                interpretability = ml_config.get('interpretability', {})
                                
                                ml_results = analyze_selector(
                                    perf_df=valid_df,
                                    feature_df=feature_df,
                                    model_type=model_type,
                                    instance_col='instance_id',
                                    algo_col='algorithm_name',
                                    perf_col='best_objective',
                                    test_size=test_size,
                                    random_state=random_state,
                                    calc_sbs=selector_analysis.get('calc_sbs', True),
                                    calc_vbs=selector_analysis.get('calc_vbs', True),
                                    calc_selector=selector_analysis.get('calc_selector', True),
                                    calc_winner_hit=selector_analysis.get('calc_winner_hit', True),
                                    calc_penalty=selector_analysis.get('calc_penalty', True),
                                    calc_risk_lambda=selector_analysis.get('calc_risk_lambda', False),
                                    feature_importance=interpretability.get('feature_importance', True),
                                    shap=interpretability.get('shap', True),
                                    perm_importance=interpretability.get('perm_importance', True),
                                    isa=interpretability.get('isa', True)
                                )
                                
                                ml_output_dir = str(project_root / "results" / "ml" / file_stem)
                                os.makedirs(ml_output_dir, exist_ok=True)
                                
                                feature_df.to_csv(os.path.join(ml_output_dir, "feature_data.csv"), index=False)
                                valid_df.to_csv(os.path.join(ml_output_dir, "performance_data.csv"), index=False)
                                
                                if 'feature_importance' in ml_results and ml_results['feature_importance'] is not None:
                                    ml_results['feature_importance'].to_csv(
                                        os.path.join(ml_output_dir, "feature_importance.csv"), index=False
                                    )
                                
                                if 'embedding' in ml_results:
                                    ml_results['embedding'].to_csv(
                                        os.path.join(ml_output_dir, "instance_embedding.csv"), index=False
                                    )
                                
                                if 'figures' in ml_results:
                                    for fig_name, fig in ml_results['figures'].items():
                                        if fig is not None:
                                            fig.savefig(os.path.join(ml_output_dir, f"{fig_name}.svg"), format='svg', dpi=300)
                                            plt.close(fig)
                                
                                results['ml_results'] = {
                                    'SBS': ml_results.get('SBS', {}),
                                    'VBS': ml_results.get('VBS', {}),
                                    'Selector': ml_results.get('Selector', {}),
                                    'feature_importance': ml_results.get('feature_importance'),
                                    'shap_values': ml_results.get('shap_values'),
                                    'permutation_importance': ml_results.get('permutation_importance'),
                                    'isa_results': ml_results.get('isa_results'),
                                    'output_dir': ml_output_dir
                                }
                                
                                self.log_signal.emit(f"ML分析完成，结果保存到: {ml_output_dir}")
                                
                except Exception as e:
                    self.log_signal.emit(f"机器学习分析错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                self.progress_signal.emit(95)
            
            self.progress_signal.emit(100)
            self.result_signal.emit(results)
            
            self.finished_signal.emit(True, "ML分析完成！结果已显示在界面上。")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"ML分析出错: {str(e)}")
        finally:
            self._is_running = False
    
    def stop(self):
        """停止分析"""
        self._is_running = False
    
    def _analyze_anytime_results(self, df, profiles, ratios) -> dict:
        """分析Anytime结果"""
        import numpy as np
        
        analysis = {}
        
        algo_stats = []
        for algo in df["algo_id"].unique():
            algo_data = df[df["algo_id"] == algo]
            
            mean_ratio = algo_data["ratio"].mean()
            median_ratio = algo_data["ratio"].median()
            std_ratio = algo_data["ratio"].std()
            
            perfect_count = int((algo_data["ratio"] == 1.0).sum())
            total_count = len(algo_data)
            perfect_rate = perfect_count / total_count if total_count > 0 else 0
            
            within_5pct = int((algo_data["ratio"] <= 1.05).sum())
            within_10pct = int((algo_data["ratio"] <= 1.10).sum())
            
            area_under_curve = np.trapz(profiles[algo], ratios)
            
            algo_stats.append({
                "algorithm": algo,
                "mean_ratio": mean_ratio,
                "median_ratio": median_ratio,
                "std_ratio": std_ratio,
                "perfect_count": perfect_count,
                "total_count": total_count,
                "perfect_rate": perfect_rate,
                "within_5pct": within_5pct,
                "within_10pct": within_10pct,
                "auc": area_under_curve
            })
        
        algo_stats.sort(key=lambda x: x["auc"], reverse=True)
        
        best_algo = algo_stats[0]["algorithm"] if algo_stats else None
        worst_algo = algo_stats[-1]["algorithm"] if algo_stats else None
        
        summary_lines = []
        summary_lines.append(f"【Anytime分析结果】")
        summary_lines.append("")
        
        if best_algo:
            best_stats = algo_stats[0]
            summary_lines.append(f"★ 最佳算法: {best_algo}")
            summary_lines.append(f"  - 平均比率: {best_stats['mean_ratio']:.4f}")
            summary_lines.append(f"  - 达到最优解的实例数: {best_stats['perfect_count']}/{best_stats['total_count']} ({best_stats['perfect_rate']*100:.1f}%)")
            summary_lines.append(f"  - 在5%误差内的实例数: {best_stats['within_5pct']}/{best_stats['total_count']}")
            summary_lines.append(f"  - 曲线下面积(AUC): {best_stats['auc']:.4f}")
        
        summary_lines.append("")
        summary_lines.append("【算法排名 (按AUC降序)】")
        for i, stats in enumerate(algo_stats, 1):
            summary_lines.append(f"  {i}. {stats['algorithm']}: AUC={stats['auc']:.4f}, 平均比率={stats['mean_ratio']:.4f}")
        
        if len(algo_stats) >= 2:
            best_auc = algo_stats[0]["auc"]
            second_auc = algo_stats[1]["auc"]
            improvement = (best_auc - second_auc) / second_auc * 100 if second_auc > 0 else 0
            summary_lines.append("")
            summary_lines.append(f"【结论】")
            summary_lines.append(f"  {best_algo}算法表现最佳，相比第二名{algo_stats[1]['algorithm']}提升了{improvement:.2f}%的AUC。")
            
            if algo_stats[0]["perfect_rate"] > 0.5:
                summary_lines.append(f"  {best_algo}在{algo_stats[0]['perfect_rate']*100:.1f}%的实例上找到了最优解，显示出强大的搜索能力。")
            elif algo_stats[0]["mean_ratio"] < 1.05:
                summary_lines.append(f"  {best_algo}的平均结果仅比最优解高{(algo_stats[0]['mean_ratio']-1)*100:.2f}%，表现出良好的稳定性。")
        
        analysis["summary"] = "\n".join(summary_lines)
        analysis["algorithm_stats"] = algo_stats
        analysis["best_algorithm"] = best_algo
        analysis["worst_algorithm"] = worst_algo
        
        return analysis


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 5px 15px;
            border: 1px solid #999999;
            border-radius: 3px;
            background-color: #e0e0e0;
        }
        QPushButton:hover {
            background-color: #d0d0d0;
        }
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 3px;
            background-color: #fafafa;
            font-family: monospace;
        }
        QProgressBar {
            border: 1px solid #cccccc;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            border-radius: 5px;
        }
        QTabBar::tab {
            padding: 8px 20px;
            margin-right: 2px;
            border: 1px solid #cccccc;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: white;
            font-weight: bold;
        }
        QTabBar::tab:!selected {
            background-color: #e0e0e0;
        }
        QScrollArea {
            border: none;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
