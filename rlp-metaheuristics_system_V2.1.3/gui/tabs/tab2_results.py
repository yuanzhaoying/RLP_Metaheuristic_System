"""
Tab 2: Results Analysis
结果分析选项卡
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QGroupBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFrame, QSplitter, QTabWidget, QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent.parent


class MplCanvas(FigureCanvas):
    """Matplotlib画布"""
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def clear(self):
        """清除图形"""
        self.axes.clear()
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)


class ResultsTab(QWidget):
    """结果分析选项卡"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_df = None
        self.results_file = None
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(15)
        
        self._create_overview_group(container_layout)
        
        self._create_statistics_group(container_layout)
        
        self._create_charts_group(container_layout)
        
        self._create_significance_group(container_layout)
        
        self._create_feasibility_group(container_layout)
        
        self._create_data_preview_group(container_layout)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        self._show_empty_state()
    
    def _create_overview_group(self, parent_layout):
        """创建数据概览组"""
        group = QGroupBox("数据概览")
        layout = QGridLayout(group)
        
        self.source_label = QLabel("-")
        self.source_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(QLabel("数据来源:"), 0, 0)
        layout.addWidget(self.source_label, 0, 1)
        
        self.total_label = QLabel("-")
        layout.addWidget(QLabel("总记录数:"), 0, 2)
        layout.addWidget(self.total_label, 0, 3)
        
        self.valid_label = QLabel("-")
        layout.addWidget(QLabel("有效记录数:"), 1, 0)
        layout.addWidget(self.valid_label, 1, 1)
        
        self.feasible_label = QLabel("-")
        layout.addWidget(QLabel("可行解比例:"), 1, 2)
        layout.addWidget(self.feasible_label, 1, 3)
        
        parent_layout.addWidget(group)
    
    def _create_statistics_group(self, parent_layout):
        """创建统计汇总组"""
        group = QGroupBox("统计汇总表格")
        layout = QVBoxLayout(group)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels(["算法", "Mean", "Std", "Min", "Max", "Rank"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setMinimumHeight(150)
        layout.addWidget(self.stats_table)
        
        self.conclusion_label = QLabel("请运行实验或加载结果文件以查看分析")
        self.conclusion_label.setWordWrap(True)
        self.conclusion_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.conclusion_label)
        
        parent_layout.addWidget(group)
    
    def _create_charts_group(self, parent_layout):
        """创建图表组"""
        group = QGroupBox("可视化分析")
        layout = QVBoxLayout(group)
        
        chart_tabs = QTabWidget()
        
        self.perf_canvas = MplCanvas(self, width=10, height=5)
        chart_tabs.addTab(self.perf_canvas, "性能剖面图")
        
        self.anytime_canvas = MplCanvas(self, width=10, height=5)
        chart_tabs.addTab(self.anytime_canvas, "Anytime曲线")
        
        self.rank_canvas = MplCanvas(self, width=10, height=5)
        chart_tabs.addTab(self.rank_canvas, "算法排名")
        
        layout.addWidget(chart_tabs)
        
        btn_layout = QHBoxLayout()
        self.save_perf_btn = QPushButton("保存性能剖面图")
        self.save_perf_btn.clicked.connect(lambda: self._save_chart(self.perf_canvas, "performance_profile"))
        btn_layout.addWidget(self.save_perf_btn)
        
        self.save_anytime_btn = QPushButton("保存Anytime曲线")
        self.save_anytime_btn.clicked.connect(lambda: self._save_chart(self.anytime_canvas, "anytime_curve"))
        btn_layout.addWidget(self.save_anytime_btn)
        
        self.save_rank_btn = QPushButton("保存排名图")
        self.save_rank_btn.clicked.connect(lambda: self._save_chart(self.rank_canvas, "algorithm_ranking"))
        btn_layout.addWidget(self.save_rank_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        parent_layout.addWidget(group)
    
    def _create_significance_group(self, parent_layout):
        """创建显著性检验组"""
        group = QGroupBox("显著性检验")
        layout = QGridLayout(group)
        
        layout.addWidget(QLabel("Friedman统计量:"), 0, 0)
        self.friedman_stat_label = QLabel("-")
        layout.addWidget(self.friedman_stat_label, 0, 1)
        
        layout.addWidget(QLabel("p值:"), 0, 2)
        self.friedman_p_label = QLabel("-")
        layout.addWidget(self.friedman_p_label, 0, 3)
        
        layout.addWidget(QLabel("结论:"), 0, 4)
        self.friedman_conclusion_label = QLabel("-")
        layout.addWidget(self.friedman_conclusion_label, 0, 5)
        
        layout.addWidget(QLabel("Wilcoxon成对检验:"), 1, 0, 1, 2)
        
        self.wilcoxon_table = QTableWidget()
        self.wilcoxon_table.setColumnCount(4)
        self.wilcoxon_table.setHorizontalHeaderLabels(["算法对", "统计量", "p值", "显著"])
        self.wilcoxon_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.wilcoxon_table.setMaximumHeight(150)
        layout.addWidget(self.wilcoxon_table, 2, 0, 1, 6)
        
        parent_layout.addWidget(group)
    
    def _create_feasibility_group(self, parent_layout):
        """创建可行性分析组"""
        group = QGroupBox("可行性与计算成本分析")
        layout = QGridLayout(group)
        
        self.feasible_ratio_label = QLabel("-")
        layout.addWidget(QLabel("可行解比例:"), 0, 0)
        layout.addWidget(self.feasible_ratio_label, 0, 1)
        
        self.infeasible_ratio_label = QLabel("-")
        layout.addWidget(QLabel("不可行评估比例:"), 0, 2)
        layout.addWidget(self.infeasible_ratio_label, 0, 3)
        
        self.avg_runtime_label = QLabel("-")
        layout.addWidget(QLabel("平均运行时间:"), 1, 0)
        layout.addWidget(self.avg_runtime_label, 1, 1)
        
        parent_layout.addWidget(group)
    
    def _create_data_preview_group(self, parent_layout):
        """创建数据预览组"""
        group = QGroupBox("原始数据预览")
        layout = QVBoxLayout(group)
        
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(["实例", "种子", "算法", "目标值", "运行时间", "RPD"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setMinimumHeight(150)
        layout.addWidget(self.data_table)
        
        parent_layout.addWidget(group)
    
    def _show_empty_state(self):
        """显示空状态"""
        self.source_label.setText("-")
        self.total_label.setText("-")
        self.valid_label.setText("-")
        self.feasible_label.setText("-")
        self.feasible_ratio_label.setText("-")
        self.infeasible_ratio_label.setText("-")
        self.avg_runtime_label.setText("-")
        self.stats_table.setRowCount(0)
        self.wilcoxon_table.setRowCount(0)
        self.data_table.setRowCount(0)
        self.perf_canvas.clear()
        self.anytime_canvas.clear()
        self.rank_canvas.clear()
    
    def set_results(self, df, file_path: str):
        """设置结果数据"""
        self.results_df = df
        self.results_file = file_path
        
        if df is None or df.empty:
            self._show_empty_state()
            return
        
        self._calculate_rpd()
        self._update_overview()
        self._update_statistics()
        self._update_charts()
        self._update_significance()
        self._update_feasibility()
        self._update_data_preview()
    
    def _calculate_rpd(self):
        """计算RPD"""
        if self.results_df is None:
            return
        
        df = self.results_df
        valid_df = df[df['best_objective'] < 1e9].copy()
        
        if valid_df.empty:
            return
        
        best_per_instance = valid_df.groupby('instance_id')['best_objective'].min()
        
        def get_rpd(row):
            if row['best_objective'] >= 1e9:
                return np.nan
            best = best_per_instance.get(row['instance_id'], row['best_objective'])
            if best == 0:
                return 0
            return ((row['best_objective'] - best) / best) * 100
        
        if 'rpd' not in df.columns:
            df['rpd'] = df.apply(get_rpd, axis=1)
    
    def _update_overview(self):
        """更新数据概览"""
        df = self.results_df
        
        if df is None:
            return
        
        self.source_label.setText(os.path.basename(self.results_file) if self.results_file else "-")
        self.total_label.setText(str(len(df)))
        
        valid_count = len(df[df['best_objective'] < 1e9])
        self.valid_label.setText(str(valid_count))
        
        feasible_ratio = valid_count / len(df) * 100 if len(df) > 0 else 0
        self.feasible_label.setText(f"{feasible_ratio:.1f}%")
    
    def _update_statistics(self):
        """更新统计汇总"""
        df = self.results_df
        
        if df is None:
            return
        
        valid_df = df[df['best_objective'] < 1e9].copy()
        
        if valid_df.empty:
            return
        
        try:
            from src.analysis import compute_statistics
            
            summary, perf_matrix = compute_statistics(
                valid_df,
                instance_col="instance_id",
                algo_col="algorithm_name",
                perf_col="best_objective"
            )
            
            if summary is not None and len(summary) > 0:
                self.stats_table.setRowCount(len(summary))
                
                for i, (algo, row) in enumerate(summary.iterrows()):
                    self.stats_table.setItem(i, 0, QTableWidgetItem(str(algo)))
                    self.stats_table.setItem(i, 1, QTableWidgetItem(f"{row.get('Mean', 0):.2f}"))
                    self.stats_table.setItem(i, 2, QTableWidgetItem(f"{row.get('Std', 0):.2f}"))
                    self.stats_table.setItem(i, 3, QTableWidgetItem(f"{row.get('Min', 0):.2f}"))
                    self.stats_table.setItem(i, 4, QTableWidgetItem(f"{row.get('Max', 0):.2f}"))
                    self.stats_table.setItem(i, 5, QTableWidgetItem(f"{row.get('Rank', 0):.2f}"))
                
                best_mean_algo = summary['Mean'].idxmin()
                best_mean_value = summary['Mean'].min()
                
                self.conclusion_label.setText(
                    f"最优均值算法: {best_mean_algo} (Mean = {best_mean_value:.2f})"
                )
                self.conclusion_label.setStyleSheet("color: green; font-weight: bold;")
        
        except Exception as e:
            self.conclusion_label.setText(f"计算统计数据时出错: {e}")
            self.conclusion_label.setStyleSheet("color: red;")
    
    def _update_charts(self):
        """更新图表"""
        df = self.results_df
        
        if df is None:
            return
        
        valid_df = df[df['best_objective'] < 1e9].copy()
        
        if valid_df.empty:
            return
        
        try:
            from src.analysis import plot_performance_profile, plot_anytime_curve, plot_rank_comparison
            
            if valid_df.duplicated(subset=['instance_id', 'algorithm_name']).any():
                perf_df_agg = valid_df.groupby(['instance_id', 'algorithm_name'])['best_objective'].min().reset_index()
            else:
                perf_df_agg = valid_df[['instance_id', 'algorithm_name', 'best_objective']]
            
            perf_matrix = perf_df_agg.pivot(
                index='instance_id',
                columns='algorithm_name',
                values='best_objective'
            )
            
            self.perf_canvas.clear()
            fig_pp = plot_performance_profile(perf_matrix, figsize=(10, 5))
            if fig_pp is not None:
                self.perf_canvas.fig = fig_pp
                self.perf_canvas.axes = fig_pp.axes[0]
                self.perf_canvas.draw()
            
            self.anytime_canvas.clear()
            time_df = valid_df[['algorithm_name', 'runtime', 'best_objective']].copy()
            time_df.columns = ['algo', 'time', 'best_so_far']
            fig_anytime = plot_anytime_curve(time_df, time_col='time', perf_col='best_so_far', algo_col='algo', figsize=(10, 5))
            if fig_anytime is not None:
                self.anytime_canvas.fig = fig_anytime
                self.anytime_canvas.axes = fig_anytime.axes[0]
                self.anytime_canvas.draw()
            
        except Exception as e:
            print(f"更新图表时出错: {e}")
    
    def _update_significance(self):
        """更新显著性检验"""
        df = self.results_df
        
        if df is None:
            return
        
        valid_df = df[df['best_objective'] < 1e9].copy()
        
        if valid_df.empty:
            return
        
        try:
            from src.analysis import perform_statistical_tests
            
            if valid_df.duplicated(subset=['instance_id', 'algorithm_name']).any():
                perf_df_agg = valid_df.groupby(['instance_id', 'algorithm_name'])['best_objective'].min().reset_index()
            else:
                perf_df_agg = valid_df[['instance_id', 'algorithm_name', 'best_objective']]
            
            perf_matrix = perf_df_agg.pivot(
                index='instance_id',
                columns='algorithm_name',
                values='best_objective'
            )
            
            stats_results = perform_statistical_tests(perf_matrix, alpha=0.05)
            
            if 'friedman' in stats_results:
                friedman = stats_results['friedman']
                self.friedman_stat_label.setText(f"{friedman['statistic']:.4f}" if friedman['statistic'] else "N/A")
                self.friedman_p_label.setText(f"{friedman['p_value']:.6f}" if friedman['p_value'] else "N/A")
                
                if friedman['significant']:
                    self.friedman_conclusion_label.setText("存在显著差异 (p < 0.05)")
                    self.friedman_conclusion_label.setStyleSheet("color: green;")
                else:
                    self.friedman_conclusion_label.setText("无显著差异 (p ≥ 0.05)")
                    self.friedman_conclusion_label.setStyleSheet("color: orange;")
            
            if 'wilcoxon' in stats_results and len(stats_results['wilcoxon']) > 0:
                wilcoxon_df = stats_results['wilcoxon']
                self.wilcoxon_table.setRowCount(len(wilcoxon_df))
                
                for i, (idx, row) in enumerate(wilcoxon_df.iterrows()):
                    self.wilcoxon_table.setItem(i, 0, QTableWidgetItem(str(row.get('Comparison', ''))))
                    self.wilcoxon_table.setItem(i, 1, QTableWidgetItem(f"{row.get('Statistic', 0):.4f}"))
                    self.wilcoxon_table.setItem(i, 2, QTableWidgetItem(f"{row.get('P-value', 0):.6f}"))
                    
                    significant = row.get('Significant', False)
                    sig_item = QTableWidgetItem("是" if significant else "否")
                    if significant:
                        sig_item.setForeground(Qt.green)
                    self.wilcoxon_table.setItem(i, 3, sig_item)
            
        except Exception as e:
            self.friedman_conclusion_label.setText(f"检验出错: {e}")
            self.friedman_conclusion_label.setStyleSheet("color: red;")
    
    def _update_feasibility(self):
        """更新可行性分析"""
        df = self.results_df
        
        if df is None:
            return
        
        try:
            from src.analysis import compute_feasibility_analysis
            
            feasibility_df = compute_feasibility_analysis(df, objective_col='best_objective', threshold=1e9)
            
            if feasibility_df is not None and len(feasibility_df) > 0:
                avg_feasible = feasibility_df['Feasible Rate (%)'].mean()
                avg_infeasible = feasibility_df['Infeasible Rate (%)'].mean()
                avg_runtime = feasibility_df['Avg Runtime (s)'].mean()
                
                self.feasible_ratio_label.setText(f"{avg_feasible:.1f}%")
                self.infeasible_ratio_label.setText(f"{avg_infeasible:.1f}%")
                self.avg_runtime_label.setText(f"{avg_runtime:.3f}s")
        
        except Exception as e:
            print(f"更新可行性分析时出错: {e}")
    
    def _update_data_preview(self):
        """更新数据预览"""
        df = self.results_df
        
        if df is None:
            return
        
        valid_df = df[df['best_objective'] < 1e9].copy()
        
        if valid_df.empty:
            return
        
        display_cols = ['instance_id', 'seed', 'algorithm_name', 'best_objective', 'runtime', 'rpd']
        display_cols = [c for c in display_cols if c in valid_df.columns]
        
        preview_df = valid_df[display_cols].head(50)
        
        self.data_table.setRowCount(len(preview_df))
        self.data_table.setColumnCount(len(display_cols))
        self.data_table.setHorizontalHeaderLabels(display_cols)
        
        for i, (idx, row) in enumerate(preview_df.iterrows()):
            for j, col in enumerate(display_cols):
                value = row[col]
                if isinstance(value, float):
                    self.data_table.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))
                else:
                    self.data_table.setItem(i, j, QTableWidgetItem(str(value)))
    
    def _save_chart(self, canvas: MplCanvas, name: str):
        """保存图表"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", f"{name}.svg", "SVG Files (*.svg);;PNG Files (*.png);;All Files (*)"
        )
        if file_path:
            try:
                canvas.fig.savefig(file_path, bbox_inches='tight', dpi=150)
                QMessageBox.information(self, "成功", f"图表已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {}
    
    def load_config(self, config: Dict[str, Any]):
        """加载配置"""
        pass
