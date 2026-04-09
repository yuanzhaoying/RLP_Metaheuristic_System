"""
Tab 3: Algorithm Selector
算法选择选项卡
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
    QFrame, QSplitter, QTabWidget, QFileDialog, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread
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


class SelectorWorker(QThread):
    """算法选择工作线程"""
    progress_signal = Signal(str)
    finished_signal = Signal(bool, object, str)
    
    def __init__(self, df, ml_model, test_size, use_tsne):
        super().__init__()
        self.df = df
        self.ml_model = ml_model
        self.test_size = test_size
        self.use_tsne = use_tsne
        self._is_running = True
    
    def run(self):
        """运行算法选择分析"""
        try:
            import pandas as pd
            from src.psp.psplib_io import load_psplib_sm
            from src.psp.features import FeatureExtractor
            from src.analysis import analyze_selector
            
            self.progress_signal.emit("正在提取实例特征...")
            
            valid_df = self.df[self.df['best_objective'] < 1e9].copy()
            
            if valid_df.empty:
                self.finished_signal.emit(False, None, "没有有效的实验数据")
                return
            
            instances = valid_df['instance_id'].unique()
            
            feature_list = []
            instance_files = []
            
            for instance_id in instances:
                if not self._is_running:
                    self.finished_signal.emit(False, None, "操作已取消")
                    return
                
                instance_id_clean = instance_id.replace('.RCP', '').replace('.rcp', '')
                if instance_id.upper().startswith('J30'):
                    instance_file = os.path.join(PROJECT_ROOT, "data", "psplib_raw", "j30", f"{instance_id_clean}.RCP")
                elif instance_id.upper().startswith('J60'):
                    instance_file = os.path.join(PROJECT_ROOT, "data", "psplib_raw", "j60", f"{instance_id_clean}.RCP")
                elif instance_id.upper().startswith('J90'):
                    instance_file = os.path.join(PROJECT_ROOT, "data", "psplib_raw", "j90", f"{instance_id_clean}.RCP")
                elif instance_id.upper().startswith('J120'):
                    instance_file = os.path.join(PROJECT_ROOT, "data", "psplib_raw", "j120", f"{instance_id_clean}.RCP")
                else:
                    continue
                
                if os.path.exists(instance_file):
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
                        instance_files.append(instance_id)
                    except Exception as e:
                        self.progress_signal.emit(f"无法提取实例 {instance_id} 的特征: {e}")
            
            if len(feature_list) == 0:
                self.finished_signal.emit(False, None, "无法提取任何实例的特征")
                return
            
            feature_df = pd.DataFrame(feature_list)
            
            self.progress_signal.emit(f"成功提取 {len(feature_list)} 个实例的特征")
            self.progress_signal.emit("正在训练算法选择器...")
            
            results = analyze_selector(
                perf_df=valid_df,
                feature_df=feature_df,
                model_type=self.ml_model,
                instance_col='instance_id',
                algo_col='algorithm_name',
                perf_col='best_objective',
                test_size=self.test_size,
                random_state=42,
                use_tsne=self.use_tsne
            )
            
            self.finished_signal.emit(True, results, "算法选择分析完成")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, None, f"分析出错: {e}")
    
    def stop(self):
        """停止分析"""
        self._is_running = False


class SelectorTab(QWidget):
    """算法选择选项卡"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_df = None
        self.results_file = None
        self.selector_results = None
        self.worker = None
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
        
        self._create_info_group(container_layout)
        
        self._create_config_group(container_layout)
        
        self._create_benchmark_group(container_layout)
        
        self._create_metrics_group(container_layout)
        
        self._create_charts_group(container_layout)
        
        self._create_conclusion_group(container_layout)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        self._show_empty_state()
    
    def _create_info_group(self, parent_layout):
        """创建信息组"""
        group = QGroupBox("算法选择模块说明")
        layout = QVBoxLayout(group)
        
        info_label = QLabel("""
本模块旨在解决"按实例动态选择最优算法"的监督学习问题。通过机器学习方法，根据每个问题实例的特征，预测最适合该实例的算法，从而尽可能接近理论最优的虚拟最佳算法（VBS）。

<b>四个关键阶段:</b>
1. <b>数据构建</b>: 基于实验结果构建性能矩阵和特征数据集
2. <b>模型训练</b>: 使用机器学习模型训练算法选择器
3. <b>性能评估</b>: 与SBS和VBS对比，评估选择器性能
4. <b>解释性分析</b>: 特征重要性分析和实例空间可视化
        """)
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        layout.addWidget(info_label)
        
        parent_layout.addWidget(group)
    
    def _create_config_group(self, parent_layout):
        """创建配置组"""
        group = QGroupBox("训练配置")
        layout = QGridLayout(group)
        
        layout.addWidget(QLabel("机器学习模型:"), 0, 0)
        self.ml_model_combo = QComboBox()
        self.ml_model_combo.addItems([
            "decision_tree", "random_forest", "gradient_boosting",
            "svm", "knn"
        ])
        layout.addWidget(self.ml_model_combo, 0, 1)
        
        layout.addWidget(QLabel("测试集比例(%):"), 0, 2)
        self.test_size_spin = QSpinBox()
        self.test_size_spin.setRange(10, 50)
        self.test_size_spin.setValue(30)
        self.test_size_spin.setSingleStep(5)
        layout.addWidget(self.test_size_spin, 0, 3)
        
        self.use_tsne_check = QCheckBox("使用t-SNE降维")
        self.use_tsne_check.setToolTip("使用t-SNE代替PCA进行降维（适用于非线性结构）")
        layout.addWidget(self.use_tsne_check, 1, 0, 1, 2)
        
        self.run_selector_btn = QPushButton("运行算法选择分析")
        self.run_selector_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_selector_btn.clicked.connect(self._run_selector_analysis)
        layout.addWidget(self.run_selector_btn, 1, 2, 1, 2)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar, 2, 0, 1, 4)
        
        self.status_label = QLabel("请先运行实验或加载结果文件")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label, 3, 0, 1, 4)
        
        parent_layout.addWidget(group)
    
    def _create_benchmark_group(self, parent_layout):
        """创建性能基准组"""
        group = QGroupBox("性能基准对比")
        layout = QGridLayout(group)
        
        self.sbs_label = QLabel("-")
        self.sbs_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(QLabel("SBS (单最佳求解器):"), 0, 0)
        layout.addWidget(self.sbs_label, 0, 1)
        
        self.selector_label = QLabel("-")
        self.selector_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.selector_label.setStyleSheet("color: blue;")
        layout.addWidget(QLabel("Selector (算法选择器):"), 1, 0)
        layout.addWidget(self.selector_label, 1, 1)
        
        self.vbs_label = QLabel("-")
        self.vbs_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.vbs_label.setStyleSheet("color: green;")
        layout.addWidget(QLabel("VBS (虚拟最佳求解器):"), 2, 0)
        layout.addWidget(self.vbs_label, 2, 1)
        
        parent_layout.addWidget(group)
    
    def _create_metrics_group(self, parent_layout):
        """创建性能指标组"""
        group = QGroupBox("选择器性能指标")
        layout = QGridLayout(group)
        
        layout.addWidget(QLabel("命中率:"), 0, 0)
        self.hit_rate_label = QLabel("-")
        layout.addWidget(self.hit_rate_label, 0, 1)
        
        layout.addWidget(QLabel("平均Regret:"), 0, 2)
        self.avg_regret_label = QLabel("-")
        layout.addWidget(self.avg_regret_label, 0, 3)
        
        layout.addWidget(QLabel("P90 Penalty:"), 1, 0)
        self.p90_penalty_label = QLabel("-")
        layout.addWidget(self.p90_penalty_label, 1, 1)
        
        layout.addWidget(QLabel("相比SBS改进:"), 1, 2)
        self.improvement_label = QLabel("-")
        layout.addWidget(self.improvement_label, 1, 3)
        
        parent_layout.addWidget(group)
    
    def _create_charts_group(self, parent_layout):
        """创建图表组"""
        group = QGroupBox("可视化分析")
        layout = QVBoxLayout(group)
        
        chart_tabs = QTabWidget()
        
        self.perf_comp_canvas = MplCanvas(self, width=10, height=5)
        chart_tabs.addTab(self.perf_comp_canvas, "性能比较")
        
        self.feature_imp_canvas = MplCanvas(self, width=10, height=5)
        chart_tabs.addTab(self.feature_imp_canvas, "特征重要性")
        
        self.instance_space_canvas = MplCanvas(self, width=10, height=5)
        chart_tabs.addTab(self.instance_space_canvas, "实例空间分析")
        
        layout.addWidget(chart_tabs)
        
        btn_layout = QHBoxLayout()
        self.save_perf_comp_btn = QPushButton("保存性能比较图")
        self.save_perf_comp_btn.clicked.connect(lambda: self._save_chart(self.perf_comp_canvas, "performance_comparison"))
        btn_layout.addWidget(self.save_perf_comp_btn)
        
        self.save_feature_btn = QPushButton("保存特征重要性图")
        self.save_feature_btn.clicked.connect(lambda: self._save_chart(self.feature_imp_canvas, "feature_importance"))
        btn_layout.addWidget(self.save_feature_btn)
        
        self.save_instance_btn = QPushButton("保存实例空间图")
        self.save_instance_btn.clicked.connect(lambda: self._save_chart(self.instance_space_canvas, "instance_space"))
        btn_layout.addWidget(self.save_instance_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        parent_layout.addWidget(group)
    
    def _create_conclusion_group(self, parent_layout):
        """创建结论组"""
        group = QGroupBox("结论与建议")
        layout = QVBoxLayout(group)
        
        self.conclusion_label = QLabel("请运行算法选择分析以查看结论")
        self.conclusion_label.setWordWrap(True)
        self.conclusion_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.conclusion_label)
        
        parent_layout.addWidget(group)
    
    def _show_empty_state(self):
        """显示空状态"""
        self.sbs_label.setText("-")
        self.selector_label.setText("-")
        self.vbs_label.setText("-")
        self.hit_rate_label.setText("-")
        self.avg_regret_label.setText("-")
        self.p90_penalty_label.setText("-")
        self.improvement_label.setText("-")
        self.perf_comp_canvas.clear()
        self.feature_imp_canvas.clear()
        self.instance_space_canvas.clear()
    
    def set_results(self, df, file_path: str):
        """设置结果数据"""
        self.results_df = df
        self.results_file = file_path
        
        if df is None or df.empty:
            self._show_empty_state()
            self.status_label.setText("请先运行实验或加载结果文件")
            self.status_label.setStyleSheet("color: gray;")
            return
        
        valid_df = df[df['best_objective'] < 1e9]
        n_instances = valid_df['instance_id'].nunique() if not valid_df.empty else 0
        n_algorithms = valid_df['algorithm_name'].nunique() if not valid_df.empty else 0
        
        self.status_label.setText(f"已加载 {n_instances} 个实例, {n_algorithms} 个算法")
        self.status_label.setStyleSheet("color: green;")
    
    def _run_selector_analysis(self):
        """运行算法选择分析"""
        if self.results_df is None or self.results_df.empty:
            QMessageBox.warning(self, "警告", "请先运行实验或加载结果文件！")
            return
        
        valid_df = self.results_df[self.results_df['best_objective'] < 1e9]
        if valid_df.empty:
            QMessageBox.warning(self, "警告", "没有有效的实验数据！")
            return
        
        self.run_selector_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("正在运行算法选择分析...")
        self.status_label.setStyleSheet("color: blue;")
        
        ml_model = self.ml_model_combo.currentText()
        test_size = self.test_size_spin.value() / 100.0
        use_tsne = self.use_tsne_check.isChecked()
        
        self.worker = SelectorWorker(self.results_df, ml_model, test_size, use_tsne)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()
    
    def _on_progress(self, message: str):
        """进度更新"""
        self.status_label.setText(message)
    
    def _on_finished(self, success: bool, results, message: str):
        """分析完成"""
        self.run_selector_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success and results is not None:
            self.selector_results = results
            self._update_results(results)
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.warning(self, "警告", message)
    
    def _update_results(self, results):
        """更新结果显示"""
        self.sbs_label.setText(
            f"{results['SBS']['score']:.2f} (算法: {results['SBS']['algorithm']})"
        )
        
        improvement = results['Selector']['improvement_over_sbs']
        self.selector_label.setText(
            f"{results['Selector']['score']:.2f} (改进: {improvement:.2f}%)"
        )
        
        gap = results['Selector']['gap_to_vbs']
        self.vbs_label.setText(
            f"{results['VBS']['score']:.2f} (Gap: {gap:.2f}%)"
        )
        
        self.hit_rate_label.setText(f"{results['Selector']['hit_rate']*100:.1f}%")
        self.avg_regret_label.setText(f"{results['Selector']['avg_regret']:.2f}")
        self.p90_penalty_label.setText(f"{results['Selector']['p90_penalty']:.2f}")
        
        if improvement > 0:
            self.improvement_label.setText(f"+{improvement:.2f}%")
            self.improvement_label.setStyleSheet("color: green;")
        else:
            self.improvement_label.setText(f"{improvement:.2f}%")
            self.improvement_label.setStyleSheet("color: red;")
        
        if 'figures' in results:
            if 'performance_comparison' in results['figures'] and results['figures']['performance_comparison'] is not None:
                self.perf_comp_canvas.clear()
                self.perf_comp_canvas.fig = results['figures']['performance_comparison']
                self.perf_comp_canvas.axes = results['figures']['performance_comparison'].axes[0]
                self.perf_comp_canvas.draw()
            
            if 'feature_importance' in results['figures'] and results['figures']['feature_importance'] is not None:
                self.feature_imp_canvas.clear()
                self.feature_imp_canvas.fig = results['figures']['feature_importance']
                self.feature_imp_canvas.axes = results['figures']['feature_importance'].axes[0]
                self.feature_imp_canvas.draw()
            
            if 'instance_space' in results['figures'] and results['figures']['instance_space'] is not None:
                self.instance_space_canvas.clear()
                self.instance_space_canvas.fig = results['figures']['instance_space']
                self.instance_space_canvas.axes = results['figures']['instance_space'].axes[0]
                self.instance_space_canvas.draw()
        
        if improvement > 5:
            self.conclusion_label.setText(f"""
🏆 算法选择器表现优秀！

- 选择器相比SBS改进了 {improvement:.2f}%
- 命中率达到 {results['Selector']['hit_rate']*100:.1f}%
- 距离理论最优VBS仅差 {gap:.2f}%

建议: 可以在实际应用中使用该选择器进行算法推荐。
            """)
            self.conclusion_label.setStyleSheet("color: green; font-weight: bold;")
        elif improvement > 0:
            self.conclusion_label.setText(f"""
✅ 算法选择器表现良好

- 选择器相比SBS改进了 {improvement:.2f}%
- 命中率为 {results['Selector']['hit_rate']*100:.1f}%

建议: 可以进一步优化模型或增加训练数据以提升性能。
            """)
            self.conclusion_label.setStyleSheet("color: blue;")
        else:
            self.conclusion_label.setText(f"""
⚠️ 算法选择器未优于SBS

- 选择器相比SBS改进了 {improvement:.2f}%
- 可能原因：实例数量不足、特征区分度不够、模型选择不当

建议: 
1. 增加实验实例数量
2. 尝试其他机器学习模型
3. 检查特征提取是否合理
            """)
            self.conclusion_label.setStyleSheet("color: orange;")
    
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
        return {
            "ml_model": self.ml_model_combo.currentText(),
            "test_size": self.test_size_spin.value(),
            "use_tsne": self.use_tsne_check.isChecked()
        }
    
    def load_config(self, config: Dict[str, Any]):
        """加载配置"""
        try:
            ml_model = config.get("ml_model", "random_forest")
            self.ml_model_combo.setCurrentText(ml_model)
            
            test_size = config.get("test_size", 30)
            self.test_size_spin.setValue(test_size)
            
            use_tsne = config.get("use_tsne", False)
            self.use_tsne_check.setChecked(use_tsne)
            
        except Exception as e:
            print(f"Error loading config in SelectorTab: {e}")
