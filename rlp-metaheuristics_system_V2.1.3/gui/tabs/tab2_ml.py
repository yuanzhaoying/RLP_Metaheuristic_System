"""
Tab 2: Machine Learning Configuration
机器学习配置选项卡
完全参照prlp-platform-v0.2/gui框架，app.py没有的功能留空并添加提示
"""
from typing import Dict, Any, List
import pandas as pd

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QGroupBox, QScrollArea, QListWidget, QListWidgetItem,
    QTabWidget, QRadioButton, QButtonGroup, QMessageBox, QFrame, QTextEdit
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtCore import Qt, QEvent


class NoScrollComboBox(QComboBox):
    """禁用滚轮事件和鼠标悬停事件的ComboBox"""
    
    def wheelEvent(self, event):
        event.ignore()
    
    def enterEvent(self, event):
        event.ignore()
    
    def leaveEvent(self, event):
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    """禁用滚轮事件的SpinBox"""
    
    def wheelEvent(self, event):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """禁用滚轮事件的DoubleSpinBox"""
    
    def wheelEvent(self, event):
        event.ignore()


class MLTab(QWidget):
    """机器学习配置选项卡"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
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
        
        self._create_instructions_group(container_layout)
        
        self._create_feature_group(container_layout)
        
        self._create_stats_group(container_layout)
        
        self._create_ml_model_group(container_layout)
        
        self._create_selector_group(container_layout)
        
        self._create_interpretability_group(container_layout)
        
        self._create_results_group(container_layout)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
    
    def _create_feature_group(self, parent_layout):
        """创建特征选择组"""
        group = QGroupBox("特征选择")
        layout = QVBoxLayout(group)
        
        desc_label = QLabel("选择要提取的实例特征:")
        desc_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(desc_label)
        
        self.feature_categories = {
            "structural": QCheckBox("结构特征"),
            "resource": QCheckBox("资源特征"),
            "slack": QCheckBox("松弛时间特征"),
            "network_topology": QCheckBox("网络拓扑特征"),
        }
        
        for checkbox in self.feature_categories.values():
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._update_feature_list)
            layout.addWidget(checkbox)
        
        structural_features = [
            "n_activities", "n_resources", "n_edges",
            "avg_out_degree", "avg_in_degree", "max_out_degree", "max_in_degree",
            "critical_path_len", "order_strength", "parallel_degree", "density"
        ]
        
        resource_features = [
            "capacity_mean", "capacity_std", "capacity_min", "capacity_max",
            "demand_mean", "demand_std", "demand_per_activity_mean", "total_work",
            "resource_strength_mean", "resource_strength_max",
            "resource_factor", "resource_usage", "avg_resource_total"
        ]
        
        slack_features = [
            "slack_mean", "slack_std", "slack_min", "slack_max", "slack_median",
            "non_zero_slack_mean", "critical_activity_ratio"
        ]
        
        network_topology_features = [
            "duration_mean", "duration_max", "duration_min",
            "max_predecessor_count", "min_predecessor_count",
            "max_successor_count", "min_successor_count",
            "network_complexity", "order_strength_new",
            "serial_parallel_indicator", "activity_distribution",
            "short_arc_indicator", "long_arc_count", "topological_floating",
            "max_progressive_level", "min_regression_level"
        ]
        
        self.feature_category_map = {
            "structural": structural_features,
            "resource": resource_features,
            "slack": slack_features,
            "network_topology": network_topology_features
        }
        
        self.feature_chinese_names = {
            "n_activities": "活动数量",
            "n_resources": "资源数量",
            "n_edges": "边数量",
            "avg_out_degree": "平均出度",
            "avg_in_degree": "平均入度",
            "max_out_degree": "最大出度",
            "max_in_degree": "最大入度",
            "critical_path_len": "关键路径长度",
            "order_strength": "顺序强度",
            "parallel_degree": "并行度",
            "density": "网络密度",
            "capacity_mean": "资源容量均值",
            "capacity_std": "资源容量标准差",
            "capacity_min": "资源容量最小值",
            "capacity_max": "资源容量最大值",
            "demand_mean": "资源需求均值",
            "demand_std": "资源需求标准差",
            "demand_per_activity_mean": "每活动平均需求",
            "total_work": "总工作量",
            "resource_strength_mean": "资源强度均值",
            "resource_strength_max": "资源强度最大值",
            "resource_factor": "资源因子",
            "resource_usage": "资源使用率",
            "avg_resource_total": "平均总资源需求",
            "slack_mean": "松弛时间均值",
            "slack_std": "松弛时间标准差",
            "slack_min": "松弛时间最小值",
            "slack_max": "松弛时间最大值",
            "slack_median": "松弛时间中位数",
            "non_zero_slack_mean": "非零松弛时间均值",
            "critical_activity_ratio": "关键活动比例",
            "duration_mean": "活动持续时间均值",
            "duration_max": "活动持续时间最大值",
            "duration_min": "活动持续时间最小值",
            "max_predecessor_count": "最大前驱数量",
            "min_predecessor_count": "最小前驱数量",
            "max_successor_count": "最大后继数量",
            "min_successor_count": "最小后继数量",
            "network_complexity": "网络复杂度",
            "order_strength_new": "新顺序强度",
            "serial_parallel_indicator": "串并行指示器",
            "activity_distribution": "活动分布",
            "short_arc_indicator": "短弧指示器",
            "long_arc_count": "长弧数量",
            "topological_floating": "拓扑浮动",
            "max_progressive_level": "最大递进层级",
            "min_regression_level": "最小回归层级",
        }
        
        self.all_features = []
        for features in self.feature_category_map.values():
            self.all_features.extend(features)
        
        list_label = QLabel("已选特征列表:")
        list_label.setStyleSheet("margin-top: 10px;")
        layout.addWidget(list_label)
        
        self.feature_list = QListWidget()
        self.feature_list.setMaximumHeight(120)
        self._update_feature_list()
        layout.addWidget(self.feature_list)
        
        info_label = QLabel(
            "注: avg_resource_0, avg_resource_1 等特征会根据实例的\n"
            "资源数量动态添加到资源特征中"
        )
        info_label.setStyleSheet("color: #666; font-size: 10px; margin-top: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        parent_layout.addWidget(group)
    
    def _create_results_group(self, parent_layout):
        """创建结果显示组"""
        group = QGroupBox("分析结果")
        layout = QVBoxLayout(group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(300)
        self.results_text.setPlaceholderText("分析结果将显示在这里...")
        layout.addWidget(self.results_text)
        
        self.image_label = QLabel("图片将显示在这里")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px;")
        self.image_label.setMinimumHeight(400)
        self.image_label.hide()
        layout.addWidget(self.image_label)
        
        self.svg_widget = QSvgWidget()
        self.svg_widget.setMinimumHeight(400)
        self.svg_widget.hide()
        layout.addWidget(self.svg_widget)
        
        parent_layout.addWidget(group)
    
    def display_results(self, results: dict):
        """显示分析结果"""
        self.results_text.clear()
        self.svg_widget.hide()
        
        if not results:
            self.results_text.setHtml("<html><body><p style='color: red;'>没有分析结果</p></body></html>")
            return
        
        html = "<html><body>"
        html += f"<p>分析完成，共 {len(results)} 项结果</p>"
        
        if 'friedman' in results:
            friedman = results['friedman']
            html += "<h3>Layer A检验结果 (Friedman)</h3>"
            html += "<table border='1' cellpadding='5' cellspacing='0'>"
            html += "<tr><th>指标</th><th>值</th></tr>"
            
            if 'statistic' in friedman:
                html += f"<tr><td>统计量</td><td>{friedman['statistic']:.4f}</td></tr>"
            if 'p_value' in friedman:
                html += f"<tr><td>p值</td><td>{friedman['p_value']:.4f}</td></tr>"
            if 'n_instances' in friedman:
                html += f"<tr><td>实例数量</td><td>{friedman['n_instances']}</td></tr>"
            if 'n_algorithms' in friedman:
                html += f"<tr><td>算法数量</td><td>{friedman['n_algorithms']}</td></tr>"
            if 'significant' in friedman:
                sig_text = "<b>是</b>" if friedman['significant'] else "否"
                html += f"<tr><td>是否显著(α=0.05)</td><td>{sig_text}</td></tr>"
            
            if 'mean_ranks' in friedman:
                html += f"<tr><td>平均排名</td><td>"
                for algo, rank in sorted(friedman['mean_ranks'].items(), key=lambda x: x[1]):
                    html += f"{algo}: {rank:.2f}<br>"
                html += "</td></tr>"
            
            html += "</table><br>"
        
        if 'pairwise' in results:
            html += "<h3>成对比较结果 (Wilcoxon + Holm校正)</h3>"
            html += "<table border='1' cellpadding='5' cellspacing='0'>"
            html += "<tr><th>算法1</th><th>算法2</th><th>p值</th><th>校正p值</th><th>胜</th><th>负</th><th>显著</th></tr>"
            
            significant_pairs = []
            for row in results['pairwise']:
                sig_text = "<b>是</b>" if row.get('significant', False) else "否"
                html += f"<tr>"
                html += f"<td>{row.get('algorithm_1', '')}</td>"
                html += f"<td>{row.get('algorithm_2', '')}</td>"
                html += f"<td>{row.get('p_value', 0):.4f}</td>"
                html += f"<td>{row.get('p_adjusted', 0):.4f}</td>"
                html += f"<td>{row.get('wins', 0)}</td>"
                html += f"<td>{row.get('losses', 0)}</td>"
                html += f"<td>{sig_text}</td>"
                html += f"</tr>"
                
                if row.get('significant', False):
                    better_algo = row.get('better', '')
                    significant_pairs.append(f"{row.get('algorithm_1', '')} vs {row.get('algorithm_2', '')}: {better_algo}更优")
            
            html += "</table><br>"
            
            if significant_pairs:
                html += "<h4>成对比较结论</h4>"
                html += "<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
                html += "<b>显著差异的算法对:</b><br>"
                for pair in significant_pairs:
                    html += f"• {pair}<br>"
                html += "</div><br>"
        
        if 'layerB_lrt' in results:
            html += "<h3>Layer B检验结果 (似然比检验)</h3>"
            html += "<table border='1' cellpadding='5' cellspacing='0'>"
            html += "<tr><th>因子</th><th>LR统计量</th><th>自由度</th><th>p值</th></tr>"
            
            for row in results['layerB_lrt']:
                html += f"<tr>"
                html += f"<td>{row.get('term', '')}</td>"
                html += f"<td>{row.get('lr_stat', 0):.4f}</td>"
                html += f"<td>{row.get('df_diff', 0)}</td>"
                html += f"<td>{row.get('p_value', 0):.4f}</td>"
                html += f"</tr>"
            
            html += "</table><br>"
        
        if 'layerB_art' in results:
            html += "<h3>Layer B检验结果 (ART ANOVA)</h3>"
            html += "<table border='1' cellpadding='5' cellspacing='0'>"
            html += "<tr><th>因子</th><th>F值</th><th>p值</th><th>偏eta²</th></tr>"
            
            for row in results['layerB_art']:
                html += f"<tr>"
                html += f"<td>{row.get('term', '')}</td>"
                html += f"<td>{row.get('F', 0):.4f}</td>"
                html += f"<td>{row.get('p_value', 0):.4f}</td>"
                html += f"<td>{row.get('partial_eta2', 0):.4f}</td>"
                html += f"</tr>"
            
            html += "</table><br>"
            
            if results['layerB_art']:
                row = results['layerB_art'][0]
                p_value = row.get('p_value', 1.0)
                eta2 = row.get('partial_eta2', 0)
                
                html += "<h4>Layer B分析结论</h4>"
                html += "<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
                
                if p_value < 0.05:
                    html += f"<b style='color: green;'>✓ 算法间存在显著差异</b><br>"
                    html += f"p值 = {p_value:.4f} < 0.05，拒绝零假设，表明不同算法的性能存在统计学上的显著差异。<br>"
                    
                    if eta2 > 0.14:
                        effect_size = "大"
                    elif eta2 > 0.06:
                        effect_size = "中等"
                    else:
                        effect_size = "小"
                    
                    html += f"偏eta² = {eta2:.4f}，效应量为<b>{effect_size}</b>（小: <0.06, 中: 0.06-0.14, 大: >0.14）"
                else:
                    html += f"<b style='color: orange;'>⚠ 算法间无显著差异</b><br>"
                    html += f"p值 = {p_value:.4f} ≥ 0.05，无法拒绝零假设，表明不同算法的性能差异不显著。"
                
                html += "</div><br>"
        
        if 'anytime_output' in results:
            html += "<h3>Anytime分析结果</h3>"
            html += f"<p>输出目录: {results['anytime_output']}</p>"
            html += "<p>已生成文件:</p>"
            html += "<ul>"
            html += "<li>data_profile.svg - Data Profile图表(SVG格式)</li>"
            html += "<li>data_profile_source.csv - 源数据(每个实例的比率)</li>"
            html += "<li>data_profile_curve.csv - 曲线数据(用于绑图)</li>"
            html += "</ul>"
            
            if 'anytime_analysis' in results:
                analysis = results['anytime_analysis']
                
                if 'algorithm_stats' in analysis:
                    html += "<h4>算法性能统计</h4>"
                    html += "<table border='1' cellpadding='5' cellspacing='0'>"
                    html += "<tr><th>排名</th><th>算法</th><th>平均比率</th><th>最优解率</th><th>5%内</th><th>10%内</th><th>AUC</th></tr>"
                    
                    for i, stats in enumerate(analysis['algorithm_stats'], 1):
                        perfect_rate = stats.get('perfect_rate', 0) * 100
                        within_5pct = stats.get('within_5pct', 0)
                        within_10pct = stats.get('within_10pct', 0)
                        total = stats.get('total_count', 0)
                        
                        html += f"<tr>"
                        html += f"<td>{i}</td>"
                        html += f"<td><b>{stats.get('algorithm', '')}</b></td>"
                        html += f"<td>{stats.get('mean_ratio', 0):.4f}</td>"
                        html += f"<td>{perfect_rate:.1f}%</td>"
                        html += f"<td>{within_5pct}/{total}</td>"
                        html += f"<td>{within_10pct}/{total}</td>"
                        html += f"<td>{stats.get('auc', 0):.4f}</td>"
                        html += f"</tr>"
                    
                    html += "</table><br>"
                
                if 'summary' in analysis:
                    html += "<h4>分析结论</h4>"
                    html += "<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
                    summary = analysis['summary'].replace('\n', '<br>')
                    html += summary
                    html += "</div>"
            
            import os
            svg_path = os.path.join(results['anytime_output'], "data_profile.svg")
            if os.path.exists(svg_path):
                self.svg_widget.load(svg_path)
                self.svg_widget.show()
                html += "<h4>Data Profile图表</h4>"
                html += "<p style='color: #666; font-size: 11px;'>↓ 下方显示Data Profile图，展示各算法在不同比率阈值下解决问题的比例</p>"
            else:
                self.svg_widget.hide()
        
        if 'ml_results' in results:
            ml = results['ml_results']
            html += "<h3>机器学习算法选择结果</h3>"
            html += f"<p>输出目录: {ml.get('output_dir', '')}</p>"
            
            html += "<h4>性能基准对比</h4>"
            html += "<table border='1' cellpadding='5' cellspacing='0'>"
            html += "<tr><th>方法</th><th>算法/得分</th><th>说明</th></tr>"
            
            if 'SBS' in ml:
                sbs = ml['SBS']
                html += f"<tr><td><b>SBS</b></td><td>{sbs.get('algorithm', '-')}: {sbs.get('score', 0):.2f}</td><td>单一最佳算法</td></tr>"
            
            if 'Selector' in ml:
                sel = ml['Selector']
                html += f"<tr><td><b>Selector</b></td><td>{sel.get('score', 0):.2f}</td><td>算法选择器</td></tr>"
            
            if 'VBS' in ml:
                vbs = ml['VBS']
                html += f"<tr><td><b>VBS</b></td><td>{vbs.get('score', 0):.2f}</td><td>虚拟最佳算法</td></tr>"
            
            html += "</table><br>"
            
            if 'Selector' in ml:
                sel = ml['Selector']
                html += "<h4>选择器性能指标</h4>"
                html += "<table border='1' cellpadding='5' cellspacing='0'>"
                html += "<tr><th>指标</th><th>值</th></tr>"
                html += f"<tr><td>命中率 (Hit Rate)</td><td>{sel.get('hit_rate', 0)*100:.1f}%</td></tr>"
                html += f"<tr><td>平均Regret</td><td>{sel.get('avg_regret', 0):.2f}</td></tr>"
                html += f"<tr><td>P90 Penalty</td><td>{sel.get('p90_penalty', 0):.2f}</td></tr>"
                html += f"<tr><td>相比SBS改进</td><td>{sel.get('improvement_over_sbs', 0):.2f}%</td></tr>"
                html += f"<tr><td>距离VBS差距</td><td>{sel.get('gap_to_vbs', 0):.2f}%</td></tr>"
                html += "</table><br>"
                
                improvement = sel.get('improvement_over_sbs', 0)
                html += "<h4>结论</h4>"
                html += "<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
                if improvement > 5:
                    html += f"<b style='color: green;'>✓ 算法选择器表现优秀！</b><br>"
                    html += f"选择器相比SBS改进了 {improvement:.2f}%，建议在实际应用中使用该选择器进行算法推荐。"
                elif improvement > 0:
                    html += f"<b style='color: blue;'>✓ 算法选择器表现良好</b><br>"
                    html += f"选择器相比SBS改进了 {improvement:.2f}%，可以进一步优化模型或增加训练数据以提升性能。"
                else:
                    html += f"<b style='color: orange;'>⚠ 算法选择器未优于SBS</b><br>"
                    html += f"建议增加实验实例数量或尝试其他机器学习模型。"
                html += "</div>"
            
            if 'feature_importance' in ml and ml['feature_importance'] is not None:
                fi = ml['feature_importance']
                if len(fi) > 0:
                    html += "<h4>特征重要性 (Top 10)</h4>"
                    html += "<table border='1' cellpadding='5' cellspacing='0'>"
                    html += "<tr><th>排名</th><th>特征</th><th>重要性</th></tr>"
                    for i, row in fi.head(10).iterrows():
                        html += f"<tr><td>{i+1}</td><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"
                    html += "</table><br>"
            
            if 'shap_values' in ml and ml['shap_values'] is not None:
                html += "<h4>SHAP分析结果</h4>"
                html += "<p>SHAP值展示了每个特征对每个预测的正负影响</p>"
                shap_df = ml['shap_values']
                if isinstance(shap_df, pd.DataFrame) and len(shap_df) > 0:
                    html += "<table border='1' cellpadding='5' cellspacing='0'>"
                    html += "<tr><th>特征</th><th>平均|SHAP|</th></tr>"
                    for i, row in shap_df.head(10).iterrows():
                        html += f"<tr><td>{row.get('feature', '')}</td><td>{row.get('mean_abs_shap', 0):.4f}</td></tr>"
                    html += "</table><br>"
            
            if 'permutation_importance' in ml and ml['permutation_importance'] is not None:
                html += "<h4>Permutation Importance结果</h4>"
                html += "<p>特征置换后的性能下降，值越大表示特征越重要</p>"
                perm_df = ml['permutation_importance']
                if isinstance(perm_df, pd.DataFrame) and len(perm_df) > 0:
                    html += "<table border='1' cellpadding='5' cellspacing='0'>"
                    html += "<tr><th>排名</th><th>特征</th><th>重要性均值</th><th>重要性标准差</th></tr>"
                    for i, row in perm_df.head(10).iterrows():
                        html += f"<tr><td>{i+1}</td><td>{row.get('feature', '')}</td><td>{row.get('importance_mean', 0):.4f}</td><td>{row.get('importance_std', 0):.4f}</td></tr>"
                    html += "</table><br>"
            
            if 'isa_results' in ml and ml['isa_results'] is not None:
                html += "<h4>实例空间分析 (ISA) 结果</h4>"
                isa = ml['isa_results']
                if isinstance(isa, dict):
                    html += "<p>实例空间分析展示了实例在降维空间中的分布和各算法的优势区域</p>"
                    if 'embedding' in isa:
                        html += f"<p>降维方法: {isa.get('method', 'PCA')}</p>"
                        html += f"<p>解释方差比: {isa.get('explained_variance', 'N/A')}</p>"
                    if 'algorithm_regions' in isa:
                        html += "<table border='1' cellpadding='5' cellspacing='0'>"
                        html += "<tr><th>算法</th><th>优势区域实例数</th><th>占比</th></tr>"
                        for algo, count in isa['algorithm_regions'].items():
                            html += f"<tr><td>{algo}</td><td>{count}</td><td>{count/isa.get('total_instances', 1)*100:.1f}%</td></tr>"
                        html += "</table><br>"
        
        html += "</body></html>"
        
        self.results_text.setHtml(html)
    
    def _create_instructions_group(self, parent_layout):
        """创建使用说明组"""
        group = QGroupBox("How to Run Algorithm Selector")
        layout = QVBoxLayout(group)
        
        instructions = QLabel(
            "<b>算法选择器运行说明:</b><br><br>"
            "1. 首先在 <b>Metaheuristics</b> 选项卡中运行实验<br>"
            "2. 实验完成后，结果会自动保存到 results/raw 文件夹<br>"
            "3. 在此选项卡中配置机器学习模型参数<br>"
            "4. 点击主界面的 <b>Run Experiment</b> 按钮运行算法选择器<br><br>"
            "<b>注意:</b> 算法选择器需要实验结果数据才能运行<br>"
            "如果没有实验结果，请先运行元启发式算法实验"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(instructions)
        
        parent_layout.addWidget(group)
    
    def _create_stats_group(self, parent_layout):
        """创建统计检验方法组"""
        group = QGroupBox("统计检验方法")
        layout = QVBoxLayout(group)
        
        layer_a_group = QGroupBox("Layer A检验 (全局检验)")
        layer_a_layout = QFormLayout(layer_a_group)
        
        self.layer_a_test = NoScrollComboBox()
        self.layer_a_test.addItems(["friedman", "quade"])
        layer_a_layout.addRow("全局检验方法:", self.layer_a_test)
        
        self.layer_a_posthoc = NoScrollComboBox()
        self.layer_a_posthoc.addItems(["wilcoxon", "nemenyi", "conover"])
        layer_a_layout.addRow("事后检验方法:", self.layer_a_posthoc)
        
        self.layer_a_correction = NoScrollComboBox()
        self.layer_a_correction.addItems(["holm", "bonferroni", "fdr"])
        layer_a_layout.addRow("多重比较校正:", self.layer_a_correction)
        
        layout.addWidget(layer_a_group)
        
        layer_b_group = QGroupBox("Layer B检验 (混合线性模型)")
        layer_b_layout = QFormLayout(layer_b_group)
        
        self.layer_b_method = NoScrollComboBox()
        self.layer_b_method.addItems(["art_anova", "mixedlm", "bootstrap"])
        layer_b_layout.addRow("Layer B方法:", self.layer_b_method)
        
        self.layer_b_by_delta = QCheckBox("按Delta分层")
        self.layer_b_by_delta.setChecked(True)
        layer_b_layout.addRow("", self.layer_b_by_delta)
        
        self.layer_b_bootstrap_samples = NoScrollSpinBox()
        self.layer_b_bootstrap_samples.setRange(10, 1000)
        self.layer_b_bootstrap_samples.setValue(200)
        layer_b_layout.addRow("Bootstrap采样次数:", self.layer_b_bootstrap_samples)
        
        layout.addWidget(layer_b_group)
        
        anytime_group = QGroupBox("Anytime分析")
        anytime_layout = QFormLayout(anytime_group)
        
        self.anytime_method = NoScrollComboBox()
        self.anytime_method.addItems(["ecdf", "data_profile", "time_to_target"])
        anytime_layout.addRow("Anytime方法:", self.anytime_method)
        
        self.anytime_target_factor = NoScrollDoubleSpinBox()
        self.anytime_target_factor.setRange(1.0, 2.0)
        self.anytime_target_factor.setValue(1.1)
        self.anytime_target_factor.setSingleStep(0.05)
        anytime_layout.addRow("目标因子:", self.anytime_target_factor)
        
        layout.addWidget(anytime_group)
        
        parent_layout.addWidget(group)
    
    def _create_ml_model_group(self, parent_layout):
        """创建机器学习模型组"""
        group = QGroupBox("Machine Learning Model")
        layout = QFormLayout(group)
        
        self.model_type = NoScrollComboBox()
        self.model_type.addItems([
            "decision_tree", "random_forest", "gradient_boosting",
            "svm", "knn", "neural_network"
        ])
        self.model_type.currentTextChanged.connect(self._on_model_changed)
        layout.addRow("Model Type:", self.model_type)
        
        self.model_params_widget = QWidget()
        self.model_params_layout = QFormLayout(self.model_params_widget)
        
        self.rf_n_estimators = NoScrollSpinBox()
        self.rf_n_estimators.setRange(10, 1000)
        self.rf_n_estimators.setValue(100)
        self.model_params_layout.addRow("N Estimators:", self.rf_n_estimators)
        
        self.rf_max_depth = NoScrollSpinBox()
        self.rf_max_depth.setRange(1, 100)
        self.rf_max_depth.setValue(20)
        self.model_params_layout.addRow("Max Depth:", self.rf_max_depth)
        
        layout.addRow(self.model_params_widget)
        
        layout.addRow("---", QLabel(""))
        
        self.test_size = NoScrollDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setValue(0.3)
        self.test_size.setSingleStep(0.05)
        layout.addRow("Test Size:", self.test_size)
        
        self.cv_folds = NoScrollSpinBox()
        self.cv_folds.setRange(2, 10)
        self.cv_folds.setValue(5)
        layout.addRow("CV Folds:", self.cv_folds)
        
        self.random_state = NoScrollSpinBox()
        self.random_state.setRange(0, 1000)
        self.random_state.setValue(42)
        layout.addRow("Random State:", self.random_state)
        
        parent_layout.addWidget(group)
    
    def _create_selector_group(self, parent_layout):
        """创建选择器分析组"""
        group = QGroupBox("选择器性能分析 (Selector Performance Analysis)")
        layout = QVBoxLayout(group)
        
        desc_label = QLabel("评估算法选择器的性能，与SBS(单一最佳算法)和VBS(虚拟最佳算法)对比")
        desc_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        self.calc_sbs = QCheckBox("计算SBS (Single Best Solver) - 所有实例上平均表现最好的单一算法")
        self.calc_sbs.setChecked(True)
        layout.addWidget(self.calc_sbs)
        
        self.calc_vbs = QCheckBox("计算VBS (Virtual Best Solver) - 每个实例上表现最好的算法(理论上限)")
        self.calc_vbs.setChecked(True)
        layout.addWidget(self.calc_vbs)
        
        self.calc_selector = QCheckBox("计算Selector - 机器学习算法选择器的实际表现")
        self.calc_selector.setChecked(True)
        layout.addWidget(self.calc_selector)
        
        self.calc_winner_hit = QCheckBox("Winner Hit Rate - 选择器选择最优算法的比例")
        self.calc_winner_hit.setChecked(True)
        layout.addWidget(self.calc_winner_hit)
        
        self.calc_penalty = QCheckBox("Penalty Analysis - 选择错误时的性能惩罚分析")
        self.calc_penalty.setChecked(True)
        layout.addWidget(self.calc_penalty)
        
        self.calc_risk_lambda = QCheckBox("Risk Lambda Ablation - 不同风险参数λ的敏感性分析")
        layout.addWidget(self.calc_risk_lambda)
        
        parent_layout.addWidget(group)
    
    def _create_interpretability_group(self, parent_layout):
        """创建解释性分析组"""
        group = QGroupBox("解释性分析 (Interpretability Analysis)")
        layout = QVBoxLayout(group)
        
        desc_label = QLabel("分析算法选择器的决策依据，解释哪些特征影响算法选择")
        desc_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        self.feature_importance = QCheckBox("特征重要性 (Feature Importance) - 各特征对选择的贡献度")
        self.feature_importance.setChecked(True)
        layout.addWidget(self.feature_importance)
        
        self.shap_analysis = QCheckBox("SHAP分析 - 各特征对每个预测的正负影响")
        self.shap_analysis.setChecked(True)
        layout.addWidget(self.shap_analysis)
        
        self.perm_importance = QCheckBox("Permutation Importance - 特征置换后的性能下降")
        self.perm_importance.setChecked(True)
        layout.addWidget(self.perm_importance)
        
        self.isa_analysis = QCheckBox("实例空间分析 (ISA) - 可视化实例分布和算法优势区域")
        self.isa_analysis.setChecked(True)
        layout.addWidget(self.isa_analysis)
        
        parent_layout.addWidget(group)
    
    def _on_model_changed(self, model_type: str):
        """模型类型改变时更新UI"""
        while self.model_params_layout.count():
            item = self.model_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if model_type == "random_forest":
            self.rf_n_estimators = NoScrollSpinBox()
            self.rf_n_estimators.setRange(10, 1000)
            self.rf_n_estimators.setValue(100)
            self.model_params_layout.addRow("N Estimators:", self.rf_n_estimators)
            
            self.rf_max_depth = NoScrollSpinBox()
            self.rf_max_depth.setRange(1, 100)
            self.rf_max_depth.setValue(20)
            self.model_params_layout.addRow("Max Depth:", self.rf_max_depth)
            
        elif model_type == "gradient_boosting":
            self.gb_n_estimators = NoScrollSpinBox()
            self.gb_n_estimators.setRange(10, 1000)
            self.gb_n_estimators.setValue(100)
            self.model_params_layout.addRow("N Estimators:", self.gb_n_estimators)
            
            self.gb_learning_rate = NoScrollDoubleSpinBox()
            self.gb_learning_rate.setRange(0.01, 1.0)
            self.gb_learning_rate.setValue(0.1)
            self.model_params_layout.addRow("Learning Rate:", self.gb_learning_rate)
            
        elif model_type == "svm":
            self.svm_c = NoScrollDoubleSpinBox()
            self.svm_c.setRange(0.1, 100.0)
            self.svm_c.setValue(1.0)
            self.model_params_layout.addRow("C:", self.svm_c)
            
            self.svm_kernel = NoScrollComboBox()
            self.svm_kernel.addItems(["linear", "rbf", "poly"])
            self.model_params_layout.addRow("Kernel:", self.svm_kernel)
            
        elif model_type == "knn":
            self.knn_n_neighbors = NoScrollSpinBox()
            self.knn_n_neighbors.setRange(1, 50)
            self.knn_n_neighbors.setValue(5)
            self.model_params_layout.addRow("N Neighbors:", self.knn_n_neighbors)
    
    def _update_feature_list(self):
        """更新特征列表显示"""
        self.feature_list.clear()
        
        for category, checkbox in self.feature_categories.items():
            if checkbox.isChecked():
                for feature in self.feature_category_map[category]:
                    chinese_name = self.feature_chinese_names.get(feature, feature)
                    item = QListWidgetItem(f"{chinese_name} ({feature})")
                    item.setData(Qt.UserRole, feature)
                    item.setCheckState(Qt.Checked)
                    self.feature_list.addItem(item)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        config = {}
        
        selected_categories = []
        for category, checkbox in self.feature_categories.items():
            if checkbox.isChecked():
                selected_categories.append(category)
        
        selected_features = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                feature = item.data(Qt.UserRole)
                selected_features.append(feature)
        
        config["features"] = {
            "enabled": len(selected_features) > 0,
            "categories": selected_categories,
            "selected_features": selected_features
        }
        
        config["stats"] = {
            "layerA": {
                "enabled": True,
                "global_test": self.layer_a_test.currentText(),
                "posthoc": self.layer_a_posthoc.currentText(),
                "correction": self.layer_a_correction.currentText()
            },
            "layerB": {
                "enabled": True,
                "method": self.layer_b_method.currentText(),
                "by_delta": self.layer_b_by_delta.isChecked(),
                "bootstrap_samples": self.layer_b_bootstrap_samples.value()
            }
        }
        
        config["anytime"] = {
            "enabled": True,
            "method": self.anytime_method.currentText(),
            "target_factor": self.anytime_target_factor.value()
        }
        
        model_type = self.model_type.currentText()
        model_params = {}
        
        if model_type == "random_forest":
            model_params = {
                "n_estimators": self.rf_n_estimators.value(),
                "max_depth": self.rf_max_depth.value()
            }
        elif model_type == "gradient_boosting":
            model_params = {
                "n_estimators": self.gb_n_estimators.value(),
                "learning_rate": self.gb_learning_rate.value()
            }
        elif model_type == "svm":
            model_params = {
                "C": self.svm_c.value(),
                "kernel": self.svm_kernel.currentText()
            }
        elif model_type == "knn":
            model_params = {
                "n_neighbors": self.knn_n_neighbors.value()
            }
        
        config["ml"] = {
            "enabled": True,
            "model_type": model_type,
            "model_params": model_params,
            "test_size": self.test_size.value(),
            "cv_folds": self.cv_folds.value(),
            "random_state": self.random_state.value(),
            "selector_analysis": {
                "calc_sbs": self.calc_sbs.isChecked(),
                "calc_vbs": self.calc_vbs.isChecked(),
                "calc_selector": self.calc_selector.isChecked(),
                "calc_winner_hit": self.calc_winner_hit.isChecked(),
                "calc_penalty": self.calc_penalty.isChecked(),
                "calc_risk_lambda": self.calc_risk_lambda.isChecked()
            },
            "interpretability": {
                "feature_importance": self.feature_importance.isChecked(),
                "shap": self.shap_analysis.isChecked(),
                "perm_importance": self.perm_importance.isChecked(),
                "isa": self.isa_analysis.isChecked()
            }
        }
        
        return config
    
    def load_config(self, config: Dict[str, Any]):
        """加载配置"""
        try:
            features = config.get("features", {})
            if features.get("enabled"):
                selected_categories = features.get("categories", [])
                for category, checkbox in self.feature_categories.items():
                    checkbox.setChecked(category in selected_categories)
                
                self._update_feature_list()
                
                selected_features = features.get("selected_features", [])
                for i in range(self.feature_list.count()):
                    item = self.feature_list.item(i)
                    feature = item.data(Qt.UserRole)
                    if feature in selected_features:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)
            
            stats = config.get("stats", {})
            layer_a = stats.get("layerA", {})
            layer_b = stats.get("layerB", {})
            
            if layer_a.get("enabled"):
                self.layer_a_test.setCurrentText(layer_a.get("global_test", "friedman"))
                self.layer_a_posthoc.setCurrentText(layer_a.get("posthoc", "wilcoxon"))
                self.layer_a_correction.setCurrentText(layer_a.get("correction", "holm"))
            
            if layer_b.get("enabled"):
                self.layer_b_method.setCurrentText(layer_b.get("method", "mixedlm"))
                self.layer_b_by_delta.setChecked(layer_b.get("by_delta", True))
                self.layer_b_bootstrap_samples.setValue(layer_b.get("bootstrap_samples", 200))
            
            anytime = config.get("anytime", {})
            if anytime.get("enabled"):
                self.anytime_method.setCurrentText(anytime.get("method", "ecdf"))
                self.anytime_target_factor.setValue(anytime.get("target_factor", 1.1))
            
            ml = config.get("ml", {})
            if ml.get("enabled"):
                self.model_type.setCurrentText(ml.get("model_type", "random_forest"))
                self._on_model_changed(ml.get("model_type", "random_forest"))
                
                model_params = ml.get("model_params", {})
                model_type = ml.get("model_type", "random_forest")
                
                if model_type == "random_forest":
                    if hasattr(self, 'rf_n_estimators'):
                        self.rf_n_estimators.setValue(model_params.get("n_estimators", 100))
                    if hasattr(self, 'rf_max_depth'):
                        self.rf_max_depth.setValue(model_params.get("max_depth", 20))
                
                self.test_size.setValue(ml.get("test_size", 0.3))
                self.cv_folds.setValue(ml.get("cv_folds", 5))
                self.random_state.setValue(ml.get("random_state", 42))
                
                self.selector_type.setCurrentText(ml.get("selector_type", "selector"))
                self.risk_lambda.setValue(ml.get("risk_lambda", 0.5))
                
                analysis = ml.get("analysis", {})
                self.winner_hit.setChecked(analysis.get("winner_hit", True))
                self.penalty_analysis.setChecked(analysis.get("penalty", True))
                self.risk_ablation.setChecked(analysis.get("risk_ablation", False))
                self.shap_analysis.setChecked(analysis.get("shap", True))
                self.perm_importance.setChecked(analysis.get("perm_importance", True))
                self.isa_analysis.setChecked(analysis.get("isa", False))
                
        except Exception as e:
            print(f"Error loading config in MLTab: {e}")
