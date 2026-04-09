"""
Tab 1: Metaheuristics Configuration
元启发式算法配置选项卡
完全参照prlp-platform-v0.2/gui框架，填充app.py内容
"""
import os
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QFileDialog, QGroupBox, QScrollArea,
    QListWidget, QListWidgetItem, QTabWidget, QRadioButton, QButtonGroup,
    QMessageBox, QFrame, QSplitter
)
from PySide6.QtCore import Qt, QEvent


class NoScrollComboBox(QComboBox):
    """禁用滚轮事件的ComboBox"""
    
    def wheelEvent(self, event):
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    """禁用滚轮事件的SpinBox"""
    
    def wheelEvent(self, event):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """禁用滚轮事件的DoubleSpinBox"""
    
    def wheelEvent(self, event):
        event.ignore()


class MetaheuristicsTab(QWidget):
    """元启发式算法配置选项卡"""
    
    DATASET_MAX_COUNTS = {
        "j30": 480,
        "j60": 480,
        "j90": 480,
        "j120": 600
    }
    
    def __init__(self, plugin_loader=None, parent=None):
        super().__init__(parent)
        self.plugin_loader = plugin_loader
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
        
        self._create_instance_group(container_layout)
        
        self._create_budget_group(container_layout)
        
        self._create_algorithm_group(container_layout)
        
        self._create_operator_group(container_layout)
        
        self._create_preview_group(container_layout)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
    
    def _create_instance_group(self, parent_layout):
        """创建实例选择组"""
        group = QGroupBox("Instance Selection")
        layout = QGridLayout(group)
        
        layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_combo = NoScrollComboBox()
        self.dataset_combo.addItems(["j30", "j60", "j90", "j120", "custom"])
        self.dataset_combo.setCurrentText("j30")
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        layout.addWidget(self.dataset_combo, 0, 1)
        
        layout.addWidget(QLabel("Instance Count:"), 0, 2)
        self.instance_count_spin = NoScrollSpinBox()
        self.instance_count_spin.setRange(1, 480)
        self.instance_count_spin.setValue(10)
        layout.addWidget(self.instance_count_spin, 0, 3)
        
        self.custom_widget = QWidget()
        custom_layout = QGridLayout(self.custom_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        
        custom_layout.addWidget(QLabel("j30:"), 0, 0)
        self.custom_j30_spin = NoScrollSpinBox()
        self.custom_j30_spin.setRange(0, 480)
        self.custom_j30_spin.setValue(0)
        custom_layout.addWidget(self.custom_j30_spin, 0, 1)
        
        custom_layout.addWidget(QLabel("j60:"), 0, 2)
        self.custom_j60_spin = NoScrollSpinBox()
        self.custom_j60_spin.setRange(0, 480)
        self.custom_j60_spin.setValue(0)
        custom_layout.addWidget(self.custom_j60_spin, 0, 3)
        
        custom_layout.addWidget(QLabel("j90:"), 1, 0)
        self.custom_j90_spin = NoScrollSpinBox()
        self.custom_j90_spin.setRange(0, 480)
        self.custom_j90_spin.setValue(0)
        custom_layout.addWidget(self.custom_j90_spin, 1, 1)
        
        custom_layout.addWidget(QLabel("j120:"), 1, 2)
        self.custom_j120_spin = NoScrollSpinBox()
        self.custom_j120_spin.setRange(0, 600)
        self.custom_j120_spin.setValue(0)
        custom_layout.addWidget(self.custom_j120_spin, 1, 3)
        
        layout.addWidget(self.custom_widget, 1, 0, 1, 4)
        self.custom_widget.setVisible(False)
        
        parent_layout.addWidget(group)
    
    def _create_budget_group(self, parent_layout):
        """创建预算设置组"""
        group = QGroupBox("Budget Settings")
        layout = QGridLayout(group)
        
        layout.addWidget(QLabel("Budget Type:"), 0, 0)
        self.budget_type_combo = NoScrollComboBox()
        self.budget_type_combo.addItems(["evaluations", "time"])
        self.budget_type_combo.currentTextChanged.connect(self._on_budget_type_changed)
        layout.addWidget(self.budget_type_combo, 0, 1)
        
        layout.addWidget(QLabel("Max Evaluations:"), 0, 2)
        self.max_evals_spin = NoScrollSpinBox()
        self.max_evals_spin.setRange(100, 100000)
        self.max_evals_spin.setValue(1000)
        self.max_evals_spin.setSingleStep(100)
        layout.addWidget(self.max_evals_spin, 0, 3)
        
        layout.addWidget(QLabel("Time Limit (sec):"), 1, 0)
        self.time_limit_spin = NoScrollSpinBox()
        self.time_limit_spin.setRange(1, 3600)
        self.time_limit_spin.setValue(60)
        self.time_limit_spin.setEnabled(False)
        layout.addWidget(self.time_limit_spin, 1, 1)
        
        layout.addWidget(QLabel("Seeds:"), 1, 2)
        self.seeds_edit = QLineEdit()
        self.seeds_edit.setText("0, 1")
        self.seeds_edit.setToolTip("Comma-separated seed values (e.g., 0, 1, 2, 3)")
        layout.addWidget(self.seeds_edit, 1, 3)
        
        parent_layout.addWidget(group)
    
    def _create_algorithm_group(self, parent_layout):
        """创建算法选择组"""
        group = QGroupBox("Algorithm Selection")
        layout = QVBoxLayout(group)
        
        algo_layout = QGridLayout()
        
        self.algo_checkboxes = {}
        algorithms = [
            ("BA", "BA (Bat Algorithm)"),
            ("PSO", "PSO (Particle Swarm Optimization)"),
            ("HS", "HS (Harmony Search)"),
            ("GA", "GA (Genetic Algorithm)"),
            ("DE", "DE (Differential Evolution)"),
            ("PR", "PR (Path Relinking)"),
            ("TS", "TS (Tabu Search)"),
        ]
        
        for i, (algo_id, algo_name) in enumerate(algorithms):
            checkbox = QCheckBox(algo_name)
            checkbox.stateChanged.connect(self._on_algorithm_changed)
            self.algo_checkboxes[algo_id] = checkbox
            row = i // 2
            col = i % 2
            algo_layout.addWidget(checkbox, row, col)
        
        layout.addLayout(algo_layout)
        
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all_algorithms)
        btn_layout.addWidget(self.select_all_btn)
        
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self._clear_all_algorithms)
        btn_layout.addWidget(self.clear_selection_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        parent_layout.addWidget(group)
    
    def _create_operator_group(self, parent_layout):
        """创建算子配置组"""
        self.operator_group = QGroupBox("Operator Configuration")
        self.operator_layout = QVBoxLayout(self.operator_group)
        
        self.operator_widgets = {}
        
        ba_widget = self._create_ba_operators()
        self.operator_widgets["BA"] = ba_widget
        self.operator_layout.addWidget(ba_widget)
        ba_widget.setVisible(False)
        
        pso_widget = self._create_pso_operators()
        self.operator_widgets["PSO"] = pso_widget
        self.operator_layout.addWidget(pso_widget)
        pso_widget.setVisible(False)
        
        hs_widget = self._create_hs_operators()
        self.operator_widgets["HS"] = hs_widget
        self.operator_layout.addWidget(hs_widget)
        hs_widget.setVisible(False)
        
        ga_widget = self._create_ga_operators()
        self.operator_widgets["GA"] = ga_widget
        self.operator_layout.addWidget(ga_widget)
        ga_widget.setVisible(False)
        
        de_widget = self._create_de_operators()
        self.operator_widgets["DE"] = de_widget
        self.operator_layout.addWidget(de_widget)
        de_widget.setVisible(False)
        
        pr_widget = self._create_pr_operators()
        self.operator_widgets["PR"] = pr_widget
        self.operator_layout.addWidget(pr_widget)
        pr_widget.setVisible(False)
        
        ts_widget = self._create_ts_operators()
        self.operator_widgets["TS"] = ts_widget
        self.operator_layout.addWidget(ts_widget)
        ts_widget.setVisible(False)
        
        parent_layout.addWidget(self.operator_group)
    
    def _create_ba_operators(self):
        """创建BA算子配置"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.ba_ls_combo = NoScrollComboBox()
        self.ba_ls_combo.addItems(["none", "tlim"])
        layout.addRow("BA Local Search:", self.ba_ls_combo)
        
        return widget
    
    def _create_pso_operators(self):
        """创建PSO算子配置"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.pso_ls_combo = NoScrollComboBox()
        self.pso_ls_combo.addItems(["none", "sa"])
        layout.addRow("PSO Local Search:", self.pso_ls_combo)
        
        self.pso_restart_combo = NoScrollComboBox()
        self.pso_restart_combo.addItems(["none", "adaptive"])
        layout.addRow("PSO Restart:", self.pso_restart_combo)
        
        return widget
    
    def _create_hs_operators(self):
        """创建HS算子配置"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.hs_param_combo = NoScrollComboBox()
        self.hs_param_combo.addItems(["fixed", "adaptive"])
        layout.addRow("HS Parameter Strategy:", self.hs_param_combo)
        
        self.hs_init_combo = NoScrollComboBox()
        self.hs_init_combo.addItems(["random", "forward"])
        layout.addRow("HS Initialization:", self.hs_init_combo)
        
        return widget
    
    def _create_ga_operators(self):
        """创建GA算子配置"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.ga_selection_combo = NoScrollComboBox()
        self.ga_selection_combo.addItems(["roulette", "tournament"])
        layout.addRow("GA Selection:", self.ga_selection_combo)
        
        self.ga_crossover_combo = NoScrollComboBox()
        self.ga_crossover_combo.addItems(["single_point", "two_point", "rcx", "hybrid"])
        layout.addRow("GA Crossover:", self.ga_crossover_combo)
        
        self.ga_mutation_combo = NoScrollComboBox()
        self.ga_mutation_combo.addItems(["random", "adaptive"])
        layout.addRow("GA Mutation:", self.ga_mutation_combo)
        
        self.ga_init_combo = NoScrollComboBox()
        self.ga_init_combo.addItems(["random", "heuristic"])
        layout.addRow("GA Initialization:", self.ga_init_combo)
        
        self.ga_ls_combo = NoScrollComboBox()
        self.ga_ls_combo.addItems(["none", "activity", "shift"])
        layout.addRow("GA Local Search:", self.ga_ls_combo)
        
        self.ga_neighborhood_check = QCheckBox()
        layout.addRow("GA Neighborhood:", self.ga_neighborhood_check)
        
        self.ga_elitism_check = QCheckBox()
        layout.addRow("GA Elitism:", self.ga_elitism_check)
        
        self.ga_sa_acceptance_check = QCheckBox()
        layout.addRow("GA SA Acceptance:", self.ga_sa_acceptance_check)
        
        return widget
    
    def _create_de_operators(self):
        """创建DE算子配置"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.de_mutation_combo = NoScrollComboBox()
        self.de_mutation_combo.addItems(["rand/1", "rand/2", "best/1", "best/2", "adaptive", "current-to-rand/2"])
        self.de_mutation_combo.currentTextChanged.connect(self._on_de_mutation_changed)
        layout.addRow("DE Mutation:", self.de_mutation_combo)
        
        self.de_crossover_combo = NoScrollComboBox()
        self.de_crossover_combo.addItems(["bin", "exp"])
        layout.addRow("DE Crossover:", self.de_crossover_combo)
        
        self.de_adaptive_f_check = QCheckBox()
        self.de_adaptive_f_check.setToolTip("Only available for rand/1, rand/2, best/1, best/2")
        layout.addRow("DE Adaptive F:", self.de_adaptive_f_check)
        
        self.de_adaptive_cr_check = QCheckBox()
        layout.addRow("DE Adaptive CR:", self.de_adaptive_cr_check)
        
        self.de_ls_check = QCheckBox()
        layout.addRow("DE Local Search:", self.de_ls_check)
        
        self._on_de_mutation_changed(self.de_mutation_combo.currentText())
        
        return widget
    
    def _create_pr_operators(self):
        """创建PR算子配置"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.pr_path_combo = NoScrollComboBox()
        self.pr_path_combo.addItems(["forward", "backward", "random", "bidirectional"])
        layout.addRow("PR Path Strategy:", self.pr_path_combo)
        
        self.pr_selection_combo = NoScrollComboBox()
        self.pr_selection_combo.addItems(["best", "random_two"])
        layout.addRow("PR Selection Strategy:", self.pr_selection_combo)
        
        self.pr_ls_check = QCheckBox()
        layout.addRow("PR Local Search:", self.pr_ls_check)
        
        return widget
    
    def _create_ts_operators(self):
        """创建TS算子配置"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.ts_strategy_combo = NoScrollComboBox()
        self.ts_strategy_combo.addItems(["static", "dynamic"])
        layout.addRow("TS Tabu Strategy:", self.ts_strategy_combo)
        
        return widget
    
    def _create_preview_group(self, parent_layout):
        """创建实验配置预览组"""
        group = QGroupBox("Experiment Preview")
        layout = QVBoxLayout(group)
        
        self.preview_label = QLabel("Click 'Update Preview' to see experiment summary")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        
        self.update_preview_btn = QPushButton("Update Preview")
        self.update_preview_btn.clicked.connect(self._update_preview)
        layout.addWidget(self.update_preview_btn)
        
        parent_layout.addWidget(group)
    
    def _on_dataset_changed(self, dataset: str):
        """数据集改变时更新UI"""
        if dataset == "custom":
            self.custom_widget.setVisible(True)
            self.instance_count_spin.setEnabled(False)
        else:
            self.custom_widget.setVisible(False)
            self.instance_count_spin.setEnabled(True)
            max_count = self.DATASET_MAX_COUNTS.get(dataset, 480)
            self.instance_count_spin.setMaximum(max_count)
            if self.instance_count_spin.value() > max_count:
                self.instance_count_spin.setValue(min(10, max_count))
    
    def _on_budget_type_changed(self, budget_type: str):
        """预算类型改变时更新UI"""
        if budget_type == "evaluations":
            self.max_evals_spin.setEnabled(True)
            self.time_limit_spin.setEnabled(False)
        else:
            self.max_evals_spin.setEnabled(False)
            self.time_limit_spin.setEnabled(True)
    
    def _on_algorithm_changed(self):
        """算法选择改变时更新算子配置区域"""
        for algo_id, widget in self.operator_widgets.items():
            checkbox = self.algo_checkboxes.get(algo_id)
            if checkbox:
                widget.setVisible(checkbox.isChecked())
    
    def _on_de_mutation_changed(self, mutation: str):
        """DE Mutation改变时更新Adaptive F选项"""
        if mutation in ["rand/1", "rand/2", "best/1", "best/2"]:
            self.de_adaptive_f_check.setEnabled(True)
        else:
            self.de_adaptive_f_check.setEnabled(False)
            self.de_adaptive_f_check.setChecked(False)
    
    def _select_all_algorithms(self):
        """选择所有算法"""
        for checkbox in self.algo_checkboxes.values():
            checkbox.setChecked(True)
    
    def _clear_all_algorithms(self):
        """清除所有算法选择"""
        for checkbox in self.algo_checkboxes.values():
            checkbox.setChecked(False)
    
    def _update_preview(self):
        """更新实验预览"""
        try:
            config = self.get_config()
            
            n_algos = len(config.get("selected_algorithms", []))
            
            dataset = config.get("instance_sets", [{}])[0].get("name", "j30")
            if dataset == "custom":
                n_instances = sum([
                    config.get("instance_sets", [{}])[0].get("j30_count", 0),
                    config.get("instance_sets", [{}])[0].get("j60_count", 0),
                    config.get("instance_sets", [{}])[0].get("j90_count", 0),
                    config.get("instance_sets", [{}])[0].get("j120_count", 0),
                ])
            else:
                n_instances = config.get("instance_sets", [{}])[0].get("limit", 0)
            
            n_seeds = len(config.get("seeds", []))
            
            total_runs = n_algos * n_instances * n_seeds
            
            budget_mode = config.get('main_budget', {}).get('mode', 'evals')
            budget_value = config.get('main_budget', {}).get('max_evals', 1000) if budget_mode == 'evals' else config.get('main_budget', {}).get('budget_sec', 60)
            
            preview_text = (
                f"<b>Experiment Summary:</b><br>"
                f"- Algorithms: {n_algos}<br>"
                f"- Instances: {n_instances}<br>"
                f"- Seeds: {n_seeds}<br>"
                f"- <b>Total Runs: {total_runs}</b><br><br>"
                f"- Budget Mode: {budget_mode}<br>"
                f"- Budget Value: {budget_value}"
            )
            
            self.preview_label.setText(preview_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate preview: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        config = {}
        
        dataset = self.dataset_combo.currentText()
        
        if dataset == "custom":
            config["instance_sets"] = [{
                "name": "custom",
                "j30_count": self.custom_j30_spin.value(),
                "j60_count": self.custom_j60_spin.value(),
                "j90_count": self.custom_j90_spin.value(),
                "j120_count": self.custom_j120_spin.value(),
                "limit": (self.custom_j30_spin.value() + self.custom_j60_spin.value() + 
                         self.custom_j90_spin.value() + self.custom_j120_spin.value())
            }]
        else:
            config["instance_sets"] = [{
                "name": dataset,
                "limit": self.instance_count_spin.value()
            }]
        
        budget_type = self.budget_type_combo.currentText()
        if budget_type == "evaluations":
            config["main_budget"] = {
                "mode": "evals",
                "max_evals": self.max_evals_spin.value()
            }
        else:
            config["main_budget"] = {
                "mode": "time",
                "budget_sec": self.time_limit_spin.value()
            }
        
        seeds_text = self.seeds_edit.text().strip()
        config["seeds"] = [int(x.strip()) for x in seeds_text.split(",") if x.strip()]
        
        selected_algos = []
        for algo_id, checkbox in self.algo_checkboxes.items():
            if checkbox.isChecked():
                selected_algos.append(algo_id)
        config["selected_algorithms"] = selected_algos
        
        config["operator_config"] = {
            "ba_ls": self.ba_ls_combo.currentText(),
            "pso_ls": self.pso_ls_combo.currentText(),
            "pso_restart": self.pso_restart_combo.currentText(),
            "hs_param": self.hs_param_combo.currentText(),
            "hs_init": self.hs_init_combo.currentText(),
            "ga_selection": self.ga_selection_combo.currentText(),
            "ga_crossover": self.ga_crossover_combo.currentText(),
            "ga_mutation": self.ga_mutation_combo.currentText(),
            "ga_init": self.ga_init_combo.currentText(),
            "ga_ls": self.ga_ls_combo.currentText(),
            "ga_neighborhood": self.ga_neighborhood_check.isChecked(),
            "ga_elitism": self.ga_elitism_check.isChecked(),
            "ga_sa_acceptance": self.ga_sa_acceptance_check.isChecked(),
            "de_mutation": self.de_mutation_combo.currentText(),
            "de_crossover": self.de_crossover_combo.currentText(),
            "de_adaptive_f": self.de_adaptive_f_check.isChecked(),
            "de_adaptive_cr": self.de_adaptive_cr_check.isChecked(),
            "de_ls": self.de_ls_check.isChecked(),
            "pr_path": self.pr_path_combo.currentText(),
            "pr_selection": self.pr_selection_combo.currentText(),
            "pr_ls": self.pr_ls_check.isChecked(),
            "ts_strategy": self.ts_strategy_combo.currentText(),
        }
        
        config["deltas"] = [0.0]
        
        config["baseline_T0"] = {
            "tries": 30,
            "seed": 0
        }
        
        config["rlp"] = {
            "renewable": True,
            "horizon_mode": "slack_over_T0"
        }
        
        config["objective"] = {
            "mode": "builtin",
            "id": "SOS"
        }
        
        config["aux_time_budget"] = {
            "enabled": False,
            "budget_sec": 1.0
        }
        
        config["trace_fracs"] = [0.05, 0.1, 0.2, 0.5, 1.0]
        
        config["stats"] = {
            "layerA": {"enabled": True},
            "layerB": {"enabled": False}
        }
        
        config["features"] = {"enabled": False}
        
        config["ml"] = {"enabled": False}
        
        config["reproducibility"] = {"enabled": False}
        
        return config
    
    def load_config(self, config: Dict[str, Any]):
        """加载配置"""
        try:
            instance_sets = config.get("instance_sets", [])
            if instance_sets:
                set_name = instance_sets[0].get("name", "j30")
                self.dataset_combo.setCurrentText(set_name)
                
                if set_name == "custom":
                    self.custom_j30_spin.setValue(instance_sets[0].get("j30_count", 0))
                    self.custom_j60_spin.setValue(instance_sets[0].get("j60_count", 0))
                    self.custom_j90_spin.setValue(instance_sets[0].get("j90_count", 0))
                    self.custom_j120_spin.setValue(instance_sets[0].get("j120_count", 0))
                else:
                    self.instance_count_spin.setValue(instance_sets[0].get("limit", 10))
            
            main_budget = config.get("main_budget", {})
            if main_budget.get("mode") == "time":
                self.budget_type_combo.setCurrentText("time")
                self.time_limit_spin.setValue(main_budget.get("budget_sec", 60))
            else:
                self.budget_type_combo.setCurrentText("evaluations")
                self.max_evals_spin.setValue(main_budget.get("max_evals", 1000))
            
            seeds = config.get("seeds", [0, 1])
            self.seeds_edit.setText(", ".join(map(str, seeds)))
            
            selected_algos = config.get("selected_algorithms", [])
            for algo_id, checkbox in self.algo_checkboxes.items():
                checkbox.setChecked(algo_id in selected_algos)
            
            operator_config = config.get("operator_config", {})
            if "ba_ls" in operator_config:
                self.ba_ls_combo.setCurrentText(operator_config["ba_ls"])
            if "pso_ls" in operator_config:
                self.pso_ls_combo.setCurrentText(operator_config["pso_ls"])
            if "pso_restart" in operator_config:
                self.pso_restart_combo.setCurrentText(operator_config["pso_restart"])
            if "hs_param" in operator_config:
                self.hs_param_combo.setCurrentText(operator_config["hs_param"])
            if "hs_init" in operator_config:
                self.hs_init_combo.setCurrentText(operator_config["hs_init"])
            if "ga_selection" in operator_config:
                self.ga_selection_combo.setCurrentText(operator_config["ga_selection"])
            if "ga_crossover" in operator_config:
                self.ga_crossover_combo.setCurrentText(operator_config["ga_crossover"])
            if "ga_mutation" in operator_config:
                self.ga_mutation_combo.setCurrentText(operator_config["ga_mutation"])
            if "ga_init" in operator_config:
                self.ga_init_combo.setCurrentText(operator_config["ga_init"])
            if "ga_ls" in operator_config:
                self.ga_ls_combo.setCurrentText(operator_config["ga_ls"])
            if "ga_neighborhood" in operator_config:
                self.ga_neighborhood_check.setChecked(operator_config["ga_neighborhood"])
            if "ga_elitism" in operator_config:
                self.ga_elitism_check.setChecked(operator_config["ga_elitism"])
            if "ga_sa_acceptance" in operator_config:
                self.ga_sa_acceptance_check.setChecked(operator_config["ga_sa_acceptance"])
            if "de_mutation" in operator_config:
                self.de_mutation_combo.setCurrentText(operator_config["de_mutation"])
            if "de_crossover" in operator_config:
                self.de_crossover_combo.setCurrentText(operator_config["de_crossover"])
            if "de_adaptive_f" in operator_config:
                self.de_adaptive_f_check.setChecked(operator_config["de_adaptive_f"])
            if "de_adaptive_cr" in operator_config:
                self.de_adaptive_cr_check.setChecked(operator_config["de_adaptive_cr"])
            if "de_ls" in operator_config:
                self.de_ls_check.setChecked(operator_config["de_ls"])
            if "pr_path" in operator_config:
                self.pr_path_combo.setCurrentText(operator_config["pr_path"])
            if "pr_selection" in operator_config:
                self.pr_selection_combo.setCurrentText(operator_config["pr_selection"])
            if "pr_ls" in operator_config:
                self.pr_ls_check.setChecked(operator_config["pr_ls"])
            if "ts_strategy" in operator_config:
                self.ts_strategy_combo.setCurrentText(operator_config["ts_strategy"])
            
        except Exception as e:
            print(f"Error loading config in MetaheuristicsTab: {e}")
