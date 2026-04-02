"""完整测试GUI ML分析流程"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit
from PySide6.QtCore import QThread, Signal

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import os

class MLAnalysisRunner(QThread):
    progress_signal = Signal(int)
    log_signal = Signal(str)
    result_signal = Signal(dict)
    finished_signal = Signal(bool, str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def run(self):
        try:
            self.log_signal.emit("正在加载实验结果...")
            self.progress_signal.emit(10)
            
            results_dir = project_root / "results" / "raw"
            csv_files = list(results_dir.glob("*.csv"))
            
            if not csv_files:
                self.finished_signal.emit(False, "没有找到实验结果文件！")
                return
            
            latest_file = max(csv_files, key=os.path.getctime)
            self.log_signal.emit(f"加载结果文件: {latest_file.name}")
            
            results_df = pd.read_csv(latest_file)
            self.log_signal.emit(f"数据行数: {len(results_df)}")
            self.progress_signal.emit(20)
            
            results = {}
            
            # Layer A
            stats_config = self.config.get('stats', {})
            if stats_config.get('layerA', {}).get('enabled', False):
                self.log_signal.emit("执行Layer A检验 (Friedman)...")
                
                pivot = results_df.pivot_table(
                    index='instance_id',
                    columns='algorithm_name',
                    values='best_objective',
                    aggfunc='median'
                ).dropna()
                
                if len(pivot.columns) >= 2:
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
                    self.log_signal.emit(f"Friedman检验完成: p={p_value:.4f}")
            
            self.progress_signal.emit(50)
            
            # Layer B
            if stats_config.get('layerB', {}).get('enabled', False):
                self.log_signal.emit("执行Layer B检验...")
                results['layerB_art'] = [{"term": "algorithm_name", "F": 10.5, "p_value": 0.001, "partial_eta2": 0.5}]
            
            self.progress_signal.emit(70)
            
            # Anytime
            anytime_config = self.config.get('anytime', {})
            if anytime_config.get('enabled', False):
                self.log_signal.emit("执行Anytime分析...")
                results['anytime_analysis'] = {"summary": "测试分析结果"}
                results['anytime_output'] = str(project_root / "results" / "anytime")
            
            self.progress_signal.emit(90)
            
            # ML
            ml_config = self.config.get('ml', {})
            if ml_config.get('enabled', False):
                self.log_signal.emit("执行机器学习分析...")
                results['ml_results'] = {
                    'SBS': {'algorithm': 'HS', 'score': 66.0},
                    'VBS': {'score': 62.0},
                    'Selector': {'score': 64.0, 'hit_rate': 0.5, 'improvement_over_sbs': 3.0, 'gap_to_vbs': 3.0}
                }
            
            self.progress_signal.emit(100)
            
            self.log_signal.emit(f"分析完成，结果键: {list(results.keys())}")
            self.result_signal.emit(results)
            self.finished_signal.emit(True, "测试完成")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"错误: {str(e)}")


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Analysis Test")
        self.resize(800, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.btn = QPushButton("Run ML Analysis")
        self.btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn)
        
        self.log_text = QTextEdit()
        layout.addWidget(self.log_text)
        
        self.results_text = QTextEdit()
        layout.addWidget(self.results_text)
    
    def log_message(self, msg):
        self.log_text.append(msg)
    
    def run_analysis(self):
        config = {
            'stats': {'layerA': {'enabled': True}, 'layerB': {'enabled': True}},
            'anytime': {'enabled': True},
            'ml': {'enabled': True}
        }
        
        self.runner = MLAnalysisRunner(config)
        self.runner.log_signal.connect(self.log_message)
        self.runner.result_signal.connect(self.display_results)
        self.runner.finished_signal.connect(self.on_finished)
        self.runner.start()
    
    def display_results(self, results):
        self.results_text.clear()
        html = f"<h2>分析结果</h2><p>共 {len(results)} 项结果</p>"
        
        if 'friedman' in results:
            html += f"<h3>Friedman检验</h3><p>p值: {results['friedman']['p_value']:.4f}</p>"
        
        if 'layerB_art' in results:
            html += f"<h3>Layer B</h3><p>F值: {results['layerB_art'][0]['F']}</p>"
        
        if 'ml_results' in results:
            html += f"<h3>ML结果</h3><p>SBS: {results['ml_results']['SBS']}</p>"
        
        self.results_text.setHtml(html)
        self.log_message(f"显示结果: {list(results.keys())}")
    
    def on_finished(self, success, msg):
        self.log_message(f"完成: {success}, {msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec()
