"""测试GUI ML分析流程"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import os

# 模拟MLAnalysisRunner
class TestRunner(QThread):
    progress_signal = Signal(int)
    log_signal = Signal(str)
    result_signal = Signal(dict)
    finished_signal = Signal(bool, str)
    
    def __init__(self):
        super().__init__()
    
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
            self.log_signal.emit("执行Layer A检验...")
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
            self.log_signal.emit("执行Layer B检验...")
            results['layerB_art'] = [{"term": "algorithm_name", "F": 10.5, "p_value": 0.001, "partial_eta2": 0.5}]
            self.progress_signal.emit(70)
            
            # Anytime
            self.log_signal.emit("执行Anytime分析...")
            results['anytime_analysis'] = {"summary": "测试分析结果"}
            results['anytime_output'] = str(project_root / "results" / "anytime")
            self.progress_signal.emit(90)
            
            self.progress_signal.emit(100)
            
            self.log_signal.emit(f"分析完成，结果键: {list(results.keys())}")
            self.result_signal.emit(results)
            self.finished_signal.emit(True, "测试完成")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"错误: {str(e)}")

def test_gui():
    app = QApplication(sys.argv)
    
    runner = TestRunner()
    
    def on_result(results):
        print(f"收到结果: {list(results.keys())}")
        print(f"friedman: {results.get('friedman', {})}")
        print(f"layerB_art: {results.get('layerB_art', [])}")
    
    def on_log(msg):
        print(f"[LOG] {msg}")
    
    def on_finished(success, msg):
        print(f"[FINISHED] success={success}, msg={msg}")
        app.quit()
    
    runner.result_signal.connect(on_result)
    runner.log_signal.connect(on_log)
    runner.finished_signal.connect(on_finished)
    
    runner.start()
    
    app.exec()

if __name__ == "__main__":
    test_gui()
