"""
Export Utils - 导出工具
导出结果到SVG/PDF/Excel格式
"""
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


class ExportUtils:
    """导出工具类"""
    
    def __init__(self, results_dir: Optional[str] = None):
        if results_dir is None:
            project_root = Path(__file__).parent.parent
            self.results_dir = project_root / "results"
        else:
            self.results_dir = Path(results_dir)
    
    def export_all(self, export_dir: Path) -> Dict[str, Any]:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "exported_files": [],
            "errors": []
        }
        
        try:
            csv_files = self._export_csv_files(export_dir)
            results["exported_files"].extend(csv_files)
        except Exception as e:
            results["errors"].append(f"CSV export error: {e}")
        
        try:
            chart_files = self._export_charts(export_dir)
            results["exported_files"].extend(chart_files)
        except Exception as e:
            results["errors"].append(f"Chart export error: {e}")
        
        return results
    
    def _export_csv_files(self, export_dir: Path) -> List[str]:
        exported = []
        
        csv_source_dir = self.results_dir / "raw"
        if csv_source_dir.exists():
            csv_export_dir = export_dir / "csv"
            csv_export_dir.mkdir(exist_ok=True)
            
            for csv_file in csv_source_dir.glob("*.csv"):
                dest = csv_export_dir / csv_file.name
                shutil.copy2(csv_file, dest)
                exported.append(str(dest))
        
        return exported
    
    def _export_charts(self, export_dir: Path) -> List[str]:
        exported = []
        
        chart_extensions = [".svg", ".png", ".pdf"]
        
        for ext in chart_extensions:
            for chart_file in self.results_dir.rglob(f"*{ext}"):
                relative_path = chart_file.relative_to(self.results_dir)
                dest = export_dir / "charts" / relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(chart_file, dest)
                exported.append(str(dest))
        
        return exported
    
    def generate_report(self, export_dir: Path) -> Path:
        report_path = export_dir / "report.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RLP Experiment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .chart { margin: 20px 0; text-align: center; }
                .chart img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>RLP Metaheuristics Experiment Report</h1>
            <p>Generated on: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <h2>Summary</h2>
            <div id="summary">
            </div>
            
            <h2>Performance Results</h2>
            <div id="performance">
            </div>
            
            <h2>Charts</h2>
            <div id="charts">
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
