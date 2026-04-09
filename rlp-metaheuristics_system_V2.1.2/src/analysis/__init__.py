"""分析模块"""
from .result_analysis import (
    compute_statistics,
    plot_performance_profile,
    plot_anytime_curve,
    perform_statistical_tests,
    plot_rank_comparison,
    compute_feasibility_analysis,
    plot_feasibility_analysis,
    generate_summary_report
)

from .algorithm_selector import (
    AlgorithmSelector,
    analyze_selector
)

__all__ = [
    'compute_statistics',
    'plot_performance_profile',
    'plot_anytime_curve',
    'perform_statistical_tests',
    'plot_rank_comparison',
    'compute_feasibility_analysis',
    'plot_feasibility_analysis',
    'generate_summary_report',
    'AlgorithmSelector',
    'analyze_selector'
]
