#!/usr/bin/env python
"""
RLP Metaheuristics GUI 启动脚本
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from gui.main import main
    main()


