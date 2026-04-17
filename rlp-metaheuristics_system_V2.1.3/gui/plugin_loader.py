"""
Plugin Loader - 插件加载器
加载和管理编码、算子、骨架、目标函数插件
"""
import os
import sys
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class PluginMeta:
    """插件元数据"""
    name: str
    version: str
    description: str
    plugin_type: str
    params_schema: Dict[str, Any]


@dataclass
class Plugin:
    """插件"""
    meta: PluginMeta
    module: Any
    compute_func: Optional[Callable] = None


class PluginLoader:
    """插件加载器"""
    
    def __init__(self, plugins_dir: Optional[str] = None):
        if plugins_dir is None:
            project_root = Path(__file__).parent.parent
            self.plugins_dir = project_root / "plugins"
        else:
            self.plugins_dir = Path(plugins_dir)
        
        self.encodings: Dict[str, Plugin] = {}
        self.operators: Dict[str, Plugin] = {}
        self.skeletons: Dict[str, Plugin] = {}
        self.objectives: Dict[str, Plugin] = {}
        
        self._load_builtin_plugins()
    
    def _load_builtin_plugins(self):
        """加载内置插件"""
        self.encodings["AL"] = Plugin(
            meta=PluginMeta(
                name="Activity List",
                version="1.0",
                description="Activity List encoding",
                plugin_type="encoding",
                params_schema={}
            ),
            module=None
        )
        
        self.encodings["RK"] = Plugin(
            meta=PluginMeta(
                name="Random Key",
                version="1.0",
                description="Random Key encoding",
                plugin_type="encoding",
                params_schema={"sigma": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0}}
            ),
            module=None
        )
        
        self.operators["swap"] = Plugin(
            meta=PluginMeta(
                name="Swap",
                version="1.0",
                description="Swap two activities",
                plugin_type="operator",
                params_schema={}
            ),
            module=None
        )
        
        self.operators["insertion"] = Plugin(
            meta=PluginMeta(
                name="Insertion",
                version="1.0",
                description="Insert an activity at a new position",
                plugin_type="operator",
                params_schema={}
            ),
            module=None
        )
        
        self.skeletons["GA"] = Plugin(
            meta=PluginMeta(
                name="Genetic Algorithm",
                version="1.0",
                description="Genetic Algorithm metaheuristic",
                plugin_type="skeleton",
                params_schema={
                    "pop_size": {"type": "int", "default": 30, "min": 10, "max": 100},
                    "tournament_k": {"type": "int", "default": 3, "min": 2, "max": 10},
                    "elitism": {"type": "int", "default": 1, "min": 0, "max": 10},
                    "crossover": {"type": "select", "options": ["ox", "blend"], "default": "ox"},
                    "mutation": {"type": "select", "options": ["swap", "insertion", "gauss"], "default": "swap"},
                    "mut_prob": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0}
                }
            ),
            module=None
        )
        
        self.objectives["SOS"] = Plugin(
            meta=PluginMeta(
                name="Sum of Squares",
                version="1.0",
                description="Sum of squares of resource usage",
                plugin_type="objective",
                params_schema={}
            ),
            module=None
        )
        
        self.objectives["PEAK"] = Plugin(
            meta=PluginMeta(
                name="Peak",
                version="1.0",
                description="Peak resource usage",
                plugin_type="objective",
                params_schema={}
            ),
            module=None
        )
    
    def load_plugin_from_file(self, file_path: str, plugin_type: str) -> Optional[Plugin]:
        try:
            spec = importlib.util.spec_from_file_location("plugin", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, "PLUGIN_META"):
                print(f"Warning: Plugin {file_path} does not define PLUGIN_META")
                return None
            
            meta_dict = module.PLUGIN_META
            meta = PluginMeta(
                name=meta_dict.get("name", "Unknown"),
                version=meta_dict.get("version", "1.0"),
                description=meta_dict.get("description", ""),
                plugin_type=plugin_type,
                params_schema=meta_dict.get("params_schema", {})
            )
            
            compute_func = getattr(module, "compute", None)
            
            plugin = Plugin(meta=meta, module=module, compute_func=compute_func)
            
            if plugin_type == "encoding":
                self.encodings[meta.name] = plugin
            elif plugin_type == "operator":
                self.operators[meta.name] = plugin
            elif plugin_type == "skeleton":
                self.skeletons[meta.name] = plugin
            elif plugin_type == "objective":
                self.objectives[meta.name] = plugin
            
            return plugin
            
        except Exception as e:
            print(f"Error loading plugin from {file_path}: {e}")
            return None
    
    def get_encoding(self, name: str) -> Optional[Plugin]:
        return self.encodings.get(name)
    
    def get_operator(self, name: str) -> Optional[Plugin]:
        return self.operators.get(name)
    
    def get_skeleton(self, name: str) -> Optional[Plugin]:
        return self.skeletons.get(name)
    
    def get_objective(self, name: str) -> Optional[Plugin]:
        return self.objectives.get(name)
    
    def list_encodings(self) -> List[str]:
        return list(self.encodings.keys())
    
    def list_operators(self) -> List[str]:
        return list(self.operators.keys())
    
    def list_skeletons(self) -> List[str]:
        return list(self.skeletons.keys())
    
    def list_objectives(self) -> List[str]:
        return list(self.objectives.keys())
