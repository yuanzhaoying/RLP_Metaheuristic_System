from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import os
import csv
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ..psp.psplib_io import RCPSPInstance, load_psplib_sm
from ..alg.start_time_algorithms import (
    create_algorithm_st,
    AlgorithmResult,
    _params_to_dict
)
from ..alg.GA import GeneticAlgorithmST, GAParamsST
from ..alg.SA import SimulatedAnnealingST, SAParamsST
from ..alg.ILS import IteratedLocalSearchST, ILSParamsST
from ..alg.TS import TabuSearchST, TSParamsST
from ..alg.PR import PathRelinkingST, PRParamsST
from ..alg.GSA import GravitationalSearchAlgorithmST, GSAParamsST
from ..alg.DE import DifferentialEvolutionST, DEParamsST


@dataclass
class RunResult:
    instance_id: str
    seed: int
    best_objective: float
    runtime: float
    algorithm_name: str
    algorithm_params: Dict
    deadline: int


@dataclass
class ExperimentConfig:
    instances: List[str]
    algorithms: List[str]
    seeds: List[int]
    deadlines: List[int]
    max_evaluations: int
    output_dir: str
    time_limit: float = 10.0
    problem_type: str = "rlp"
    use_delay_factors: bool = False


def _params_to_dict(params) -> Dict:
    result = {}
    for key, value in asdict(params).items():
        if isinstance(value, (int, float, str, bool)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def generate_all_algorithm_configs():
    """生成所有算法的所有算子组合配置"""
    configs = []
    
    # DE算法的所有算子组合（80种）
    mutation_configs = [
        ("rand/1", True, "rand/1_adaptiveF"),
        ("rand/1", False, "rand/1_fixedF"),
        ("rand/2", True, "rand/2_adaptiveF"),
        ("rand/2", False, "rand/2_fixedF"),
        ("best/1", True, "best/1_adaptiveF"),
        ("best/1", False, "best/1_fixedF"),
        ("best/2", True, "best/2_adaptiveF"),
        ("best/2", False, "best/2_fixedF"),
        ("adaptive", True, "adaptive"),
        ("current-to-rand/2", False, "current-to-rand/2"),
    ]
    
    crossover_configs = [
        ("bin", True, "bin_adaptiveCR"),
        ("bin", False, "bin_fixedCR"),
        ("exp", True, "exp_adaptiveCR"),
        ("exp", False, "exp_fixedCR"),
    ]
    
    local_search_configs = [
        (True, "withLS"),
        (False, "noLS"),
    ]
    
    for mut_strat, use_adapt_F, mut_name in mutation_configs:
        for cross_strat, use_adapt_CR, cross_name in crossover_configs:
            for use_ls, ls_name in local_search_configs:
                config_name = f"DE_{mut_name}_{cross_name}_{ls_name}"
                configs.append((config_name, "DE", {
                    "mutation_strategy": mut_strat,
                    "use_adaptive_F": use_adapt_F,
                    "crossover_strategy": cross_strat,
                    "use_adaptive_CR": use_adapt_CR,
                    "use_local_search": use_ls,
                }))
    
    # GA算法的所有算子组合（96种）
    selection_strategies = ["roulette", "tournament"]
    crossover_strategies = ["single_point", "two_point", "rcx", "hybrid"]
    mutation_strategies = ["random", "adaptive"]
    initialization_strategies = ["random", "heuristic"]
    local_search_options = [True, False]
    neighborhood_options = [True, False]
    
    for sel in selection_strategies:
        for cross in crossover_strategies:
            for mut in mutation_strategies:
                for init in initialization_strategies:
                    for use_ls in local_search_options:
                        for use_neighborhood in neighborhood_options:
                            config_name = f"GA_{sel}_{cross}_{mut}_{init}_{'withLS' if use_ls else 'noLS'}_{'withNeighborhood' if use_neighborhood else 'noNeighborhood'}"
                            configs.append((config_name, "GA", {
                                "selection_strategy": sel,
                                "crossover_strategy": cross,
                                "mutation_strategy": mut,
                                "initialization_strategy": init,
                                "use_local_search": use_ls,
                                "neighborhood_size": 2 if use_neighborhood else 0,
                            }))
    
    # SA算法（1种，没有明显算子选择）
    configs.append(("SA_default", "SA", {}))
    
    # ILS算法（5种扰动强度）
    for strength in [3, 5, 7, 10, 15]:
        config_name = f"ILS_strength{strength}"
        configs.append((config_name, "ILS", {
            "perturbation_strength": strength,
        }))
    
    # TS算法（2种禁忌策略）
    for strategy in ["static", "dynamic"]:
        config_name = f"TS_{strategy}"
        configs.append((config_name, "TS", {
            "tabu_strategy": strategy,
        }))
    
    # PR算法（8种组合）
    path_strategies = ["forward", "backward"]
    selection_strategies = ["best", "random"]
    local_search_options = [True, False]
    
    for path_strat in path_strategies:
        for sel_strat in selection_strategies:
            for use_ls in local_search_options:
                config_name = f"PR_{path_strat}_{sel_strat}_{'withLS' if use_ls else 'noLS'}"
                configs.append((config_name, "PR", {
                    "path_strategy": path_strat,
                    "selection_strategy": sel_strat,
                    "use_local_search": use_ls,
                }))
    
    # GSA算法（1种，没有明显算子选择）
    configs.append(("GSA_default", "GSA", {}))
    
    return configs


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[RunResult] = []
        self.all_configs = generate_all_algorithm_configs()

    def run_single(
        self,
        instance: RCPSPInstance,
        algorithm_config: tuple,
        seed: int,
        deadline: int,
        max_evaluations: int
    ) -> RunResult:
        config_name, algo_type, extra_params = algorithm_config
        
        # 根据算法类型创建参数
        if algo_type == "DE":
            params = DEParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                population_size=50,
                time_limit=self.config.time_limit,
                F=0.1,
                CR=0.9,
                **extra_params
            )
        elif algo_type == "GA":
            params = GAParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                population_size=50,
                time_limit=self.config.time_limit,
                **extra_params
            )
        elif algo_type == "SA":
            params = SAParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                time_limit=self.config.time_limit,
                **extra_params
            )
        elif algo_type == "ILS":
            params = ILSParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                time_limit=self.config.time_limit,
                **extra_params
            )
        elif algo_type == "TS":
            params = TSParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                time_limit=self.config.time_limit,
                **extra_params
            )
        elif algo_type == "PR":
            params = PRParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                time_limit=self.config.time_limit,
                **extra_params
            )
        elif algo_type == "GSA":
            params = GSAParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                population_size=50,
                time_limit=self.config.time_limit,
                **extra_params
            )
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}")
        
        algorithm, _ = create_algorithm_st(algo_type.lower(), instance, deadline, params)
        
        result = algorithm.run()
        
        return RunResult(
            instance_id=instance.name,
            seed=seed,
            best_objective=result.best_objective,
            runtime=result.runtime,
            algorithm_name=config_name,
            algorithm_params=result.algorithm_params,
            deadline=deadline
        )

    def run_batch(
        self,
        data_dir: str,
        verbose: bool = True
    ) -> pd.DataFrame:
        results = []
        
        # 过滤出用户指定的算法
        if "all" in self.config.algorithms:
            selected_configs = self.all_configs
        else:
            selected_configs = [
                cfg for cfg in self.all_configs
                if any(algo.lower() in cfg[0].lower() for algo in self.config.algorithms)
            ]
        
        total_runs = (
            len(self.config.instances) *
            len(selected_configs) *
            len(self.config.seeds)
        )
        
        iterator = tqdm(total=total_runs, disable=not verbose)
        
        for idx, instance_id in enumerate(self.config.instances):
            instance_path = instance_id
            
            if not os.path.exists(instance_path):
                print(f"Warning: Instance not found: {instance_path}")
                iterator.update(len(selected_configs) * len(self.config.seeds))
                continue
            
            try:
                instance = load_psplib_sm(instance_path)
            except Exception as e:
                print(f"Error loading {instance_id}: {e}")
                continue
            
            if idx < len(self.config.deadlines):
                deadline = self.config.deadlines[idx]
            else:
                print(f"Warning: No deadline for instance {instance_id}, skipping")
                iterator.update(len(selected_configs) * len(self.config.seeds))
                continue
            
            for algo_config in selected_configs:
                for seed in self.config.seeds:
                    try:
                        result = self.run_single(
                            instance, algo_config, seed, deadline, self.config.max_evaluations
                        )
                        row = {
                            "instance_id": result.instance_id,
                            "seed": result.seed,
                            "best_objective": result.best_objective,
                            "runtime": result.runtime,
                            "algorithm_name": result.algorithm_name,
                            "deadline": result.deadline,
                        }
                        for key, value in result.algorithm_params.items():
                            row[f"param_{key}"] = value
                        results.append(row)
                    except Exception as e:
                        print(f"Error running {instance_id}/{algo_config[0]}/{seed}: {e}")
                        row = {
                            "instance_id": instance_id,
                            "seed": seed,
                            "best_objective": float('inf'),
                            "runtime": 0,
                            "algorithm_name": algo_config[0],
                            "deadline": deadline,
                        }
                        results.append(row)
                    
                    iterator.update(1)
        
        iterator.close()
        
        df = pd.DataFrame(results)
        self.results = results
        return df

    def save_results(self, output_path: str, format: str = "csv"):
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Results saved to {output_path}")
