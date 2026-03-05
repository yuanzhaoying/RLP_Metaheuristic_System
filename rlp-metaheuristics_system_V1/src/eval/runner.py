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
    GAParamsST,
    SAParamsST,
    ILSParamsST
)


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


def _params_to_dict(params: AlgorithmParams) -> Dict:
    result = {}
    for key, value in asdict(params).items():
        if isinstance(value, (int, float, str, bool)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[RunResult] = []

    def run_single(
        self,
        instance: RCPSPInstance,
        algorithm_name: str,
        seed: int,
        deadline: int,
        max_evaluations: int
    ) -> RunResult:
        params_map = {
            "ils": ILSParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                perturbation_strength=5,
                time_limit=self.config.time_limit
            ),
            "ga": GAParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                population_size=50,
                time_limit=self.config.time_limit
            ),
            "sa": SAParamsST(
                max_evaluations=max_evaluations,
                seed=seed,
                initial_temperature=10000,
                time_limit=self.config.time_limit
            )
        }

        params = params_map.get(algorithm_name)
        if params is None:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        algorithm, _ = create_algorithm_st(algorithm_name, instance, deadline, params)

        result = algorithm.run()

        return RunResult(
            instance_id=instance.name,
            seed=seed,
            best_objective=result.best_objective,
            runtime=result.runtime,
            algorithm_name=algorithm_name,
            algorithm_params=result.algorithm_params,
            deadline=deadline
        )

    def run_batch(
        self,
        data_dir: str,
        verbose: bool = True
    ) -> pd.DataFrame:
        results = []

        total_runs = (
            len(self.config.instances) *
            len(self.config.algorithms) *
            len(self.config.seeds)
        )

        iterator = tqdm(total=total_runs, disable=not verbose)

        for idx, instance_id in enumerate(self.config.instances):
            instance_path = instance_id

            if not os.path.exists(instance_path):
                print(f"Warning: Instance not found: {instance_path}")
                iterator.update(len(self.config.algorithms) * len(self.config.seeds))
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
                iterator.update(len(self.config.algorithms) * len(self.config.seeds))
                continue

            for algo in self.config.algorithms:
                for seed in self.config.seeds:
                    try:
                        result = self.run_single(
                            instance, algo, seed, deadline, self.config.max_evaluations
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
                        print(f"Error running {instance_id}/{algo}/{seed}: {e}")
                        row = {
                            "instance_id": instance_id,
                            "seed": seed,
                            "best_objective": float('inf'),
                            "runtime": 0,
                            "algorithm_name": algo,
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
