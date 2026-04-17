# src/prlp/eval/anytime.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_ecdf(data: np.ndarray):
    """
    Compute empirical cumulative distribution function.
    """
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def data_profile(runs: pd.DataFrame, out_dir: str):
    """
    Generate data profile: fraction of instances solved within a certain factor of best.
    """
    os.makedirs(out_dir, exist_ok=True)

    # For each instance, compute the best known objective
    best_per_instance = runs.groupby("instance_id")["best_obj"].min()

    # For each algorithm, compute for each instance the ratio of its best_obj to the instance's best
    rows = []
    for algo in runs["algo_id"].unique():
        algo_runs = runs[runs["algo_id"] == algo]
        for inst_id, group in algo_runs.groupby("instance_id"):
            best_algo = group["best_obj"].min()
            best_known = best_per_instance.loc[inst_id]
            ratio = best_algo / best_known
            rows.append({"algo_id": algo, "instance_id": inst_id, "ratio": ratio})

    df = pd.DataFrame(rows)

    # Generate data profile
    ratios = np.linspace(1.0, 2.0, 100)
    profiles = {}

    for algo in df["algo_id"].unique():
        algo_ratios = df[df["algo_id"] == algo]["ratio"].values
        profile = []
        for r in ratios:
            profile.append(np.mean(algo_ratios <= r))
        profiles[algo] = profile

    # Plot
    plt.figure(figsize=(10, 6))
    for algo, profile in profiles.items():
        plt.plot(ratios, profile, label=algo)
    plt.xlabel("Ratio to best known")
    plt.ylabel("Fraction of instances")
    plt.title("Data Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "data_profile.png"))
    plt.close()

def time_to_target(runs: pd.DataFrame, out_dir: str, target_factor: float = 1.1):
    """
    Compute time to reach target objective (e.g., 10% above best).
    """
    os.makedirs(out_dir, exist_ok=True)

    # For each instance, compute the best known objective
    best_per_instance = runs.groupby("instance_id")["best_obj"].min()
    targets = best_per_instance * target_factor

    # For each algorithm and instance, find the earliest time it reaches the target
    rows = []
    for algo in runs["algo_id"].unique():
        algo_runs = runs[runs["algo_id"] == algo]
        for inst_id, group in algo_runs.groupby("instance_id"):
            target = targets.loc[inst_id]
            # Assuming we have trace data with time points
            # ... (code to find time to reach target)
            pass

    # Plot
    # ... (plotting code)

def run_anytime_analysis(runs: pd.DataFrame, out_dir: str):
    """
    Run all anytime analyses.
    """
    data_profile(runs, out_dir)
    time_to_target(runs, out_dir)