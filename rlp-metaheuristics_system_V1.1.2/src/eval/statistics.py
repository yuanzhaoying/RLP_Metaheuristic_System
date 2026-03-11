from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp


def friedman_test(data: pd.DataFrame, performance_col: str = "best_objective",
                  instance_col: str = "instance_id", algo_col: str = "algorithm_name") -> Dict:
    pivot = data.pivot_table(
        index=instance_col,
        columns=algo_col,
        values=performance_col,
        aggfunc="median"
    )

    pivot = pivot.dropna()

    if len(pivot.columns) < 2:
        return {"error": "Need at least 2 algorithms"}

    algos = list(pivot.columns)
    performance_data = [pivot[a].values for a in algos]

    stat, p_value = friedmanchisquare(*performance_data)

    ranks = pivot.rank(axis=1, ascending=True)
    mean_ranks = ranks.mean()

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_instances": len(pivot),
        "n_algorithms": len(algos),
        "mean_ranks": mean_ranks.to_dict(),
        "significant": p_value < 0.05
    }


def pairwise_wilcoxon(data: pd.DataFrame, performance_col: str = "best_objective",
                      instance_col: str = "instance_id", algo_col: str = "algorithm_name",
                      alpha: float = 0.05) -> pd.DataFrame:
    pivot = data.pivot_table(
        index=instance_col,
        columns=algo_col,
        values=performance_col,
        aggfunc="median"
    )

    pivot = pivot.dropna()
    algos = list(pivot.columns)

    n = len(algos)
    results = []

    for i in range(n):
        for j in range(i+1, n):
            algo1, algo2 = algos[i], algos[j]
            diff = pivot[algo1].values - pivot[algo2].values

            if np.all(diff == 0):
                continue

            try:
                stat, p = wilcoxon(diff, alternative='two-sided')
            except Exception:
                continue

            wins = np.sum(diff < 0)
            losses = np.sum(diff > 0)

            results.append({
                "algorithm_1": algo1,
                "algorithm_2": algo2,
                "statistic": float(stat),
                "p_value": float(p),
                "wins": int(wins),
                "losses": int(losses),
                "better": algo1 if wins > losses else algo2
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df = holm_correction_df(results_df, alpha=alpha)

    return results_df


def holm_correction_df(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if "p_value" not in df.columns:
        return df

    df = df.sort_values("p_value").reset_index(drop=True)
    m = len(df)
    adjusted_p = []

    for k, row in df.iterrows():
        adjusted = min(1.0, (m - k) * row["p_value"])
        adjusted_p.append(adjusted)

    df["p_adjusted"] = adjusted_p
    df["significant"] = df["p_adjusted"] < alpha

    return df


def critical_difference(algos: List[str], n_instances: int, alpha: float = 0.05) -> float:
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }

    k = len(algos)
    if k in q_alpha:
        q = q_alpha[k]
    else:
        q = 2.728 + 0.3 * (k - 5)

    cd = q * np.sqrt(k * (k + 1) / (6 * n_instances))
    return cd


def compute_ranks(data: pd.DataFrame, performance_col: str = "best_objective",
                  instance_col: str = "instance_id", algo_col: str = "algorithm_name") -> pd.DataFrame:
    pivot = data.pivot_table(
        index=instance_col,
        columns=algo_col,
        values=performance_col,
        aggfunc="median"
    )

    ranks = pivot.rank(axis=1, ascending=True)
    mean_ranks = ranks.mean().sort_values()

    return pd.DataFrame({
        "algorithm": mean_ranks.index,
        "mean_rank": mean_ranks.values
    })


def nemenyi_test(data: pd.DataFrame, performance_col: str = "best_objective",
                 instance_col: str = "instance_id", algo_col: str = "algorithm_name") -> pd.DataFrame:
    pivot = data.pivot_table(
        index=instance_col,
        columns=algo_col,
        values=performance_col,
        aggfunc="median"
    )

    result = sp.posthoc_nemenyi_friedman(pivot.values)
    result.columns = pivot.columns
    result.index = pivot.columns

    return result


def vargha_delaney_a(A: np.ndarray, B: np.ndarray) -> float:
    m, n = len(A), len(B)

    if m == 0 or n == 0:
        return 0.5

    more = 0
    equal = 0

    for a in A:
        for b in B:
            if a > b:
                more += 1
            elif a == b:
                equal += 1

    a_score = (more + 0.5 * equal) / (m * n)
    return a_score


def effect_size_matrix(data: pd.DataFrame, performance_col: str = "best_objective",
                       instance_col: str = "instance_id", algo_col: str = "algorithm_name") -> pd.DataFrame:
    pivot = data.pivot_table(
        index=instance_col,
        columns=algo_col,
        values=performance_col,
        aggfunc="median"
    )

    pivot = pivot.dropna()
    algos = list(pivot.columns)

    n = len(algos)
    effect_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                effect_matrix[i, j] = vargha_delaney_a(
                    pivot[algos[i]].values,
                    pivot[algos[j]].values
                )

    result = pd.DataFrame(effect_matrix, columns=algos, index=algos)
    return result


def generate_cd_diagram_data(ranks_df: pd.DataFrame, n_instances: int,
                             alpha: float = 0.05) -> Dict:
    algos = ranks_df["algorithm"].tolist()
    mean_ranks = ranks_df["mean_rank"].tolist()

    cd = critical_difference(algos, n_instances, alpha)

    groups = []
    current_group = [algos[0]]

    for i in range(1, len(algos)):
        if mean_ranks[i] - mean_ranks[0] < cd:
            current_group.append(algos[i])
        else:
            groups.append(current_group)
            current_group = [algos[i]]
            mean_ranks = [mean_ranks[i]]

    groups.append(current_group)

    return {
        "algorithms": algos,
        "mean_ranks": dict(zip(algos, mean_ranks)),
        "critical_difference": cd,
        "groups": groups
    }


def statistical_summary(data: pd.DataFrame, performance_col: str = "best_objective",
                       instance_col: str = "instance_id", algo_col: str = "algorithm_name") -> pd.DataFrame:
    summary = data.groupby(algo_col)[performance_col].agg([
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("count", "count")
    ])

    summary["cv"] = summary["std"] / summary["mean"]

    return summary
