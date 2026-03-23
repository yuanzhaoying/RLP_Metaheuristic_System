from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import networkx as nx
import pandas as pd
from ..psp.psplib_io import RCPSPInstance


class FeatureExtractor:
    def __init__(self, inst: RCPSPInstance, horizon: int):
        self.inst = inst
        self.horizon = horizon
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(range(self.inst.n_activities))

        for i in range(self.inst.n_activities):
            for j in self.inst.successors[i]:
                if j < self.inst.n_activities:
                    G.add_edge(i, j)

        return G

    def extract_all(self) -> Dict[str, float]:
        features = {}

        features.update(self._structural_features())
        features.update(self._resource_features())
        features.update(self._slack_features())
        features.update(self._network_topology_features())

        return features

    def _structural_features(self) -> Dict[str, float]:
        n = self.inst.n_activities
        R = self.inst.n_resources

        n_edges = self.graph.number_of_edges()
        out_degrees = [self.graph.out_degree(i) for i in range(n)]
        in_degrees = [self.graph.in_degree(i) for i in range(n)]

        topo_order = list(nx.topological_sort(self.graph))

        critical_path = self._compute_critical_path_length()

        try:
            order_strength = self._compute_order_strength()
        except:
            order_strength = 0.0

        parallel_degree = n_edges / max(1, n * (n - 1) / 2)

        depth = 0
        for node in nx.topological_sort(self.graph):
            if self.graph.in_degree(node) == 0:
                longest_path = 0
                for successor in self.graph.successors(node):
                    path_len = self._longest_path_from(node)
                    longest_path = max(longest_path, path_len)
                depth = max(depth, longest_path)

        return {
            "n_activities": n,
            "n_resources": R,
            "n_edges": n_edges,
            "avg_out_degree": np.mean(out_degrees),
            "avg_in_degree": np.mean(in_degrees),
            "max_out_degree": max(out_degrees),
            "max_in_degree": max(in_degrees),
            "critical_path_len": critical_path,
            "order_strength": order_strength,
            "parallel_degree": parallel_degree,
            "density": nx.density(self.graph),
        }

    def _compute_critical_path_length(self) -> float:
        if self.graph.number_of_nodes() == 0:
            return 0.0

        topo_order = list(nx.topological_sort(self.graph))
        dist = {i: 0 for i in topo_order}

        for node in topo_order:
            for succ in self.graph.successors(node):
                dur = int(self.inst.durations[node])
                dist[succ] = max(dist[succ], dist[node] + dur)

        return float(max(dist.values())) if dist else 0.0

    def _compute_order_strength(self) -> float:
        n = self.inst.n_activities
        if n <= 1:
            return 0.0

        reachable_pairs = 0
        total_pairs = n * (n - 1) // 2

        for i in range(n):
            for j in range(i+1, n):
                if nx.has_path(self.graph, i, j) or nx.has_path(self.graph, j, i):
                    reachable_pairs += 1

        return reachable_pairs / max(1, total_pairs)

    def _longest_path_from(self, start: int) -> int:
        if start not in self.graph:
            return 0

        longest = 0
        for succ in self.graph.successors(start):
            path_len = self.inst.durations[start] + self._longest_path_from(succ)
            longest = max(longest, path_len)

        return int(self.inst.durations[start]) + longest

    def _resource_features(self) -> Dict[str, float]:
        capacities = self.inst.capacity.astype(float)
        demands = self.inst.demands.astype(float)
        durations = self.inst.durations.astype(float)

        total_work = (demands.T * durations).sum(axis=1)

        resource_strength = []
        for r in range(self.inst.n_resources):
            if capacities[r] > 0 and self.horizon > 0:
                rs = total_work[r] / (capacities[r] * self.horizon)
                resource_strength.append(rs)

        demand_per_activity = demands.sum(axis=1)
        activity_demand_mean = demand_per_activity[demand_per_activity > 0].mean() if len(demand_per_activity[demand_per_activity > 0]) > 0 else 0

        return {
            "capacity_mean": float(capacities.mean()),
            "capacity_std": float(capacities.std()),
            "capacity_min": float(capacities.min()),
            "capacity_max": float(capacities.max()),
            "demand_mean": float(demands.mean()),
            "demand_std": float(demands.std()),
            "demand_per_activity_mean": float(activity_demand_mean),
            "total_work": float(total_work.sum()),
            "resource_strength_mean": float(np.mean(resource_strength)) if resource_strength else 0.0,
            "resource_strength_max": float(np.max(resource_strength)) if resource_strength else 0.0,
        }

    def _slack_features(self) -> Dict[str, float]:
        es, ls = self._compute_time_windows()

        if len(es) == 0:
            return {
                "slack_mean": 0.0,
                "slack_std": 0.0,
                "slack_min": 0.0,
                "slack_max": 0.0,
                "critical_activity_ratio": 0.0,
            }

        slacks = ls - es
        non_zero_slacks = slacks[slacks > 0]

        critical_count = np.sum(slacks == 0)
        critical_ratio = critical_count / len(slacks) if len(slacks) > 0 else 0.0

        return {
            "slack_mean": float(slacks.mean()),
            "slack_std": float(slacks.std()) if len(slacks) > 1 else 0.0,
            "slack_min": float(slacks.min()),
            "slack_max": float(slacks.max()),
            "slack_median": float(np.median(slacks)),
            "non_zero_slack_mean": float(non_zero_slacks.mean()) if len(non_zero_slacks) > 0 else 0.0,
            "critical_activity_ratio": float(critical_ratio),
        }

    def _compute_time_windows(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.inst.n_activities
        durations = self.inst.durations

        es = np.zeros(n, dtype=int)
        ls = np.full(n, self.horizon, dtype=int)

        for j in range(n):
            for pred in self.inst.predecessors[j]:
                es[j] = max(es[j], es[pred] + durations[pred])

        for j in reversed(range(n)):
            if j == n - 1:
                ls[j] = self.horizon - durations[j]
            else:
                min_ls = self.horizon
                for succ in self.inst.successors[j]:
                    min_ls = min(min_ls, ls[succ] - durations[j])
                ls[j] = min_ls

        return es, ls

    def _network_topology_features(self) -> Dict[str, float]:
        n = self.inst.n_activities
        features = {}

        durations = self.inst.durations.astype(float)
        demands = self.inst.demands.astype(float)
        capacities = self.inst.capacity.astype(float)

        features["duration_mean"] = float(np.mean(durations))
        features["duration_max"] = float(np.max(durations))
        features["duration_min"] = float(np.min(durations))

        predecessor_counts = [len(self.inst.predecessors[i]) for i in range(n)]
        successor_counts = [len(self.inst.successors[i]) for i in range(n)]

        features["max_predecessor_count"] = float(max(predecessor_counts))
        features["min_predecessor_count"] = float(min(predecessor_counts))
        features["max_successor_count"] = float(max(successor_counts))
        features["min_successor_count"] = float(min(successor_counts))

        n_edges = sum(successor_counts)
        features["network_complexity"] = n_edges / max(1, n)

        first_successor_count = len(self.inst.successors[0])
        last_predecessor_count = len(self.inst.predecessors[n - 1])
        order_strength_new = (n_edges - first_successor_count - last_predecessor_count) / max(1, n * (n - 1) / 2)
        features["order_strength_new"] = order_strength_new

        PL, max_PL = self._compute_progressive_level()
        RL = self._compute_regression_level(max_PL)

        SP = (max_PL - 1) / max(1, n - 1)
        features["serial_parallel_indicator"] = SP

        mean_w = n / max(1, max_PL)
        total_w = self._count_elements(PL)
        AD = self._activity_distribution(total_w, mean_w, max_PL, n)
        features["activity_distribution"] = AD

        short_arc, long_arc = self._calculate_arc_lengths(PL)
        SA = self._short_arc_indicator(total_w, n, short_arc)
        features["short_arc_indicator"] = SA
        features["long_arc_count"] = float(long_arc)

        TF = self._topological_floating(max_PL, PL, RL, n)
        features["topological_floating"] = TF

        features["max_progressive_level"] = float(max_PL)
        features["min_regression_level"] = float(min(RL))

        RF, RU = self._compute_rf_ru()
        features["resource_factor"] = RF
        features["resource_usage"] = RU

        total_resource_request = demands.sum(axis=0)
        avg_resource_request = total_resource_request / max(1, n - 2)
        features["avg_resource_total"] = float(np.mean(avg_resource_request))

        for r in range(len(avg_resource_request)):
            features[f"avg_resource_{r}"] = float(avg_resource_request[r])

        return features

    def _compute_progressive_level(self) -> Tuple[List[int], int]:
        n = self.inst.n_activities
        PL = [1]
        for i in range(1, n):
            pl_i = 0
            for pred in self.inst.predecessors[i]:
                pl_i = max(PL[pred], pl_i)
            PL.append(pl_i + 1)
        return PL, max(PL)

    def _compute_regression_level(self, max_PL: int) -> List[int]:
        n = self.inst.n_activities
        RL = [max_PL]
        for i in range(1, n):
            rl_i = 0
            for succ in self.inst.successors[n - 1 - i]:
                rl_i = max(RL[n - 1 - succ], rl_i)
            RL.append(rl_i - 1)
        RL.reverse()
        return RL

    def _count_elements(self, lst: List[int]) -> List[int]:
        counts = OrderedDict()
        for num in lst:
            counts[num] = counts.get(num, 0) + 1
        return list(counts.values())

    def _activity_distribution(self, total_w: List[int], mean_w: float, max_PL: int, n: int) -> float:
        if max_PL <= 1 or max_PL >= n:
            return 0.0
        processed = [abs(x - mean_w) for x in total_w]
        return sum(processed) / (2 * (max_PL - 1) * (mean_w - 1)) if mean_w > 1 else 0.0

    def _calculate_arc_lengths(self, PL: List[int]) -> Tuple[int, int]:
        short_arc = 0
        long_arc = 0
        for i in range(len(PL)):
            for j in self.inst.successors[i]:
                if j < len(PL):
                    if PL[j] - PL[i] == 1:
                        short_arc += 1
                    else:
                        long_arc += 1
        return short_arc, long_arc

    def _short_arc_indicator(self, total_w: List[int], n: int, short_arc: int) -> float:
        D = 0
        for a in range(len(total_w) - 1):
            D += total_w[a] * total_w[a + 1]
        if n - total_w[0] < D:
            return (short_arc - n + total_w[0]) / (D - n + total_w[0])
        return 1.0

    def _topological_floating(self, max_PL: int, PL: List[int], RL: List[int], n: int) -> float:
        if max_PL <= 1 or max_PL >= n:
            return 0.0
        return sum(RL[i] - PL[i] for i in range(len(RL))) / ((max_PL - 1) * (n - max_PL))

    def _compute_rf_ru(self) -> Tuple[float, float]:
        n = self.inst.n_activities
        R = self.inst.n_resources
        demands = self.inst.demands

        b = []
        for i in range(n):
            for r in range(R):
                if demands[i, r] > 0:
                    b.append(1)
                else:
                    b.append(0)

        RF = sum(b) / max(1, (n - 2) * R)
        RU = sum(b) / max(1, n - 2)

        return RF, RU


def extract_features_batch(instances: List[RCPSPInstance], horizons: List[int]) -> pd.DataFrame:
    all_features = []

    for inst, horizon in zip(instances, horizons):
        try:
            extractor = FeatureExtractor(inst, horizon)
            features = extractor.extract_all()
            features["instance_id"] = inst.name
            features["horizon"] = horizon
            all_features.append(features)
        except Exception as e:
            print(f"Error extracting features for {inst.name}: {e}")

    return pd.DataFrame(all_features)


def get_feature_names() -> List[str]:
    return [
        "n_activities", "n_resources", "n_edges",
        "avg_out_degree", "avg_in_degree", "max_out_degree", "max_in_degree",
        "critical_path_len", "order_strength", "parallel_degree", "density",
        "capacity_mean", "capacity_std", "capacity_min", "capacity_max",
        "demand_mean", "demand_std", "demand_per_activity_mean", "total_work",
        "resource_strength_mean", "resource_strength_max",
        "slack_mean", "slack_std", "slack_min", "slack_max", "slack_median",
        "non_zero_slack_mean", "critical_activity_ratio",
        "duration_mean", "duration_max", "duration_min",
        "max_predecessor_count", "min_predecessor_count",
        "max_successor_count", "min_successor_count",
        "network_complexity", "order_strength_new",
        "serial_parallel_indicator", "activity_distribution",
        "short_arc_indicator", "long_arc_count", "topological_floating",
        "max_progressive_level", "min_regression_level",
        "resource_factor", "resource_usage", "avg_resource_total"
    ]
