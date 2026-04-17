"""
优先级编码算法统一接口

算子组合汇总（共463种）：
    GA: 384种 = 2 selection × 3 crossover × 4 mutation × 2 initialization × 2 repair × 2 elitism × 2 local_search
    DE: 48种 = 6 mutation × 2 crossover × 2 initialization × 2 local_search
    PR: 16种 = 4 path × 2 selection × 2 local_search
    TS: 2种 = 2 tabu_strategy
    PSO: 4种 = 2 local_search × 2 restart
    BA: 2种 = 2 local_search
    HS: 4种 = 2 parameter × 2 initialization
    SA: 3种 = 3 neighborhood
"""

from typing import List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict

if TYPE_CHECKING:
    from psp.psplib_io import RCPSPInstance


GA_SELECTION = ["roulette", "tournament"]
GA_CROSSOVER = ["arithmetic", "blend", "sbx"]
GA_MUTATION = ["uniform", "gaussian", "polynomial", "swap"]
GA_INIT = ["random", "zero"]
GA_REPAIR = [True, False]
GA_ELITISM = [True, False]
GA_LOCAL = ["none", "uniform"]

DE_MUTATION = ["rand/1", "best/1", "rand/2", "best/2", "current-to-best/1", "adaptive"]
DE_CROSSOVER = ["bin", "exp"]
DE_INIT = ["random", "zero"]
DE_LOCAL = [True, False]

SA_NEIGHBOR = ["uniform", "gaussian", "swap"]

PSO_LOCAL = ["none", "uniform"]
PSO_RESTART = ["none", "adaptive"]

BA_LOCAL = ["none", "uniform"]

HS_PARAM = ["fixed", "adaptive"]
HS_INIT = ["random", "zero"]

TS_TABU = ["static", "dynamic"]

PR_PATH = ["forward", "backward", "random", "bidirectional"]
PR_SEL = ["best", "random_two"]
PR_LOCAL = [True, False]


def get_all_ga_pri() -> List[Dict]:
    c = []
    for sel in GA_SELECTION:
        for cross in GA_CROSSOVER:
            for mut in GA_MUTATION:
                for init in GA_INIT:
                    for rep in GA_REPAIR:
                        for elite in GA_ELITISM:
                            for ls in GA_LOCAL:
                                c.append({"selection_strategy": sel, "crossover_strategy": cross,
                                         "mutation_strategy": mut, "initialization_strategy": init,
                                         "use_repair": rep, "elitism": elite, "local_search_strategy": ls})
    return c


def get_all_de_pri() -> List[Dict]:
    c = []
    for mut in DE_MUTATION:
        for cross in DE_CROSSOVER:
            for init in DE_INIT:
                for ls in DE_LOCAL:
                    c.append({"mutation_strategy": mut, "crossover_strategy": cross,
                             "initialization_strategy": init, "use_local_search": ls})
    return c


def get_all_sa_pri() -> List[Dict]:
    return [{"neighborhood_strategy": n} for n in SA_NEIGHBOR]


def get_all_pso_pri() -> List[Dict]:
    c = []
    for ls in PSO_LOCAL:
        for r in PSO_RESTART:
            c.append({"local_search_strategy": ls, "restart_strategy": r})
    return c


def get_all_ba_pri() -> List[Dict]:
    return [{"local_search_strategy": n} for n in BA_LOCAL]


def get_all_hs_pri() -> List[Dict]:
    c = []
    for p in HS_PARAM:
        for i in HS_INIT:
            c.append({"parameter_strategy": p, "initialization_strategy": i})
    return c


def get_all_ts_pri() -> List[Dict]:
    return [{"tabu_strategy": t} for t in TS_TABU]


def get_all_pr_pri() -> List[Dict]:
    c = []
    for path in PR_PATH:
        for sel in PR_SEL:
            for ls in PR_LOCAL:
                c.append({"path_strategy": path, "selection_strategy": sel, "use_local_search": ls})
    return c


def get_all_pri_combinations() -> Dict[str, List[Dict]]:
    return {
        "GA": get_all_ga_pri(),
        "DE": get_all_de_pri(),
        "SA": get_all_sa_pri(),
        "PSO": get_all_pso_pri(),
        "BA": get_all_ba_pri(),
        "HS": get_all_hs_pri(),
        "TS": get_all_ts_pri(),
        "PR": get_all_pr_pri()
    }


def count_all_pri() -> Dict[str, int]:
    combos = get_all_pri_combinations()
    return {k: len(v) for k, v in combos.items()}


def create_algorithm_pri(algo_type: str, instance: "RCPSPInstance", deadline: int, params):
    algo_type = algo_type.lower()
    if algo_type == "ga":
        from .GA_PRI import GeneticAlgorithmPRI
        return GeneticAlgorithmPRI(instance, deadline, params), "GA_PRI"
    elif algo_type == "de":
        from .DE_PRI import DifferentialEvolutionPRI
        return DifferentialEvolutionPRI(instance, deadline, params), "DE_PRI"
    elif algo_type == "sa":
        from .SA_PRI import SimulatedAnnealingPRI
        return SimulatedAnnealingPRI(instance, deadline, params), "SA_PRI"
    elif algo_type == "pso":
        from .PSO_PRI import ParticleSwarmOptimizationPRI
        return ParticleSwarmOptimizationPRI(instance, deadline, params), "PSO_PRI"
    elif algo_type == "ba":
        from .BA_PRI import BatAlgorithmPRI
        return BatAlgorithmPRI(instance, deadline, params), "BA_PRI"
    elif algo_type == "hs":
        from .HS_PRI import HarmonySearchPRI
        return HarmonySearchPRI(instance, deadline, params), "HS_PRI"
    elif algo_type == "ts":
        from .TS_PRI import TabuSearchPRI
        return TabuSearchPRI(instance, deadline, params), "TS_PRI"
    elif algo_type == "pr":
        from .PR_PRI import PathRelinkingPRI
        return PathRelinkingPRI(instance, deadline, params), "PR_PRI"
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")


def create_default_params_pri(algo_type: str, **kwargs) -> Any:
    algo_type = algo_type.lower()
    if algo_type == "ga":
        from .GA_PRI import GAParamsPRI
        return GAParamsPRI(**kwargs)
    elif algo_type == "de":
        from .DE_PRI import DEParamsPRI
        return DEParamsPRI(**kwargs)
    elif algo_type == "sa":
        from .SA_PRI import SAParamsPRI
        return SAParamsPRI(**kwargs)
    elif algo_type == "pso":
        from .PSO_PRI import PSOParamsPRI
        return PSOParamsPRI(**kwargs)
    elif algo_type == "ba":
        from .BA_PRI import BAParamsPRI
        return BAParamsPRI(**kwargs)
    elif algo_type == "hs":
        from .HS_PRI import HSParamsPRI
        return HSParamsPRI(**kwargs)
    elif algo_type == "ts":
        from .TS_PRI import TSParamsPRI
        return TSParamsPRI(**kwargs)
    elif algo_type == "pr":
        from .PR_PRI import PRParamsPRI
        return PRParamsPRI(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
