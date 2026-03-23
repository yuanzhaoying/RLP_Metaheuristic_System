from __future__ import annotations
from typing import List, Callable, Tuple
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .psplib_io import RCPSPInstance


class RandomGenerator:
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def random(self):
        return self.rng.random()

    def integers(self, low: int, high: int = None, size: int = 1):
        if high is None:
            high = low
            low = 0
        result = self.rng.integers(low, high, size=size)
        if size == 1:
            return int(result.item() if hasattr(result, 'item') else result)
        return result.tolist()

    def choice(self, a: int, size: int = 1, replace: bool = True):
        if isinstance(a, int):
            a = np.arange(a)
        result = self.rng.choice(a, size=size, replace=replace)
        if size == 1:
            return int(result.item() if hasattr(result, 'item') else result)
        return result.tolist()

    def shuffle(self, x: List):
        x = x.copy()
        self.rng.shuffle(x)
        return x

    def permutation(self, n: int):
        return self.rng.permutation(n).tolist()


def op_swap(chromosome: List[int], rng: RandomGenerator) -> List[int]:
    n = len(chromosome)
    i = rng.integers(0, n)
    j = (i + rng.integers(1, n)) % n
    chromosome = chromosome.copy()
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


def op_insertion(chromosome: List[int], rng: RandomGenerator) -> List[int]:
    n = len(chromosome)
    i = rng.integers(0, n)
    j = rng.integers(0, n)
    chromosome = chromosome.copy()
    gene = chromosome.pop(i)
    chromosome.insert(j, gene)
    return chromosome


def _get_two_indices(rng: RandomGenerator, n: int) -> Tuple[int, int]:
    indices = rng.integers(0, n, size=2)
    if isinstance(indices, list):
        return sorted(indices)
    else:
        return sorted(indices.tolist())


def op_inversion(chromosome: List[int], rng: RandomGenerator) -> List[int]:
    n = len(chromosome)
    i, j = _get_two_indices(rng, n)
    chromosome = chromosome.copy()
    chromosome[i:j+1] = reversed(chromosome[i:j+1])
    return chromosome


def op_scramble(chromosome: List[int], rng: RandomGenerator) -> List[int]:
    n = len(chromosome)
    if n <= 1:
        return chromosome.copy()
    i, j = _get_two_indices(rng, n)
    chromosome = chromosome.copy()
    subset = chromosome[i:j+1]
    rng.shuffle(subset)
    chromosome[i:j+1] = subset
    return chromosome


def op_swap_based(chromosome: List[int], rng: RandomGenerator, n_swaps: int = 2) -> List[int]:
    result = chromosome.copy()
    for _ in range(n_swaps):
        result = op_swap(result, rng)
    return result


def op_insertion_based(chromosome: List[int], rng: RandomGenerator, n_inserts: int = 2) -> List[int]:
    result = chromosome.copy()
    for _ in range(n_inserts):
        result = op_insertion(result, rng)
    return result


def crossover_ox1(parent1: List[int], parent2: List[int], rng: RandomGenerator) -> Tuple[List[int], List[int]]:
    n = len(parent1)
    if n <= 2:
        return parent1.copy(), parent2.copy()

    i, j = _get_two_indices(rng, n)

    child1 = [None] * n
    child2 = [None] * n

    child1[i:j+1] = parent1[i:j+1]
    child2[i:j+1] = parent2[i:j+1]

    def fill_child(child, other_parent):
        remaining = [x for x in other_parent if x not in child]
        remaining_idx = 0
        for k in range(n):
            if child[k] is None:
                while remaining_idx < len(remaining) and remaining[remaining_idx] in child:
                    remaining_idx += 1
                if remaining_idx < len(remaining):
                    child[k] = remaining[remaining_idx]
                    remaining_idx += 1
                else:
                    for x in other_parent:
                        if x not in child:
                            child[k] = x
                            break
        return child

    child1 = fill_child(child1, parent2)
    child2 = fill_child(child2, parent1)

    return [x for x in child1], [x for x in child2]


def crossover_pmx(parent1: List[int], parent2: List[int], rng: RandomGenerator) -> Tuple[List[int], List[int]]:
    n = len(parent1)
    i, j = _get_two_indices(rng, n)

    child1 = [None] * n
    child2 = [None] * n

    child1[i:j+1] = parent1[i:j+1]
    child2[i:j+1] = parent2[i:j+1]

    def fill_pmx(child, seg1, seg2):
        mapping = {}
        for idx in range(i, j+1):
            if child[idx] != seg1[idx - i]:
                mapping[seg1[idx - i]] = child[idx]
                for k in range(n):
                    if child[k] is None and seg2[k] in mapping:
                        child[k] = mapping[seg2[k]]
        return child

    child1 = fill_pmx(child1, parent1[i:j+1], parent2[i:j+1])
    child2 = fill_pmx(child2, parent2[i:j+1], parent1[i:j+1])

    for idx in range(n):
        if child1[idx] is None:
            child1[idx] = parent2[idx]
        if child2[idx] is None:
            child2[idx] = parent1[idx]

    return child1, child2


def crossover_order(parent1: List[int], parent2: List[int], rng: RandomGenerator) -> Tuple[List[int], List[int]]:
    n = len(parent1)
    i, j = _get_two_indices(rng, n)

    child1 = parent1[:]
    child2 = parent2[:]

    mask = np.zeros(n, dtype=bool)
    mask[i:j+1] = True

    def preserve_order(parent, other_parent, mask):
        result = []
        other_remaining = [x for i, x in enumerate(other_parent) if not mask[i]]
        for i, x in enumerate(parent):
            if mask[i]:
                result.append(x)
            else:
                while other_remaining and other_remaining[0] in result:
                    other_remaining.pop(0)
                if other_remaining:
                    result.append(other_remaining.pop(0))
        return result

    child1 = preserve_order(parent1, parent2, mask)
    child2 = preserve_order(parent2, parent1, mask)

    return child1, child2


OPERATORS = {
    "swap": op_swap,
    "insertion": op_insertion,
    "inversion": op_inversion,
    "scramble": op_scramble,
    "swap_based": op_swap_based,
    "insertion_based": op_insertion_based,
}

CROSSOVERS = {
    "ox1": crossover_ox1,
    "pmx": crossover_pmx,
    "order": crossover_order,
}


def get_operator(name: str) -> Callable:
    if name not in OPERATORS:
        raise ValueError(f"Unknown operator: {name}. Available: {list(OPERATORS.keys())}")
    return OPERATORS[name]


def get_crossover(name: str) -> Callable:
    if name not in CROSSOVERS:
        raise ValueError(f"Unknown crossover: {name}. Available: {list(CROSSOVERS.keys())}")
    return CROSSOVERS[name]
