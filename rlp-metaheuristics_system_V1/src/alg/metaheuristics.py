from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
import numpy as np
import time

from ..psp.psplib_io import RCPSPInstance
from ..psp.objective import ScheduleResult, evaluate_schedule
from ..psp.ssgs import SerialSSGS
from ..psp.rlp_decoder import RLPDecoder, evaluate_rlp_schedule
from .operators import RandomGenerator, get_operator, get_crossover


N_SOLUTIONS = 1000


@dataclass
class AlgorithmParams:
    max_evaluations: int = N_SOLUTIONS
    time_limit: float = 300.0
    seed: int = 0
    problem_type: str = "rlp"
    use_delay_factors: bool = False


@dataclass
class OptimizationResult:
    best_objective: float
    best_chromosome: List[int]
    best_delay_factors: Optional[List[float]]
    best_schedule: Optional[np.ndarray]
    n_evaluations: int
    n_iterations: int
    runtime: float
    convergence_history: List[float]
    is_feasible: bool
    algorithm_name: str
    makespan: int = 0


class FitnessEvaluator:
    def __init__(
        self,
        instance: RCPSPInstance,
        deadline: int,
        decoder: SerialSSGS,
        objective_type: str = "variance",
        max_evaluations: int = N_SOLUTIONS,
        problem_type: str = "rlp",
        use_delay_factors: bool = False
    ):
        self.instance = instance
        self.deadline = deadline
        self.decoder = decoder
        self.objective_type = objective_type
        self.max_evaluations = max_evaluations
        self.problem_type = problem_type
        self.use_delay_factors = use_delay_factors
        self.n_evaluations = 0

        if problem_type == "rlp":
            self.rlp_decoder = RLPDecoder(instance, deadline)

    def evaluate(self, chromosome: List[int], delay_factors: List[float] = None) -> Tuple[float, ScheduleResult]:
        self.n_evaluations += 1
        try:
            if self.problem_type == "rlp":
                repaired = self.rlp_decoder._repair_topological(chromosome)
                
                if self.use_delay_factors and delay_factors is not None:
                    start_times, is_feasible = self.rlp_decoder.decode(repaired, delay_factors)
                else:
                    start_times, is_feasible = self.rlp_decoder.decode_es(repaired)
                
                if start_times is None:
                    return float('inf'), ScheduleResult(
                        start_times=None,
                        makespan=0,
                        is_feasible=False,
                        objective_value=float('inf')
                    )
                
                obj_value, usage, is_feasible = evaluate_rlp_schedule(
                    self.instance,
                    start_times,
                    self.deadline,
                    self.objective_type
                )
                
                makespan = int((start_times + self.instance.durations).max())
                
                return obj_value, ScheduleResult(
                    start_times=start_times,
                    makespan=makespan,
                    is_feasible=is_feasible,
                    objective_value=obj_value,
                    resource_usage=usage
                )
            else:
                repaired = self.decoder._repair_topological(chromosome)
                start_times = self.decoder.decode(repaired)
                result = evaluate_schedule(
                    self.instance,
                    start_times,
                    self.deadline,
                    self.objective_type
                )
                return result.objective_value, result
        except Exception as e:
            return float('inf'), ScheduleResult(
                start_times=None,
                makespan=0,
                is_feasible=False,
                objective_value=float('inf')
            )

    def reset_counter(self):
        self.n_evaluations = 0


class LocalSearch:
    def __init__(
        self,
        evaluator: FitnessEvaluator,
        operator_name: str = "swap",
        max_iterations: int = 200,
        time_limit: float = 300.0
    ):
        self.evaluator = evaluator
        self.operator = get_operator(operator_name)
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.rng = RandomGenerator()

    def search(
        self,
        initial_chromosome: List[int],
        initial_delay_factors: List[float],
        start_time: float
    ) -> Tuple[List[int], List[float], float, int]:
        current = initial_chromosome.copy()
        current_delays = initial_delay_factors.copy() if initial_delay_factors else None
        current_fitness, _ = self.evaluator.evaluate(current, current_delays)
        n_iterations = 0

        while n_iterations < self.max_iterations and self.evaluator.n_evaluations < self.evaluator.max_evaluations:
            elapsed = time.time() - start_time
            if elapsed >= self.time_limit:
                break

            neighbor = self.operator(current, self.rng)
            neighbor_delays = self._mutate_delays(current_delays) if self.evaluator.use_delay_factors else None
            neighbor_fitness, result = self.evaluator.evaluate(neighbor, neighbor_delays)
            n_iterations += 1

            if neighbor_fitness < current_fitness:
                current = neighbor
                current_delays = neighbor_delays
                current_fitness = neighbor_fitness
                n_iterations = 0

        return current, current_delays, current_fitness, n_iterations

    def _mutate_delays(self, delays: List[float]) -> List[float]:
        if delays is None:
            return None
        result = delays.copy()
        if len(result) > 0:
            idx = self.rng.integers(0, len(result))
            result[idx] = max(0.0, min(1.0, result[idx] + (self.rng.rng.random() - 0.5) * 0.2))
        return result


@dataclass
class ILSParams(AlgorithmParams):
    perturbation_strength: int = 5
    local_search_operator: str = "swap"
    perturbation_operator: str = "swap"
    acceptance: str = "better"


class IteratedLocalSearch:
    def __init__(
        self,
        instance: RCPSPInstance,
        deadline: int,
        params: ILSParams
    ):
        self.instance = instance
        self.deadline = deadline
        self.params = params
        self.decoder = SerialSSGS(instance, deadline)
        self.evaluator = FitnessEvaluator(
            instance, deadline, self.decoder, 
            max_evaluations=params.max_evaluations,
            problem_type=params.problem_type,
            use_delay_factors=params.use_delay_factors
        )
        self.rng = RandomGenerator(params.seed)

    def run(self) -> OptimizationResult:
        start_time = time.time()
        convergence = []

        n = self.instance.n_activities
        initial = list(self.rng.permutation(n))
        initial_delays = [self.rng.rng.random() for _ in range(n)] if self.params.use_delay_factors else None
        
        best_chromosome = initial
        best_delay_factors = initial_delays
        best_fitness, _ = self.evaluator.evaluate(best_chromosome, best_delay_factors)

        local_chromosome = best_chromosome.copy()
        local_delay_factors = best_delay_factors.copy() if best_delay_factors else None
        local_fitness = best_fitness

        n_iterations = 0
        while self.evaluator.n_evaluations < self.params.max_evaluations:
            elapsed = time.time() - start_time
            if elapsed >= self.params.time_limit:
                break

            perturbed, perturbed_delays = self._perturb(local_chromosome, local_delay_factors)

            improved_chrom, improved_delays, improved_fitness, _ = self._local_search(
                perturbed, perturbed_delays, start_time
            )

            if self._accept(improved_fitness, local_fitness):
                local_chromosome = improved_chrom
                local_delay_factors = improved_delays
                local_fitness = improved_fitness

                if improved_fitness < best_fitness:
                    best_chromosome = improved_chrom
                    best_delay_factors = improved_delays
                    best_fitness = improved_fitness

            convergence.append(best_fitness)
            n_iterations += 1

        runtime = time.time() - start_time
        _, best_result = self.evaluator.evaluate(best_chromosome, best_delay_factors)

        return OptimizationResult(
            best_objective=best_fitness,
            best_chromosome=best_chromosome,
            best_delay_factors=best_delay_factors,
            best_schedule=best_result.start_times,
            n_evaluations=self.evaluator.n_evaluations,
            n_iterations=n_iterations,
            runtime=runtime,
            convergence_history=convergence,
            is_feasible=best_result.is_feasible,
            algorithm_name="ILS",
            makespan=best_result.makespan
        )

    def _perturb(self, chromosome: List[int], delays: List[float]) -> Tuple[List[int], List[float]]:
        op_perturb = get_operator(self.params.perturbation_operator)
        result = chromosome.copy()
        result_delays = delays.copy() if delays else None
        for _ in range(self.params.perturbation_strength):
            result = op_perturb(result, self.rng)
            if result_delays and len(result_delays) > 0:
                idx = self.rng.integers(0, len(result_delays))
                result_delays[idx] = self.rng.rng.random()
        return result, result_delays

    def _local_search(
        self,
        chromosome: List[int],
        delays: List[float],
        start_time: float
    ) -> Tuple[List[int], List[float], float, int]:
        local_search = LocalSearch(
            self.evaluator,
            self.params.local_search_operator,
            max_iterations=50,
            time_limit=self.params.time_limit
        )
        return local_search.search(chromosome, delays, start_time)

    def _accept(self, candidate_fitness: float, current_fitness: float) -> bool:
        if self.params.acceptance == "always":
            return True
        elif self.params.acceptance == "better":
            return candidate_fitness < current_fitness
        else:
            return candidate_fitness < current_fitness


@dataclass
class GAParams(AlgorithmParams):
    population_size: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    crossover: str = "ox1"
    mutation: str = "swap"
    elitism: int = 2


class GeneticAlgorithm:
    def __init__(
        self,
        instance: RCPSPInstance,
        deadline: int,
        params: GAParams
    ):
        self.instance = instance
        self.deadline = deadline
        self.params = params
        self.decoder = SerialSSGS(instance, deadline)
        self.evaluator = FitnessEvaluator(
            instance, deadline, self.decoder, 
            max_evaluations=params.max_evaluations,
            problem_type=params.problem_type,
            use_delay_factors=params.use_delay_factors
        )
        self.rng = RandomGenerator(params.seed)

    def run(self) -> OptimizationResult:
        start_time = time.time()
        convergence = []

        n = self.instance.n_activities
        n_generations = self.params.max_evaluations // self.params.population_size
        if self.params.max_evaluations % self.params.population_size != 0:
            n_generations += 1

        population = self._initialize_population()
        population = self._evaluate_population(population)
        population.sort(key=lambda x: x[2])

        best_chromosome = population[0][0]
        best_delays = population[0][1]
        best_fitness = population[0][2]

        gen = 0
        while gen < n_generations and self.evaluator.n_evaluations < self.params.max_evaluations:
            elapsed = time.time() - start_time
            if elapsed >= self.params.time_limit:
                break

            new_population = [(ind[0], ind[1]) for ind in population[:self.params.elitism]]

            while len(new_population) < self.params.population_size:
                parent1_chrom, parent1_delays, _ = self._tournament_selection(population)
                parent2_chrom, parent2_delays, _ = self._tournament_selection(population)

                if self.rng.rng.random() < self.params.crossover_rate:
                    crossover_fn = get_crossover(self.params.crossover)
                    child1_chrom, child2_chrom = crossover_fn(parent1_chrom, parent2_chrom, self.rng)
                    if self.params.use_delay_factors:
                        child1_delays = [(parent1_delays[i] + parent2_delays[i]) / 2 for i in range(len(parent1_delays))]
                        child2_delays = child1_delays.copy()
                    else:
                        child1_delays, child2_delays = None, None
                else:
                    child1_chrom, child1_delays = parent1_chrom.copy(), parent1_delays.copy() if parent1_delays else None
                    child2_chrom, child2_delays = parent2_chrom.copy(), parent2_delays.copy() if parent2_delays else None

                if self.rng.rng.random() < self.params.mutation_rate:
                    mutation_fn = get_operator(self.params.mutation)
                    child1_chrom = mutation_fn(child1_chrom, self.rng)
                    if self.params.use_delay_factors:
                        child1_delays = self._mutate_delays(child1_delays)

                if self.rng.rng.random() < self.params.mutation_rate:
                    mutation_fn = get_operator(self.params.mutation)
                    child2_chrom = mutation_fn(child2_chrom, self.rng)
                    if self.params.use_delay_factors:
                        child2_delays = self._mutate_delays(child2_delays)

                new_population.append((child1_chrom, child1_delays))
                if len(new_population) < self.params.population_size:
                    new_population.append((child2_chrom, child2_delays))

            new_population = new_population[:self.params.population_size]
            new_population = self._evaluate_population(new_population)
            new_population.sort(key=lambda x: x[2])

            population = new_population

            if population[0][2] < best_fitness:
                best_chromosome = population[0][0]
                best_delays = population[0][1]
                best_fitness = population[0][2]

            convergence.append(best_fitness)
            gen += 1

        runtime = time.time() - start_time
        _, best_result = self.evaluator.evaluate(best_chromosome, best_delays)

        return OptimizationResult(
            best_objective=best_fitness,
            best_chromosome=best_chromosome,
            best_delay_factors=best_delays,
            best_schedule=best_result.start_times,
            n_evaluations=self.evaluator.n_evaluations,
            n_iterations=gen,
            runtime=runtime,
            convergence_history=convergence,
            is_feasible=best_result.is_feasible,
            algorithm_name="GA",
            makespan=best_result.makespan
        )

    def _initialize_population(self) -> List[Tuple[List[int], Optional[List[float]]]]:
        population = []
        n = self.instance.n_activities
        for _ in range(self.params.population_size):
            chromosome = list(self.rng.permutation(n))
            delays = [self.rng.rng.random() for _ in range(n)] if self.params.use_delay_factors else None
            population.append((chromosome, delays))
        return population

    def _evaluate_population(
        self,
        population: List[Tuple[List[int], Optional[List[float]]]]
    ) -> List[Tuple[List[int], Optional[List[float]], float]]:
        evaluated = []
        for chromosome, delays in population:
            fitness, _ = self.evaluator.evaluate(chromosome, delays)
            evaluated.append((chromosome, delays, fitness))
        return evaluated

    def _tournament_selection(
        self,
        population: List[Tuple[List[int], Optional[List[float]], float]]
    ) -> Tuple[List[int], Optional[List[float]], float]:
        candidates = self.rng.choice(
            len(population),
            size=self.params.tournament_size,
            replace=False
        )
        best = min(candidates, key=lambda i: population[i][2])
        return population[best][0].copy(), population[best][1].copy() if population[best][1] else None, population[best][2]

    def _mutate_delays(self, delays: List[float]) -> List[float]:
        if delays is None:
            return None
        result = delays.copy()
        if len(result) > 0:
            idx = self.rng.integers(0, len(result))
            result[idx] = max(0.0, min(1.0, result[idx] + (self.rng.rng.random() - 0.5) * 0.3))
        return result


@dataclass
class SAParams(AlgorithmParams):
    initial_temperature: float = 10000.0
    cooling_rate: float = 0.995
    min_temperature: float = 1.0
    iterations_per_temperature: int = 100
    mutation_operator: str = "swap"


class SimulatedAnnealing:
    def __init__(
        self,
        instance: RCPSPInstance,
        deadline: int,
        params: SAParams
    ):
        self.instance = instance
        self.deadline = deadline
        self.params = params
        self.decoder = SerialSSGS(instance, deadline)
        self.evaluator = FitnessEvaluator(
            instance, deadline, self.decoder, 
            max_evaluations=params.max_evaluations,
            problem_type=params.problem_type,
            use_delay_factors=params.use_delay_factors
        )
        self.rng = RandomGenerator(params.seed)

    def run(self) -> OptimizationResult:
        start_time = time.time()
        convergence = []

        n = self.instance.n_activities
        current = list(self.rng.permutation(n))
        current_delays = [self.rng.rng.random() for _ in range(n)] if self.params.use_delay_factors else None
        current_fitness, _ = self.evaluator.evaluate(current, current_delays)

        best = current.copy()
        best_delays = current_delays.copy() if current_delays else None
        best_fitness = current_fitness

        temperature = self.params.initial_temperature
        n_iterations = 0

        while self.evaluator.n_evaluations < self.params.max_evaluations:
            elapsed = time.time() - start_time
            if elapsed >= self.params.time_limit:
                break

            for _ in range(self.params.iterations_per_temperature):
                if self.evaluator.n_evaluations >= self.params.max_evaluations:
                    break

                mutation_fn = get_operator(self.params.mutation_operator)
                neighbor = mutation_fn(current, self.rng)
                neighbor_delays = self._mutate_delays(current_delays) if self.params.use_delay_factors else None
                neighbor_fitness, _ = self.evaluator.evaluate(neighbor, neighbor_delays)

                delta = neighbor_fitness - current_fitness

                if delta < 0 or self._acceptance_probability(delta, temperature) > self.rng.rng.random():
                    current = neighbor
                    current_delays = neighbor_delays
                    current_fitness = neighbor_fitness

                    if current_fitness < best_fitness:
                        best = current.copy()
                        best_delays = current_delays.copy() if current_delays else None
                        best_fitness = current_fitness

            convergence.append(best_fitness)
            temperature *= self.params.cooling_rate
            n_iterations += 1

        runtime = time.time() - start_time
        _, best_result = self.evaluator.evaluate(best, best_delays)

        return OptimizationResult(
            best_objective=best_fitness,
            best_chromosome=best,
            best_delay_factors=best_delays,
            best_schedule=best_result.start_times,
            n_evaluations=self.evaluator.n_evaluations,
            n_iterations=n_iterations,
            runtime=runtime,
            convergence_history=convergence,
            is_feasible=best_result.is_feasible,
            algorithm_name="SA",
            makespan=best_result.makespan
        )

    def _mutate_delays(self, delays: List[float]) -> List[float]:
        if delays is None:
            return None
        result = delays.copy()
        if len(result) > 0:
            idx = self.rng.integers(0, len(result))
            result[idx] = max(0.0, min(1.0, result[idx] + (self.rng.rng.random() - 0.5) * 0.3))
        return result

    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        if delta < 0:
            return 1.0
        return np.exp(-delta / temperature)


ALGORITHMS = {
    "ils": (IteratedLocalSearch, ILSParams),
    "ga": (GeneticAlgorithm, GAParams),
    "sa": (SimulatedAnnealing, SAParams),
}


def create_algorithm(
    name: str,
    instance: RCPSPInstance,
    deadline: int,
    params: AlgorithmParams = None
) -> Tuple[object, AlgorithmParams]:
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}")

    algo_class, params_class = ALGORITHMS[name]

    if params is None:
        params = params_class()

    algorithm = algo_class(instance, deadline, params)
    return algorithm, params
