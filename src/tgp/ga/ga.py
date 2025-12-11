import heapq
from math import ceil
import numpy as np
import numpy.typing as npt
import networkx as nx
from numba import njit, types, prange
from numba.experimental import jitclass
from numba.typed import List

from tgp.types import SolutionType


@njit
def optimal_fraction_size(alpha: float, beta: float, distance: float) -> list[float]:
    return (1 / alpha) * ((2 * distance ** (1 / beta)) / (beta - 1)) ** (1 / beta)


@njit
def cost(dist: float, weight: float, alpha: float, beta: float) -> float:
    return dist + (dist * alpha * weight) ** beta


@njit
def optimal_split(
    tour: list[int],
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
) -> tuple[float, npt.NDArray[np.int32]]:
    """
    Converts a Giant Tour (permutation of cities) into an optimal set of trips.
    Uses Dijkstra-like logic on a DAG (Directed Acyclic Graph).
    """
    N = len(tour)
    V = np.full(N + 1, np.inf, dtype=np.float32)
    V[0] = 0.0

    predecessors = np.zeros(N + 1, dtype=np.int32)

    LIMIT = N if beta <= 1.0 else 32  # TODO: come up with an exact heuristic

    for i in range(N):
        if V[i] == np.inf:
            continue

        load = 0.0
        curr_cost = 0.0
        prev = 0

        for j in range(i, min(N, i + LIMIT)):
            curr = tour[j]

            d_ij = dist_matrix[prev, curr]
            curr_cost += cost(d_ij, load, alpha, beta)

            load += float(golds[curr])

            d_j0 = dist_matrix[curr, 0]
            return_cost = cost(d_j0, load, alpha, beta)

            total_trip_cost = curr_cost + return_cost

            if V[i] + total_trip_cost < V[j + 1]:
                V[j + 1] = V[i] + total_trip_cost
                predecessors[j + 1] = i

            prev = curr

    return V[N], predecessors


@njit
def predecessors_to_solution(
    tour: list[int],
    predecessors: npt.NDArray[np.int32],
    golds: npt.NDArray[np.float32],
) -> SolutionType:
    """
    Converts the optimal split result into a solution path.

    NOTE: The solution only includes the cities where we pick up gold (and depot).
    It does NOT include intermediate nodes in shortest paths, as those would cause
    incorrect cost calculations. The cost function is non-linear, so traversing
    intermediate nodes explicitly would give different costs than using the
    shortest path distance directly.

    Args:
        tour: The giant tour (permutation of non-depot cities)
        predecessors: The optimal split points computed by optimal_split
        golds: Gold available at each node
        paths: Shortest paths between all pairs of nodes (used for validation only)

    Returns:
        A solution path with only the cities where gold is collected
    """
    split_points = [len(tour)]
    curr = len(tour)
    while curr > 0:
        curr = predecessors[curr]
        split_points.append(curr)

    sol = [(0, 0.0)]

    for trip_idx in range(len(split_points) - 1, 0, -1):
        start = split_points[trip_idx]
        end = split_points[trip_idx - 1]

        trip_cities = [int(tour[k]) for k in range(start, end)]

        load = 0.0

        for city in trip_cities:
            load += float(golds[city])
            sol.append((city, load))

        sol.append((0, load))

        if trip_idx > 1:
            sol.append((0, 0.0))

    return sol


def remove_bulk(
    G: nx.Graph,
    alpha: float,
    beta: float,
    paths: dict[int, dict[int, list[int]]],
    dists: dict[int, dict[int, float]],
) -> tuple[nx.Graph, SolutionType]:
    new_graph = G.copy()
    sol: SolutionType = []

    L_stars = [optimal_fraction_size(alpha, beta, dists[0][node]) for node in G.nodes]

    for node in G.nodes:
        if node == 0:
            continue

        num_fractions = int(G.nodes[node]["gold"] // L_stars[node])
        # TODO: It would be nice to remove nodes with almost no gold to reduce the graph size
        new_graph.nodes[node]["gold"] = G.nodes[node]["gold"] % L_stars[node]

        for _ in range(num_fractions):
            for city in paths[0][node]:
                sol.append((city, 0.0))

            sol[-1] = (node, L_stars[node])

            for city in paths[node][0][1:]:
                sol.append((city, L_stars[node]))

    return new_graph, sol


spec = [
    ("tour", types.ListType(types.int32)),
    ("fitness", types.float64),
    ("predecessors", types.int32[:]),
]


@jitclass(spec)  # type: ignore
class Individual:
    def __init__(self, tour, fitness, predecessors):
        self.tour = tour
        self.fitness = fitness
        self.predecessors = predecessors

    def __lt__(self, other):
        return self.fitness < other.fitness


@njit
def tournament_selection(population: list[Individual], tau: int = 5) -> Individual:
    """Select the best individual from a random subset of the population.

    Args:
        population (list[tuple[list[int], float]]): The population of
            individuals with their fitness values.
        tau (int): The number of individuals to select for the tournament.

    Returns:
        Individual: The best individual from the tournament.
    """

    selected = np.random.choice(len(population), size=tau, replace=False)
    return min([population[i] for i in selected])


@njit
def swap_mutation(path: list[int]) -> None:
    """
    Swap in-place two cities in the path to create a mutation.

    Args:
        path (list[int]): The current path representing a TSP solution. (mut)
    """

    a, b = np.random.choice(len(path), size=2, replace=False)
    path[a], path[b] = path[b], path[a]


@njit
def iox(parent1: Individual, parent2: Individual) -> Individual:
    """
    Implements the Inver-over crossover operator.

    The algorithm works as follows:
    1. Select a random value from parent1.
    2. Find the position of this value in parent2.
    3. Find the position of the next value of parent2 from the first value in
        parent1.
    4. Reverse the segment in parent1 between these two positions in a circular
        manner such that the edge in parent2 is preserved and the ones in
        parent1 reversed.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: The child individual resulting from the crossover.
    """

    start1 = np.random.randint(len(parent1.tour))
    start2 = parent2.tour.index(parent1.tour[start1])
    next = parent2.tour[(start2 + 1) % len(parent2.tour)]
    end1 = parent1.tour.index(next)

    child = Individual(
        parent1.tour.copy(), 0.0, np.zeros(len(parent1.tour), dtype=np.int32)
    )

    if start1 < end1:
        segment = child.tour[start1 + 1 : end1 + 1]
        segment.reverse()
        child.tour[start1 + 1 : end1 + 1] = segment
    else:
        # Wrap-around case: reverse segment from start1+1 wrapping to end1
        segment = child.tour[start1 + 1 :]  # + child.tour[: end1 + 1]
        segment.extend(child.tour[: end1 + 1])
        segment.reverse()
        child.tour[start1 + 1 :] = segment[: len(child.tour) - start1 - 1]
        child.tour[: end1 + 1] = segment[len(child.tour) - start1 - 1 :]

    return child


@njit(parallel=True)
def create_offspring(
    population: list[Individual],
    num_offspring: int,
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
    mutation_rate: float,
) -> list[Individual]:
    dummy_tour = List(np.array([0], dtype=np.int32))  # type: ignore
    dummy_preds = np.zeros(1, dtype=np.int32)
    dummy_ind = Individual(dummy_tour, 0.0, dummy_preds)
    offspring = List([dummy_ind for _ in range(num_offspring)])  # type: ignore

    for i in prange(num_offspring):
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child = iox(parent1, parent2)

        if np.random.random() < mutation_rate:
            swap_mutation(child.tour)

        child.fitness, child.predecessors = optimal_split(
            child.tour, dist_matrix, golds, alpha, beta
        )
        offspring[i] = child

    return offspring


@njit(parallel=True)
def ga(
    nodes: npt.NDArray[np.int32],
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
    population_size_percent: float = 0.5,
    generations: int = 100,
    elitism_rate: float = 0.2,
    mutation_rate: float = 0.1,
    seed: int = 42,
):
    np.random.seed(seed)

    non_depot_nodes = nodes[nodes != 0]
    population_size = ceil(len(non_depot_nodes) * population_size_percent)

    dummy_tour = List(np.array([0], dtype=np.int32))  # type: ignore
    dummy_preds = np.zeros(1, dtype=np.int32)
    dummy_ind = Individual(dummy_tour, 0.0, dummy_preds)
    population = List([dummy_ind])  # type: ignore
    population.pop(0)

    for _ in range(population_size):
        tour_arr = non_depot_nodes.copy()
        np.random.shuffle(tour_arr)
        tour = List(tour_arr)  # type: ignore
        heapq.heappush(
            population,
            Individual(tour, *optimal_split(tour, dist_matrix, golds, alpha, beta)),
        )

    for _ in range(generations):
        elite_size = int(elitism_rate * population_size)
        num_offspring = population_size - elite_size

        offspring = create_offspring(
            population, num_offspring, dist_matrix, golds, alpha, beta, mutation_rate
        )

        new_population = population.copy()
        new_population = new_population[:elite_size]

        for child in offspring:
            heapq.heappush(new_population, child)

        population = new_population

    return population[0]
