import heapq
from math import ceil
from typing import Callable

import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import njit, prange
from numba.typed import List

from tgp.solution.ga.crossovers import iox
from tgp.solution.ga.individual import Individual, create_individual_list
from tgp.solution.ga.mutations import swap_mutation
from tgp.types import SolutionType


@njit
def optimal_fraction_size(alpha: float, beta: float, distance: float) -> float:
    """
    Computes the optimal fraction size to solve the TGP problem. For beta -> 1
    the fraction size goes to infinity, meaning that dividing the nodes in
    fractions is useful only for beta > 1.

    Args:
        alpha (float): parameter of the cost function
        beta (float): parameter of the cost function
        distance (float): distance between nodes

    Returns:
        float: optimal fraction size
    """

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
    Uses Prin's algorithm to convert a giant tour (permutation of cities) into
    an optimal split of trips minimizing the total cost.
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
    dists: npt.NDArray[np.float32],
) -> tuple[nx.Graph, SolutionType]:
    """
    Removes the bulk of the gold from each node. This optimization is useful for
    beta > 1 as splitting the tour in multiple short trips become more efficient
    as beta grows.

    Args:
        G (nx.Graph): the problem graph
        alpha (float): the problem alpha
        beta (float): the problem beta
        paths (dict[int, dict[int, list[int]]]): a dictionary of shortest paths
            between all pairs of nodes
        dists (npt.NDArray[np.float32]): a matrix of shortest path distances
            between all pairs of nodes

    Returns:
        tuple[nx.Graph, SolutionType]: a tuple containing the new graph with
            reduced gold amounts and a solution path representing the steps
            taken to collect the bulk gold.
    """

    new_graph = G.copy()
    sol: SolutionType = []

    L_stars = [optimal_fraction_size(alpha, beta, dists[0, node]) for node in G.nodes]

    for node in G.nodes:
        if node == 0:
            continue

        num_fractions = int(G.nodes[node]["gold"] // L_stars[node])
        # TODO: We could overstuff the last fraction to reduce the number of
        # cities we have to visit. This would slightly decrease the optimality
        # but would highly reduce the solution space, which is more important
        # given O(N!). The problem is finding a mathematical threshold for that
        # and also separate the cities that we have to visit from those that are
        # actually in the graph. If we were to just remove nodes we would end up
        # influencing the shortest paths and distances and could even end up
        # with an unconnected graph.
        new_graph.nodes[node]["gold"] = G.nodes[node]["gold"] % L_stars[node]

        for _ in range(num_fractions):
            for city in paths[0][node]:
                sol.append((city, 0.0))

            sol[-1] = (node, L_stars[node])

            for city in paths[node][0][1:]:
                sol.append((city, L_stars[node]))

    return new_graph, sol


@njit
def tournament_selection(population: list[Individual], tau: int = 4) -> Individual:
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


@njit(parallel=True)
def ga(
    nodes: npt.NDArray[np.int32],
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
    population_size_percent: float = 0.5,
    generations_percent: float = 0.5,
    elitism_rate: float = 0.2,
    mutation_rate: float = 0.1,
    seed: int = 42,
    crossover: Callable[[Individual, Individual], Individual] = iox,
    mutation: Callable[[list[int]], None] = swap_mutation,
):
    np.random.seed(seed)

    non_depot_nodes = nodes[nodes != 0]
    population_size: int = ceil(len(non_depot_nodes) * population_size_percent)

    population = create_individual_list()

    for _ in range(population_size):
        tour_arr = non_depot_nodes.copy()
        np.random.shuffle(tour_arr)
        tour = List(tour_arr)
        heapq.heappush(
            population,
            Individual(tour, *optimal_split(tour, dist_matrix, golds, alpha, beta)),
        )

    for _ in range(int(generations_percent * population_size)):
        elite_size = int(elitism_rate * population_size)
        num_offspring = population_size - elite_size

        offspring = create_individual_list(num_offspring)

        for i in prange(num_offspring):  # ty:ignore[not-iterable]
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1, parent2)

            if np.random.random() < mutation_rate:
                mutation(child.tour)

            child.fitness, child.predecessors = optimal_split(
                child.tour, dist_matrix, golds, alpha, beta
            )
            offspring[i] = child

        new_population = population.copy()
        new_population = new_population[:elite_size]

        for child in offspring:
            heapq.heappush(new_population, child)

        population = new_population

    return population[0]
