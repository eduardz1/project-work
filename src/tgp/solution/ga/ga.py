import heapq
from typing import Callable

import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import njit, prange
from numba.typed import List

from tgp.solution.ga.common import cost, get_routes, optimal_fraction_size
from tgp.solution.ga.crossovers import iox
from tgp.solution.ga.individual import Individual, create_individual_list
from tgp.solution.ga.lns import lns
from tgp.solution.ga.mutations import inversion_mutation
from tgp.types import SolutionType


@njit
def compute_prin_visits_limit(
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
) -> int:
    """
    Computes the average limit where we should stop looking for consecutive
    routes in the Prin algorithm. It takes a "pessimistic" approach by sampling
    the bottom 25th percentile of the cities' gold. It allows us to limit the
    complexity to O(Nk) instead of O(N^2)

    Args:
        dist_matrix (npt.NDArray[np.float32]): The matrix of distances between cities.
        golds (npt.NDArray[np.float32]): The gold available at each city.
        alpha (float): The alpha parameter of the cost function.
        beta (float): The beta parameter of the cost function.

    Returns:
        int: The computed limit for consecutive routes in the Prin algorithm.
    """

    avg_dist_from_0 = np.mean(dist_matrix[0, 1:])

    N = len(golds)
    total_nn_dist = 0.0

    for i in range(1, N):
        # Takes the second closes distance (the closest is the node itself)
        row = dist_matrix[i, 1:]
        total_nn_dist += np.partition(row, 1)[1]

    avg_nearest_neighbor_dist = total_nn_dist / (N - 1)

    if avg_nearest_neighbor_dist < np.finfo(np.float32).eps:
        return len(golds)  # Nodes are overlapping

    # Take the bottom 25th percentile just to be pessimistic in our estimate
    avg_gold = np.percentile(golds[1:], 25.0)

    rhs = (
        2 * avg_dist_from_0
        - avg_nearest_neighbor_dist
        + (avg_dist_from_0 * alpha * avg_gold) ** beta
    )

    denominator = alpha * avg_nearest_neighbor_dist

    w_limit = np.power(rhs, 1 / beta) / denominator

    return max(5, int(w_limit / avg_gold))


@njit
def optimal_split(
    tour: list[int],
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
    limit: int,
) -> tuple[float, npt.NDArray[np.int32]]:
    """
    Uses Prin's algorithm to convert a giant tour (permutation of cities) into
    an optimal split of trips minimizing the total cost.

    Args:
        tour (list[int]): The giant tour (permutation of non-depot cities).
        dist_matrix (npt.NDArray[np.float32]): The matrix of distances between each node.
        golds (npt.NDArray[np.float32]): The gold values associated with each city.
        alpha (float): The alpha parameter in the cost function.
        beta (float): The beta parameter in the cost function.
        limit (int): The maximum number of consecutive cities to consider for trips.

    Returns:
        tuple[float, npt.NDArray[np.int32]]: A tuple containing the total cost of the
            optimal split and an array of predecessors indicating the split points.
    """

    N = len(tour)
    V = np.full(N + 1, np.inf, dtype=np.float32)
    V[0] = 0.0

    predecessors = np.zeros(N + 1, dtype=np.int32)

    for i in range(N):
        if V[i] == np.inf:
            continue

        load = 0.0
        curr_cost = 0.0
        prev = 0

        max_j = min(N, i + limit)

        for j in range(i, max_j):
            curr = tour[j]

            d_ij = dist_matrix[prev, curr]
            curr_cost += cost(d_ij, load, alpha, beta)

            load += golds[curr]

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

    routes = get_routes(tour, predecessors)
    sol = [(0, 0.0)]

    for i in range(len(routes)):
        trip_cities = routes[i]
        load = 0.0

        for city in trip_cities:
            load += float(golds[city])
            sol.append((city, load))

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
    sol: SolutionType = [(0, 0.0)]

    L_stars = [optimal_fraction_size(alpha, beta, dists[0, node]) for node in G.nodes]

    for node in G.nodes:
        if node == 0:
            continue

        num_fractions = int(G.nodes[node]["gold"] // L_stars[node])
        # TODO: We could "overstuff" the last fraction to reduce the number of
        # cities we have to visit. This would slightly decrease the optimality
        # but would highly reduce the solution space, which is more important
        # given O(N!). The problem is finding a mathematical threshold for that
        # and separate the cities that we have to visit from those that are
        # actually in the graph. If we were to just remove nodes we would end up
        # influencing the shortest paths and distances and could even end up
        # with an unconnected graph.
        new_graph.nodes[node]["gold"] = G.nodes[node]["gold"] % L_stars[node]

        for _ in range(num_fractions):
            # Depot -> node (skip depot, already there from previous trip)
            for city in paths[0][node][1:]:
                sol.append((city, 0.0))
            sol[-1] = (node, L_stars[node])

            # Node -> depot (skip node, already there)
            for city in paths[node][0][1:]:
                sol.append((city, L_stars[node]))
            sol[-1] = (0, 0.0)

    return new_graph, sol


@njit
def tournament_selection(population: list[Individual], tau: int) -> Individual:
    """Select the best individual from a random subset of the population.

    Args:
        population (list[tuple[list[int], float]]): The population of
            individuals with their fitness values.
        tau (int): The number of individuals to select for the tournament.

    Returns:
        Individual: The best individual from the tournament.
    """

    selected = np.random.choice(len(population), size=tau, replace=False)
    best = population[selected[0]]
    for i in range(1, tau):
        candidate = population[selected[i]]
        if candidate < best:
            best = candidate
    return best


@njit(parallel=True)
def ga(
    nodes: npt.NDArray[np.int32],
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
    population_size_percent: float = 0.23,
    generations_percent: float = 0.21,
    elitism_rate: float = 0.17,
    mutation_rate: float = 0.04,
    lns_rate: float = 0.04,
    lns_num_to_remove_percent: float = 0.25,
    tournament_size_percent: float = 0.14,
    crossover: Callable[[Individual, Individual], Individual] = iox,
    mutation: Callable[[list[int]], None] = inversion_mutation,
):
    nnz_nodes = nodes[nodes != 0]
    p_size = int(len(nnz_nodes) * population_size_percent)
    gens = int(len(nnz_nodes) * generations_percent)
    tau = max(2, int(tournament_size_percent * p_size))
    elite_size = int(elitism_rate * p_size)
    num_offspring = p_size - elite_size
    lns_num_to_remove = int(len(nnz_nodes) * lns_num_to_remove_percent)

    population = create_individual_list()
    limit = compute_prin_visits_limit(dist_matrix, golds, alpha, beta)

    for _ in range(p_size):
        tour_arr = nnz_nodes.copy()
        np.random.shuffle(tour_arr)
        tour = List(tour_arr)
        heapq.heappush(
            population,
            Individual(
                tour, *optimal_split(tour, dist_matrix, golds, alpha, beta, limit)
            ),
        )

    for _ in range(gens):
        offspring = create_individual_list(num_offspring)

        for i in prange(num_offspring):  # ty:ignore[not-iterable]
            parent1 = tournament_selection(population, tau)
            parent2 = tournament_selection(population, tau)
            child = crossover(parent1, parent2)

            if np.random.random() < mutation_rate:
                mutation(child.tour)

            child.fitness, child.predecessors = optimal_split(
                child.tour, dist_matrix, golds, alpha, beta, limit
            )

            if np.random.random() < lns_rate:
                child.tour = lns(
                    child.tour,
                    child.predecessors,
                    dist_matrix,
                    golds,
                    alpha,
                    beta,
                    lns_num_to_remove,
                )
                child.fitness, child.predecessors = optimal_split(
                    child.tour, dist_matrix, golds, alpha, beta, limit
                )

            offspring[i] = child

        new_population = population[:elite_size]
        for child in offspring:
            new_population.append(child)

        heapq.heapify(new_population)
        population = new_population

    return population[0]
