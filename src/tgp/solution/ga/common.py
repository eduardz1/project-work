import numpy as np
import numpy.typing as npt
from numba import njit
from numba.typed import List

from tgp.types import SolutionType


@njit
def get_routes(tour: List, predecessors: npt.NDArray[np.int32]) -> List:
    routes = List()
    curr = len(tour)
    while curr > 0:
        prev = predecessors[curr]
        # trip from prev to curr (indices in tour)
        trip = List()
        for k in range(prev, curr):
            trip.append(tour[k])
        routes.append(trip)
        curr = prev

    # Reverse routes to match the original tour order
    n = len(routes)
    for i in range(n // 2):
        routes[i], routes[n - 1 - i] = routes[n - 1 - i], routes[i]

    return routes


@njit(inline="always")
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


@njit(inline="always")
def cost(dist: float, weight: float, alpha: float, beta: float) -> float:
    return dist + (dist * alpha * weight) ** beta


@njit
def evaluate_solution(
    solution: SolutionType,
    distance_matrix: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
) -> float:
    """
    Evaluate the total cost of a solution path.

    Because the cost function is non-linear, we cannot sum costs per edge when
    intermediate nodes are present. Instead, we accumulate the distance for each
    consecutive edge where the carried weight stays constant and compute
    the cost once with the full shortest-path distance for that section.

    Args:
        solution (SolutionType): The solution path to evaluate, as a list of (city, weight) tuples.
        distance_matrix (npt.NDArray[np.float32]): A precomputed matrix of distances between cities.
        alpha (float): The alpha parameter of the cost function.
        beta (float): The beta parameter of the cost function.

    Returns:
        float: The total cost of the solution path.
    """

    if len(solution) < 2:
        return 0.0

    total_cost = 0.0
    accumulated_dist = np.float32(0.0)
    current_weight = solution[0][1]

    for i in range(len(solution) - 1):
        city_from, weight_from = solution[i]
        city_to = solution[i + 1][0]

        if weight_from != current_weight:
            total_cost += cost(accumulated_dist, current_weight, alpha, beta)
            accumulated_dist = np.float32(0.0)
            current_weight = weight_from

        accumulated_dist += distance_matrix[city_from, city_to]

    if accumulated_dist > 0.0:
        total_cost += cost(accumulated_dist, current_weight, alpha, beta)

    return total_cost


@njit
def calculate_route_cost(
    route: list[int],
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
) -> float:
    if len(route) == 0:
        return 0.0

    total_cost = 0.0
    load = 0.0
    prev = 0  # depot

    for node in route:
        d = dist_matrix[prev, node]
        total_cost += cost(d, load, alpha, beta)
        load += golds[node]
        prev = node

    # return to depot
    d = dist_matrix[prev, 0]
    total_cost += cost(d, load, alpha, beta)

    return total_cost
