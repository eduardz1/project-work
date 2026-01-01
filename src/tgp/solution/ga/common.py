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
    total_cost = 0.0

    for i in range(len(solution) - 1):
        city_from, weight_from = solution[i]
        city_to, _ = solution[i + 1]

        dist = distance_matrix[city_from, city_to]
        total_cost += cost(dist, weight_from, alpha, beta)

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
