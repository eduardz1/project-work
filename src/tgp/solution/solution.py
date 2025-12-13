import numpy as np
import numpy.typing as npt
from numba import njit

from tgp.problem.problem import Problem
from tgp.solution.ga.ga import cost, ga, predecessors_to_solution, remove_bulk
from tgp.types import SolutionType


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


def solution(P: Problem, **kwargs) -> SolutionType:
    assert P.beta >= 0.0 and P.alpha >= 0.0

    G = P.graph
    sol: SolutionType = []

    if P.beta > 1.0:
        new_graph, bulk_sol = remove_bulk(G, P.alpha, P.beta, P.paths, P.dists)

        G = new_graph
        sol = bulk_sol

    nodes_array = np.array(list(G.nodes), dtype=np.int32)
    golds_array = np.array(
        [G.nodes[node]["gold"] for node in G.nodes], dtype=np.float32
    )

    best_ga = ga(nodes_array, P.dists, golds_array, P.alpha, P.beta, **kwargs)

    ga_sol = predecessors_to_solution(best_ga.tour, best_ga.predecessors, golds_array)

    return sol + ga_sol
