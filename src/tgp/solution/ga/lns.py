import numpy as np
import numpy.typing as npt
from numba import njit
from numba.typed import List

from tgp.solution.ga.common import calculate_route_cost, get_routes


@njit
def destroy(routes: List, num_to_remove: int) -> tuple[List, List]:
    """
    Destroy phase of the LNS algorithm.

    Args:
        routes (List): The city routes.
        num_to_remove (int): The number of cities to remove, picked randomly,
            from the routes.

    Returns:
        tuple[List, List]: A tuple containing the new routes after removal and
            the list of removed nodes.
    """

    all_nodes = List()
    for r in routes:
        for n in r:
            all_nodes.append(n)

    if len(all_nodes) <= num_to_remove:
        return routes[:0], all_nodes

    indices = np.random.choice(len(all_nodes), size=num_to_remove, replace=False)
    mask = np.ones(len(all_nodes), dtype=np.bool_)
    for idx in indices:
        mask[idx] = False

    removed_nodes = List()
    for idx in indices:
        removed_nodes.append(all_nodes[idx])

    new_routes = List()
    curr_idx = 0
    for r in routes:
        new_r = List()
        for _ in r:
            if mask[curr_idx]:
                new_r.append(all_nodes[curr_idx])
            curr_idx += 1
        if len(new_r) > 0:
            new_routes.append(new_r)

    return new_routes, removed_nodes


@njit
def repair(
    routes: List,
    removed_nodes: List,
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
) -> List:
    """
    "Repairs" the routes by re-inserting the removed cities at the optimal place.

    Args:
        routes (List): Routes to "repair".
        removed_nodes (List): Nodes removed in the "destroy" phase.
        dist_matrix (npt.NDArray[np.float32]): The matrix of distances between each node
        golds (npt.NDArray[np.float32]): The gold values associated with each city.
        alpha (float): The alpha parameter in the cost function.
        beta (float): The beta parameter in the cost function.

    Returns:
        List: The repaired routes.
    """

    for node in removed_nodes:
        best_cost_increase = np.inf
        best_route_idx = -1
        best_pos_idx = -1

        for r_idx, r in enumerate(routes):
            base_cost = calculate_route_cost(r, dist_matrix, golds, alpha, beta)

            for pos in range(len(r) + 1):
                temp_r = List()
                for k in range(pos):
                    temp_r.append(r[k])
                temp_r.append(node)
                for k in range(pos, len(r)):
                    temp_r.append(r[k])

                new_cost = calculate_route_cost(temp_r, dist_matrix, golds, alpha, beta)
                increase = new_cost - base_cost

                if increase < best_cost_increase:
                    best_cost_increase = increase
                    best_route_idx = r_idx
                    best_pos_idx = pos

        new_route_cost = calculate_route_cost(
            List([node]), dist_matrix, golds, alpha, beta
        )
        if new_route_cost < best_cost_increase:
            best_cost_increase = new_route_cost
            best_route_idx = len(routes)
            best_pos_idx = 0

        if best_route_idx == len(routes):
            routes.append(List([node]))
        else:
            r = routes[best_route_idx]
            r.insert(best_pos_idx, node)

    return routes


@njit
def lns(
    tour: List,
    predecessors: npt.NDArray[np.int32],
    dist_matrix: npt.NDArray[np.float32],
    golds: npt.NDArray[np.float32],
    alpha: float,
    beta: float,
    num_to_remove: int,
) -> List:
    """
    Implements Large Neighborhood Search

    Args:
        tour (List): The giant tour of cities (permutation).
        predecessors (npt.NDArray[np.int32]): The predecessor array indicating route breaks.
        dist_matrix (npt.NDArray[np.float32]): The distance matrix between cities.
        golds (npt.NDArray[np.float32]): The gold values associated with cities.
        alpha (float): The alpha parameter in the cost function.
        beta (float): The beta parameter in the cost function.
        num_to_remove (int): The number of nodes to remove during the destroy phase.

    Returns:
        List: The improved giant tour after applying Large Neighborhood Search.
    """

    routes = get_routes(tour, predecessors)

    routes, removed = destroy(routes, num_to_remove)

    routes = repair(routes, removed, dist_matrix, golds, alpha, beta)

    new_tour = List()
    for r in routes:
        for n in r:
            new_tour.append(n)

    return new_tour
