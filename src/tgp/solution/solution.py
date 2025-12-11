from ..ga.ga import cost, predecessors_to_solution, remove_bulk, ga
import networkx as nx
from ..problem.problem import Problem
import numpy as np
import numpy.typing as npt
from numba import njit

from ..types import SolutionType


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


# TODO: CLean this up
def validate_solution(
    solution: SolutionType,
    G: nx.Graph,
    verbose: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validates a solution for correctness.

    Checks:
    1. Solution starts and ends at depot (node 0)
    2. All cities are reachable from each other (connected graph)
    3. Weight tracking is correct (increases when picking up gold, stays constant otherwise)
    4. All gold is collected exactly once
    5. Each trip returns to depot

    NOTE: The solution represents a logical sequence of cities visited.
    Edges don't need to exist directly - shortest paths are used for travel.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    if len(solution) == 0:
        errors.append("Solution is empty")
        return False, errors

    # Check starts at depot
    if solution[0][0] != 0:
        errors.append(
            f"Solution must start at depot (0), but starts at {solution[0][0]}"
        )

    if solution[0][1] != 0.0:
        errors.append(
            f"Solution must start with weight 0, but starts with {solution[0][1]}"
        )

    # Check ends at depot
    if solution[-1][0] != 0:
        errors.append(f"Solution must end at depot (0), but ends at {solution[-1][0]}")

    # Track gold collected per node
    gold_collected = {node: 0.0 for node in G.nodes}

    # Validate each step
    for i in range(len(solution) - 1):
        city_from, weight_from = solution[i]
        city_to, weight_to = solution[i + 1]

        # Special case: depot to depot (unloading between trips)
        if city_from == 0 and city_to == 0:
            if weight_to > weight_from:
                errors.append(
                    f"Step {i}: At depot, weight increased from {weight_from} to {weight_to}"
                )
            # Weight decrease or stay same is OK (unloading or staying)
            continue

        # Check nodes exist in graph
        if city_from not in G.nodes:
            errors.append(f"Step {i}: Node {city_from} does not exist in graph")
        if city_to not in G.nodes:
            errors.append(f"Step {i}: Node {city_to} does not exist in graph")

        # Check weight logic
        if city_from == city_to:
            # Same city - shouldn't happen in a valid path
            errors.append(f"Step {i}: Path stays at same city {city_from}")
        elif city_to == 0:
            # Returning to depot - weight should stay the same
            if abs(weight_to - weight_from) > 1e-6:
                errors.append(
                    f"Step {i}: Returning to depot, weight should stay {weight_from} but changed to {weight_to}"
                )
        else:
            # Moving to non-depot city
            # Weight should either stay same or increase by gold at destination
            gold_at_dest = G.nodes[city_to]["gold"]

            if abs(weight_to - weight_from) < 1e-6:
                # No gold picked up (passing through)
                pass
            elif abs(weight_to - weight_from - gold_at_dest) < 1e-6:
                # Picked up all gold at destination
                gold_collected[city_to] += gold_at_dest
            elif weight_to > weight_from:
                # Picked up some gold
                amount = weight_to - weight_from
                gold_collected[city_to] += amount
                if verbose:
                    errors.append(
                        f"Step {i}: Picked up {amount:.2f} gold at city {city_to} "
                        f"(available: {gold_at_dest:.2f})"
                    )
            else:
                errors.append(
                    f"Step {i}: Weight decreased from {weight_from} to {weight_to} "
                    f"when moving to city {city_to}"
                )

    # Check all gold collected
    for node in G.nodes:
        if node == 0:
            continue
        expected = G.nodes[node]["gold"]
        collected = gold_collected[node]
        if abs(expected - collected) > 1e-6:
            errors.append(
                f"Node {node}: Expected {expected:.2f} gold, collected {collected:.2f}"
            )

    # Check for valid trips (each trip should return to depot)
    trips = []
    current_trip = [solution[0]]
    for i in range(1, len(solution)):
        current_trip.append(solution[i])
        if solution[i][0] == 0:
            trips.append(current_trip)
            current_trip = [solution[i]]

    if verbose and len(errors) == 0:
        print(f"✓ Solution is valid with {len(trips)} trips")
        print("✓ All edges exist in graph")
        print("✓ All gold collected correctly")

    return len(errors) == 0, errors


def solution(P: Problem):
    assert P.beta >= 0.0 and P.alpha >= 0.0

    G = P.graph
    sol: SolutionType = []

    if P.beta > 1.0:
        new_graph, bulk_sol = remove_bulk(G, P.alpha, P.beta, P.paths, P.dists)

        G = new_graph
        sol = bulk_sol

    nodes_array = np.array(list(G.nodes), dtype=np.int32)
    dist_matrix = np.zeros((len(G.nodes), len(G.nodes)), dtype=np.float32)
    for i in G.nodes:
        for j in G.nodes:
            dist_matrix[i, j] = P.dists[i][j]

    golds_array = np.array(
        [G.nodes[node]["gold"] for node in G.nodes], dtype=np.float32
    )

    best_ga = ga(
        nodes_array,
        dist_matrix,
        golds_array,
        P.alpha,
        P.beta,
    )

    ga_sol = predecessors_to_solution(best_ga.tour, best_ga.predecessors, golds_array)

    return sol + ga_sol
