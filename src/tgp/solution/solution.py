import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from tgp.problem.problem import Problem
from tgp.solution.ga.ga import ga, predecessors_to_solution, remove_bulk
from tgp.types import SolutionType


def expand_solution(
    s: SolutionType, shortest_paths: dict[int, dict[int, list[int]]]
) -> SolutionType:
    """Expand solution to include all intermediate steps between cities."""
    if not s:
        return []

    expanded = [s[0]]
    for i in range(1, len(s)):
        c_prev, gold_prev = s[i - 1]
        c_curr, gold_curr = s[i]
        path = shortest_paths[c_prev][c_curr]

        for c in path[1:-1]:
            expanded.append((c, gold_prev))

        expanded.append((c_curr, gold_curr))

    return expanded


def solution(P: Problem, **kwargs) -> SolutionType:
    """
    Compute the solution of the Traveling Goblin Problem.

    The function uses a Genetic Algorithm to find an approximate solution to the
    Traveling Goblin Problem.

    If the beta parameter is greater than 1, it first removes the bulk of the
    gold from each node to optimize the solution. It can be proven that for
    beta > 1, splitting the tour into multiple short trips becomes more
    efficient as beta grows.

    For 0 <= beta <= 1, the standard Genetic Algorithm is applied directly to
    the problem. The population is initialized with random permutations of all
    cities different from the starting one. Concepts of the Vehicle Routing
    Problem are used here, in particular the giant tour is split using Prins'
    algorithm to find the optimal split points. The fitness of each individual
    is computed using the cost of this optimal split. The split is constrained
    to O(N*k) complexity, with k being a constant that is computed based on the
    problem characteristics. Parents are selected using tournament selection.
    Crossover is used to generate offspring, mutation is applied to the child
    to maintain diversity. Large Neighborhood Search (LNS) is applied to the
    optimal split of the child with a certain probability. Elitism is used to
    carry only the best individuals to the next generation.

    Args:
        P (Problem): The Traveling Goblin Problem instance to solve.

    Keyword Args:
        population_size_percent (float, optional): The size of the population as
            a percentage of the number of nodes.
        generations_percent (float, optional): The number of generations as a
            percentage of the number of nodes.
        elitism_rate (float, optional): The rate of elitism in the genetic algorithm.
        mutation_rate (float, optional): The mutation rate in the genetic algorithm.
        lns_rate (float, optional): The Large Neighborhood Search rate in the
            genetic algorithm.
        lns_num_to_remove_percent (float, optional): The percentage of nodes to
            remove in the "destroy" phase of the Large Neighborhood Search.
        tournament_size_percent (float, optional): The size of the tournament as
            a percentage of the population size.
        crossover (Callable, optional): The crossover function to use in the genetic
            algorithm.
        mutation (Callable, optional): The mutation function to use in the genetic
            algorithm.

    Returns:
        SolutionType: The computed solution path.
    """

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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task("Running Genetic Algorithm...", total=1)
        best_ga = ga(nodes_array, P.dists, golds_array, P.alpha, P.beta, **kwargs)

    ga_sol = predecessors_to_solution(best_ga.tour, best_ga.predecessors, golds_array)
    ga_sol_expanded = expand_solution(ga_sol, P.paths)

    if sol: # skip duplicate ending
        ga_sol_expanded = ga_sol_expanded[1:]

    return sol + ga_sol_expanded
