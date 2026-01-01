import networkx as nx

from tgp.problem.problem import Problem


def greedy(problem: Problem) -> float:
    """
    Greedy approach which takes 100% of the gold from each node and then
    goes back to node 0 using Dijkstra's algorithm to find the shortest path
    between each node and node 0. The cost is computed using the formula:
        cost = dist + (alpha * dist * gold) ** beta

    Returns:
        float: Total cost of the baseline solution.
    """

    cost = 0

    for dest, path in nx.single_source_dijkstra_path(
        problem.graph, source=0, weight="dist"
    ).items():
        if dest == 0:
            continue

        for c1, c2 in zip(path, path[1:]):
            dist = problem.graph[c1][c2]["dist"]

            cost += (  # Cost (0 -> dest -> 0)
                2 * dist
                + (problem.alpha * dist * problem.graph.nodes[dest]["gold"])
                ** problem.beta
            )

    return cost
