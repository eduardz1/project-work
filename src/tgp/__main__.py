import itertools
import numpy as np
import networkx as nx
from numba import cuda

from tgp.problem.problem import Problem
from tgp.solution.solution import solution, evaluate_solution


def main():
    graph_sizes = [50, 100, 200, 400]
    densities = [0.3, 0.5, 0.7, 1.0]
    alphas = [0.1, 1.0, 10.0]
    betas = [0.5, 1.0, 2.0]

    problems = [
        Problem(*p) for p in itertools.product(graph_sizes, alphas, betas, densities)
    ]

    if cuda.is_available():
        nx.config.backend_priority = [
            "cugraph",
        ]
        nx.config.warnings_to_ignore.add("cache")

    for p in problems:
        print(
            f"Solving Problem: |V|={len(p.graph.nodes)}, alpha={p.alpha}, beta={p.beta}, density={nx.density(p.graph):.2f}"
        )
        sol = solution(p)

        dist_matrix = np.zeros(
            (len(p.graph.nodes), len(p.graph.nodes)), dtype=np.float32
        )
        for i in p.graph.nodes:
            for j in p.graph.nodes:
                dist_matrix[i, j] = p.dists[i][j]

        total_cost = evaluate_solution(
            sol,
            dist_matrix,
            p.alpha,
            p.beta,
        )
        print("Total Cost of Solution:", total_cost)
        base = p.baseline()
        print("Baseline Cost:", base)
        print("Improvement: {:.2f}%".format((base - total_cost) / base * 100))
        print("-" * 40)


if __name__ == "__main__":
    main()
