import itertools

import networkx as nx
from numba import cuda

from tgp.problem.problem import Problem
from tgp.solution.baseline import baseline
from tgp.solution.ga.crossovers import iox, ox, pmx
from tgp.solution.ga.mutations import inversion_mutation, swap_mutation
from tgp.solution.solution import evaluate_solution, solution


def main():
    graph_sizes = [50, 100, 200, 400]
    densities = [0.3, 0.5, 0.7, 1.0]
    alphas = [0.1, 1.0, 10.0]
    betas = [0.5, 1.0, 2.0]

    mutations = [swap_mutation, inversion_mutation]
    crossovers = [pmx, ox, iox]

    problems = [
        Problem(*p) for p in itertools.product(graph_sizes, alphas, betas, densities)
    ]

    if cuda.is_available():
        nx.config.backend_priority = ["cugraph"]
        nx.config.warnings_to_ignore.add("cache")

    for p in problems:
        print(
            f"Solving Problem: |V|={len(p.graph.nodes)}, alpha={p.alpha}, beta={p.beta}, density={nx.density(p.graph):.2f}"
        )
        sol = solution(p)

        total_cost = evaluate_solution(
            sol,
            p.dists,
            p.alpha,
            p.beta,
        )
        print("Total Cost of Solution:", total_cost)
        base = baseline(p)
        print("Baseline Cost:", base)
        improvement = (base - total_cost) / base * 100
        print(f"Improvement: {improvement:.2f}%")
        print("-" * 40)


if __name__ == "__main__":
    main()
