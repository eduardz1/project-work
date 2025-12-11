import networkx as nx
import numpy as np
from itertools import combinations
from typing import cast
from functools import cached_property
import matplotlib.pyplot as plt

MIN_GOLD = 1.0
MAX_GOLD = 1000.0


class Problem:
    graph: nx.Graph
    alpha: float
    beta: float

    def __init__(
        self,
        num_cities: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.beta = beta
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5

        self.graph = nx.Graph()
        self.graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            self.graph.add_node(
                c,
                pos=(cities[c, 0], cities[c, 1]),
                gold=(MIN_GOLD + (MAX_GOLD - MIN_GOLD) * rng.random()),
            )

        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1), dtype=np.float32)
        for c1, c2 in combinations(range(num_cities), 2):
            if rng.random() < density or c2 == c1 + 1:
                self.graph.add_edge(c1, c2, dist=d[c1, c2])

        assert nx.is_connected(self.graph)

    @cached_property
    def paths(self) -> dict[int, dict[int, list[int]]]:
        return cast(
            dict[int, dict[int, list[int]]],
            dict(nx.shortest_path(self.graph, weight="dist")),
        )

    @cached_property
    def dists(self) -> dict[int, dict[int, float]]:
        return cast(
            dict[int, dict[int, float]],
            {
                u: {
                    v: nx.path_weight(self.graph, self.paths[u][v], weight="dist")
                    for v in self.graph
                }
                for u in self.graph
            },
        )

    def cost(self, path, weight):
        dist = nx.path_weight(self.graph, path, weight="dist")
        return dist + (self.alpha * dist * weight) ** self.beta

    def baseline(self) -> float:
        """
        Greedy approach which takes 100% of the gold from each node and then
        goes back to node 0 using Dijkstra's algorithm to find the shortest path
        between each node and node 0.

        Returns:
            float: Total cost of the baseline solution.
        """

        cost = 0

        for dest, path in nx.single_source_dijkstra_path(
            self.graph, source=0, weight="dist"
        ).items():
            if dest == 0:
                continue
            cost += self.cost(path, 0) + self.cost(path, self.graph.nodes[dest]["gold"])

        return cost

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self.graph, "pos")
        size = [100] + [self.graph.nodes[n]["gold"] for n in range(1, len(self.graph))]
        color = ["red"] + ["lightblue"] * (len(self.graph) - 1)
        return nx.draw(
            self.graph, pos, with_labels=True, node_color=color, node_size=size
        )
