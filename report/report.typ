#import "@preview/lilaq:0.5.0" as lq
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.8": *
#import "@preview/gentle-clues:1.2.0": *
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#import "lib.typ": *

#let okabe = lq.color.map.okabe-ito

#set outline.entry(fill: repeat(gap: .5em)[#sym.dot.c])
#show outline.entry.where(level: 1): it => {
  if it.element.func() != heading {
    return it
  }
  set block(above: 5em)
  line(length: 100%, stroke: 1pt + okabe.at(2))
  set text(weight: "bold", fill: okabe.at(2))
  align(center, it.body())
}
#show outline.entry.where(level: 2): it => {
  if it.element.func() != heading {
    return it
  }
  set block(above: 1em)
  set text(weight: "bold")
  link(it.element.location(), it.indented(it.prefix(), {
    (it.body() + h(1fr) + it.page())
  }))
}

// // Lato does not support small-caps
// #let fakesc(s, scaling: 0.8, expansion: 1.1) = {
//   show regex("\p{Ll}+"): it => {
//     context {
//       box(scale(x: expansion * 100%, reflow: true, text(
//         scaling * 1em,
//         upper(it),
//       )))
//     }
//   }
//   text(s)
// }
#let fakesc = smallcaps
#set document(title: "Traveling Goblin Problem", author: "Eduard Occhipinti")

#set text(font: "New Computer Modern", size: 10pt)
// #set text(font: "Lato", lang: "en", size: 10pt)
// #show math.equation: set text(font: "Lete Sans Math")

#set page(paper: "a4", margin: auto, numbering: "1/1")

#show figure.caption: emph
// Floating figures appear as `place` instead of `block` so we
// need this workaround, see https://github.com/typst/typst/issues/6095
// #show figure.caption: balance.with(is-figure: true)
#show figure: it => {
  if it.placement == none {
    block(it, inset: (y: .75em))
  } else {
    place(
      it.placement + center,
      float: true,
      block(it, inset: (y: .75em)),
    )
  }
}

#set par(justify: true, first-line-indent: 1.8em)

#show figure: set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
#show table.cell.where(y: 0): strong
#show table.cell.where(y: 0): smallcaps
#show table: it => {
  set par(justify: false)
  it
}
#set table(
  stroke: (_, y) => (
    top: if y == 0 { 1pt } else if y == 1 { none } else { 0pt },
    bottom: .5pt,
  ),
  align: center + horizon,
)

#set math.equation(numbering: "(1)")

#set heading(numbering: "1.1")
#show heading: smallcaps
#show heading: set block(above: 1.4em, below: 1em)
#show link: set text(okabe.at(0))
#show cite: set text(okabe.at(3))
#show ref: set text(okabe.at(5))

#show raw: set text(font: "Fira Code")
#show raw.where(block: true): set text(0.9em)
#show: codly-init.with()
#codly(
  breakable: true,
  languages: codly-languages,
  zebra-fill: none,
  // lang-outset: (x: -5pt, y: 5pt),
  number-align: right + horizon,
  number-format: it => text(fill: luma(200), str(it)),
)
#show: style-algorithm.with(
  hlines: (
    grid.hline(stroke: 1pt + black),
    grid.hline(stroke: .5pt + black),
    grid.hline(stroke: 1pt + black),
  ),
  // breakable: false,
)
#show figure.where(kind: "algorithm"): set text(size: 0.8em)

#show heading: it => {
  let num = counter(heading).display()
  block({
    if num != "0" and it.body != [Bibliography] {
      set text(fill: okabe.at(2).lighten(60%))
      // place(num, dx: -(measure(num).width + 0.3em))
      num + h(0.3em)
    }
    set text(fill: okabe.at(2))
    smallcaps(it.body)
  })
}

#show bibliography: set heading(outlined: false)

#{
  set align(center)
  set page(footer: text(
    fill: gray,
  )[ Report for the Computational Intelligence Course Project \ #datetime.today().display()])

  v(.4fr)


  let width = 85%

  line(length: width, stroke: 4pt + okabe.at(2))
  block(
    smallcaps(
      text(
        size: 2.5em,
        fill: okabe.at(2),
        weight: "bold",
      )[Traveling Goblin Problem],
    ),
  )
  line(length: width, stroke: okabe.at(2))
  box(
    width: width,
    grid(
      columns: 2,
      column-gutter: 1fr,
      [s332100\@studenti.polito.it], [Eduard Antonovic Occhipinti],
    ),
  )

  v(.3fr)

  show link: set text(black)
  show ref: set text(black)
  outline(title: none)

  v(1fr)
}


#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  set align(center)
  v(4em)
  it
  v(1em)
}


= Laboratories Report

In this first section I will summarize the work done for each of the laboratories assigned during the course.

== The Joke

- https://github.com/eduardz1/CI2025_lab0

Not much to say about that one.

== Knapsack Problem

- https://github.com/eduardz1/CI2025_lab1

The second lab required us to implement a solution for an N-dimensional knapsack problem. To solve it, I decided to use a _Simulated Annealing_ approach. This approach, inspired by physics, works by integrating a certain degree of randomness into the optimization process, which gradually decreases over time, much like particle movement in a metal that is cooling down.

=== The Algorithm

The algorithm provides a relatively good solution. It can be summarized with the pseudocode we see in @knapsack-algorithm.

#algorithm-figure(
  [Solve the N-Dimensional Knapsack Problem],
  vstroke: .5pt + luma(200),
  inset: .6em,
  indent: 1em,
  {
    import algorithmic: *
    Procedure(
      fakesc[KnapsackSA],
      [],
      {
        Assign[sol][#CallInline[#fakesc[RandomSolution]][]]
        While([*not* #CallInline[#fakesc[IsFeasible]][sol]], {
          Call[#fakesc[RemoveRandomItem]][sol]
        })
        LineComment(Assign[T][T\*], [Assign initial temperature])
        Assign[best_sol][sol]
        For([\_ in #CallInline[#fakesc[Range]][n_iterations]], {
          Assign[new_sol][#CallInline[#fakesc[Mutate]][sol]]
          If([#CallInline[#fakesc[IsFeasible]][new_sol]], {
            Assign[$Delta$][#CallInline[#fakesc[ComputeDelta]][sol, new_sol]]
            If(
              [$Delta >= 0 or$ #CallInline[#fakesc[RandomFloat]][] < $e^(display(Delta) / display(T))$],
              {
                Assign[sol][new_sol]
                If([$"new_sol" > "best_sol"$], {
                  Assign[best_sol][new_sol]
                })
              },
            )
          })
          Assign[T][T \* cooling_rate]
        })
        Return[best_sol]
      },
    )
  },
) <knapsack-algorithm>

=== Comments

In the review I was asked to integrate a way to visualize the convergence of the algorithm over time. This can be useful to understand if the hyperparameters chosen are appropriate for the problem at hand.

== Traveling Salesperson Problem

- https://github.com/eduardz1/CI2025_lab2

The third lab required us to implement the solution for the Traveling Salesperson Problem (TSP) that generalizes also for graphs with negative weights, self loops and non-symmetric edges.

=== The Algorithm

To solve it, I decided to use a Genetic Algorithm (GA) approach. I created the graph using the `NetworkX` library, explicitly specifying whether to make the graph directed or undirected and whether to add self-loops or not. The graph generated can also be labeled with the city names like in the example provided in the lab. The `Python` code looks as follows:

```python
def generate_graph(
    data: npt.NDArray,
    directed: bool = False,
    loop: bool = False,
    labels: list[str] | None = None,
) -> nx.Graph | nx.DiGraph:
    """Generate a NetworkX graph from a 2D numpy array.

    Args:
        data (npt.NDArray): A 2D numpy array representing the adjacency matrix.
        directed (bool): Whether to create a directed graph. Default is False.
        loop (bool): Whether to include self-loops. Default is False.
        labels (list[str] | None): Optional list of labels for the nodes.

    Returns:
        nx.Graph | nx.DiGraph: The generated NetworkX graph.
    """

    assert data.ndim == 2, "Input data must be a 2D array."
    assert data.shape[0] == data.shape[1], "Input data must be a square matrix."

    graph = nx.DiGraph() if directed else nx.Graph()

    n = data.shape[0]
    for i, j in combinations(range(n), 2):
        graph.add_edge(i, j, weight=data[i, j])

        if directed:
            graph.add_edge(j, i, weight=data[j, i])

        if loop:
            # add edges to self to handle cases where the weight is not zero
            graph.add_edge(i, i, weight=data[i, i])
            graph.add_edge(j, j, weight=data[j, j])

    if labels:
        mapping = {i: labels[i] for i in range(len(labels))}
        graph = nx.relabel_nodes(graph, mapping)

    return graph
```

By using `NetworkX`, the fitness of a TSP solution can be computed in a straightforward way using `nx.path_weight`. For the GA I used swaps as the mutation operator, tournament selection as the parent selection operator and a combination of Partially Mapped Crossover (PMX) and Inver-over Crossover (IOX) as the crossover operators.

#info(accent-color: okabe.at(0))[
  / Swap Mutation: swaps two random cities in the tour.

  / Tournament Selection: selects the best individual (2 in the case of parent selection) from a random subset of the population.
  / Partially Mapped Crossover: designed as a recombination operator for TSP-like problems, it preserves relative order and position of cities from the parents.
  / Inver-over Crossover: specifically designed for TSP problems, presented by #cite(<inver-over>, form: "prose"), it works by inverting segments of the tour based on information from both parents. In the work they present, it outperforms all previous traditional crossover operators for TSP and it's the one that was suggested in class. #cite(<inver-over-aco-ga>, form: "prose") also obtain good results by combining the GA with an Ant Colony Optimization (ACO) approach.
]

=== Comments <tsp-comments>

A common criticism of my implementation was the usage of the swap operator instead of an inversion operator for mutation. My reasoning was that the IOX crossover already introduces inversions in the offspring, but nevertheless I will explore the usage of an inversion mutation operator in the project work. Another criticism was the usage of PMX instead of Order Crossover (OX). Personally I don't see how OX would provide a significant advantage over PMX in preserving adjaciencies between cities, but I will also explore this in the project work. Tournament size and elitism rate were also deemed too high, so I will experiment with lower values for both of them.

== Shortest Path

- https://github.com/eduardz1/CI2025_lab3

The goal of this lab was to implement a shortest path algorithm that could handle negative weights and handle negative cycles.

=== The Algorithm

To solve it, I implemented an A\* algorithm that uses the Bellman-Ford algorithm as the heuristic to guide the search. The Bellman-Ford algorithm is able to handle negative weights and detect negative cycles, making it a suitable choice for this problem. The `Python` code for it is relatively straightforward:

```python
@dataclass
class Node:
    """
    Represents a node in the A* search algorithm.

    Attributes:
        id (int): The identifier of the node.
        g (float): The cost from the start node to this node.
        h (float): The heuristic estimate from this node to the target node.
        parent (Optional[Node]): The parent node in the path.
    """

    id: int
    g: float
    h: float
    parent: Optional["Node"] = None

    def __post_init__(self):
        self._f = self.g + self.h

    def __lt__(self, other: "Node"):
        return self._f < other._f


def heuristic(G: nx.DiGraph, node: int, target: int) -> float:
    """
    Computes an heuristic for the A* algorithm.

    Uses networkx's implementation of the Bellman-Ford algorithm to compute the
    shortest-path distances to the target node from all other nodes by
    running the algorithm on a reverse view of the graph. This heuristic should
    handle negative edge weights and detect negative cycles. When a negative
    cycle is detected, the returned heuristic is zero, which is admissible.

    Args:
        G (nx.DiGraph): the graph. The object should be assumed mutable. A cache
            of the Bellman-Ford results is stored in
            `G.graph["_bf_to_target_cache"]` for each target node to avoid
            recomputing the same distances multiple times.
        node (int): the starting node
        target (int): the target node

    Returns:
        float: the heuristic estimate of the distance from `node` to `target`
    """

    cache = G.graph.setdefault("_bf_to_target_cache", {})
    if target not in cache:
        try:
            dist = nx.single_source_bellman_ford_path_length(
                G.reverse(copy=False), target
            )
        except nx.NetworkXUnbounded:
            cache[target] = {}
        else:
            cache[target] = dist

    return float(cache[target].get(node, 0.0))


def astar(
    G: nx.DiGraph, source: int, target: int
) -> tuple[Optional[List[int]], float] | None:
    """
    Implements the A* search algorithm.

    Args:
        G (nx.DiGraph): The directed graph on which to perform the search.
        source (int): The starting node identifier.
        target (int): The target node identifier.

    Returns:
        (tuple[Optional[List[int]], float] | None): A tuple containing the list
            of node identifiers representing the shortest path from source to
            target and the total cost of that path. If no path exists, returns
            None.
    """

    start = Node(id=source, g=0, h=0)
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, start)

    while open_set:
        current = heapq.heappop(open_set)

        if current.id == target:
            path = []
            cost = current.g
            while current:
                path.append(current.id)
                current = current.parent
            return path[::-1], cost

        closed_set.add(current.id)

        for neighbor in G.neighbors(current.id):
            if neighbor in closed_set:
                continue

            g = current.g + G[current.id][neighbor]["weight"]
            h = heuristic(G, neighbor, target)

            neighbor_node = Node(id=neighbor, g=g, h=h, parent=current)

            if any(node.id == neighbor and node.g <= g for node in open_set):
                continue

            heapq.heappush(open_set, neighbor_node)
```

=== Comments

I received a comment about improving the efficiency of the A\* implementation by reducing `heapq` operations. I also seem to have missed the requirement of finding a strictly positive path as solution.

= Project Work

In this section we will explore the work done for the project assignment.

== Problem Definition

The problem consists in finding the optimal path for a goblin to steal all the gold from a set of nodes and bring it back its hut. The goblin can carry any amount of gold and can take any amount of it from each node but it cannot leave any gold behind. The cost function to travel between each node is given by

$
  C = D + (D alpha W)^beta | alpha >= 0 and beta >= 0.
$ <cost>

== Performance Considerations

To improve the performance of the algorithm, I used the `Numba` library to Just-In-Time compile most of the performance-critical code using `LLVM`. For `Networkx` operations, I added support for the `cugraph` backend, providing GPU acceleration when a CUDA-capable GPU is available. I also added support for `Intel`-specific libraries for better `Numba` compilation. You can enable one or both with the following `uv` command:


```bash
uv sync --extra gpu --extra intel
```

== Solution Approach

In this section we will explore the different intuitions and the approaches that led to a final algorithm that nicely approximates a good solution for the problem.

=== First Intuitions

My first intuition when approaching this problem was to consider similar problems, in doing so I identified this problem as a special case of the vehicle routing problem with capacity constraints (CVRP).

==== Prin's Algorithm and Giant Tours <prin-sec>

A common approach to solve this kind of problem is to generate a "giant tour", which consists in taking a permutation of all the nodes in the graph, and then split it into smaller routes. To compute the split, Prin's algorithm is a good choice as is detailed in #cite(<prin>, form: "prose"). The nice thing about this algorithm is that the split computed is optimal for the given permutation. The algorithm has complexity $cal(O)(N^2)$ but can be simplified to $cal(O)(N)$ with the approach in #cite(<split>, form: "prose"). This approach, unfortunately, is not applicable to our non-linear cost function.

==== Fractional Nodes <frac-nodes>

Part of the problem specification is the ability to take any amount of gold from each node. To convert this continuous optimization problem into a discrete one, we can split each node into multiple "fractional nodes", each containing a fraction of the gold from the original node. To avoid having the path finding always end up traversing all the fractions of a node, we also have to make all the fractions fully connected with edges of zero distance. We can also add a fraction of zero gold to each node to allow the solver to skip nodes entirely if needed. This approach allows us to completely transparently optimize the giant tour approach without any knowledge of the internal working of our TSP solver.

#warning(
  accent-color: okabe.at(1),
)[Given that the TSP problem has already complexity $cal(O)(N!)$, any increase in the number of nodes should be carefully considered and possibly avoided.]

=== Analyzing the Cost Function <remove-bulk>

Starting from the assumption that we want to split each node into the minimum number of fractional nodes possible, we define the total cost $T$ as the sum of

+ The cost of the outbound trip from $0$ to $i$, which corresponds to the distance $D$.
+ The return cost using @cost, which comes out to $D + (D alpha W)^beta$.

$
  T = (D) + (D + (D alpha W)^beta) = 2D + (D alpha W)^beta.
$

We can notice that, for $beta < 1$, the cost of the load is "discounted", with $(W_1 + W_2)^beta < W_1^beta + W_2^beta$. This means that combining multiple loads can lead to a lower overall cost.

For $beta >= 1$, we can imagine having two separate scenarios:

1. Taking multiple single trips from base to node $i$ and back, each of them carrying an optimal load that we define as $L^*$.
2. Chaining together multiple cities where we carry an optimal load $L^*$ for each city.

Computing this optimal load $L^*$ allows us to mathematically decide on the fraction factor that we mentioned in @frac-nodes. The split algorithm that we mentioned in @prin-sec will then be able to optimally decide how to group these fractions together. To do so, we focus on minimizing the cost per unit of gold carried, which is a function $U(W)$ defined as

$
  U(W) = T(W) / W = (2D + (D alpha W)^beta) / W = (2D)/W + (D alpha)^beta W^(beta-1).
$

If we assume $D > 0 and alpha >= 0 and W >= 0 and beta >= 1$, the limits for $W -> 0$ and $W -> +infinity$ are both $+infinity$. Therefore, finding the zeros of the derivative $U'(W)$ will give us the minimum cost per unit of gold carried, which will define our optimal load $L^*$.

$
  U'(W) = d / (d W) ( (2D)/W + (D alpha)^beta W^(beta-1) ) & = 0 \
  - (2D) / W^2 + (beta - 1) (D alpha)^beta W^(beta - 2) & = 0 \
  (beta - 1) (D alpha)^beta W^(beta - 2) W^2 & = 2D \
  W^beta & = (2D) / ((beta - 1) (D alpha)^beta) \
  L^* & = 1 / alpha ((2 D^(1 - beta)) / (beta - 1))^(1 / beta).
$

The existence of this result has an important implication: for $beta > 1$ we can find an optimal packet size $L^*$, meaning that chaining two optimal packets is at best only equivalent to taking two separate trips with optimal packets. This means that we can completely exclude the "fractional nodes" approach detailed in @frac-nodes as the best strategy is to make single "greedy" trips with optimal packet size $L^*$ until we are left with a remainder smaller than $L^*$.

Empirically we see that for $beta = 2$, the optimal load is $approx 1$ for every node on our graph defined on a unit square with $alpha = 1$. Given that the amount of gold on each node is sampled from $[1, 1000)$, this means that the optimal strategy in this case consists in thousands of small trips.

=== Constraining Prin's Algorithm <constrain-prin>

As we have established in @prin-sec, due to our cost function being non-linear, we cannot directly apply @split to our problem. However, given that we have a lot of information about the problem characteristics, we can try to constrain the problem to $cal(O)(N k)$, with $k$ being a problem-dependent constant.

We can solve this by saying that we want to find the threshold where the cost associated with carrying an additional load becomes higher than the cost of making an additional trip. We can formalize this as such:

$
  delta + (delta alpha W)^beta > 2 D + (D alpha g)^beta,
$

where $delta$ is the distance from the previous and current city, $D$ is the distance from base to the current city, $W$ is the current load and $g$ is the gold at the current city.

Solving for $W$ we get

$
  W > (2 D - delta + (D alpha g)^beta)^(1 / beta) / (delta alpha).
$

More concretely, this means that we can use the average distance from node zero as our $D$, the average distance between nodes as our $delta$, and the average gold per node as our $g$. The threshold then becomes the $W$ limit divided by the average gold per node. To give a bit more "leeway" to our solver, we take the bottom 25th percentile as the average amount of gold per node and we cap the threshold to a minimum of 5 nodes.

In practice, we notice that this limit works quite well. While there are cases where our optimal split algorithm would have preferred to continue adding nodes, the final cost difference is minimal, proving that the improvement gained by these potential longer routes is negligible.

=== Large Neighborhood Search

A very common approach to solve VRP problems is the usage of Large Neighborhood Search (LNS) or its variants. Inspired by the work in #cite(<lns>, form: "prose"), and its application to Genetic Algorithms in #cite(<ga_lns>, form: "prose"), I implemented a basic LNS algorithm that is used to further optimize the path of the children of each generation in my Genetic Algorithm implementation.

The algorithm works in the following way:

1. A subset of nodes is randomly selected and removed from a tour derived from the optimal split algorithm. This phase is known as _destroy_.
2. The removed nodes are reinserted into the tour in the position that yields the lowest increase in the solution cost. This phase is known as _repair_.

=== Genetic Algorithm

To solve the overall problem, I chose to implement a Genetic Algorithm (GA). The solution can be summarized with the following pseudocode:


#algorithm-figure(
  [Compute Solution for TGP],
  vstroke: .5pt + luma(200),
  inset: .6em,
  indent: 1em,
  {
    import algorithmic: *
    Procedure(
      fakesc[Solution],
      [$P : "Problem"$],
      {
        Assign[sol][[]]
        If([$beta$ > 1], {
          LineComment(
            Assign([sol], CallInline[#fakesc[RemoveBulkGold]][$P$]),
            [See @remove-bulk],
          )
        })
        Assign[ga_sol][#CallInline[#fakesc[GA]][$P$]]
        Return[sol + ga_sol]
      },
    )
    LineBreak
    Procedure(fakesc[GA], [$P: "Problem"$], {
      LineComment(
        Assign[limit][#CallInline[#fakesc[ComputePrinLimit]][$P$]],
        [See @constrain-prin],
      )
      Assign[population][#CallInline[#fakesc[Permutations]][P.graph.nodes]\[:p_size\]]
      For([\_ in #CallInline[#fakesc[Range]][n_generations]], {
        Assign[offsprings][[]]
        For([i in #CallInline[#fakesc[Range]][n_offsprings] *in parallel*], {
          Assign[parent1][#CallInline[#fakesc[TournamentSelection]][population]]
          Assign[parent2][#CallInline[#fakesc[TournamentSelection]][population]]
          Assign[child][#CallInline[#fakesc[Crossover]][parent1, parent2]]
          If([#CallInline[#fakesc[RandomFloat]][] < mutation_rate], {
            Assign[child][#CallInline[#fakesc[Mutate]][child]]
          })
          LineComment(
            Call[#fakesc[ComputeOptimalSplit]][child, $P$, limit],
            [See @prin-sec],
          )
          If([#CallInline[#fakesc[RandomFloat]][] < some_threshold], {
            Assign[child][#CallInline[#fakesc[LNS]][child, $P$, n_to_remove]]
          })
          Assign[offspring[i]][child]
        })
        Assign[population][offsprings + elites]
      })
      Return[best_individual]
    })
  },
)

==== Optimizing the Hyperparameters <hyperparam>

Our Genetic Algorithm has quite a few hyperparameters that need to be tuned to obtain good results. To do so, I found a very interesting library called `Optuna`, by #cite(<optuna>, form: "prose"). This library manages automatically the search space for the hyperparameters to reduce the number of simulations and smartly prune out dead ends. I run 5000 simulations with a combination of problems that I found to be representative for our use case that we see in @hyperparam-problems.

#figure(
  ```python
  # A sufficiently challenging city set without being too large.
  city_sizes = [100]

  # We test both very low alpha which would allow us to take longer routes and
  # high alpha which would force us to take shorter routes.
  alphas = [0.05, 10.0]

  # Both low and high density graphs, should generalize well.
  densities = [0.2, 0.9]

  # We only optimize beta <= 1.0. For beta > 1.0 most of the gains come from
  # removing the bulk gold.
  betas = [0.05, 0.5, 1.0]

  problems = [
      Problem(num_cities=n, alpha=a, beta=b, density=d)
      for n in city_sizes
      for a in alphas
      for b in betas
      for d in densities
  ]
  ```,
  caption: [Set of problems used for hyperparameter optimization.],
) <hyperparam-problems>

Instead of introducing an early stopping criteria in the GA, we optimize for both the average percent improvement over all the problems compared to the greedy baseline, and the total program runtime. By doing so we avoid having the optimizer favoring ever larger populations and number of generations. The Pareto front obtained is shown in @pareto-front.

#figure(
  image("imgs/pareto.png"),
  caption: [Pareto front for the hyperparameter optimization. On the X axis we have the negative of the average percent improvement over the baseline. On the Y axis we have the total runtime in seconds.],
) <pareto-front>

// The combination chosen on the Pareto front, with an average improvement of 27% over the baseline and a cumulative runtime of 1.85 seconds, is the following:

We choose three preset on the Pareto front with an average improvement ranging from 26 to 28 percent over the baseline. We call these presets `fast`, `balanced` and `quality`, with them taking, respectively, $approx 0.3s$, $approx 1.9s$ and $approx 6.7s$ to solve all the benchmark problems. The values of the hyperparameters are the ones shown in @presets.

#figure(
  ```json
  {
    "fast": {
      "population_size_percent": 0.23,
      "generations_percent": 0.21,
      "elitism_rate": 0.17,
      "mutation_rate": 0.04,
      "lns_rate": 0.04,
      "lns_num_to_remove_percent": 0.25,
      "tournament_size_percent": 0.14,
      "crossover": "iox",
      "mutation": "inversion",
    },
    "balanced": {
      "population_size_percent": 0.23,
      "generations_percent": 0.37,
      "elitism_rate": 0.11,
      "mutation_rate": 0.01,
      "lns_rate": 0.48,
      "lns_num_to_remove_percent": 0.24,
      "tournament_size_percent": 0.14,
      "crossover": "iox",
      "mutation": "swap",
    },
    "quality": {
      "population_size_percent": 0.51,
      "generations_percent": 0.86,
      "elitism_rate": 0.32,
      "mutation_rate": 0.04,
      "lns_rate": 0.48,
      "lns_num_to_remove_percent": 0.24,
      "tournament_size_percent": 0.14,
      "crossover": "iox",
      "mutation": "inversion",
    },
  }
  ```,
  caption: [
    Hyperparameters of the three presets offered by default by the program. The `_percent` suffix indicates that the value is a percentage of the total number of cities.
  ],
) <presets>

Even if the problem is not exactly the same, we can now address some of the comments in @tsp-comments. Focusing on the `balanced` preset, keeping the number of elites low seems to provide a good balance between exploration and exploitation. Mutation rate should also be kept low but using inversion instead of swaps doesn't provide any advantage and if we look at the Pareto front we can see either one with around the same frequency. Between PMX, IOX and OX, IOX dominates the Pareto front (and could also contribute to the reason why inversions don't help much). Tournament size has been chosen to be relatively high and consistent for all the presets. LNS is heavily prioritized in the higher quality presets but it's also very costly, so for the `fast` preset it's almost completely ignored.

== Evaluating the Solution

We can now finally evaluate the cost of the solution found by our algorithm compared to the greedy baseline. We run both algorithms on a set of 81 problems with a combination of city sizes $N in [100, 300, 1000]$, $alpha in [0.05, 1.0, 10.0]$, $beta in [0.2, 1.0, 3.0]$, and density $rho in [0.2, 0.5, 0.9]$. For $beta = 1$ we notice that the greedy approach is already close to optimal. We see some slight improvements where $alpha$ is also very low and some paths can benefit from multiple stops. For $beta = 0.2$ some significant improvements can be seen but those pale in comparison to the results for $beta = 3.0$, where the "remove bulk gold" optimization really shines. For example, for one of the problems with $beta = 3.0$, the optimal load was computed to be $approx 0.91$. This result means that for each node, the algorithm that "removes the bulk gold" is going to take a small trip to city $i$ carrying just $0.91$ of gold: $(0, 0) -> (i, 0.91) -> (0, 0.91) -> ...$

We also notice that the hyperparameters of our choice generalize well to larger problems, where we don't see a drop in performance compared to smaller ones.

#info(
  accent-color: okabe.at(0),
)[All the results presented here have been computed using the `balanced` preset.]

#align(center, benchmark-table(json("benches/out.json")))

#bibliography("works.yaml")
