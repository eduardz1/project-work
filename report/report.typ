#set document(title: "Traveling Goblin Problem", author: "Eduard Occhipinti")

#set text(font: "New Computer Modern", lang: "en", size: 11pt)

#set page(paper: "a4", margin: 3.5cm)

#show figure.caption: emph
// // Floating figures appear as `place` instead of `block` so we
// // need this workaround, see https://github.com/typst/typst/issues/6095
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

= Problem Definition

The problem consists in finding the optimal path for a goblin to steal all the gold from a set of nodes and bring it back to base. The goblin can carry any amount of gold and can take any amount of it from each node but it cannot leave any gold behind. The cost function to travel between each node is given by

$
  C = D + (D alpha W)^beta | alpha >= 0 and beta >= 0.
$ <cost>

= Solution Approach

== First Intuitions

=== Prin's Algorithm and Giant Tours <prin-sec>

My first intuition when approaching this problem was to consider similar problems, in doing so I identified this problem as a special case of the vehicle routing problem with capacity constraints (CVRP). A common approach to solve this kind of problem is to generate a "giant tour", which consists in taking a permutation of all the nodes in the graph, and then split it into smaller routes. A common approach to compute the split is done using Prin's algorithm as is detailed in #cite(<prin>, form: "prose"). The nice thing about this algorithm is that the split computed is optimal for the given permutation. Therefore, the optimization part can be simplified to a traditional traveling salesman problem (TSP). The algorithm has complexity $cal(O)(N^2)$ but can be simplified to $cal(O)(N)$ with the approach in #cite(<split>, form: "prose"). This approach, unfortunately, is not applicable to our non-linear cost function.

=== Fractional Nodes <frac-nodes>

Part of the problem specification is the ability to take any amount of gold from each node. To convert this continuous optimization problem into a discrete one, we can split each node into multiple "fractional nodes", each containing a fraction of the gold from the original node. To avoid having the path finding always end up traversing all the fractions of a node, we also have to make all the fractions fully connected with edges of zero distance. We can also add a fraction of zero gold to each node to allow the solver to skip nodes entirely if needed. This approach allows us to completely transparently optimize the giant tour approach without any knowledge of the internal working of our TSP solver.

A big problem of this approach is the fact that this splitting increases the size of the graph significantly. Being the TSP problem already $cal(O)(N!)$ in complexity, any increase in the number of nodes should be carefully considered.

== Analyzing the Cost Function

Starting from the assumption that we want to split each node into the minimum number of fractional nodes possible, we define the total cost $T$ as the sum of

+ The cost of the outbound trip from $0$ to $i$, which we'll call $D$.
+ The return cost using @cost, which comes out to $D + (D alpha W)^beta$.

$
  T = (D) + (D + (D alpha W)^beta) = 2D + (D alpha W)^beta.
$

We can notice that, for $beta < 1$, the cost of the load is "discounted", with $(W_1 + W_2)^beta < W_1^beta + W_2^beta$. This means that combining multiple loads can lead to a lower overall cost.

For $beta >= 1$, we can imagine having two separate scenarios:

1. Taking multiple single trips from base to node $i$ and back, each carrying an optimal load that we define as $L^*$.
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

The existance of this result has an important implication: for $beta > 1$ we can find an optimal packet size $L^*$, meaning that chaining two optimal packets is at best only equivalent to taking two separate trips with optimal packets. This means that we can completely exclude the "fractional nodes" approach detailed in @frac-nodes as the best strategy is to make single "greedy" trips with optimal packet size $L^*$ until we are left with a remainder smaller than $L^*$.

Empirically we see that for $beta = 2$, the optimal load is $approx 1$ for every node on our graph defined on a unit square with $alpha = 1$. Given that the amount of gold on each node is sampled from $[1, 1000)$, this means that the optimal strategy in this case consists in thousands of small trips.

#bibliography("works.yaml")
