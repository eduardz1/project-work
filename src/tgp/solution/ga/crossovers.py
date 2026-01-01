import numpy as np
from numba import njit
from numba.typed import List

from tgp.solution.ga.individual import Individual


@njit
def iox(parent1: Individual, parent2: Individual) -> Individual:
    """
    Implements the Inver-over crossover operator.

    The algorithm works as follows:
    1. Select a random value from parent1.
    2. Find the position of this value in parent2.
    3. Find the position of the next value of parent2 from the first value in
        parent1.
    4. Reverse the segment in parent1 between these two positions in a circular
        manner such that the edge in parent2 is preserved and the ones in
        parent1 reversed.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: The child individual resulting from the crossover.
    """

    start1 = np.random.randint(len(parent1.tour))
    start2 = parent2.tour.index(parent1.tour[start1])
    next = parent2.tour[(start2 + 1) % len(parent2.tour)]
    end1 = parent1.tour.index(next)

    child = Individual(
        parent1.tour.copy(), 0.0, np.zeros(len(parent1.tour), dtype=np.int32)
    )

    if start1 < end1:
        segment = child.tour[start1 + 1 : end1 + 1]
        segment.reverse()
        child.tour[start1 + 1 : end1 + 1] = segment
    else:
        # Wrap-around case: reverse segment from start1+1 wrapping to end1
        segment = child.tour[start1 + 1 :]  # + child.tour[: end1 + 1]
        segment.extend(child.tour[: end1 + 1])
        segment.reverse()
        child.tour[start1 + 1 :] = segment[: len(child.tour) - start1 - 1]
        child.tour[: end1 + 1] = segment[len(child.tour) - start1 - 1 :]

    return child


@njit
def pmx(parent1: Individual, parent2: Individual) -> Individual:
    """
    Partially Mapped Crossover (PMX) between two parents.

    The algorithm works as follows:
    1. Select two crossover points randomly that define a segment.
    2. Copy the a segment from parent1 to the child.
    3. For each position outside the segment, fill in genes from parent2, making
        sure that no duplicates occur in the child.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: The child individual resulting from the crossover.
    """

    n = len(parent1.tour)
    start, end = sorted(np.random.choice(n, size=2, replace=False))

    seg1 = parent1.tour[start:end]
    seg2 = parent2.tour[start:end]
    child = List()
    for _ in range(n):
        child.append(np.int32(-1))
    child[start:end] = seg1

    # iterate outside the copied segment
    for i in list(range(0, start)) + list(range(end, n)):
        gene = parent2.tour[i]
        # iterate until we have a new gene in the child
        while gene in seg1:
            j = seg1.index(gene)
            gene = seg2[j]
        child[i] = gene

    return Individual(child, 0.0, np.zeros(n, dtype=np.int32))


@njit
def ox(parent1: Individual, parent2: Individual) -> Individual:
    """
    Order Crossover (OX) between two parents.

    The algorithm works as follows:
    1. Select two crossover points randomly that define a segment.
    2. Copy the segment from parent1 to the child.
    3. Fill in the remaining positions in the child with genes from parent2,
        preserving their order and skipping duplicates.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        Individual: The child individual resulting from the crossover.
    """

    n = len(parent1.tour)
    start, end = sorted(np.random.choice(n, size=2, replace=False))

    child = List()
    for _ in range(n):
        child.append(np.int32(-1))
    child[start:end] = parent1.tour[start:end]

    current_pos = end % n
    for i in range(n):
        gene = parent2.tour[(end + i) % n]
        if gene not in child:
            child[current_pos] = gene
            current_pos = (current_pos + 1) % n

    return Individual(child, 0.0, np.zeros(n, dtype=np.int32))
