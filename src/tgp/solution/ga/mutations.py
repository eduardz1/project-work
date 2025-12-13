import numpy as np
from numba import njit


@njit
def swap_mutation(path: list[int]) -> None:
    """
    Swap in-place two cities in the path to create a mutation.

    Args:
        path (list[int]): The current path representing a TSP solution. (mut)
    """

    a, b = np.random.choice(len(path), size=2, replace=False)
    path[a], path[b] = path[b], path[a]


@njit
def inversion_mutation(path: list[int]) -> None:
    """
    Perform an inversion mutation on the path by reversing a segment.

    Args:
        path (list[int]): The current path representing a TSP solution. (mut)
    """

    a, b = sorted(np.random.choice(len(path), size=2, replace=False))
    path[a:b] = path[a:b][::-1]
