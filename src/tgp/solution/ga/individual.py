import numpy as np
from numba import njit, types
from numba.experimental import jitclass
from numba.typed import List

spec = [
    ("tour", types.ListType(types.int32)),
    ("fitness", types.float64),
    ("predecessors", types.int32[:]),
]


@jitclass(spec)
class Individual:
    # Can't define default values for fitness and predecessors fields due to
    # https://github.com/numba/numba/issues/4820
    def __init__(self, tour, fitness, predecessors):
        self.tour = tour
        self.fitness = fitness
        self.predecessors = predecessors

    def __lt__(self, other):
        return self.fitness < other.fitness


@njit
def create_individual_list(size: int = 0) -> list[Individual]:
    """
    Create a typed list for Individual objects.

    Workaround for https://github.com/numba/numba/issues/8734

    Args:
        size: If 0, creates empty list. Otherwise pre-allocates with dummy objects.

    Returns:
        Numba compatible typed list of Individual objects.
    """

    dummy_tour = List(np.array([0], dtype=np.int32))
    dummy_preds = np.zeros(1, dtype=np.int32)
    dummy_ind = Individual(dummy_tour, 0.0, dummy_preds)

    if size == 0:
        lst = List([dummy_ind])
        lst.pop(0)
    else:
        lst = List([dummy_ind for _ in range(size)])

    return lst
