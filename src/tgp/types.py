import numpy as np
import numpy.typing as npt
from numba import njit

type SolutionType = list[tuple[int, float]]  # List of (city, gold collected at city)
type DistMatrix = npt.NDArray[np.float32]


@njit
def DistMatrix(data: dict[int, dict[int, float]]) -> npt.NDArray[np.float32]:  # noqa: F811
    """Convert nested distance dict to matrix."""
    size = len(data)
    matrix = np.zeros((size, size), dtype=np.float32)
    for i, row in data.items():
        for j, value in row.items():
            matrix[i, j] = value
    return matrix
