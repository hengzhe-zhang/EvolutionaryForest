import numpy as np


def compute_mixup_ratio(
    kernel_matrix: np.ndarray, indices_a: np.ndarray, indices_b: np.ndarray
) -> np.ndarray:
    if not (0 <= kernel_matrix.min() and kernel_matrix.max() <= 1):
        raise ValueError("Kernel matrix values must be between 0 and 1.")

    ratio = 1 - 0.5 * kernel_matrix[indices_a, indices_b]
    return ratio
