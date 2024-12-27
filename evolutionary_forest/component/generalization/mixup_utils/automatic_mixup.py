import numpy as np


def compute_mixup_ratio(
    kernel_matrix: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    inverse=False,
) -> np.ndarray:
    if not (0 <= kernel_matrix.min() and kernel_matrix.max() <= 1):
        raise ValueError("Kernel matrix values must be between 0 and 1.")

    if inverse:
        ratio = 0.5 + 0.5 * kernel_matrix[indices_a, indices_b]
    else:
        ratio = 1 - 0.5 * kernel_matrix[indices_a, indices_b]
    return ratio


if __name__ == "__main__":
    # Define a sample kernel matrix with values between 0 and 1
    kernel_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # Define index arrays
    indices_a = np.array([0, 1, 2])
    indices_b = np.array([2, 1, 0])

    # Compute mixup ratio without inverse
    non_inverse_ratio = compute_mixup_ratio(
        kernel_matrix=kernel_matrix,
        indices_a=indices_a,
        indices_b=indices_b,
        inverse=False,
    )
    print("Mixup Ratio (Non-Inverse):")
    print(non_inverse_ratio)

    # Compute mixup ratio with inverse
    inverse_ratio = compute_mixup_ratio(
        kernel_matrix=kernel_matrix,
        indices_a=indices_a,
        indices_b=indices_b,
        inverse=True,
    )
    print("Mixup Ratio (Inverse):")
    print(inverse_ratio)
