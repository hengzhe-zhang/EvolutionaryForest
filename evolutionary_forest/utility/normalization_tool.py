import numpy as np


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # Return the zero vector as is
    normalized_v = v / norm
    # Find the first non-zero element
    for i in range(len(normalized_v)):
        if normalized_v[i] != 0:
            # If the first non-zero element is negative, invert the vector
            if normalized_v[i] < 0:
                normalized_v = -normalized_v
            break
    return normalized_v


if __name__ == "__main__":
    # Example usage
    residual_1 = np.array([-1, -1])
    residual_2 = np.array([1, 1])
    residual_3 = np.array([-1, 1])

    normalized_residual_1 = normalize_vector(residual_1)
    normalized_residual_2 = normalize_vector(residual_2)
    normalized_residual_3 = normalize_vector(residual_3)

    print(normalized_residual_1)  # Output should be [1/sqrt(2), 1/sqrt(2)]
    print(normalized_residual_2)  # Output should be [1/sqrt(2), 1/sqrt(2)]
    print(normalized_residual_3)  # Output should be [-1/sqrt(2), 1/sqrt(2)]
