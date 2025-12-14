import numpy as np


def dropout_features(x: np.ndarray, dropout_rate: float) -> np.ndarray:
    """
    Applies dropout to the input array `x` by setting a fraction of its elements to 0.

    Parameters:
    - x: Input array.
    - dropout_rate: Fraction of the input units to drop, must be between 0 and 1.

    Returns:
    - x_dropped: Array after applying dropout.
    """
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be between 0 and 1.")

    # Generate a mask where some elements will be set to 0 based on the dropout_rate
    mask = np.random.rand(*x.shape) >= dropout_rate

    # Apply the mask and scale the remaining elements to keep the same expected value
    x_dropped = (x * mask) / (1 - dropout_rate)

    return x_dropped
