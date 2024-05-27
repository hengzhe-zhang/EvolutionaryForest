import numpy as np
import torch


def contains_tensor(result):
    """Check if the list contains any tensor."""
    return any(isinstance(yp, torch.Tensor) for yp in result)


def impute_values(result, data_length):
    """Impute values for non-array elements or adjust arrays of length 1."""
    for i in range(len(result)):
        yp = result[i]
        if not isinstance(yp, (np.ndarray, torch.Tensor)):
            yp = np.full(data_length, yp).astype(np.float32)
        elif (isinstance(yp, np.ndarray) and yp.size == 1) or (
            not isinstance(yp, np.ndarray) and len(yp) == 1
        ):
            yp = np.full(data_length, yp.item() if isinstance(yp, torch.Tensor) else yp)
        result[i] = yp
    return result


def handle_inf_nan(result, include_tensor):
    """Handle infinities and NaNs in numpy arrays or tensors."""
    if not include_tensor:
        result = np.array(
            [
                np.nan_to_num(yp, posinf=0, neginf=0)
                if isinstance(yp, np.ndarray)
                else yp
                for yp in result
            ]
        )
    else:
        result = [
            torch.nan_to_num(
                torch.from_numpy(yp) if isinstance(yp, np.ndarray) else yp,
                posinf=0,
                neginf=0,
            )
            for yp in result
        ]
    return result


def quick_fill(result: list, data: np.ndarray):
    """Impute data based on tensor presence and data length."""
    include_tensor = contains_tensor(result)
    data_length = len(data)

    result = impute_values(result, data_length)
    result = handle_inf_nan(result, include_tensor)

    return result


if __name__ == "__main__":
    # Example usage
    data = np.array([1, 2, 3])
    result = [np.array([np.nan]), torch.tensor([float("inf")]), 5]
    updated_result = quick_fill(result, data)
    print(updated_result)
