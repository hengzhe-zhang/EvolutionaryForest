def perform_extrapolation(
    algorithm_X, algorithm_y, indices_a, indices_b, data, label, ratio
):
    # Simulate extrapolation where ratio could exceed 1
    temp_ratio = (1 + ratio).reshape(-1, 1)
    data_extrapolation = temp_ratio * algorithm_X[indices_a] + (
        (1 - temp_ratio) * algorithm_X[indices_b]
    )
    label_extrapolation = temp_ratio.flatten() * algorithm_y[indices_a] + (
        (1 - temp_ratio).flatten() * algorithm_y[indices_b]
    )

    # Identify out-of-distribution samples
    replace_index = (label_extrapolation > algorithm_y.max()) | (
        label_extrapolation < algorithm_y.min()
    )

    # Update the ratio, data, and label for out-of-distribution samples
    temp_ratio = temp_ratio.flatten()
    ratio[replace_index] = temp_ratio[replace_index]
    data[replace_index] = data_extrapolation[replace_index]
    label[replace_index] = label_extrapolation[replace_index]

    return data, label, ratio
