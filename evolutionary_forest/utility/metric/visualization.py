import numpy as np


def average_loss(predictions_all, true_values):
    return np.mean((predictions_all - true_values) ** 2)


def calculate_ambiguity(predicted_values_list, true_values):
    # Calculate variance per sample
    variance_per_sample = np.var(predicted_values_list, axis=0)

    # Ensure the length of variance per sample matches the number of true values
    assert len(variance_per_sample) == true_values.shape[0]

    # Calculate the mean variance (ambiguity)
    ambiguity = np.mean(variance_per_sample)

    return ambiguity


if __name__ == "__main__":
    # Example predicted values from 3 models for 5 samples
    predicted_values_list = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.1, 2.1, 2.9, 3.9, 5.1]),
        np.array([0.9, 1.9, 3.1, 4.1, 4.9]),
    ]

    # Example true values
    true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Calculate ambiguity
    ambiguity_value = calculate_ambiguity(predicted_values_list, true_values)
    print(f"Ambiguity: {ambiguity_value}")
