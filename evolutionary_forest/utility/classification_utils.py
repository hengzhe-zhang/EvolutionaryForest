import numpy as np


def calculate_zero_one_loss(target_labels: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the 0-1 loss for a batch of predictions.

    :param target_labels: 1D array of integers, where each integer is the correct class index for each sample.
    :param y_pred: 2D array of shape (n_samples, n_classes) containing the predicted probabilities for each class.
    :return: The 0-1 loss, which is the fraction of incorrect predictions.
    """
    predicted_labels = np.argmax(
        y_pred, axis=1
    )  # Get the predicted class for each sample
    incorrect_predictions = (
        predicted_labels != target_labels
    )  # Compare with true labels
    return incorrect_predictions


def calculate_cross_entropy(target_labels: np.ndarray, y_pred: np.ndarray):
    """
    Calculate the cross entropy loss for a batch of predictions.

    :param target_labels: 1D array of integers, where each integer is the correct class index for each sample.
    :param y_pred: 2D array of shape (n_samples, n_classes) containing the predicted probabilities for each class.
    :return: Cross entropy loss for each sample.
    """
    eps = 1e-8
    # Correct indexing for each sample's predicted probability of the true class
    correct_class_probs = y_pred[np.arange(len(target_labels)), target_labels]
    # Cross entropy calculation
    cross_entropy = -np.log(np.clip(correct_class_probs, eps, 1))
    return cross_entropy


if __name__ == "__main__":
    # Example predicted probabilities for 5 samples and 3 classes
    y_pred = np.array(
        [
            [0.1, 0.6, 0.3],  # Sample 1
            [0.8, 0.1, 0.1],  # Sample 2
            [0.3, 0.4, 0.3],  # Sample 3
            [0.2, 0.2, 0.6],  # Sample 4
            [0.1, 0.7, 0.2],  # Sample 5
        ]
    )

    # Example true labels for the 5 samples
    target_labels = np.array([1, 0, 2, 2, 1])  # Correct classes for each sample

    # Calculate the cross-entropy loss
    cross_entropy_losses = calculate_cross_entropy(target_labels, y_pred)

    # Print the cross-entropy loss for each sample
    print("Cross-Entropy Losses:", cross_entropy_losses)
