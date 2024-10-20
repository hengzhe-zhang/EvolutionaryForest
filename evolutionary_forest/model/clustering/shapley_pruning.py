import itertools
import numpy as np
from sklearn.metrics import mean_squared_error


def ensemble_predict_regression(predictions):
    """
    Perform ensemble predictions by averaging predictions for regression tasks.
    Args:
        predictions (array): An array of shape (n_models, n_samples) containing predictions of each model.
    Returns:
        array: Final ensemble predictions using averaging.
    """
    # Average predictions for regression
    averaged_predictions = np.mean(predictions, axis=0)

    return averaged_predictions


def calculate_shapley_values_for_regression(
    predictions, y, performance_function=mean_squared_error
):
    """
    Calculate Shapley values for each model using their predictions for regression tasks.
    Args:
        predictions (array): A matrix of predictions from each model (n_models, n_samples).
        y (array): True labels.
        performance_function (callable): Function to calculate performance metric (e.g., mean_squared_error).
    Returns:
        shapley_values (dict): Shapley values for each model.
    """
    n_models = predictions.shape[0]
    model_indices = list(range(n_models))
    shapley_values = {i: 0 for i in model_indices}

    # Calculate performance (MSE, MAE, or other) of the full ensemble
    full_ensemble_performance = performance_function(
        y, ensemble_predict_regression(predictions)
    )

    # Loop over all models to calculate their marginal contribution
    for i in model_indices:
        for subset in itertools.combinations(
            [j for j in model_indices if j != i], len(model_indices) - 1
        ):
            subset = list(subset)

            # Predictions of the subset without model i
            subset_predictions = predictions[subset, :]
            performance_without_i = performance_function(
                y, ensemble_predict_regression(subset_predictions)
            )

            # Performance of the subset with model i included
            subset_with_i_predictions = np.vstack(
                [subset_predictions, predictions[i, :]]
            )
            performance_with_i = performance_function(
                y, ensemble_predict_regression(subset_with_i_predictions)
            )

            # Marginal contribution of model i
            marginal_contribution = performance_without_i - performance_with_i
            shapley_values[i] += marginal_contribution / (n_models * 1.0)

    return shapley_values


def prune_models_based_on_shapley_for_regression(predictions, y, threshold=0):
    """
    Prune models based on their Shapley values calculated from their predictions for regression tasks.
    Args:
        predictions (array): A matrix of predictions from each model (n_models, n_samples).
        y (array): True labels.
        threshold (float): Minimum Shapley value for a model to be retained.
    Returns:
        pruned_predictions (array): Predictions from the pruned models.
        shapley_values (dict): Shapley values for each model.
    """
    shapley_values = calculate_shapley_values_for_regression(predictions, y)

    # Prune models with Shapley values below the threshold
    pruned_indices = [
        i for i, shapley_value in shapley_values.items() if shapley_value > threshold
    ]

    return pruned_indices
