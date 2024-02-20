from typing import List

import numpy as np
from scipy.special import softmax

from evolutionary_forest.multigene_gp import MultipleGeneGP


def calculate_instance_weights(target, population: List[MultipleGeneGP], mode="Easy"):
    """
    Calculate weights for each data point based on the loss of a population of GP individuals.

    Parameters:
    - data: An iterable of data points.
    - population: An iterable of GP individuals in the population.

    Returns:
    - weights: A numpy array of weights for each data point.
    """
    criterion, mode = mode.split("-")
    # Initialize a matrix to hold the losses for each individual on each data point
    length = len(population[0].predicted_values)
    losses = np.zeros((len(population), length))

    # Evaluate each individual on each data point and record the loss
    for i, individual in enumerate(population):
        loss = (individual.predicted_values - target) ** 2
        losses[i] = loss

    # Aggregate losses for each data point across the population
    if criterion == "Min":
        min_losses = np.min(losses, axis=0)
    elif criterion == "Median":
        min_losses = np.median(losses, axis=0)
    else:
        raise Exception
    epsilon = 1e-10

    if mode == "Hard":
        weights = min_losses / (np.sum(min_losses) + epsilon)
    elif mode == "Easy":
        # Invert the minimum losses to reflect difficulty (add a small constant to avoid division by zero)
        inverted_losses = 1 / (min_losses + epsilon)

        # Normalize the inverted losses to get the final weights
        weights = inverted_losses / np.sum(inverted_losses)
    elif mode == "HardSoftmax":
        weights = softmax(min_losses)
    elif mode == "EasySoftmax":
        weights = softmax(-min_losses)
    else:
        raise ValueError("Mode should be either 'Easy' or 'Hard'")
    # weights = min_losses / (np.sum(min_losses) + epsilon)

    return weights
