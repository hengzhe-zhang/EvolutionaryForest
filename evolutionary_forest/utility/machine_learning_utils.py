from typing import List

import numpy as np
from sklearn.metrics import r2_score

from evolutionary_forest.multigene_gp import MultipleGeneGP

"""
1. Boostrap: Simple re-sampling loss
2. OOB Error: Has some hold-out (out-of-bag) data
"""


def bootstrap_fitness(Yp, Y_true):
    num = len(Y_true)
    sum = []
    for i in range(20):
        index = np.random.randint(0, num, num)
        sum.append(np.mean((Yp[index] - Y_true[index]) ** 2))
    return np.mean(sum)


def oob_error(pop: List[MultipleGeneGP], y):
    # Create array to store predictions for each individual in the population
    prediction = np.full((len(pop), len(y)), np.nan, dtype=np.float)
    # Iterate through individuals in the population
    for i, x in enumerate(pop):
        # Get indices of out-of-bag samples
        index = x.out_of_bag
        # Store individual's out-of-bag predictions in prediction array
        prediction[i][index] = x.oob_prediction
    # Calculate mean label value across all predictions (ignoring NaN values)
    label = np.nanmean(prediction, axis=0)
    label = np.nan_to_num(label)
    # Calculate R^2 score using mean label values and actual labels
    accuracy = r2_score(y, label)
    return accuracy
