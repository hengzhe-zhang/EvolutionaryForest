import numpy as np


def feature_importance_process(coefficient):
    if sum(coefficient) == 0:
        coefficient = np.ones_like(coefficient)
    coefficient = coefficient / np.sum(coefficient)
    coefficient = np.nan_to_num(coefficient, posinf=0, neginf=0)
    return coefficient
