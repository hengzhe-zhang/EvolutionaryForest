import numpy as np


def get_leaf_size_of_dt(leaf_size, y):
    if leaf_size == "Adaptive":
        if len(np.unique(y)) < 20:
            leaf = 20
        else:
            leaf = 5
    else:
        leaf = int(leaf_size)
    return leaf
