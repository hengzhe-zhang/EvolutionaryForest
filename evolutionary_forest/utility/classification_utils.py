from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import os
import seaborn as sns


def calculate_cross_entropy(target_label, y_pred):
    one_hot_targets = OneHotEncoder(sparse_output=False).fit_transform(
        target_label.reshape(-1, 1)
    )
    eps = 1e-8
    # Cross entropy
    cross_entropy = -1 * np.sum(
        one_hot_targets * np.log(np.clip(y_pred, eps, 1 - eps)), axis=1
    )
    return cross_entropy
