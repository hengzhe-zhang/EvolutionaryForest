import math
import random
from functools import partial
from typing import Union, Callable

import numpy as np
from deap import base
from deap.tools import cxUniform, mutUniformInt
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from evolutionary_forest.multigene_gp import result_calculation


class SpacePartition():
    def __init__(self):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.base_learner: str = None
        self.dynamic_partition: str = None
        self.current_gen: int = None
        self.n_gen: int = None
        self.ps_tree_ratio: float = None
        self.interleaving_period: int = None
        self.partition_model: BaseEstimator = None

    def partition_number_calculation(self):
        """
        Dynamically calculate the partition number based on training data
        :return:
        """
        if isinstance(self.partition_number, str) and 'Sample' in self.partition_number:
            min_samples_leaf = int(self.partition_number.split('-')[1])
            partition_number = int(self.partition_number.split('-')[2])
            partition_number = int(np.clip(math.floor(self.X.shape[0] / min_samples_leaf), 1, partition_number))
            self.partition_number = partition_number
            return partition_number
        else:
            assert isinstance(self.partition_number, int)
            return self.partition_number

    def partition_scheme_initialization(self) -> Union[np.ndarray, Callable, None]:
        """
        Initialize a space partition scheme
        """
        if self.base_learner == 'Soft-PLTree' or 'Fast-Soft-PLTree' in self.base_learner:
            if self.dynamic_partition == 'Self-Adaptive' or self.dynamic_partition is None:
                # Initialization with decision tree
                if isinstance(self.partition_number, str) and 'Sample' in self.partition_number:
                    min_samples_leaf = int(self.partition_number.split('-')[1])
                    partition_number = int(self.partition_number.split('-')[2])
                    dt = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf,
                                               max_leaf_nodes=partition_number)
                    dt.fit(self.X, self.y)
                    self.partition_number = dt.tree_.n_leaves
                else:
                    dt = DecisionTreeRegressor(max_leaf_nodes=self.partition_number)
                    dt.fit(self.X, self.y)
                self.partition_model = dt
                partition_scheme = LabelEncoder().fit_transform(dt.apply(self.X))
                assert np.all(partition_scheme < self.partition_number)
            elif self.dynamic_partition == 'GA':
                self.partition_number = self.partition_number_calculation()

                def partition_scheme():
                    return np.random.randint(0, self.partition_number, len(self.X))
            else:
                raise Exception
        else:
            partition_scheme = None
        return partition_scheme

    def partition_scheme_updating(self):
        # Full evaluation after some iterations
        if (self.interleaving_period is None or self.interleaving_period == 0) \
            and self.current_gen > self.n_gen * (1 - self.ps_tree_ratio):
            if self.base_learner.startswith('Fast-'):
                self.base_learner = self.base_learner.replace('Fast-', '')

        if self.interleaving_period > 0:
            # Interleaving changing
            if self.current_gen % self.interleaving_period == 0:
                if not self.base_learner.startswith('Fast'):
                    self.base_learner = 'Fast-' + self.base_learner
            else:
                self.base_learner = self.base_learner.replace('Fast-', '')
            if self.verbose:
                print(self.current_gen, self.base_learner, self.interleaving_period)


def partition_scheme_varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
    return offspring


def partition_scheme_toolbox(partition_num, cxpb=0.1, indpb=0.1):
    toolbox = base.Toolbox()
    # toolbox.register("mate", cxTwoPoint)
    toolbox.register("mate", cxUniform, indpb=cxpb)
    toolbox.register("mutate", partial(mutUniformInt, low=0, up=partition_num - 1, indpb=indpb))
    return toolbox
