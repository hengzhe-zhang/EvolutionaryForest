from abc import abstractmethod

import numpy as np


class Fitness():
    @abstractmethod
    def fitness_value(self, individual, estimators, Y, y_pred):
        pass


class MTLR2(Fitness):
    def __init__(self, number_of_tasks):
        self.number_of_tasks = number_of_tasks

    def fitness_value(self, individual, estimators, Y, y_pred):
        number_of_tasks = self.number_of_tasks
        Y_partitions = np.array_split(Y, number_of_tasks)
        y_pred_partitions = np.array_split(y_pred, number_of_tasks)

        r_squared_values = []

        for i in range(number_of_tasks):
            mean_y = np.mean(Y_partitions[i])
            tss = np.sum((Y_partitions[i] - mean_y) ** 2)
            rss = np.sum((Y_partitions[i] - y_pred_partitions[i]) ** 2)
            r_squared = 1 - (rss / tss)
            r_squared_values.append(r_squared)
        return -1*np.mean(r_squared_values),
