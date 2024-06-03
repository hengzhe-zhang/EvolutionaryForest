from _operator import eq
from abc import abstractmethod

import numpy as np
from deap.tools import HallOfFame

from evolutionary_forest.model.attention_layer import AttentionMetaWrapper


class MetaLearner(HallOfFame):
    def __init__(self, base_hof: HallOfFame, maxsize):
        self.base_hof = base_hof
        super().__init__(maxsize)

    @abstractmethod
    def fit(self, models, X, y):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray[float]) -> np.ndarray[float]:
        pass

    def __iter__(self):
        return self.base_hof.__iter__()

    def __len__(self):
        return self.base_hof.__len__()

    def __getitem__(self, i):
        return self.base_hof.__getitem__(i)

    def __reversed__(self):
        return self.base_hof.__reversed__()

    def insert(self, item):
        self.base_hof.insert(item)

    def remove(self, index):
        self.base_hof.remove(index)

    def clear(self):
        self.base_hof.clear()

    def __str__(self):
        return self.base_hof.__str__()

    def update(self, population):
        self.base_hof.update(population)


class AttentionMetaLearner(
    MetaLearner,
):
    def __init__(self, base_hof: HallOfFame, maxsize, similar=eq):
        self.base_hof = base_hof

        super().__init__(maxsize, similar)

    def fit(self, models, X, y):
        attention_net_params = {
            "input_dim": X.shape[1],
            "model_output_dim": len(models),  # Number of base models
            "hidden_dim": 10,
            "output_dim": 1,
            "max_epochs": 500,
            "lr": 0.01,
        }
        self.attention_model = AttentionMetaWrapper(models, attention_net_params, True)
        self.attention_model.fit(X, y)

    def predict(self, x: np.ndarray[float]) -> np.ndarray[float]:
        return self.attention_model.predict(x)
