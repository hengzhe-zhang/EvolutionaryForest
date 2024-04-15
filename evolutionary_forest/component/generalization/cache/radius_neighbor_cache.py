from sklearn.neighbors import RadiusNeighborsRegressor
import numpy as np

from evolutionary_forest.component.generalization.cache.sharpness_memory import (
    TreeLRUCache,
)


def tree_to_array(tree):
    return tree


class LearningTreeCache:
    def __init__(self, radius=0.001, capacity=1000):
        self.models = {}
        self.data = {}
        self.labels = {}
        self.capacity = capacity
        self.radius = radius
        self.semantics_cache = TreeLRUCache(
            capacity=int(1e10), noise_evaluation_cache=False
        )

        self.cache_hits = 0
        self.cache_misses = 0

    def query(self, key, random_noise, random_seed):
        if random_noise <= 0 or random_seed not in self.models:
            return None
        assert random_noise > 0

        query_point = self.semantics_cache.query(key, 0, 0)
        model = self.models.get(random_seed)

        result = model.predict([query_point])[0]
        if np.any(np.isnan(result)):
            # Handle the case where no neighbors are found
            result = None
            self.cache_misses += 1
        else:
            self.cache_hits += 1

        # self._log_cache_performance()
        return result

    def store(self, key, value, random_noise, random_seed):
        if random_noise <= 0:
            assert random_noise == 0
            self.semantics_cache.store(key, value, 0, 0)
            return

        new_data_point = self.semantics_cache.query(key, 0, 0)
        if random_seed not in self.data:
            self.data[random_seed] = []
            self.labels[random_seed] = []

        self.data[random_seed].append(new_data_point)
        self.labels[random_seed].append(value)

        # Maintain capacity limits
        if len(self.data[random_seed]) > self.capacity:
            self.data[random_seed].pop(0)
            self.labels[random_seed].pop(0)

    def retrain(self):
        self.semantics_cache.cache.clear()
        for random_seed in self.data:
            self._retrain_model(random_seed)

    def _retrain_model(self, random_seed):
        if random_seed not in self.models:
            self.models[random_seed] = RadiusNeighborsRegressor(radius=self.radius)

        self.models[random_seed].fit(
            np.array(self.data[random_seed]), np.array(self.labels[random_seed])
        )

    def _log_cache_performance(self):
        if (self.cache_misses + self.cache_hits) % 1000 == 0:
            print(
                f"Cache hits: {self.cache_hits}, Cache misses: {self.cache_misses}, "
                f"Cache miss ratio: {self.get_cache_miss_ratio()}"
            )
            print("Semantics cache size: ", len(self.semantics_cache.cache))

    def get_cache_miss_ratio(self):
        total_queries = self.cache_hits + self.cache_misses
        return self.cache_misses / total_queries if total_queries > 0 else 0
