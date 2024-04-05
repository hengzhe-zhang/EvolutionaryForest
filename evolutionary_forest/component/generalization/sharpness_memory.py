import numpy as np
from cachetools import LRUCache
from deap.gp import PrimitiveTree

from evolutionary_forest.component.primitive_functions import tree_to_tuple


class TreeLRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = LRUCache(capacity)
        self.noise_evaluation_cache = True

        self.cache_hits = 0
        self.cache_misses = 0

    def query(self, key: PrimitiveTree, random_noise, random_seed):
        if self.noise_evaluation_cache and random_noise <= 0:
            return None
        key = tree_to_tuple(key) + (random_noise, random_seed)
        result = self.cache.get(key)
        verbose = False
        if verbose:
            if result is not None:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            if (self.cache_misses + self.cache_hits) % 1000 == 0:
                print(
                    f"Cache hits: {self.cache_hits}, Cache misses: {self.cache_misses}",
                    "Cache miss ratio: ",
                    self.get_cache_miss_ratio(),
                )
        return result

    def get_cache_miss_ratio(self):
        total_queries = self.cache_hits + self.cache_misses
        if total_queries == 0:
            return 0
        return self.cache_misses / total_queries

    def store(
        self, key: PrimitiveTree, value: np.ndarray, random_noise, random_seed
    ) -> None:
        if self.noise_evaluation_cache and random_noise <= 0:
            return
        key = tree_to_tuple(key) + (random_noise, random_seed)
        self.cache[key] = value
