import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class Strategy:
    @staticmethod
    def do(self):
        pass


class Clearing(Strategy):
    # a niching strategy for population diversity

    def __init__(self, clearing_cluster_size=0, **kwargs):
        self.clearing_cluster_size = clearing_cluster_size

    def do(self, population):
        # Clearing strategy: only the best one in each cluster will survive
        if self.clearing_cluster_size > 1:
            # Collect case values and sum of fitness values for all individuals in the population
            all_case_values = np.array([p.case_values for p in population])
            sum_fitness = np.array([p.fitness.wvalues[0] for p in population])
            key = np.arange(0, len(population))

            # Use K-means clustering to assign labels to each individual based on their case values
            label = KMeans(len(population) // self.clearing_cluster_size).fit_predict(
                all_case_values
            )
            df = pd.DataFrame(
                np.array([key, label, sum_fitness]).T,
                columns=["key", "label", "fitness"],
            )

            # Sort individuals in descending order based on fitness and keep only the best in each cluster
            df = df.sort_values("fitness", ascending=False).drop_duplicates(["label"])

            # Update the population by selecting the best individuals
            population = [population[int(k)] for k in list(df["key"])]
        return population
