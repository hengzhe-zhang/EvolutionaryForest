import math

import numpy as np
from deap.tools import HallOfFame, selBest
from sklearn.decomposition import KernelPCA

from evolutionary_forest.component.archive_operators.test_utils import (
    generate_random_individuals,
)


class GridMAPElites(HallOfFame):
    def __init__(self, k, map_archive_candidate_size=100, **kwargs):
        super().__init__(k)
        self.map_archive_candidate_size = map_archive_candidate_size
        self.grid_size = math.ceil(np.sqrt(k))
        self.elites = np.empty((self.grid_size, self.grid_size), dtype=object)

    def update(self, population):
        individuals = selBest(
            population,
            self.map_archive_candidate_size,
        ) + list(self.items)
        # Perform Kernel PCA to reduce GP outputs to two dimensions
        data = np.array([ind.predicted_values for ind in individuals])
        pca = KernelPCA(n_components=2, kernel="cosine")
        reduced_data = pca.fit_transform(data)

        # Determine the bounds for grid discretization
        min_vals = np.min(reduced_data, axis=0)
        max_vals = np.max(reduced_data, axis=0)
        grid_width = (max_vals - min_vals) / self.grid_size

        # Assign individuals to grid cells and find elites
        for ind, coords in zip(population, reduced_data):
            grid_x = int((coords[0] - min_vals[0]) / grid_width[0])
            grid_y = int((coords[1] - min_vals[1]) / grid_width[1])
            grid_x = min(grid_x, self.grid_size - 1)  # Avoid edge case
            grid_y = min(grid_y, self.grid_size - 1)  # Avoid edge case

            current_elite = self.elites[grid_x, grid_y]
            if current_elite is None or ind.fitness > current_elite.fitness:
                self.elites[grid_x, grid_y] = ind

        self.clear()
        for elite in sorted(
            self.get_elites(), key=lambda x: x.fitness.wvalues, reverse=True
        )[: self.maxsize]:
            self.insert(elite)

    def get_elites(self):
        # Flatten the grid and filter out None values
        return [ind for ind in self.elites.flatten() if ind is not None]


if __name__ == "__main__":
    hof = GridMAPElites(k=5)  # k should be a perfect square for a square grid
    population, y_target = generate_random_individuals()

    hof.update(population)
    print("Selected Individuals:")
    for ind in hof:
        print(ind.predicted_values)
