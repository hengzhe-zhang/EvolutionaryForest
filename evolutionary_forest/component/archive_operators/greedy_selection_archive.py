import numpy as np
from deap.tools import HallOfFame

from evolutionary_forest.component.archive_operators.test_utils import (
    generate_random_individuals,
)


class GreedyHallOfFame(HallOfFame):
    def __init__(self, maxsize, y, **kwargs):
        self.y = y
        super().__init__(maxsize)

    def update(self, population):
        total_population = population + list(self.items)

        # Greedily select individuals to minimize the residual to self.y
        new_hof = []
        current_residual_vector = np.copy(self.y)

        for _ in range(self.maxsize):
            residuals = [
                self.calculate_residual(ind, current_residual_vector)
                for ind in total_population
            ]
            min_residual_index = np.argmin(residuals)
            candidate = total_population[min_residual_index]

            new_residual = np.mean(
                (current_residual_vector - candidate.predicted_values) ** 2
            )
            no_modification = np.mean(current_residual_vector**2)
            # print(f"New residual: {new_residual}, No modification: {no_modification}")

            # Only add this candidate if it improves the residual
            if len(new_hof) == 0 or new_residual < no_modification:
                new_hof.append(candidate)
                current_residual_vector -= candidate.predicted_values
            else:
                break

        # Clear the old hall of fame and update with new individuals
        self.clear()
        for individual in new_hof:
            self.insert(individual)

    def calculate_residual(self, individual, residual):
        loss = np.mean((residual - individual.predicted_values) ** 2)
        return loss


if __name__ == "__main__":
    population, y_target = generate_random_individuals()

    hof = GreedyHallOfFame(maxsize=3, y=y_target)
    hof.update(population)

    print("Selected Individuals:")
    for ind in hof:
        print(ind.predicted_values)
