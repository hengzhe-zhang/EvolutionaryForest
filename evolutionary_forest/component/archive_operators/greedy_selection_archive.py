import numpy as np
from deap.tools import HallOfFame

from evolutionary_forest.component.archive_operators.test_utils import (
    generate_random_individuals,
)


class GreedyHallOfFame(HallOfFame):
    def __init__(self, maxsize, y, nuber_of_initial_individuals=1, **kwargs):
        self.y = y
        self.nuber_of_initial_individuals = nuber_of_initial_individuals
        super().__init__(maxsize)

    def update(self, population):
        total_population = population + list(self.items)

        # select number of initial individuals based on top fitness
        total_population = list(
            sorted(total_population, key=lambda ind: ind.fitness.wvalues, reverse=True)
        )
        selected_population = total_population[: self.nuber_of_initial_individuals]
        assert len(selected_population) == self.nuber_of_initial_individuals

        # Greedily select individuals to minimize the residual to self.y
        new_hof = list(selected_population)
        # Initialize the residual vector
        current_prediction = np.mean([ind.predicted_values for ind in new_hof], axis=0)

        for _ in range(self.maxsize):
            residuals = [
                self.calculate_residual(ind, current_prediction, len(new_hof))
                for ind in total_population
            ]
            min_residual_index = np.argmin(residuals)
            candidate = total_population[min_residual_index]

            # Check if the candidate improves the prediction
            new_residual = residuals[min_residual_index]
            # Calculate the residual if we don't add the candidate
            no_modification = np.mean((self.y - current_prediction) ** 2)
            print(f"New residual: {new_residual}, No modification: {no_modification}")

            # Only add this candidate if it improves the residual
            if len(new_hof) == 0 or new_residual < no_modification:
                new_hof.append(candidate)
                # Update the residual vector
                current_prediction = np.mean(
                    [ind.predicted_values for ind in new_hof], axis=0
                )
                # self.residual_correct_check(new_hof, self.y, new_residual)
            else:
                break

        # Clear the old hall of fame and update with new individuals
        self.clear()
        for individual in new_hof:
            self.insert(individual)

    def residual_correct_check(self, individuals, target, reference_residual):
        prediction = np.mean([ind.predicted_values for ind in individuals], axis=0)
        residual = np.mean((target - prediction) ** 2)
        assert np.allclose(
            residual, reference_residual
        ), f"Residuals do not match: {residual} != {reference_residual}"

    def calculate_residual(
        self, individual, current_prediction, number_of_current_items
    ):
        # Calculate the ensemble prediction
        ensemble_prediction = (
            number_of_current_items * current_prediction + individual.predicted_values
        ) / (number_of_current_items + 1)
        # Calculate the loss
        loss = np.mean((ensemble_prediction - self.y) ** 2)
        return loss


if __name__ == "__main__":
    population, y_target = generate_random_individuals()

    hof = GreedyHallOfFame(maxsize=3, y=y_target, nuber_of_initial_individuals=5)
    hof.update(population)

    print("Selected Individuals:")
    for ind in hof:
        print(ind.predicted_values)
