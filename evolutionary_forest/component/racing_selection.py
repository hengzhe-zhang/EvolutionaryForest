from typing import List

from deap import gp
from deap.gp import Primitive, PrimitiveTree
from scipy import stats

from evolutionary_forest.multigene_gp import MultipleGeneGP


class RacingFunctionSelector:
    def __init__(self, pset):
        self.function_fitness_lists = {}
        self.best_individuals_fitness_list = []
        self.MAX_SIZE = 100
        self.pset = pset

    def update_function_fitness_list(self, individual: MultipleGeneGP):
        """
        Updates the fitness list of functions in the individual's GP tree.
        """
        fitness_value = individual.fitness.wvalues[0]
        for tree in individual.gene:
            tree: PrimitiveTree
            for function in tree:
                # If the function is a primitiv, then add its fitness value
                if isinstance(function, Primitive):
                    if function not in self.function_fitness_lists:
                        self.function_fitness_lists[function] = []
                    self.function_fitness_lists[function].append(fitness_value)

                    # Ensure the list does not exceed maximum size by removing the worst fitness value
                    if len(self.function_fitness_lists[function]) > self.MAX_SIZE:
                        self.function_fitness_lists[function].remove(
                            min(self.function_fitness_lists[function])
                        )

    def update_best_individuals_list(self, individual: MultipleGeneGP):
        """
        Updates the fitness list of the best individuals.
        """
        fitness_value = individual.fitness.wvalues[0]
        self.best_individuals_fitness_list.append(fitness_value)
        self.best_individuals_fitness_list.sort()

        # Ensure the list does not exceed the maximum size by removing the worst fitness value
        if len(self.best_individuals_fitness_list) > self.MAX_SIZE:
            self.best_individuals_fitness_list.pop()

    def update(self, individuals: List[MultipleGeneGP]):
        """
        Updates function fitness lists and the best individuals' list for a given population.
        """
        for individual in individuals:
            self.update_function_fitness_list(individual)

        # Find the best individuals in the provided population
        sorted_individuals = sorted(
            individuals, key=lambda ind: ind.fitness.wvalues[0], reverse=True
        )
        for individual in sorted_individuals[: self.MAX_SIZE]:
            self.update_best_individuals_list(individual)

        self.eliminate_functions()

    def eliminate_functions(self):
        """
        Eliminate functions that have significantly worse fitness values.
        """
        functions_to_remove = []
        for function, fitness_list in self.function_fitness_lists.items():
            _, p_value = stats.mannwhitneyu(
                self.best_individuals_fitness_list, fitness_list, alternative="greater"
            )

            if p_value < 0.05:
                functions_to_remove.append(function)

        for function in functions_to_remove:
            self.pset.primitives[function.ret].remove(function)
            del self.function_fitness_lists[function]


def valid_individual(individual: MultipleGeneGP, valid_functions):
    """
    Check if an individual only uses valid (non-eliminated) functions.
    """
    for tree in individual.gene:
        tree: PrimitiveTree
        for function in tree:
            if isinstance(function, gp.Primitive) and function not in valid_functions:
                return False
    return True
