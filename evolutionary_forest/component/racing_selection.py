from typing import List

from deap import gp
from deap.gp import Primitive, PrimitiveTree
from scipy import stats

from evolutionary_forest.multigene_gp import MultipleGeneGP


class RacingFunctionSelector:
    def __init__(self):
        self.function_fitness_lists = {}
        self.best_individuals_fitness_list = []
        self.MAX_SIZE = 100

    def update_function_fitness_list(self, individual: MultipleGeneGP):
        """
        Updates the fitness list of functions and terminals in the individual's GP tree.
        """
        fitness_value = individual.fitness.wvalues[0]
        for tree in individual.gene:
            tree: PrimitiveTree
            for element in tree:
                # If the element is a primitive or terminal, then add its fitness value
                if isinstance(element, (Primitive, gp.Terminal)):
                    # Use a string representation as the key
                    if isinstance(element, gp.Terminal):
                        element_key = str(element.value)
                    else:
                        element_key = str(element.name)

                    if element_key not in self.function_fitness_lists:
                        self.function_fitness_lists[element_key] = []

                    self.function_fitness_lists[element_key].append(fitness_value)

                    # Ensure the list does not exceed maximum size by removing the worst fitness value
                    if len(self.function_fitness_lists[element_key]) > self.MAX_SIZE:
                        self.function_fitness_lists[element_key].remove(
                            min(self.function_fitness_lists[element_key])
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
        Eliminate functions and terminals that have significantly worse fitness values.
        """
        elements_to_remove = []
        for element, fitness_list in self.function_fitness_lists.items():
            _, p_value = stats.mannwhitneyu(
                self.best_individuals_fitness_list, fitness_list, alternative="greater"
            )

            if p_value < 0.05:
                elements_to_remove.append(element)

        print("Removed {} elements".format(elements_to_remove))

        for element in elements_to_remove:
            if isinstance(element, gp.Primitive):
                self.pset.primitives[element.ret].remove(element)
            elif isinstance(element, gp.Terminal):
                self.pset.terminals[element.ret].remove(element)
            del self.function_fitness_lists[element]

    def pset_update(self, pset: gp.PrimitiveSet):
        """
        Updates the pset based on eliminated functions and terminals.
        """
        self.pset = pset
        for element in self.function_fitness_lists.keys():
            if isinstance(element, gp.Primitive):
                if element not in self.pset.primitives[element.ret]:
                    del self.function_fitness_lists[element]
            elif isinstance(element, gp.Terminal):
                if element not in self.pset.terminals[element.ret]:
                    del self.function_fitness_lists[element]

    def isValid(self, individual: MultipleGeneGP) -> bool:
        """
        Checks if the given individual uses any of the eliminated functions or terminals.
        """
        for tree in individual.gene:
            tree: PrimitiveTree
            for element in tree:
                element_key = str(element)
                if element_key not in self.function_fitness_lists:
                    return False
        return True

    def environmental_selection(
        self, parents: List[MultipleGeneGP], offspring: List[MultipleGeneGP], n: int
    ) -> List[MultipleGeneGP]:
        """
        Selects individuals from a combination of parents and offspring, filtering out those that use eliminated functions,
        and returns the top n individuals based on fitness values.
        """
        combined_population = parents + offspring
        valid_individuals = [ind for ind in combined_population if self.isValid(ind)]

        # Sort individuals by their fitness values in descending order and select the top n
        top_individuals = sorted(
            valid_individuals, key=lambda ind: ind.fitness.wvalues[0], reverse=True
        )[:n]

        return top_individuals


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
