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
        Updates the fitness list of functions and terminals in the individual's GP tree.
        """
        fitness_value = individual.fitness.wvalues[0]
        for tree in individual.gene:
            tree: PrimitiveTree
            for element in tree:
                # If the element is a primitive or terminal, then add its fitness value
                if isinstance(element, (Primitive, gp.Terminal)):
                    # Check if the element is a constant terminal
                    if (
                        isinstance(element, gp.Terminal)
                        and hasattr(element, "value")
                        and isinstance(element.value, (int, float))
                    ):
                        continue  # Skip the constant terminal

                    element_key = self.get_element_name(element)

                    if element_key not in self.function_fitness_lists:
                        self.function_fitness_lists[element_key] = []

                    self.function_fitness_lists[element_key].append(fitness_value)

                    # Ensure the list does not exceed maximum size by removing the worst fitness value
                    if len(self.function_fitness_lists[element_key]) > self.MAX_SIZE:
                        self.function_fitness_lists[element_key].remove(
                            min(self.function_fitness_lists[element_key])
                        )

    def get_element_name(self, element):
        # Use a string representation as the key
        if isinstance(element, gp.Terminal):
            element_key = str(element.value)
        else:
            element_key = str(element.name)
        return element_key

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
        primitive_fitness_lists = {
            key: value
            for key, value in self.function_fitness_lists.items()
            if self.is_primitive(key)
        }
        terminal_fitness_lists = {
            key: value
            for key, value in self.function_fitness_lists.items()
            if not self.is_primitive(key)
        }

        # Identify the list with the best average fitness from primitive_fitness_lists
        best_primitive_key = max(
            primitive_fitness_lists,
            key=lambda k: sum(primitive_fitness_lists[k])
            / len(primitive_fitness_lists[k]),
        )
        best_primitive_fitness_list = primitive_fitness_lists[best_primitive_key]

        # Identify the list with the best average fitness from terminal_fitness_lists
        best_terminal_key = max(
            terminal_fitness_lists,
            key=lambda k: sum(terminal_fitness_lists[k])
            / len(terminal_fitness_lists[k]),
        )
        best_terminal_fitness_list = terminal_fitness_lists[best_terminal_key]

        elements_to_remove = []

        # Check primitives
        for element, fitness_list in primitive_fitness_lists.items():
            _, p_value = stats.mannwhitneyu(
                best_primitive_fitness_list, fitness_list, alternative="greater"
            )
            if (
                p_value < 0.01 and element != best_primitive_key
            ):  # Ensure we don't remove the best one itself
                elements_to_remove.append(element)

        # Check terminals
        for element, fitness_list in terminal_fitness_lists.items():
            _, p_value = stats.mannwhitneyu(
                best_terminal_fitness_list, fitness_list, alternative="greater"
            )
            if (
                p_value < 0.01 and element != best_terminal_key
            ):  # Ensure we don't remove the best one itself
                elements_to_remove.append(element)

        print("Removed elements:", elements_to_remove)

        for element_name in elements_to_remove:
            # Check and remove from pset.primitives
            for return_type, primitives in list(self.pset.primitives.items()):
                for p in primitives:
                    if p.name == element_name:
                        self.pset.primitives[return_type].remove(p)
                        break

            # Check and remove from pset.terminals
            for return_type, terminals in list(self.pset.terminals.items()):
                for t in terminals:
                    if t.name == element_name:
                        self.pset.terminals[return_type].remove(t)
                        break

            del self.function_fitness_lists[element_name]

    def is_primitive(self, element_key: str) -> bool:
        primitives = [p.name for p in self.pset.primitives[object]]
        if element_key in primitives:
            return True

    def isValid(self, individual: MultipleGeneGP) -> bool:
        """
        Checks if the given individual uses any of the eliminated functions or terminals.
        """
        for tree in individual.gene:
            tree: PrimitiveTree
            for element in tree:
                # Check if the element is a constant terminal
                if (
                    isinstance(element, gp.Terminal)
                    and hasattr(element, "value")
                    and isinstance(element.value, (int, float))
                ):
                    continue  # Skip the constant terminal
                element_key = self.get_element_name(element)
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
        invalid_individuals = [
            ind for ind in combined_population if not self.isValid(ind)
        ]

        # Sort valid individuals by their fitness values in descending order
        top_valid_individuals = sorted(
            valid_individuals, key=lambda ind: ind.fitness.wvalues[0], reverse=True
        )

        # If there are not enough valid individuals to fill 'n' slots, add some of the top invalid individuals
        if len(top_valid_individuals) < n:
            required_count = n - len(top_valid_individuals)
            top_invalid_individuals = sorted(
                invalid_individuals,
                key=lambda ind: ind.fitness.wvalues[0],
                reverse=True,
            )[:required_count]
            top_individuals = top_valid_individuals + top_invalid_individuals
        else:
            top_individuals = top_valid_individuals[:n]

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
