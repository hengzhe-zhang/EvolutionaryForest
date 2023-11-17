import copy
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from deap import gp
from deap.gp import Primitive, PrimitiveTree
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utility.priority_queue import MinPriorityQueue


class RacingFunctionSelector:
    def __init__(
        self,
        pset,
        content,
        remove_primitives=False,
        remove_terminals=False,
        racing_list_size=100,
        importance_level="Inv",
        use_global_fitness=True,
        frequency_threshold=0,
        use_sensitivity_analysis=False,
        verbose=False,
        racing_environmental_selection=True,
        p_threshold=1e-2,
        ts_num_predictions=0,
        priority_queue=False,
        only_better=False,
        **kwargs
    ):
        self.ts_num_predictions = ts_num_predictions
        self.p_threshold = p_threshold
        self.racing_environmental_selection = racing_environmental_selection
        self.verbose = verbose
        self.frequency_threshold = frequency_threshold
        self.importance_level = importance_level
        self.remove_primitives = remove_primitives
        self.remove_terminals = remove_terminals
        self.function_fitness_lists = {}
        if priority_queue:
            self.best_individuals_fitness_list = MinPriorityQueue()
        else:
            self.best_individuals_fitness_list = []
        self.racing_list_size = racing_list_size
        self.pset = pset
        self.backup_pset = copy.deepcopy(pset)
        self.content = content
        self.function_importance_list = {}
        self.use_global_fitness = use_global_fitness
        self.use_sensitivity_analysis = use_sensitivity_analysis
        self.priority_queue = priority_queue
        self.only_better = only_better

    def sensitivity_analysis(self, pop: List[MultipleGeneGP]):
        model = RandomForestRegressor()
        num_features = len(self.pset.terminals[object]) + len(
            self.pset.primitives[object]
        )
        X = np.zeros((len(pop), num_features))  # Feature matrix
        y = np.zeros(len(pop))  # Target vector

        for ind_idx, ind in enumerate(pop):
            for tree in ind.gene:
                for element in tree:
                    # We check for constant value terminals which we skip
                    if (
                        isinstance(element, gp.Terminal)
                        and hasattr(element, "value")
                        and isinstance(element.value, (int, float))
                    ):
                        continue
                    feature_idx = self.get_index(element)
                    X[ind_idx, feature_idx] += 1  # Mark the feature as present
            y[ind_idx] = ind.fitness.wvalues[0]

        # Train the Random Forest
        model.fit(X, y)

        # Get feature importances
        feature_importances = model.feature_importances_
        return feature_importances

    def get_index(self, element):
        all_list = self.pset.terminals[object] + self.pset.primitives[object]
        all_list = [x.name for x in all_list]
        return all_list.index(element.name)

    def update_function_fitness_list(self, individual: MultipleGeneGP):
        """
        Updates the fitness list of functions and terminals in the individual's GP tree.
        """
        fitness_value = individual.fitness.wvalues[0]
        frequency = defaultdict(int)
        importance = defaultdict(int)
        coefs = abs(individual.coef)
        coefs = coefs / np.sum(coefs)
        for tree, coef in zip(individual.gene, coefs):
            if self.importance_level == "Square" and coef < 1 / (
                np.sqrt(len(individual.gene)) ** 2
            ):
                continue
            if self.importance_level == "Inv" and coef < 1 / len(individual.gene):
                continue
            if self.importance_level == "Sqrt" and coef < 1 / np.sqrt(
                len(individual.gene)
            ):
                continue
            if self.importance_level == "Cbrt" and coef < 1 / np.cbrt(
                len(individual.gene)
            ):
                continue

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
                    frequency[element_key] += 1
                    importance[element_key] += coef

            for element_key, freq in frequency.items():
                if self.frequency_threshold > 0 and freq <= self.frequency_threshold:
                    continue
                if self.frequency_threshold < 0 and freq > -self.frequency_threshold:
                    continue

                if element_key not in self.function_fitness_lists:
                    if self.priority_queue:
                        self.function_fitness_lists[element_key] = MinPriorityQueue()
                    else:
                        self.function_fitness_lists[element_key] = []

                self.function_fitness_lists[element_key].append(fitness_value)

                if element_key not in self.function_importance_list:
                    self.function_importance_list[element_key] = []

                self.function_importance_list[element_key].append(
                    importance[element_key] / freq
                )

                # Ensure the list does not exceed maximum size
                if (
                    len(self.function_fitness_lists[element_key])
                    > self.racing_list_size
                ):
                    # Remove the fitness value using the index
                    self.function_fitness_lists[element_key].pop(0)
                    # Remove the corresponding frequency using the same index
                    self.function_importance_list[element_key].pop(0)

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

        # Ensure the list does not exceed the maximum size by removing the worst fitness value
        if len(self.best_individuals_fitness_list) > self.racing_list_size:
            self.best_individuals_fitness_list.pop(0)

    def update(self, individuals: List[MultipleGeneGP]):
        """
        Updates function fitness lists and the best individuals' list for a given population.
        """
        self.pset.primitives = copy.deepcopy(self.backup_pset.primitives)
        self.pset.terminals = copy.deepcopy(self.backup_pset.terminals)
        for individual in individuals:
            self.update_function_fitness_list(individual)
        for individual in individuals:
            self.update_best_individuals_list(individual)

        sensitivity = self.sensitivity_analysis(individuals)

        self.eliminate_functions(sensitivity)

    def eliminate_functions(self, sensitivity: np.ndarray):
        """
        Eliminate functions and terminals that have significantly worse fitness values.
        """
        remove_primitives = self.remove_primitives
        remove_terminals = self.remove_terminals

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

        elements_to_remove = []

        if remove_primitives:
            # Identify the list with the best average fitness from primitive_fitness_lists
            (
                best_primitive_fitness_list,
                best_primitive_key,
            ) = self.get_reference_element(primitive_fitness_lists)

            self.elminate_elements(
                best_primitive_fitness_list,
                best_primitive_key,
                elements_to_remove,
                primitive_fitness_lists,
            )

        if remove_terminals:
            best_terminal_fitness_list, best_terminal_key = self.get_reference_element(
                terminal_fitness_lists
            )
            self.elminate_elements(
                best_terminal_fitness_list,
                best_terminal_key,
                elements_to_remove,
                terminal_fitness_lists,
            )

        if self.use_sensitivity_analysis:
            elements_to_remove = self.remove_by_sensitivity_analysis(
                elements_to_remove, sensitivity
            )

        # print("Elements to remove:", elements_to_remove)
        if self.verbose:
            print("Removed elements:", len(elements_to_remove), elements_to_remove)

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

    def remove_by_sensitivity_analysis(self, elements_to_remove, sensitivity):
        terminal_sensitivity = sensitivity[: len(self.pset.terminals[object])]
        primitive_sensitivity = sensitivity[len(self.pset.terminals[object]) :]
        good_terminal = [
            x.name
            for x, s in zip(self.pset.terminals[object], terminal_sensitivity)
            if s > np.percentile(terminal_sensitivity, 80)
        ]
        bad_terminal = [
            x.name
            for x, s in zip(self.pset.terminals[object], terminal_sensitivity)
            if s < np.percentile(terminal_sensitivity, 20)
        ]
        good_primitive = [
            x.name
            for x, s in zip(self.pset.primitives[object], primitive_sensitivity)
            if s > np.percentile(primitive_sensitivity, 80)
        ]
        bad_primitive = [
            x.name
            for x, s in zip(self.pset.primitives[object], primitive_sensitivity)
            if s < np.percentile(primitive_sensitivity, 20)
        ]
        elements_to_remove = list(
            filter(
                lambda x: x not in good_terminal and x not in good_primitive,
                elements_to_remove,
            )
        )
        remove_list = []
        if self.remove_primitives:
            remove_list += bad_primitive
        if self.remove_terminals:
            remove_list += bad_terminal
        for e in remove_list:
            if e not in elements_to_remove:
                elements_to_remove.append(e)
        return elements_to_remove

    def ts_predict(self, fitness_list):
        auto_regressive_order = 5
        integrate_order = 1
        if (
            self.ts_num_predictions > 0
            and len(fitness_list) > auto_regressive_order + integrate_order
        ):
            series = pd.Series(fitness_list)
            model = ARIMA(series, order=(auto_regressive_order, integrate_order, 0))
            model_fit = model.fit()
            # Forecast future data points
            forecast = model_fit.forecast(steps=self.ts_num_predictions)
            extended_fitness_list = fitness_list + list(forecast)
            return extended_fitness_list
        else:
            return fitness_list

    def elminate_elements(
        self,
        best_primitive_fitness_list: list,
        best_primitive_key: str,
        elements_to_remove: list,
        primitive_fitness_lists: dict,
    ):
        # Check primitives
        for element, fitness_list in primitive_fitness_lists.items():
            fitness_list = self.ts_predict(fitness_list)
            if self.priority_queue:
                best_primitive_fitness_list = list(best_primitive_fitness_list)
                fitness_list = list(fitness_list)
            if self.only_better:
                _, p_value = stats.mannwhitneyu(
                    best_primitive_fitness_list,
                    fitness_list,
                    alternative="smaller",
                )
            else:
                _, p_value = stats.mannwhitneyu(
                    best_primitive_fitness_list,
                    fitness_list,
                    # alternative="greater",
                )
            p_threshold = self.p_threshold
            if (
                p_value > p_threshold and element != best_primitive_key
            ):  # Ensure we don't remove the best one itself
                elements_to_remove.append(element)

    def get_reference_element(self, terminal_fitness_lists) -> [list, str]:
        # Identify the list with the best average fitness from terminal_fitness_lists
        best_terminal_key = max(
            terminal_fitness_lists,
            key=lambda k: sum(terminal_fitness_lists[k])
            / len(terminal_fitness_lists[k]),
        )
        if self.use_global_fitness:
            best_terminal_fitness_list = self.best_individuals_fitness_list
        else:
            best_terminal_fitness_list = terminal_fitness_lists[best_terminal_key]
        return best_terminal_fitness_list, best_terminal_key

    def is_primitive(self, element_key: str) -> bool:
        primitives = [p.name for p in self.pset.primitives[object]]
        if element_key in primitives:
            return True

    def is_terminal(self, element_key: str) -> bool:
        terminals = [p.name for p in self.pset.terminals[object]]
        if element_key in terminals:
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

                if not self.is_primitive(element_key) and not self.is_terminal(
                    element_key
                ):
                    return False
        return True

    def remove_duplicates(self, combined_population):
        unique_population = []
        encountered_keys = set()

        for individual in combined_population:
            key = tuple(str(gene) for gene in individual.gene)
            if key not in encountered_keys:
                unique_population.append(individual)
                encountered_keys.add(key)
        return unique_population

    def environmental_selection(
        self, parents: List[MultipleGeneGP], offspring: List[MultipleGeneGP], n: int
    ) -> List[MultipleGeneGP]:
        """
        Selects individuals from a combination of parents and offspring, filtering out those that use eliminated functions,
        and returns the top n individuals based on fitness values.
        """
        if not self.racing_environmental_selection:
            return offspring
        combined_population = parents + offspring
        combined_population = self.remove_duplicates(combined_population)
        valid_individuals = [ind for ind in combined_population if self.isValid(ind)]
        invalid_individuals = [
            ind for ind in combined_population if not self.isValid(ind)
        ]

        # Sort valid individuals by their fitness values in descending order
        top_valid_individuals = sorted(
            valid_individuals, key=lambda ind: ind.fitness.wvalues[0], reverse=True
        )
        top_invalid_individuals = sorted(
            invalid_individuals, key=lambda ind: ind.fitness.wvalues[0], reverse=True
        )

        # If there are not enough valid individuals to fill 'n' slots, add some of the top invalid individuals
        if len(top_valid_individuals) >= n:
            return top_valid_individuals[:n]
        else:
            return (
                top_valid_individuals[:n]
                + top_invalid_individuals[: n - len(top_valid_individuals)]
            )


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
