from deap.tools import sortNondominated

from evolutionary_forest.component.environmental_selection import EnvironmentalSelection
from evolutionary_forest.component.fitness import RademacherComplexityR2, R2PACBayesian
from evolutionary_forest.multigene_gp import *

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class ParetoFrontTool():
    @staticmethod
    def calculate_test_pareto_front(self: "EvolutionaryForestRegressor", test_x, test_y):
        """
        Calculate the Pareto front for test data based on the evolutionary forest regressor.

        This function determines the Pareto front from the current population of the model and
        computes the normalized prediction error for the test dataset. The results are appended
        to the `pareto_front` attribute of the `EvolutionaryForestRegressor` object.

        Parameters:
        - self (EvolutionaryForestRegressor): An instance of the EvolutionaryForestRegressor class.
        - test_x (array-like): Test input data. Each row represents an instance and each column represents a feature.
        - test_y (array-like): True target values for the test input data.
        """

        first_pareto_front = sortNondominated(self.pop, len(self.pop))[0]
        normalization_factor_scaled = np.mean((self.y - np.mean(self.y)) ** 2)
        predictions = self.individual_prediction(test_x, first_pareto_front)
        normalization_factor_test = np.mean((test_y - np.mean(test_y)) ** 2)
        for ind, prediction in zip(first_pareto_front, predictions):
            errors = (test_y - prediction) ** 2
            test_error_normalized_by_test = np.mean(errors) / normalization_factor_test
            self.pareto_front.append((float(np.mean(ind.case_values) / normalization_factor_scaled),
                                      float(test_error_normalized_by_test)))

    @staticmethod
    def calculate_sharpness_pareto_front(self: "EvolutionaryForestRegressor", x_train, y_train):
        if self.experimental_configuration.pac_bayesian_comparison and \
            isinstance(self.environmental_selection, EnvironmentalSelection):
            pac = R2PACBayesian(self, **self.param)
            self.pac_bayesian.objective = 'R2,MaxSharpness-1-Base'
            first_pareto_front = sortNondominated(self.pop, len(self.pop))[0]
            predictions = self.individual_prediction(x_train, first_pareto_front)
            normalization_factor = np.mean((y_train - np.mean(y_train)) ** 2)
            normalization_factor_scaled = np.mean((self.y - np.mean(self.y)) ** 2)
            for ind, prediction in zip(first_pareto_front, predictions):
                if not isinstance(self.score_func, R2PACBayesian):
                    if isinstance(self.score_func, RademacherComplexityR2):
                        ind.rademacher_fitness_list = ind.fitness_list
                    pac.assign_complexity(ind, ind.pipe)
                sharpness_value = ind.fitness_list[1][0]

                errors = (y_train - prediction) ** 2
                assert len(errors) == len(y_train)
                train_mse = np.mean(errors)

                """
                1. Normalized fitness values (case values represent cross-validation squared error)
                2. Normalized sharpness values
                3. Normalized training MSE
                """
                self.pareto_front.append((float(np.mean(ind.case_values) / normalization_factor_scaled),
                                          float(sharpness_value / normalization_factor_scaled),
                                          float(train_mse / normalization_factor)))

    @staticmethod
    def calculate_sharpness_pareto_front_on_test(self: "EvolutionaryForestRegressor", test_x, test_y, y_train):
        if self.experimental_configuration.pac_bayesian_comparison and \
            isinstance(self.environmental_selection, EnvironmentalSelection):
            self.test_pareto_front = []
            first_pareto_front = sortNondominated(self.pop, len(self.pop))[0]
            predictions = self.individual_prediction(test_x, first_pareto_front)
            """
            very important:
            Please ensure training data and test data are scaled in the same way.
            If test data is calculated on unscaled data, whereas training data is calculated on scaled data,
            the result will be wrong.
            """
            normalization_factor_train = np.mean((y_train - np.mean(y_train)) ** 2)
            normalization_factor_test = np.mean((test_y - np.mean(test_y)) ** 2)
            normalization_factor_scaled = np.mean((self.y - np.mean(self.y)) ** 2)
            for ind, prediction in zip(first_pareto_front, predictions):
                # the mean 1-sharpness across all samples
                sharpness_value = ind.fitness_list[1][0]
                errors = (test_y - prediction) ** 2
                assert len(errors) == len(test_y)
                test_error_normalized_by_train = np.mean(errors) / normalization_factor_train
                test_error_normalized_by_test = np.mean(errors) / normalization_factor_test
                """
                1. Normalized test error (1-Test R2)
                2. Normalized sharpness values / model size
                3. Normalized test error, but use same normalization factor as training data
                """
                self.test_pareto_front.append((test_error_normalized_by_test,
                                               float(sharpness_value / normalization_factor_scaled),
                                               test_error_normalized_by_train))
                self.size_pareto_front.append((test_error_normalized_by_test,
                                               sum([len(gene) for gene in ind.gene]),
                                               test_error_normalized_by_train))
