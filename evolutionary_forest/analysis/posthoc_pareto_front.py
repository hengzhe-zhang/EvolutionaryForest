from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class ParetoFrontTool():
    @staticmethod
    def calculate_pareto_front(self: "EvolutionaryForestRegressor"):
        self.pareto_front = []
        if self.experimental_configuration.pac_bayesian_comparison and \
            isinstance(self.environmental_selection, EnvironmentalSelection):
            pac = R2PACBayesian(self, **self.param)
            self.pac_bayesian.objective = 'R2,MaxSharpness-1-Base'
            first_pareto_front = sortNondominated(self.pop, len(self.pop))[0]
            normalization_factor = np.mean((self.y - np.mean(self.y)) ** 2)
            for ind in first_pareto_front:
                if not isinstance(self.score_func, R2PACBayesian):
                    if isinstance(self.score_func, RademacherComplexityR2):
                        ind.rademacher_fitness_list = ind.fitness_list
                    pac.assign_complexity(ind, ind.pipe)
                sharpness_value = ind.fitness_list[1][0]
                self.pareto_front.append((float(np.mean(ind.case_values) / normalization_factor),
                                          float(sharpness_value / normalization_factor)))

    @staticmethod
    def calculate_test_pareto_front(self: "EvolutionaryForestRegressor", test_x, test_y):
        self.test_pareto_front = []
        self.size_pareto_front = []
        if self.experimental_configuration.pac_bayesian_comparison and \
            isinstance(self.environmental_selection, EnvironmentalSelection):
            self.test_pareto_front = []
            first_pareto_front = sortNondominated(self.pop, len(self.pop))[0]
            predictions = self.individual_prediction(test_x, first_pareto_front)
            normalization_factor = np.mean((self.y - np.mean(self.y)) ** 2)
            test_normalization_factor = np.mean((test_y - np.mean(test_y)) ** 2)
            for ind, prediction in zip(first_pareto_front, predictions):
                # the mean 1-sharpness across all samples
                sharpness_value = ind.fitness_list[1][0]
                errors = (test_y - prediction) ** 2
                assert len(errors) == len(test_y)
                normalized_test_error = float(np.mean(errors) / test_normalization_factor)
                normalized_sharpness = float(sharpness_value / normalization_factor)
                self.test_pareto_front.append((normalized_test_error, normalized_sharpness))
                self.size_pareto_front.append((normalized_test_error, sum([len(gene) for gene in ind.gene])))
