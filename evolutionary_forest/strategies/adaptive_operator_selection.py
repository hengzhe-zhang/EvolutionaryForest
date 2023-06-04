from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np

from evolutionary_forest.component.configuration import MABConfiguration

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class MultiArmBandit():
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **kwargs):
        self.algorithm = algorithm
        self.mab_configuration = MABConfiguration(**algorithm.mab_parameter)
        self.selection_operators = self.mab_configuration.selection_operators.split(',')
        self.selection_data = np.ones((2, len(self.selection_operators)))
        self.operator_selection_history = []
        self.best_value = None
        self.worst_value = None

    def update(self, population, offspring):
        comparison_criterion = self.mab_configuration.comparison_criterion
        """
        Fitness: Using the fitness improvement as the criterion of a success trial
        Case: Using the fitness improvement on a single case as the criterion of a success trial
        Case-Simple: Only consider fitness in each generation
        """
        self.best_value_initialization(population, comparison_criterion)
        best_value, parent_case_values = self.best_value_update(comparison_criterion, population)

        selection_data = self.selection_data
        mode = self.mab_configuration.mode
        cnt = Counter({
            id: 0
            for id in range(0, len(selection_data[0]))
        })
        if mode == 'Decay':
            selection_data[0] *= self.mab_configuration.decay_ratio
            selection_data[1] *= self.mab_configuration.decay_ratio
        C = self.mab_configuration.threshold

        for o in offspring:
            cnt[o.selection_operator] += 1
            if (comparison_criterion in ['Fitness', 'Fitness-Simple'] and o.fitness.wvalues[0] > best_value) or \
                (comparison_criterion in ['Case', 'Case-Simple'] and np.any(o.case_values < best_value)) or \
                (comparison_criterion in ['Parent', 'Single-Parent'] and
                 np.all(o.fitness.wvalues[0] < o.parent_fitness)) or \
                (isinstance(comparison_criterion, int) and
                 np.sum(o.case_values < best_value) > comparison_criterion):
                selection_data[0][o.selection_operator] += 1
            else:
                selection_data[1][o.selection_operator] += 1
        self.operator_selection_history.append(tuple(cnt.values()))

        if self.algorithm.verbose:
            print(selection_data[0], selection_data[1])
            print(cnt)

        if mode == 'Threshold':
            # fixed threshold
            for op in range(selection_data.shape[1]):
                if selection_data[0][op] + selection_data[1][op] > C:
                    sum_data = selection_data[0][op] + selection_data[1][op]
                    selection_data[0][op] = selection_data[0][op] / sum_data * C
                    selection_data[1][op] = selection_data[1][op] / sum_data * C
        # avoid trivial probability
        selection_data = np.clip(selection_data, 1e-2, None)
        self.selection_data = selection_data

    def best_value_update(self, comparison_criterion, population):
        best_value = self.best_value
        parent_case_values = None
        # record historical best values
        if comparison_criterion == 'Case':
            # consider the best fitness across all generations
            best_value = np.min([np.min([p.case_values for p in population], axis=0), self.best_value], axis=0)
            worst_value = np.max([np.max([p.case_values for p in population], axis=0), self.best_value], axis=0)
        if comparison_criterion == 'Fitness':
            # historical best fitness values
            best_value = max(*[p.fitness.wvalues[0] for p in population], self.best_value)
        if comparison_criterion == 'Fitness-Simple':
            # consider the best fitness in each generation
            best_value = max([p.fitness.wvalues[0] for p in population])
        if comparison_criterion == 'Case-Simple' or isinstance(comparison_criterion, int):
            # consider the best fitness in each generation
            best_value = np.min([p.case_values for p in population], axis=0)
        return best_value, parent_case_values

    def best_value_initialization(self, population, comparison_criterion):
        if self.best_value is None:
            if comparison_criterion in ['Fitness', 'Fitness-Simple']:
                self.best_value = np.max([p.fitness.wvalues[0] for p in population])
            elif comparison_criterion in ['Case', 'Case-Simple'] or isinstance(comparison_criterion, int):
                self.best_value = np.min([p.case_values for p in population], axis=0)
                self.worst_value = np.max([p.case_values for p in population], axis=0)
            elif comparison_criterion in ['Parent', 'Single-Parent']:
                pass
            else:
                raise Exception

    def select(self, parent):
        selection_operators = self.selection_operators
        selection_data = self.selection_data
        selection_operator_id = np.argmax(np.random.beta(selection_data[0], selection_data[1]))
        selection_operator = selection_operators[selection_operator_id]

        offspring = self.algorithm.custom_selection(parent, selection_operator)
        for o in offspring:
            o.selection_operator = selection_operator_id
        return offspring


class MCTS(MultiArmBandit):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **kwargs):
        super().__init__(algorithm)
        candidate_selection_operators = 'MAP-Elite-Lexicase,Tournament-7,Tournament-15,Lexicase'.split(',')
        candidate_survival_operators = 'AFP,Best,None'.split(',')
        mcts_dict = {}
        mcts_dict['Survival Operators'] = np.ones((2, len(candidate_survival_operators)))
        mcts_dict['Selection Operators'] = np.ones((2, len(candidate_selection_operators)))
        self.candidate_selection_operators = candidate_selection_operators
        self.candidate_survival_operators = candidate_survival_operators
        self.mcts_dict = mcts_dict

    def update(self, population, offspring):
        comparison_criterion = self.mab_configuration.comparison_criterion
        mcts_dict = self.mcts_dict
        C = self.mab_configuration.threshold
        selection_operator_counter = defaultdict(int)
        survival_operator_counter = defaultdict(int)

        self.best_value_initialization(population, comparison_criterion)
        best_value, parent_case_values = self.best_value_update(comparison_criterion, population)

        for o in offspring:
            if (comparison_criterion in ['Fitness', 'Fitness-Simple'] and o.fitness.wvalues[0] > best_value) or \
                (comparison_criterion in ['Case', 'Case-Simple'] and np.any(o.case_values < best_value)) or \
                (isinstance(comparison_criterion, int) and
                 np.sum(o.case_values < best_value) > comparison_criterion):
                mcts_dict['Survival Operators'][0][o.survival_operator_id] += 1
                mcts_dict['Selection Operators'][0][o.selection_operator] += 1
            else:
                mcts_dict['Survival Operators'][1][o.survival_operator_id] += 1
                mcts_dict['Selection Operators'][1][o.selection_operator] += 1
            selection_operator_counter[o.selection_operator] += 1
            survival_operator_counter[o.survival_operator_id] += 1

        if self.algorithm.verbose:
            print(selection_operator_counter, survival_operator_counter)

        mode = self.mab_configuration.mode
        # fixed threshold
        for k, data in mcts_dict.items():
            if mode == 'Threshold':
                # fixed threshold
                for op in range(len(data[0])):
                    if data[0][op] + data[1][op] > C:
                        sum_data = data[0][op] + data[1][op]
                        data[0][op] = data[0][op] / sum_data * C
                        data[1][op] = data[1][op] / sum_data * C
            if mode == 'Decay':
                data *= self.mab_configuration.decay_ratio
            # avoid trivial solutions
            data = np.clip(data, 1e-2, None)
            mcts_dict[k] = data
        if self.algorithm.verbose:
            print('MCTS Result', mcts_dict)
        self.mcts_dict = mcts_dict

    def select(self, parent):
        parent = self.survival_selection(parent)
        # MCTS
        mcts_dict = self.mcts_dict
        candidate_selection_operators = self.selection_operators
        candidates = mcts_dict['Selection Operators']
        selection_operator_id = np.argmax(np.random.beta(candidates[0], candidates[1]))
        selection_operator = candidate_selection_operators[selection_operator_id]

        offspring = self.algorithm.custom_selection(parent, selection_operator)
        for o in offspring:
            o.selection_operator = selection_operator_id
        return offspring

    def survival_selection(self, parent):
        mcts_dict = self.mcts_dict
        candidate_survival_opearators = self.candidate_survival_operators

        # Select the best survival operator based on the Thompson sampling
        candidates = mcts_dict[f'Root']
        survival_operator_id = np.argmax(np.random.beta(candidates[0], candidates[1]))
        parent_archive = candidate_survival_opearators[survival_operator_id]

        if parent_archive == 'Fitness-Size':
            parent = self.nsga_archive
        if parent_archive == 'AFP':
            parent = self.afp_archive
        if parent_archive == 'Best':
            parent = self.best_archive

        for o in parent:
            o.survival_operator_id = survival_operator_id
        return parent
