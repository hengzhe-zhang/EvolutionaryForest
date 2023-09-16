import copy
import math
import operator
from bisect import bisect_right
from collections import defaultdict
from functools import partial
from itertools import chain, compress
from operator import eq

import numpy as np
from deap.tools import HallOfFame
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from evolutionary_forest.component.configuration import ArchiveConfiguration
from evolutionary_forest.component.evaluation import quick_result_calculation
from evolutionary_forest.component.primitive_functions import individual_to_tuple
from evolutionary_forest.component.subset_selection import EnsembleSelectionADE


class StrictlyImprovementHOF(HallOfFame):

    def update(self, population):
        population = list(filter(lambda x: (not hasattr(x, 'parent_fitness')) or
                                           np.all(x.fitness.wvalues[0] > np.array(x.parent_fitness)),
                                 population))
        super().update(population)


class CustomHOF(HallOfFame):
    def __init__(self, maxsize, comparison_function, key_metric,
                 similar=eq):
        self.comparison_function = comparison_function
        self.key_metric = key_metric
        super().__init__(maxsize, similar)

    def update(self, population):
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if self.comparison_function(ind, self[-1]) or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)

    def insert(self, item):
        item = copy.deepcopy(item)
        i = bisect_right(self.keys, self.key_metric(item))
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, self.key_metric(item))


class GeneralizationHOF(HallOfFame):
    def __init__(self, X, y, pset, maxsize=1, verbose=False):
        super().__init__(maxsize)
        self.X = X
        self.y = y
        self.pset = pset
        self.verbose = verbose

    def update(self, population):
        best_ind = sorted(population, key=lambda x: x.fitness.wvalues)[-1]
        X_hat = quick_result_calculation(best_ind.gene, self.pset, self.X)
        lr_r2_scores = []
        for _ in range(5):
            id = np.random.permutation(np.arange(0, len(self.X)))
            y = self.y[id]
            lr = LinearRegression()
            lr_r2_score = r2_score(y, lr.fit(X_hat, y).predict(X_hat))
            lr_r2_scores.append(lr_r2_score)
        best_ind.complexity_score = np.mean(lr_r2_scores)
        if len(self) == 0:
            super().update(population)
        else:
            if best_ind.fitness.wvalues[0] - best_ind.complexity_score > \
                self[0].fitness.wvalues[0] - self[0].complexity_score:
                super().update(population)
            else:
                if self.verbose:
                    print('Complexity Score', best_ind.complexity_score, self[0].complexity_score)


class LexicaseHOF(HallOfFame):
    """
    Select all individuals with one case speciality
    """

    def __init__(self):
        HallOfFame.__init__(self, None)

    def update(self, population):
        for p in population:
            self.insert(p)

        hofer_arr: np.ndarray = None
        for i, hofer in enumerate(self):
            if hofer_arr is None:
                hofer_arr = np.array(-1 * hofer.case_values).reshape(1, -1)
            else:
                hofer_arr = np.concatenate([hofer_arr, np.array(-1 * hofer.case_values).reshape(1, -1)])
        max_hof = np.max(hofer_arr, axis=0)

        del_ind = []
        max_value = np.full_like(max_hof, -np.inf)
        for index, x in enumerate(self):
            fitness_wvalues = np.array(-1 * x.case_values)
            if np.any(fitness_wvalues >= max_hof) and np.any(fitness_wvalues > max_value):
                loc = np.where(fitness_wvalues > max_value)
                max_value[loc] = fitness_wvalues[loc]
                continue
            # determine the deletion individual list
            del_ind.append(index)

        for i in reversed(del_ind):
            self.remove(i)


class RandomObjectiveHOF(HallOfFame):
    """
    An archive which updating based on random projected fitness values
    """

    def update(self, population):
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if np.any(ind.case_values < self[-1].case_values) or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)


"""
There are two places need to consider data imbalance
"""


class EnsembleSelectionHallOfFame(HallOfFame):
    def __init__(self, maxsize, label, similar=eq, multitask=False, diversity_ratio=0,
                 unique=True, bagging_iteration=1, loss_function=None, inner_sampling=0,
                 outer_sampling=0.5, limited_sample=False, initial_size=5):
        """
        Three strategies:
        1. Selection with Replacement
        2. Sorted Ensemble Initialization
        3. Bagged Ensemble Selection

        unique: Selection without bootstrapping, and thus avoiding weighted selection
        """
        super().__init__(maxsize, similar)
        self.label = label
        self.categories = None
        self.novelty_weight = None
        self.unique = unique
        self.ensemble_weight = defaultdict(int)
        self.initial_size = initial_size
        self.bagging_iteration = bagging_iteration
        self.instances_num = 0
        self.verbose = False
        self.class_weight = None
        self.task_type = 'Regression'
        self.loss_function = loss_function
        self.multitask = multitask
        self.diversity_ratio = diversity_ratio
        self.inner_sampling = inner_sampling
        self.outer_sampling = outer_sampling
        self.limited_sample = limited_sample

    def mse(self, prediction):
        return (self.label - prediction) ** 2

    def cross_entropy(self, prediction):
        eps = 1e-15
        return -1 * self.label * np.log(np.clip(prediction, eps, 1 - eps))

    def zero_one(self, prediction):
        return np.argmax(self.label, axis=1) != np.argmax(prediction, axis=1)

    def hinge_loss(self, prediction):
        return np.mean(np.clip(self.label - prediction, 0, None))

    def loss(self, prediction):
        # Both classification and regression use MSE as the ensemble selection metric
        # The reason is due to an additive model need to use MSE as the loss function
        if self.task_type == 'Regression':
            return self.mse(prediction)
        else:
            if self.loss_function == 'Hinge':
                return self.hinge_loss(prediction)
            elif self.loss_function == 'CrossEntropy':
                return self.cross_entropy(prediction)
            elif self.loss_function == 'ZeroOne':
                return self.zero_one(prediction)
            elif self.loss_function == 'MSE':
                return self.mse(prediction)
            else:
                raise Exception

    def ensemble_selection(self, population):
        new_inds = []
        new_ind_tuples = defaultdict(int)
        instances_num = self.instances_num
        assert instances_num > 0
        if self.categories is None:
            # regression
            sum_prediction = np.zeros(instances_num)
        else:
            # classification
            sum_prediction = np.zeros((instances_num, self.categories))

        all_inds = population
        population_size = len(all_inds)
        current_error = np.inf
        # we need to define a variable to record
        selection_count = 0

        if self.loss_function == 'ZeroOne':
            # ZeroOne loss is beneficial for decision tree
            for ind in population:
                # ind.predicted_values = (ind.predicted_values == np.max(ind.predicted_values, axis=0)).astype(int)
                argmax = np.argmax(ind.predicted_values, axis=1)
                ind.predicted_values[:] = 0
                ind.predicted_values[np.arange(0, len(ind.predicted_values)), argmax] = 1

        def error_calculation(x, diversity=False):
            prediction = ((x.predicted_values + sum_prediction) / (selection_count + 1))
            if self.class_weight is not None:
                loss = self.loss(prediction)
                if len(loss.shape) == 1:
                    error = np.sum(self.class_weight.flatten() * loss)
                elif len(loss.shape) == 2:
                    error = np.sum(self.class_weight * loss)
                else:
                    raise Exception
            else:
                error = np.sum(self.loss(prediction))
            if diversity:
                error -= self.diversity_ratio * \
                         np.sum((x.predicted_values - sum_prediction / selection_count) ** 2)
            return error

        """
        目前依然需要确定多样性丧失的问题根源
        """
        while selection_count < self.maxsize and len(all_inds) > 0:
            if selection_count == 0:
                if isinstance(self, GreedySelectionHallOfFame):
                    # calculate loss summation
                    if self.loss_function in ['ZeroOne', 'CrossEntropy', 'MSE']:
                        errors = [error_calculation(x) for x in all_inds]
                    else:
                        errors = [np.sum(x.case_values[:instances_num]) for x in all_inds]
                    # For multitask optimization, we may need to consider them respectively
                    if self.multitask and all_inds[0].base_model is not None:
                        initial_individuals = self.initial_size * 2
                        args = np.argsort(errors)
                        all_base_models = {ind.base_model for ind in all_inds}
                        inds = []
                        for m in all_base_models:
                            inds.extend(list(filter(lambda id: all_inds[id].base_model == m, args))
                                        [:self.initial_size])
                    else:
                        initial_individuals = self.initial_size
                    inds = np.argsort(errors)[:initial_individuals]
                    for ind in inds:
                        ind = all_inds[ind]
                        sum_prediction += ind.predicted_values
                        selection_count += 1
                        # using tuple to determine which individuals are selected
                        ind_tuple = individual_to_tuple(ind)
                        new_ind_tuples[ind_tuple] += 1
                        if self.unique:
                            # single instance selection
                            new_inds.append(ind)
                            all_inds.remove(ind)
                            self.ensemble_weight[ind_tuple] += 1
                        else:
                            # allow sample with replacement
                            if ind_tuple not in self.ensemble_weight:
                                new_inds.append(ind)
                                self.ensemble_weight[ind_tuple] = 1
                            else:
                                self.ensemble_weight[ind_tuple] += 1
                    if self.unique:
                        assert len(all_inds) < population_size, f'{len(all_inds)} {population_size}'
                    assert selection_count == initial_individuals, selection_count
                    # update current error (first iteration)
                    prediction = (sum_prediction / initial_individuals)
                    if self.class_weight is not None:
                        current_error = np.sum(self.class_weight * self.loss(prediction))
                    else:
                        current_error = np.sum(self.loss(prediction))
                    continue
                else:
                    ind = min(all_inds, key=lambda x: np.sum(x.case_values[:instances_num]))
            else:
                if isinstance(self, DREPHallOfFame):
                    if self.class_weight is not None or self.task_type == 'Classification':
                        raise Exception
                    # First select individuals according to diversity
                    elitist = sorted(all_inds,
                                     key=lambda x: -1 * np.sum((x.predicted_values -
                                                                sum_prediction / selection_count) ** 2)) \
                        [:math.ceil(self.r * len(all_inds))]
                    # Then select individuals according to accuracy
                    ind = min(elitist, key=lambda x: np.sum(x.case_values[:instances_num]))
                elif isinstance(self, GreedySelectionHallOfFame):
                    # Select based on loss reduction
                    # list(map(lambda x: (x.base_model, error_calculation(x)), all_inds))
                    if self.inner_sampling > 0:
                        selected_index = np.random.random(len(all_inds)) < self.inner_sampling
                        if np.sum(selected_index) == 0:
                            # if all zero, then random set one position to be true
                            selected_index[np.random.randint(len(all_inds))] = True
                        ind_pool = list(compress(all_inds, selected_index))
                    else:
                        ind_pool = all_inds
                    if self.diversity_ratio > 0:
                        ind = min(ind_pool, key=partial(error_calculation, diversity=True))
                    else:
                        ind = min(ind_pool, key=error_calculation)
                else:
                    raise Exception

            # common early stop strategy
            early_stop = True
            skip_selection = True
            if error_calculation(ind) < current_error:
                current_error = error_calculation(ind)
            else:
                # warning: different order will not produce different results
                all_inds.remove(ind)
                if isinstance(self, GreedySelectionHallOfFame):
                    # If the selection strategy is Greedy Selection, then we don't need to check further more
                    break
                if skip_selection:
                    continue
                if early_stop:
                    break

            sum_prediction += ind.predicted_values
            selection_count += 1
            # using tuple to determine which individuals are selected
            ind_tuple = individual_to_tuple(ind)
            new_ind_tuples[ind_tuple] += 1
            if self.unique:
                # single instance selection
                new_inds.append(ind)
                all_inds.remove(ind)
                self.ensemble_weight[ind_tuple] += 1
            else:
                if ind_tuple not in self.ensemble_weight:
                    new_inds.append(ind)
                    self.ensemble_weight[ind_tuple] = 1
                else:
                    self.ensemble_weight[ind_tuple] += 1
                    if self.limited_sample and new_ind_tuples[ind_tuple] >= 5:
                        # limit the number of same model each round
                        all_inds.remove(ind)
        # assert_almost_equal(np.sum(sum_prediction / selection_count, axis=1), 1)
        if self.class_weight is not None:
            loss = np.mean(self.class_weight * self.loss(sum_prediction / selection_count))
        else:
            loss = np.mean(self.loss(sum_prediction / selection_count))
        return new_inds, loss

    def update(self, population):
        self.ensemble_weight.clear()
        self.instances_num = len(population[0].predicted_values)
        if self.bagging_iteration > 1:
            new_inds = []
            new_inds_set = set()
            for _ in range(self.bagging_iteration):
                probability = self.outer_sampling
                all_inds = list(chain(population, self))
                selected_index = np.random.random(len(all_inds)) < probability
                temp_inds, _ = self.ensemble_selection(list(compress(all_inds, selected_index)))
                for ind in temp_inds:
                    ind_tuple = individual_to_tuple(ind)
                    if ind_tuple not in new_inds_set:
                        new_inds.append(ind)
                        new_inds_set.add(ind_tuple)
            # [(self.ensemble_weight.get(individual_to_tuple(a), 0), a) for a in all_inds]
            # print('All Ensemble Weight', self.ensemble_weight.values())
            assert len(self.ensemble_weight.keys()) == len(new_inds_set)
        else:
            new_inds, _ = self.ensemble_selection(population)
        if self.verbose:
            print('Ensemble Size', len(new_inds))
        self.clear()
        for x in new_inds:
            self.insert(x)


class DREPHallOfFame(EnsembleSelectionHallOfFame):
    # A hall of fame based on DREP algorithm
    def update(self, population):
        paradigm = []
        for r in np.arange(0.2, 0.5 + 0.01, 0.1):
            self.r = r
            paradigm.append(self.ensemble_selection(population))
        # print([x[1] for x in paradigm])
        new_inds, _ = min(paradigm, key=lambda x: x[1])
        super(DREPHallOfFame, self).update(new_inds)


class MultiTaskHallOfFame(HallOfFame):

    def __init__(self, maxsize, model_list, similar=eq):
        self.model_list = model_list
        super().__init__(maxsize, similar)
        number_of_models = len(self.model_list.split(','))
        self.real_hof = [HallOfFame(maxsize // number_of_models) for _ in range(number_of_models)]

    def update(self, population):
        all_models = []
        for id, model in enumerate(self.model_list.split(',')):
            new_ind = list(filter(lambda x: x.base_model == model, population))
            self.real_hof[id].update(new_ind)
            all_models.extend(list(self.real_hof[id].items))
        self.items = all_models


class ValidationHallOfFame(HallOfFame):
    def __init__(self, maxsize, validation_function, archive_configuration: ArchiveConfiguration,
                 similar=eq):
        # Currently, only supports size of one
        self.validation_function = validation_function
        self.archive_configuration = archive_configuration
        assert maxsize == 1
        super().__init__(maxsize, similar)

    def update(self, population):
        if self.archive_configuration.dynamic_validation and len(self) > 0:
            # update the fitness of individuals in the hall of fame
            self[0].fitness.values = (-1 * self.validation_function(self[0], force_training=True),)
        best_individual = max(population, key=lambda x: x.fitness.wvalues)
        best_individual = copy.deepcopy(best_individual)
        best_individual.fitness.values = (-1 * self.validation_function(best_individual),)
        super().update([best_individual])


class NoveltyHallOfFame(EnsembleSelectionHallOfFame):
    pass


class GreedySelectionHallOfFame(EnsembleSelectionHallOfFame):
    pass


class HybridHallOfFame(HallOfFame):
    """
    A hall of fame tailored for hybrid base models,
    i.e., retain best models for each class of base models
    """

    def __init__(self, maxsize, similar=operator.eq):
        super().__init__(maxsize, similar=similar)
        hof_num = 2
        self.hofs = [HallOfFame(maxsize // hof_num) for i in range(hof_num)]

    def update(self, population):
        for cls, hof in zip([RidgeCV, DecisionTreeRegressor], self.hofs):
            hof.update(list(filter(lambda x: isinstance(x.pipe['Ridge'], cls), population)))

    def __len__(self):
        return len(list(chain.from_iterable(self.hofs)))

    def __getitem__(self, i):
        return list(chain.from_iterable(self.hofs))[i]

    def __iter__(self):
        return iter(chain.from_iterable(self.hofs))

    def __reversed__(self):
        return reversed(list(chain.from_iterable(self.hofs)))


class OOBHallOfFame(HallOfFame):
    """
    A hall of fame based on OOB error
    """

    def __init__(self, X, y, toolbox, maxsize, similar=operator.eq):
        super().__init__(maxsize, similar)
        self.X = X
        self.y = y
        self.toolbox = toolbox

    # @timeit
    def update(self, population):
        if len(self.items) < self.maxsize:
            super().update(population)
        else:
            classes = population[0].pipe['Ridge'].classes_
            n_classes = population[0].pipe['Ridge'].n_classes_

            # generate one-hot label
            encoded_label = OneHotEncoder(sparse_output=False).fit_transform(self.y.reshape(-1, 1))
            assert np.all(classes[np.argmax(encoded_label, axis=1)] == self.y)
            assert encoded_label.shape[1] == n_classes

            current_pop = list(self.items) + population
            prediction = np.full((len(current_pop), len(self.y), n_classes), np.nan)
            for i, x in enumerate(current_pop):
                index = x.out_of_bag
                prediction[i][index] = x.oob_prediction

            def evaluation_func(x):
                # calculate the OOB error for a given selection scheme
                x_index = np.argsort(x)[:self.maxsize]
                return np.sum((np.nanmean(prediction[x_index], axis=0) - encoded_label) ** 2)

            # generate an initial vector based on fitness values
            initial_vector = np.zeros(len(current_pop))
            initial_vector[np.argsort(-1 * np.array([p.fitness.wvalues[0] for p in current_pop]))[:self.maxsize]] = -1
            e = EnsembleSelectionADE(len(current_pop), 20, 100, evaluation_function=evaluation_func,
                                     verbose=False, de_algorithm='JSO')
            result = e.run(initial_vector=initial_vector)
            selected_index = np.argsort(result)[:self.maxsize]

            self.clear()
            for s in selected_index:
                self.insert(current_pop[s])
            assert len(self.items) == self.maxsize, len(self.items)


class BootstrapHallOfFame(HallOfFame):
    """
    A hall of fame based on different optimal targets
    """

    def __init__(self, x_train, maxsize, similar=operator.eq):
        super().__init__(maxsize, similar)
        self.x_train = x_train
        sample_indices = np.arange(self.x_train.shape[0])
        bootstrap_indices = np.random.choice(sample_indices,
                                             size=(maxsize, sample_indices.shape[0]),
                                             replace=True)
        self.bootstrap_indices = bootstrap_indices
        self.items = [None for _ in range(maxsize)]
        self.items_score = [np.inf for _ in range(maxsize)]

    def update(self, population):
        for p in population:
            for i, s in enumerate(self.bootstrap_indices):
                score = np.sum(p.case_values[s])
                if score < self.items_score[i]:
                    self.items[i] = p
                    self.items_score[i] = score
