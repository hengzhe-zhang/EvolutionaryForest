import copy
from abc import abstractmethod
from inspect import isclass
from typing import List, Union

import numpy as np
import scipy.special
from deap.gp import Primitive, Terminal
from deap.tools import HallOfFame
from mlxtend.evaluate import feature_importance_permutation

from evolutionary_forest.component.evaluation import quick_result_calculation, get_cv_splitter
from evolutionary_forest.component.tree_utils import construct_tree, TreeNode, get_parent_of_leaves, \
    StringDecisionTreeClassifier
from evolutionary_forest.multigene_gp import MultiplePrimitiveSet, MultipleGeneGP
from evolutionary_forest.strategies.multiarm_bandit import MultiArmBandit


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def sum_by_group(array, x):
    i = 0
    while i < len(array):
        group_sum = np.sum(array[i:i + x], axis=0)
        for j in range(i, i + x):
            if j >= len(array):
                break
            array[j] = group_sum
        i += x
    return array


def merge_arrays(arrays):
    """
    There are n groups of arrays, each array represents the weight of each x_i,
    and each array has a corresponding array, which represents the specific index containing x_i.
    Now, this function combines these n arrays into one array, the value at each position is the average value of all arrays.
    """
    n = len(arrays)
    merged = {}
    counts = {}
    for i in range(n):
        weights = arrays[i][0]
        indices = arrays[i][1]
        for j in range(len(indices)):
            if indices[j] not in merged:
                merged[indices[j]] = weights[j]
                counts[indices[j]] = 1
            else:
                merged[indices[j]] += weights[j]
                counts[indices[j]] += 1
    for index in merged:
        merged[index] /= counts[index]
    return merged


class IEstimationOfDistribution:
    @abstractmethod
    def eda_lazy_init(self, X):
        """
        Lazily initializes the EDA with the given dataset.

        :param X: the input dataset.
        """
        pass

    @abstractmethod
    def permutation_importance_calculation(self, X, Y, individual):
        """
        Calculates the permutation importance of the given individual on the given dataset.

        :param X: the input dataset.
        :param Y: the target dataset.
        :param individual: the individual to evaluate.
        """
        pass

    @abstractmethod
    def update_decision_tree(self, best_pop):
        """
        Updates the decision tree with the best population.

        :param best_pop: the best population.
        """
        pass

    @abstractmethod
    def update_frequency_count(self, best_pop, factor, importance_weight, pset=None):
        """
        Updates the frequency count of the population.

        :param best_pop: the best population.
        :param factor: the scaling factor.
        :param importance_weight: the importance weight.
        :param pset: the primitive set.
        """
        pass

    @abstractmethod
    def frequency_counting(self, importance_weight=True):
        """
        Update the frequency count of primitives and terminals.

        :param importance_weight: A boolean value indicating whether to use importance weighting.
        """
        pass

    @abstractmethod
    def probability_elimination(self, terminal_counts):
        """
        Eliminate the probability of building blocks from the terminal counts.

        :param terminal_counts: A dictionary mapping terminals to their counts.
        """
        pass

    @abstractmethod
    def probability_smoothing(self, terminal_counts):
        """
        Smooth the probability of building blocks in the terminal counts.

        :param terminal_counts: A dictionary mapping terminals to their counts.
        """
        pass

    @abstractmethod
    def probability_sampling(self):
        """
        Update the probabilities of selecting primitives and terminals based on their frequency counts.
        """
        pass


class EstimationOfDistribution():
    mutation_scheme: str
    hof: HallOfFame
    primitive_prob: Union[np.ndarray, dict]
    terminal_prob: Union[np.ndarray, dict, StringDecisionTreeClassifier]
    pset: MultiplePrimitiveSet
    original_features: bool
    mgp_mode: bool
    layer_mgp: bool
    mgp_scope: int
    shared_eda: bool
    cv: int
    pop: List[MultipleGeneGP]
    X: np.ndarray
    y: np.ndarray
    ensemble_size: int
    verbose: bool
    gene_num: int

    """
    The optimal EDA operator is to share the EDA distribution
    """

    def __init__(self,
                 elite_ratio=0.5,
                 softmax_smoothing=False,
                 weighted_by_fitness=False,
                 weight_by_size=False,
                 no_building_blocks=False,
                 eda_archive_size=0,
                 decision_tree_mode=False,
                 multi_armed_bandit=False,
                 **params):
        # good individuals have larger weights
        self.weighted_by_fitness = weighted_by_fitness
        # each individual has equal weight, not matter the tree size
        self.weight_by_size = weight_by_size
        # no counting building blocks in MGP
        self.no_building_blocks = no_building_blocks
        # using softmax smoothing to smooth probabilities
        self.softmax_smoothing = softmax_smoothing
        self.elite_ratio = elite_ratio
        # an additional EDA archive is possible
        self.eda_archive_size = eda_archive_size
        if self.eda_archive_size > 0:
            self.eda_archive = HallOfFame(self.eda_archive_size)
        else:
            self.eda_archive = None
        # using decision tree
        self.decision_tree_mode = decision_tree_mode
        self.multi_armed_bandit = multi_armed_bandit
        self.mab = MultiArmBandit()

    def permutation_importance_calculation(self, X, Y, individual):
        # The correct way is to calculate the terminal importance based on the test data
        estimators = individual.estimators
        kcv = get_cv_splitter(estimators[0]['Ridge'], self.cv)
        all_importance_values = []
        for id, index in enumerate(kcv.split(X, Y)):
            def prediction_function(X):
                # quickly calculate permutation importance
                Yp = quick_result_calculation(individual.gene, self.pset, X, self.original_features)
                return estimators[id].predict(Yp)

            train_index, test_index = index
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            # get permutation importance
            importance_value = feature_importance_permutation(X_test, y_test, prediction_function, 'r2')[0]
            all_importance_values.append(importance_value)
        all_importance_values = np.mean(all_importance_values, axis=0)
        all_importance_values = np.clip(all_importance_values, 0, None)
        individual.terminal_importance_values = all_importance_values

    def update_decision_tree(self, best_pop):
        # construct a decision tree based on the best population
        X = []
        y = []
        weights = []
        for h in best_pop:
            for coef, gene in zip(h.coef, h.gene):
                tree: TreeNode = construct_tree(copy.deepcopy(gene))
                all_leaves = get_parent_of_leaves(tree)
                for leaf, parent in all_leaves.items():
                    if parent is None:
                        X.append('None')
                        y.append(leaf.val.name)
                    else:
                        X.append(parent.val.name)
                        y.append(leaf.val.name)
                    weights.append(coef)
        dt = StringDecisionTreeClassifier()
        dt.fit(np.array(X).reshape(-1, 1), y, sample_weight=weights)
        return dt

    def update_frequency_count(self, best_pop, factor, importance_weight, pset=None):
        if self.decision_tree_mode:
            dt = self.update_decision_tree(best_pop)
            self.primitive_prob = None
            self.terminal_prob = dt
            return None, dt
        weighted_by_fitness = self.weighted_by_fitness
        weight_by_size = self.weight_by_size

        # Improved version: using fitness value and mean feature importance
        # count the frequency of primitives and terminals
        primitive_dict = {}
        if pset is None:
            pset = self.pset
            pset_id = None
            primitive_prob_count = self.primitive_prob_count
            terminal_prob_count = self.terminal_prob_count
        else:
            pset_id = pset.pset_id
            primitive_prob_count = np.zeros(len(pset.primitives[object]))
            terminal_prob_count = np.zeros(len(pset.terminals[object]))
        constant_terminals = [index for index, var in enumerate(pset.terminals[object])
                              if not isinstance(var, Terminal)]
        assert len(constant_terminals) <= 1, constant_terminals

        for k in pset.primitives.keys():
            primitive_dict.update({v.name: k for k, v in enumerate(pset.primitives[k])})
        terminal_dict = {}
        for k in pset.terminals.keys():
            terminal_dict.update({v.name: k for k, v in enumerate(pset.terminals[k])})
        for h in best_pop:
            # R2 score in default, larger is better
            fitness = h.fitness.wvalues[0]
            if weighted_by_fitness:
                coef = h.mean_coef * max(fitness, 0)
            else:
                coef = h.coef
            for gid, importance, g in zip(range(0, len(h.gene)), coef, h.gene):
                if pset_id != None and gid != pset_id:
                    continue
                count_of_terminals = len(list(filter(lambda x: isinstance(x, Terminal), g)))
                assert count_of_terminals > 0, str(g)
                for x in g:
                    if importance_weight:
                        # weighted by the importance of feature
                        weight = importance
                    else:
                        weight = 1
                    weight *= factor
                    if weight_by_size:
                        # larger gene will have smaller weight
                        weight *= 1 / count_of_terminals
                    if isinstance(x, Primitive):
                        primitive_prob_count[primitive_dict[x.name]] += weight
                    elif isinstance(x, Terminal):
                        if x.name not in terminal_dict:
                            assert is_number(x.name)
                            # constant(position may not be one)
                            terminal_prob_count[constant_terminals[0]] += weight
                        else:
                            terminal_prob_count[terminal_dict[x.name]] += weight
                    else:
                        raise Exception
        return primitive_prob_count, terminal_prob_count

    def frequency_counting(self, importance_weight=True):
        if self.eda_archive is not None:
            self.eda_archive.update(self.pop)

        if self.mutation_scheme == 'EDA-Terminal-PMI':
            # Permutation importance based probability estimation (Slow!)
            for h in self.hof:
                if not hasattr(h, 'terminal_importance_values'):
                    self.permutation_importance_calculation(self.X, self.y, h)
            self.primitive_prob_count = np.ones(self.pset.prims_count)
            self.terminal_prob_count = np.mean([x.terminal_importance_values for x in self.hof], axis=0)
        else:
            # Feature Importance based on frequency
            self.primitive_prob_count = np.ones(self.pset.prims_count)
            self.terminal_prob_count = np.ones(self.pset.terms_count)
            if self.hof == None:
                return
            if self.mutation_scheme == 'EDA-Terminal-Balanced':
                # Separate individuals to two groups, good ones and bad ones
                positive_sample = list(sorted(self.hof, key=lambda x: x.fitness.wvalues[0]))[len(self.hof) // 2:]
                negative_sample = list(sorted(self.hof, key=lambda x: x.fitness.wvalues[0]))[:len(self.hof) // 2]
                self.update_frequency_count(positive_sample, 1, importance_weight)
                self.update_frequency_count(negative_sample, -1, importance_weight)
                min_max_scaling = lambda v: (v - v.min())
                self.primitive_prob_count = min_max_scaling(self.primitive_prob_count) + 1
                self.terminal_prob_count = min_max_scaling(self.terminal_prob_count) + 1
            else:
                # Check based on ensemble size instead of hall of fame.
                # Sometimes, hof is used for keeping redundant individuals.
                if self.ensemble_size == 1:
                    best_size = int(self.elite_ratio * len(self.pop))
                    if self.eda_archive is not None:
                        top_individuals = self.eda_archive
                    else:
                        top_individuals = sorted(self.pop, key=lambda x: x.fitness.wvalues[0],
                                                 reverse=True)[:best_size]
                    if self.mgp_mode:
                        terminal_counts = []
                        # MGP mode, has many primitives
                        for pset in self.pset.pset_list:
                            _, terminal_count = self.update_frequency_count(
                                top_individuals,
                                1, importance_weight, pset=pset
                            )
                            terminal_counts.append(terminal_count)
                        if self.layer_mgp:
                            # if layer-wise MGP, then we can merge information in each layer
                            terminal_counts = sum_by_group(terminal_counts, self.mgp_scope)
                        if self.shared_eda:
                            # probability vector should be comprehensive
                            # even if some features are missing on some locations
                            pset_list = [[t.__name__ if isclass(t) else t.name
                                          for (id, t) in enumerate(ps.terminals[object])]
                                         for ps in self.pset.pset_list]
                            pset_list = [(terminal_counts[pid], ps)
                                         for pid, ps in enumerate(pset_list)]
                            terminal_counts = merge_arrays(pset_list)

                            # do not estimate distribution for building blocks
                            if self.no_building_blocks:
                                self.probability_elimination(terminal_counts)

                            # smooth the probability of basic components
                            if self.softmax_smoothing:
                                self.probability_smoothing(terminal_counts)

                        if isinstance(terminal_counts, dict):
                            # if shared probability matrix, then normalize all values
                            count_sum = np.sum(list(terminal_counts.values()))
                            for k in terminal_counts.keys():
                                terminal_counts[k] = terminal_counts[k] / count_sum
                        if self.multi_armed_bandit:
                            if isinstance(terminal_counts, dict):
                                # shared probabilities
                                for k, v in terminal_counts.items():
                                    self.mab.update(k, v)
                                self.terminal_prob_count = self.mab.normalized_probability()
                            elif isinstance(terminal_counts, list):
                                if not isinstance(self.mab, list):
                                    self.mab = MultiArmBandit.create_multiple(len(terminal_counts))
                                for i, c in enumerate(terminal_counts):
                                    terminal_counts[i] = c / c.sum()
                                mabs = self.mab
                                MultiArmBandit.update_multiple(mabs, terminal_counts)
                                self.terminal_prob_count = MultiArmBandit.normalized_probability_multiple_list(mabs)
                                self.terminal_prob_count = np.array([np.array(x) for x in self.terminal_prob_count])
                            else:
                                raise Exception()
                        else:
                            self.terminal_prob_count = terminal_counts
                    else:
                        self.update_frequency_count(top_individuals, 1, importance_weight)
                else:
                    # ensemble learning mode
                    if self.mgp_mode:
                        raise ValueError("Unimplemented mgp mode!")
                    else:
                        self.update_frequency_count(self.hof, 1, importance_weight)

    def probability_elimination(self, terminal_counts):
        average_value = np.mean([terminal_counts[f"ARG{i}"] for i in range(self.X.shape[1])])
        for k in terminal_counts.keys():
            terminal_id = k.replace('ARG', '')
            # replace high-level terminals with average probability
            if is_number(terminal_id) and int(terminal_id) >= self.X.shape[1]:
                terminal_counts[k] = average_value

    def probability_smoothing(self, terminal_counts):
        sum = 0
        terminal_ids = []
        # smoothing building block parts to avoid some building blocks have zero probability
        for k in terminal_counts.keys():
            terminal_id = k.replace('ARG', '')
            if is_number(terminal_id) and int(terminal_id) >= self.X.shape[1]:
                terminal_ids.append(k)
                sum += terminal_counts[k]
        terminal_probs = scipy.special.softmax([terminal_counts[k] for k in terminal_ids])
        for p, k in zip(terminal_probs, terminal_ids):
            terminal_counts[k] = p * sum

    def probability_sampling(self):
        if self.decision_tree_mode:
            return
        # Sampling primitive and terminal nodes based on the probability matrix
        if self.mutation_scheme in ['EDA-Terminal']:
            self.primitive_prob /= np.sum(self.primitive_prob)
            self.terminal_prob += self.terminal_prob_count / np.sum(self.terminal_prob_count)
        elif self.mutation_scheme in ['EDA-Terminal-Balanced', 'EDA-Terminal-PMI']:
            self.primitive_prob[:] = self.primitive_prob_count / np.sum(self.primitive_prob_count)
            if self.mutation_scheme == 'EDA-Terminal-PMI':
                # add the probability of constant nodes
                terminal_prob = np.append(self.terminal_prob_count, np.mean(self.terminal_prob_count))
            else:
                terminal_prob = self.terminal_prob_count
            assert np.all(terminal_prob >= 0)
            if np.sum(terminal_prob) == 0:
                # When there are no important features
                self.terminal_prob[:] = np.full_like(terminal_prob, 1 / len(terminal_prob))
            else:
                self.terminal_prob[:] = terminal_prob / np.sum(terminal_prob)
        elif self.mutation_scheme.startswith('EDA-PM'):
            self.primitive_prob[:] = self.primitive_prob_count / np.sum(self.primitive_prob_count)
            self.terminal_prob[:] = self.terminal_prob_count / np.sum(self.terminal_prob_count)
            if self.verbose:
                print('primitive_prob', self.primitive_prob)
                print('terminal_prob', self.terminal_prob)
        elif self.mutation_scheme.startswith('EDA-Terminal-PM'):
            self.primitive_prob /= np.sum(self.primitive_prob)
            if isinstance(self.terminal_prob_count, dict):
                # if it is a dict don't do normalization
                self.terminal_prob = self.terminal_prob_count
            elif isinstance(self.terminal_prob_count[0], np.ndarray):
                terminal_counts = []
                for t in self.terminal_prob_count:
                    # normalize each vector
                    if np.sum(t) == 0:
                        # problematic probability values
                        t = np.full_like(t, fill_value=1 / len(t))
                    else:
                        t = t / np.sum(t)
                    terminal_counts.append(t)
                self.terminal_prob[:] = terminal_counts[:]
            else:
                self.terminal_prob[:] = self.terminal_prob_count / np.sum(self.terminal_prob_count)
            ## Eliminate trivial terminals
            # self.terminal_prob[self.terminal_prob < np.mean(self.terminal_prob) / 10] = 0
            # self.terminal_prob[self.terminal_prob > 0] = 1
            # self.terminal_prob[:] = self.terminal_prob / np.sum(self.terminal_prob)
        else:
            raise Exception
        if self.verbose:
            if isinstance(self.terminal_prob, list):
                print('Terminal Probability\n')
                print(self.terminal_prob[0])
                # for t in self.terminal_prob:
                #     print(t, '\n')
            else:
                print('Terminal Probability', self.terminal_prob)


if __name__ == '__main__':
    arrays = [([0.1, 0.2, 0.3], [1, 2, 3]),
              ([0.4, 0.5, 0.6], [2, 3, 4]),
              ([0.7, 0.8, 0.9], [3, 4, 5]),
              ]
    print(merge_arrays(arrays))
