import copy
import itertools
import random
import sys
from functools import wraps, partial
from inspect import isclass
from typing import Callable, List
from typing import TYPE_CHECKING

import numpy as np
from deap import base
from deap.gp import PrimitiveTree, compile, cxOnePoint, mutUniform, mutShrink, mutInsert, cxOnePointLeafBiased, \
    PrimitiveSet, Primitive, Terminal
from deap.tools import selTournament, selRandom
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tpot.base import TPOTBase

from evolutionary_forest.component.configuration import CrossoverConfiguration, MutationConfiguration, \
    MAPElitesConfiguration
from evolutionary_forest.component.crossover_mutation import intron_mutation, intron_crossover, cxOnePointWithRoot, \
    random_combination, cxOnePointSizeSafe, mutUniformSizeSafe
from evolutionary_forest.component.primitives import individual_to_tuple
from evolutionary_forest.component.syntax_tools import TransformerTool
from evolutionary_forest.component.tree_utils import StringDecisionTreeClassifier

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


class GPPipeline(Pipeline):
    active_gene: np.ndarray
    partition_scheme: np.ndarray


class FailureCounter:
    def __init__(self):
        self.value = 0


class IntronPrimitive(Primitive):
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq', 'corr', 'level', 'equal_subtree', 'hash_id')

    @property
    def intron(self):
        return self.corr < 0.01

    def __init__(self, name, args, ret):
        super().__init__(name, args, ret)
        self.corr = 0
        self.level = 0
        self.equal_subtree = -1
        self.hash_id = 0


class IntronTerminal(Terminal):
    __slots__ = ('name', 'value', 'ret', 'conv_fct', 'corr', 'level', 'hash_id')

    @property
    def intron(self):
        return self.corr < 0.01

    def __init__(self, terminal, symbolic, ret):
        super().__init__(terminal, symbolic, ret)
        self.corr = 0
        self.level = 0
        self.hash_id = 0


def selTournamentFeature(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=lambda x: x[1]))
    return chosen


class FitnessMin(base.Fitness):
    weights = (-1.0,)


class MultipleGeneGP():
    introns_results: List[dict]
    case_values: np.ndarray
    coef: np.ndarray
    hash_result: list
    mgp_mode: bool
    semantics: np.ndarray
    pipe: GPPipeline

    @property
    def gene_num(self):
        return len(self.gene)

    def gene_addition(self):
        if len(self.gene) < self.max_gene_num:
            self.gene.append(PrimitiveTree(self.content()))

    def gene_deletion(self):
        if len(self.gene) > 1:
            random_index = random.randrange(self.gene_num)
            del self.gene[random_index]

    def __init__(self, content, gene_num, tpot_model: TPOTBase = None, partition_scheme=None,
                 base_model_list=None, number_of_register=0, active_gene_num=0, intron_probability=0,
                 intron_threshold=0, algorithm: "EvolutionaryForestRegressor" = None):
        configuration = algorithm.mutation_configuration
        self.gene: List[PrimitiveTree] = []
        self.fitness = FitnessMin()
        self.mgp_mode = False
        self.layer_mgp = False
        self.mgp_scope = 0
        self.max_gene_num = gene_num
        self.active_gene_num = active_gene_num
        self.content = content
        if configuration.gene_addition_rate > 0:
            gene_num = random.randint(1, gene_num)
        self.tree_initialization(content, gene_num)
        if tpot_model != None:
            self.base_model = tpot_model._toolbox.individual()
        if base_model_list != None:
            self.base_model = random.choice(base_model_list.split(','))
        self.dynamic_leaf_size = random.randint(1, 10)
        self.dynamic_regularization = 1
        self.intron_threshold = intron_threshold
        if partition_scheme is not None:
            # self.partition_scheme = np.random.randint(0, partition_scheme[1] , partition_scheme[0])
            # initialize partition scheme by decision tree
            if callable(partition_scheme):
                self.partition_scheme = partition_scheme()
            else:
                self.partition_scheme = partition_scheme
        else:
            self.partition_scheme = None
        # some active genes
        self.intron_probability = intron_probability
        self.active_gene = np.random.randn(self.max_gene_num) < intron_probability

        # initialize some registers
        self.number_of_register = number_of_register
        if self.number_of_register > 0:
            self.register = np.random.randint(0, self.number_of_register, self.gene_num),
        else:
            self.register = None

        # self-adaptive evolution (for Lasso)
        self.parameters = {
            'Lasso': np.random.uniform(-5, -2, 1)[0]
        }
        self.parent_fitness: tuple[float] = None
        self.crossover_type = None

    def tree_initialization(self, content, gene_num):
        # This flag is only used for controlling the mutation and crossover
        for i in range(gene_num):
            pset: MultiplePrimitiveSet = content.keywords['pset']
            if isinstance(pset, MultiplePrimitiveSet):
                if hasattr(pset, 'layer_mgp') and hasattr(pset, 'mgp_scope'):
                    self.layer_mgp = pset.layer_mgp
                    self.mgp_scope = pset.mgp_scope
                    self.mgp_mode = True
                self.gene.append(PrimitiveTree(content(pset=pset.pset_list[i])))
            else:
                self.gene.append(PrimitiveTree(content()))

    def random_select_index(self):
        return random.randint(0, self.gene_num - 1)

    def random_select(self, with_id=False):
        if with_id:
            id = random.randint(0, len(self.gene) - 1)
            return self.gene[id], id
        else:
            if self.intron_threshold > 0:
                return self.gene[random.choice(list(filter(lambda id: self.coef[id] > self.intron_threshold,
                                                           range(0, self.gene_num))))]
            else:
                return self.gene[random.randint(0, len(self.gene) - 1)]

    def tournament_selection(self, tournsize=3, reverse=False):
        """
        """
        coef = np.abs(self.coef)
        if reverse:
            key = selTournamentFeature(list(enumerate(-1 * coef)), 1, tournsize=tournsize)
        else:
            key = selTournamentFeature(list(enumerate(coef)), 1, tournsize=tournsize)
        return self.gene[key[0][0]]

    def best_gene(self, with_id=False):
        best_index = max(range(len(self.gene)), key=lambda x: np.abs(self.coef)[x])
        if with_id:
            return self.gene[best_index], best_index
        else:
            return self.gene[best_index]

    def worst_gene(self, with_id=False):
        worst_index = min(range(len(self.gene)), key=lambda x: np.abs(self.coef)[x])
        if with_id:
            return self.gene[worst_index], worst_index
        else:
            return self.gene[worst_index]

    def replace_worst_gene(self, gene):
        worst_index = min(range(len(self.gene)), key=lambda x: np.abs(self.coef)[x])
        self.gene[worst_index] = gene

    def softmax_selection(self, reverse=False, temperature=1 / 20, with_id=False):
        if temperature == 0:
            weight = np.abs(self.coef)
        else:
            if reverse:
                weight = softmax(-(1 / temperature) * np.abs(self.coef))
            else:
                weight = softmax((1 / temperature) * np.abs(self.coef))
        if weight.sum() == 0:
            # If there is an error in the feature importance vector
            index = np.random.choice(np.arange(len(self.gene)), 1)
        else:
            weight = weight / weight.sum()
            index = np.random.choice(np.arange(len(self.gene)), 1, p=weight)
        if with_id:
            return self.gene[index[0]], index[0]
        else:
            return self.gene[index[0]]

    def weighted_selection(self, reverse=False, with_id=False):
        weight = np.abs(self.coef)
        p = weight / np.sum(weight)
        if reverse:
            p[p == 0] = p[p > 0].min()
            p = 1 / p
            index = np.random.choice(np.arange(len(weight)), p=p / p.sum())
        else:
            index = np.random.choice(np.arange(len(weight)), p=p)
        if with_id:
            return self.gene[index], index
        else:
            return self.gene[index]

    def replace_weight_gene_inverse(self, gene):
        index = np.random.choice(np.arange(len(self.gene)), 1, p=softmax(-1 * np.abs(self.coef)))
        self.gene[index] = gene

    def deterministic_select(self):
        weight = np.abs(self.coef)
        return self.gene[np.argmax(-weight)]

    def __len__(self):
        return sum([len(g) for g in self.gene])

    def __str__(self):
        return str([str(g) for g in self.gene])

    def __repr__(self):
        return str([str(g) for g in self.gene])


def get_cross_point(gene, inverse=False, min_tournament_size=1):
    tournsize = min(max(min_tournament_size, round(0.1 * len(gene))), len(gene))
    aspirants = selRandom(list([(k, getattr(g, 'corr', 0)) for k, g in enumerate(gene)]), tournsize)
    if inverse:
        point = min(aspirants, key=lambda x: x[1])
    else:
        point = max(aspirants, key=lambda x: x[1])
    return point[0]


def get_random_from_interval(x, c):
    interval_start = (x // c) * c
    interval_end = ((x // c) + 1) * c
    return random.randint(interval_start, interval_end - 1)


def cxOnePoint_multiple_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP,
                             pset=None, crossover_configuration: CrossoverConfiguration = None):
    """
    :param ind1: parent A
    :param ind2: parent B
    :param intron_parameters: using intron-aware operator
    :return:
    """
    if crossover_configuration is None:
        crossover_configuration = CrossoverConfiguration()
    intron_parameters = crossover_configuration.intron_parameters
    same_index = crossover_configuration.same_index

    if intron_parameters is not None and random.random() < intron_parameters.get("intron_crossover_pb", 0):
        gene1, id1 = ind1.random_select(with_id=True)
        gene2, id2 = ind2.random_select(with_id=True)
        if intron_parameters.get('exon_tournament', False):
            # Warning: After mutation, there may exist some undefined parameters
            introns_a = [get_cross_point(gene1)]
            introns_b = [get_cross_point(gene2)]
            intron_crossover(gene1, gene2, introns_a, introns_b, cross_intron=False, avoid_fallback=True)
        elif intron_parameters.get('exon_tournament_V2', False):
            # Warning: After mutation, there may exist some undefined parameters
            introns_a = [get_cross_point(gene1, min_tournament_size=2)]
            introns_b = [get_cross_point(gene2, min_tournament_size=2)]
            intron_crossover(gene1, gene2, introns_a, introns_b, cross_intron=False, avoid_fallback=True)
        elif intron_parameters.get("intron_crossover", False):
            # Using exons to replace introns
            introns_a, introns_b = get_intron_id(gene1), get_intron_id(gene2)
            intron_crossover(gene1, gene2, introns_a, introns_b, cross_intron=True)
        elif intron_parameters.get("random_combination", False):
            random_combination(gene1, gene2, pset)
        elif intron_parameters.get("exon_crossover", False):
            introns_a, introns_b = get_intron_id(gene1), get_intron_id(gene2)
            intron_crossover(gene1, gene2,
                             list(filter(lambda x: x not in introns_a, range(0, len(gene1)))),
                             list(filter(lambda x: x not in introns_b, range(0, len(gene2)))))
        else:
            raise Exception
    else:
        if ind1.mgp_mode == True:
            gene1, id1 = ind1.random_select(with_id=True)
            if ind1.layer_mgp:
                gene2 = get_random_from_interval(id1, ind1.mgp_scope)
                gene2 = ind2.gene[min(gene2, len(ind2.gene) - 1)]
            else:
                gene2 = ind2.gene[id1]
            # must be the same index
            id2 = id1
        else:
            if crossover_configuration.tree_selection == 'Random':
                gene1, id1 = ind1.random_select(with_id=True)
                if same_index:
                    gene2 = ind2.gene[id1]
                    id2 = id1
                else:
                    gene2, id2 = ind2.random_select(with_id=True)
            elif crossover_configuration.tree_selection == 'Self-Competitive':
                gene1_a, id1_a = ind1.best_gene(with_id=True)
                gene2_a, id2_a = ind2.best_gene(with_id=True)
                gene1_b, id1_b = ind1.worst_gene(with_id=True)
                gene2_b, id2_b = ind2.worst_gene(with_id=True)
                gene1_a = copy.deepcopy(gene1_a)
                gene2_a = copy.deepcopy(gene2_a)
                gene1_b = copy.deepcopy(gene1_b)
                gene2_b = copy.deepcopy(gene2_b)
                # mutate worst, preserve best
                ind1.gene[id1_b], _ = gene_crossover(gene1_b, gene2_a,
                                                     configuration=crossover_configuration)
                _, ind2.gene[id2_b] = gene_crossover(gene1_a, gene2_b,
                                                     configuration=crossover_configuration)
                return ind1, ind2
            elif crossover_configuration.tree_selection == 'Probability':
                gene1, id1 = ind1.weighted_selection(with_id=True)
                gene2, id2 = ind2.weighted_selection(with_id=True)
            else:
                raise Exception
        ind1.gene[id1], ind2.gene[id2] = gene_crossover(gene1, gene2, configuration=crossover_configuration)
    return ind1, ind2


def gene_crossover(gene1, gene2, configuration: CrossoverConfiguration):
    if configuration.safe_crossover:
        gene1, gene2 = cxOnePointSizeSafe(gene1, gene2, configuration)
    elif configuration.root_crossover:
        gene1, gene2 = cxOnePointWithRoot(gene1, gene2, configuration)
    else:
        gene1, gene2 = cxOnePoint(gene1, gene2)
    return gene1, gene2


# Perform mutation that takes a MultipleGeneGP individual and a primitive set as inputs
def mutProbability_multiple_gene(individual: MultipleGeneGP, pset, parsimonious_probability=0.9):
    # Get the frequency vector of the terminals in the individual using a helper function
    terminal_prob = get_frequency_vector(individual, pset)

    # Sampling: choose between a full tree with only terminals with non-zero frequency or a full tree with any terminals
    if random.random() < parsimonious_probability:
        sub_tree = partial(genFull_with_prob, pset=pset, min_=0, max_=2,
                           terminal_probs=(terminal_prob > 0))
    else:
        sub_tree = partial(genFull_with_prob, pset=pset, min_=0, max_=2)

    # Mutation: select a random gene from the individual and apply the mutation using the sub_tree function
    gene, id = individual.random_select(with_id=True)
    mutUniform(gene, sub_tree, pset)
    return individual,


def get_frequency_vector(individual, pset):
    terminal_prob = np.zeros(pset.terms_count)
    terminal_dict = {}
    for k in pset.terminals.keys():
        terminal_dict.update({v.name: k for k, v in enumerate(pset.terminals[k])})
    # Get used variables
    for g in individual.gene:
        fitness = individual.fitness.wvalues[0]
        for coef, node in zip(individual.coef, g):
            if isinstance(node, Terminal):
                if node.name not in terminal_dict:
                    terminal_prob[-1] += coef * fitness
                else:
                    terminal_prob[terminal_dict[node.name]] += coef * fitness
    terminal_prob = terminal_prob / np.sum(terminal_prob)
    return terminal_prob


def mutUniform_multiple_gene(individual: MultipleGeneGP, expr, pset,
                             tree_generation=None, configuration=None):
    if configuration is None:
        configuration = MutationConfiguration()
    intron_parameters = configuration.intron_parameters
    if random.random() < configuration.gene_addition_rate:
        individual.gene_addition()
        return individual,
    elif random.random() < configuration.gene_deletion_rate:
        individual.gene_deletion()
        return individual,

    if intron_parameters is not None and random.random() < intron_parameters.get("intron_mutation_pb", 0):
        gene, id = individual.random_select(with_id=True)

        if intron_parameters.get('exon_tournament', False):
            introns = [get_cross_point(gene, inverse=True)]
            intron_mutation(gene, expr, pset, introns)
        elif intron_parameters.get('exon_tournament_V2', False):
            introns = [get_cross_point(gene, inverse=True, min_tournament_size=2)]
            intron_mutation(gene, expr, pset, introns)
        elif intron_parameters.get('mutation_worst', False):
            introns = [min(list([(k, getattr(g, 'corr', 0)) for k, g in enumerate(gene)]), key=lambda x: x[1])[0]]
            intron_mutation(gene, expr, pset, introns)
        elif intron_parameters.get('intron_mutation', False):
            introns = get_intron_id(gene)
            intron_mutation(gene, expr, pset, introns)
        elif intron_parameters.get('insert_mutation', False):
            mutInsert(gene, pset)
        elif intron_parameters.get('exon_mutation', False):
            # only mutate exons
            introns = get_intron_id(gene)
            list_of_exons = list(filter(lambda x: x not in introns, range(0, len(gene))))
            intron_mutation(gene, expr, pset, list_of_exons)
        else:
            raise Exception
    else:
        if isinstance(pset, MultiplePrimitiveSet):
            gene, id = individual.random_select(with_id=True)
            gene_mutation(gene, pset.pset_list[id], expr, tree_generation, configuration)
        else:
            gene, id = individual.random_select(with_id=True)
            gene_mutation(gene, pset, expr, tree_generation, configuration)
    if hasattr(individual, 'introns_results'):
        individual.introns_results[id].clear()
    return individual,


def gene_mutation(gene, pset, expr, tree_generation, configuration):
    if configuration.safe_mutation:
        genes = mutUniformSizeSafe(gene, tree_generation, pset, configuration)
    else:
        genes = mutUniform(gene, expr, pset)
    return genes


def multiple_gene_evaluation(compiled_genes, x):
    result = []
    for gene in compiled_genes:
        result.append(gene(*x))
    return result


def multiple_gene_initialization(cls, generator, gene_num, **kwargs):
    return cls(generator, gene_num, **kwargs)


def multiple_gene_compile(expr: MultipleGeneGP, pset):
    """
    """
    gene_compiled = []
    if hasattr(expr, 'active_gene_num') and expr.active_gene_num > 0:
        # some genes may not be activated
        for gene in expr.gene[:expr.active_gene_num]:
            gene_compiled.append(compile(gene, pset))
    else:
        for gene in expr.gene:
            gene_compiled.append(compile(gene, pset))
    return gene_compiled


def pearson_check(x, y):
    if np.abs(pearsonr(x, y)[0]) < 0.95:
        return True
    else:
        return False


def cxOnePoint_multiple_gene_novelty(ind1: MultipleGeneGP, ind2: MultipleGeneGP, test_func):
    a = ind1.random_select()
    b = ind2.random_select()
    x, y = test_func(a), test_func(b)
    while True:
        cxOnePoint(a, b)
        z, w = test_func(a), test_func(b)
        if pearson_check(x, z) and pearson_check(y, z) and pearson_check(x, w) and pearson_check(y, w):
            break
    return ind1, ind2


def cxOnePoint_multiple_gene_diversity(ind1: MultipleGeneGP, ind2: MultipleGeneGP, visited_features: set):
    while True:
        a = ind1.random_select()
        b = ind2.random_select()
        cxOnePoint(a, b)
        if individual_to_tuple(a) in visited_features or individual_to_tuple(b) in visited_features:
            continue
    return ind1, ind2


def get_intron_id(gene):
    introns_results = []
    for id, g in enumerate(gene):
        if isinstance(g, (IntronPrimitive, IntronTerminal)) and g.intron:
            introns_results.append(id)
    return introns_results


def cxOnePoint_multiple_gene_tournament(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    """
    A feature importance aware crossover operator based on the tournament selection
    """
    cxOnePoint(ind1.tournament_selection(), ind2.tournament_selection())
    return ind1, ind2


def selTournamentGenePool(coef_list, tournsize=7):
    individuals = [i for i in range(len(coef_list))]
    aspirants = selRandom(individuals, tournsize)
    return max(aspirants, key=lambda x: coef_list[x])


def selRouletteGenePool(coef_list):
    individuals = [i for i in range(len(coef_list))]
    s_inds = sorted(individuals, key=lambda x: coef_list[x], reverse=True)
    sum_fits = sum(coef_list[ind] for ind in individuals)
    u = random.random() * sum_fits
    sum_ = 0
    for ind in s_inds:
        sum_ += coef_list[ind]
        if sum_ >= u:
            return ind
    raise Exception('No solution found!')


def cxOnePoint_multiple_gene_pool(ind1: MultipleGeneGP, ind2: MultipleGeneGP,
                                  remove_zero_features=False):
    """
    Pay attention to remove redundant features and irrelevant features.
    """
    ind1_list, ind2_list = [], []
    ind1_set, ind2_set = set(), set()
    while len(ind1_list) < len(ind1.gene) or len(ind2_list) < len(ind2.gene):
        coef_list = np.array(list(ind1.coef) + list(ind2.coef))
        gene_list = ind1.gene + ind2.gene
        if remove_zero_features:
            # Remove completely irrelevant features
            gene_list = list(itertools.compress(ind1.gene + ind2.gene, coef_list > 0))
            coef_list = coef_list[coef_list > 0]
        a, b = selTournamentGenePool(coef_list), selTournamentGenePool(coef_list)
        if (a, b) in ind1_set:
            # Remove potential redundant features
            continue
        else:
            ind1_set.add((a, b))
        a, b = copy.deepcopy(gene_list[a]), copy.deepcopy(gene_list[b])
        cxOnePoint(a, b)
        if str(a) not in ind1_set and len(ind1_list) < len(ind1.gene):
            ind1_list.append(a)
            ind1_set.add(str(a))
        if str(b) not in ind2_set and len(ind2_list) < len(ind2.gene):
            ind2_list.append(b)
            ind2_set.add(str(b))
    ind1.gene = ind1_list
    ind2.gene = ind2_list
    return ind1, ind2


def cxOnePoint_multiple_gene_SC(ind1: MultipleGeneGP, ind2: MultipleGeneGP,
                                crossover_configuration: CrossoverConfiguration):
    """
    self-competitive crossover operator
    Using the worst individual as the base individual and migrate a portion of useful materials from well-behaved one
    """
    temperature = crossover_configuration.sc_temperature
    gene_crossover(ind1.softmax_selection(reverse=True, temperature=temperature),
                   copy.deepcopy(ind2.softmax_selection(temperature=temperature)),
                   crossover_configuration)
    gene_crossover(ind2.softmax_selection(reverse=True, temperature=temperature),
                   copy.deepcopy(ind1.softmax_selection(temperature=temperature)),
                   crossover_configuration)
    return ind1, ind2


def cxOnePoint_multiple_gene_TSC(ind1: MultipleGeneGP, ind2: MultipleGeneGP,
                                 crossover_configuration: CrossoverConfiguration):
    """
    tournament-based self-competitive crossover operator
    Using the worst individual as the base individual and migrate a portion of useful materials from well-behaved one
    """
    tournsize = crossover_configuration.sc_tournament_size
    gene_crossover(ind1.tournament_selection(tournsize, reverse=True),
                   copy.deepcopy(ind2.tournament_selection(tournsize)),
                   crossover_configuration)
    gene_crossover(ind2.tournament_selection(tournsize, reverse=True),
                   copy.deepcopy(ind1.tournament_selection(tournsize)),
                   crossover_configuration)
    return ind1, ind2


def cxOnePoint_multiple_gene_SBC(ind1: MultipleGeneGP, ind2: MultipleGeneGP, temperature=1 / 20):
    """
    self-competitive biased crossover operator
    """
    a_bad, a_bad_id = ind1.softmax_selection(reverse=True, temperature=temperature, with_id=True)
    b_good = copy.deepcopy(ind2.softmax_selection(temperature=temperature))
    cxOnePoint(a_bad, b_good)
    ind1.gene[a_bad_id] = b_good

    b_bad, b_bad_id = ind2.softmax_selection(reverse=True, temperature=temperature, with_id=True)
    a_good = copy.deepcopy(ind1.softmax_selection(temperature=temperature))
    cxOnePoint(b_bad, a_good)
    ind2.gene[b_bad_id] = a_good
    return ind1, ind2


def cxOnePoint_multiple_gene_same_index(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Crossover on the same location
    index = random.randint(0, ind1.gene_num - 1)
    cxOnePoint(ind1.gene[index], ind2.gene[index])
    return ind1, ind2


def cxOnePoint_multiple_gene_biased(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePointLeafBiased(ind1.random_select(), ind2.random_select(), 0.1)
    return ind1, ind2


def cxOnePoint_multiple_gene_same_weight(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Only cross features with same weight index
    index = random.randint(0, ind1.gene_num - 1)
    cxOnePoint(ind1.gene[np.argsort(ind1.coef)[index]], ind2.gene[np.argsort(ind2.coef)[index]])
    return ind1, ind2


def cxOnePoint_multiple_gene_best(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.best_gene(), ind2.best_gene())
    return ind1, ind2


def cxOnePoint_multiple_gene_worst(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.worst_gene(), ind2.worst_gene())
    return ind1, ind2


# extract all useful features from individuals
def extract_features(ind: MultipleGeneGP, useful=True, threshold=None):
    results = []
    if threshold is not None:
        mean_coef = threshold * np.mean(ind.coef)
    else:
        mean_coef = np.mean(ind.coef)
    for i, x in zip(ind.gene, ind.coef):
        if useful and x >= mean_coef:
            results.append(i)
        if not useful and x < mean_coef:
            results.append(i)
    return results


def feature_to_tuple(x):
    return tuple(a.name for a in x)


def cxOnePoint_multiple_gene_threshold(ind1: MultipleGeneGP, ind2: MultipleGeneGP, mutation: Callable,
                                       cross_pb: float, threshold: float):
    # extract all useful features from individuals for crossover
    useful_features_a, useful_features_b = extract_features(ind1, threshold=threshold), \
        extract_features(ind2, threshold=threshold)
    # replace useless features with useful features
    all_features = set(feature_to_tuple(x) for x in extract_features(ind1, False, threshold) +
                       extract_features(ind2, False, threshold))

    # print('Count Useful Features', len(useful_features_a) + len(useful_features_b))
    # print('Count Useless Features', len(all_features))

    def replace_useless_features(ind):
        # generated_features = set(feature_to_tuple(x) for x in extract_features(ind))
        generated_features = set()
        # replace useless feature with useful features
        for i in range(len(ind.gene)):
            variation = True
            # check whether a feature is a useless feature
            while feature_to_tuple(ind.gene[i]) in all_features or variation:
                variation = False
                a, b = random.choice(useful_features_a), random.choice(useful_features_b)
                # if random.random() < (len(useful_features_a) + len(useful_features_b)) / all_features:
                if random.random() < cross_pb:
                    gene1, gene2 = cxOnePoint(copy.deepcopy(a), copy.deepcopy(b))
                else:
                    gene1, gene2 = mutation(copy.deepcopy(a))[0], mutation(copy.deepcopy(b))[0]
                # ensure the generated features are not redundant features
                l = list(filter(lambda x: (feature_to_tuple(x) not in all_features)
                                          and
                                          (feature_to_tuple(x) not in generated_features)
                                , [gene1, gene2]))
                if len(l) > 0:
                    ind.gene[i] = random.choice(l)
                    generated_features.add(feature_to_tuple(ind.gene[i]))

    replace_useless_features(ind1)
    replace_useless_features(ind2)
    return ind1, ind2


def cxOnePoint_multiple_gene_cross(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    gene1 = copy.deepcopy(ind1.best_gene())
    gene2 = copy.deepcopy(ind2.best_gene())
    gene1, gene2 = cxOnePoint(gene1, gene2)
    ind1.replace_worst_gene(gene1)
    ind2.replace_worst_gene(gene2)
    return ind1, ind2


def cxOnePoint_all_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP, permutation=False):
    if permutation:
        a_list = np.random.permutation(np.arange(0, len(ind1.gene)))
        b_list = np.random.permutation(np.arange(0, len(ind2.gene)))
    else:
        a_list = np.arange(0, len(ind1.gene))
        b_list = np.arange(0, len(ind2.gene))
    for a, b in zip(a_list, b_list):
        cxOnePoint(ind1.gene[a], ind2.gene[b])
    return ind1, ind2


def cxOnePoint_all_gene_with_importance_probability(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Each tree has a probability to be varied
    probability = (ind1.coef + ind2.coef) / 2
    a_list = np.arange(0, len(ind1.gene))
    b_list = np.arange(0, len(ind2.gene))
    for i, a, b in zip(range(len(a_list)), a_list, b_list):
        if random.random() < probability[i]:
            cxOnePoint(ind1.gene[a], ind2.gene[b])
    return ind1, ind2


def cxOnePoint_all_gene_with_probability(ind1: MultipleGeneGP, ind2: MultipleGeneGP, probability):
    # Each tree has a probability to be varied
    a_list = np.arange(0, len(ind1.gene))
    b_list = np.arange(0, len(ind2.gene))
    for a, b in zip(a_list, b_list):
        if random.random() < probability:
            cxOnePoint(ind1.gene[a], ind2.gene[b])
            # cxOnePointLeafBiased(ind1.gene[a], ind2.gene[b], 0.1)
    return ind1, ind2


def mutUniform_multiple_gene_with_probability(individual: MultipleGeneGP, expr, pset, probability):
    for g in individual.gene:
        if random.random() < probability:
            mutUniform(g, expr, pset)
    return individual,


def mutShrink_multiple_gene(individual: MultipleGeneGP, expr, pset):
    if random.random() < 0.5:
        mutUniform(individual.random_select(), expr, pset)
    else:
        mutShrink(individual.random_select())
    return individual,


# transformer-based mutation
def mutUniform_multiple_gene_transformer(individual: MultipleGeneGP, expr, pset,
                                         condition_probability: Callable[[], float],
                                         transformer: TransformerTool):
    if random.random() > condition_probability():
        return mutUniform_multiple_gene(individual, expr, pset)
    else:
        ind = transformer.sample(1)
        individual.replace_worst_gene(PrimitiveTree(ind[0]))
        return individual,


def mutUniform_multiple_gene_worst(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.worst_gene(), expr, pset)
    return individual,


def mutUniform_multiple_gene_threshold(individual: MultipleGeneGP, expr, pset):
    s = sum([1 - c for c in individual.coef])
    for g, c in zip(individual.gene, individual.coef):
        if random.random() < (1 - c) / s:
            mutUniform(individual.worst_gene(), expr, pset)
    return individual,


def mutUniform_multiple_gene_with_prob(individual: MultipleGeneGP, expr, pset, terminal_probs, primitive_probs):
    root_individual = individual
    individual = individual.random_select()
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_, terminal_probs=terminal_probs, primitive_probs=primitive_probs)
    return root_individual,


def mutInsert_multiple_gene(individual: MultipleGeneGP, pset):
    mutInsert(individual.random_select(), pset)
    return individual,


def cxOnePoint_multiple_gene_weighted(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Potential issue: This operator may overly cross important features and ignore useless features
    cxOnePoint(ind1.weighted_selection(), ind2.weighted_selection())
    return ind1, ind2


def mutWeight_multiple_gene(individual: MultipleGeneGP, expr, pset, threshold_ratio=0.2):
    good_features, threshold = construct_feature_pools([individual], True, threshold_ratio=threshold_ratio)

    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            positive = False
            if (positive and c >= threshold) or (not positive and c < threshold):
                new_features = mutUniform(copy.deepcopy(random.choice(good_features)), expr, pset)
                ind.gene[i] = new_features[0]

    replaces_features(individual)
    return individual,


def construct_feature_pools(pop, positive, threshold_ratio=0.2,
                            good_features_threshold=None):
    # positive: get all important features
    # negative: get all unimportant features
    good_features = []
    good_features_str = set()
    if good_features_threshold == None:
        threshold = np.quantile([ind.coef for ind in pop], threshold_ratio)
    elif good_features_threshold == 'mean':
        threshold = np.mean([ind.coef for ind in pop])
    else:
        # threshold for good features
        threshold = np.quantile([ind.coef for ind in pop], good_features_threshold)

    def add_features(ind):
        for c, x in zip(ind.coef, ind.gene):
            if (positive and c >= threshold) or (not positive and c < threshold):
                if str(x) in good_features_str:
                    continue
                # assign a fitness value for each feature
                x.fitness = c
                good_features.append(x)
                good_features_str.add(str(x))

    for ind in pop:
        add_features(ind)

    # calculate the threshold for crossover
    threshold = np.quantile([ind.coef for ind in pop], threshold_ratio)
    return good_features, threshold


def feature_crossover_cross(ind1, ind2, threshold_ratio):
    pop = [ind1, ind2]
    good_features, threshold = construct_feature_pools(pop, True, threshold_ratio=threshold_ratio)
    new_pop = []
    for ind in pop:
        ind = cxOnePoint_multiple_gene_weight_plus(ind, good_features, threshold, False)
        new_pop.append(ind)
    return new_pop


def feature_crossover_cross_global(ind1, ind2, regressor):
    pop = [ind1, ind2]
    new_pop = []
    for ind in pop:
        ind = cxOnePoint_multiple_gene_weight_plus(ind, regressor.good_features, regressor.cx_threshold, False)
        new_pop.append(ind)
    return new_pop


def feature_mutation_global(individual: MultipleGeneGP, expr, pset, regressor):
    threshold = regressor.cx_threshold

    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            if c < threshold:
                new_features = mutUniform(copy.deepcopy(random.choice(regressor.good_features)), expr, pset)
                ind.gene[i] = new_features[0]

    replaces_features(individual)
    return individual,


def pool_based_mutation(individual: MultipleGeneGP, expr, pset, regressor, pearson_selection=False,
                        feature_evaluation=None, tournament_size=0):
    # construct features from the pool of good features
    # in addition to that, we force new features are not equivalent to old features and useless features
    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            if str(ind.gene[i]) in regressor.generated_features or c < regressor.cx_threshold:
                # new_features = copy.deepcopy(ind.gene[i])
                while True:
                    def feature_selection():
                        if tournament_size == 0:
                            return random.choice(regressor.good_features)
                        else:
                            return selTournament(regressor.good_features, tournament_size, 1)[0]

                    if random.random() < 0.2:
                        # mutation (in order to deal with the case of infinite loop)
                        new_features = mutUniform(copy.deepcopy(feature_selection()),
                                                  expr, pset)[0]
                    else:
                        # crossover
                        new_features = cxOnePoint(copy.deepcopy(feature_selection()),
                                                  copy.deepcopy(feature_selection()))
                        new_features = random.choice(new_features)
                    regressor.repetitive_feature_count[-1] += 1
                    if not str(new_features) in regressor.generated_features:
                        # Pre-selection by Pearson correlation to ensure the synthesized feature is useful
                        # However, such a process might be misleading
                        if pearson_selection:
                            func = compile(new_features, pset)
                            Yp = result_calculation([func], regressor.X[:20], False).flatten()
                            if np.abs(pearsonr(Yp, regressor.y[:20])[0]) <= 0.05:
                                # useless features
                                regressor.generated_features.add(str(new_features))
                            else:
                                break
                        elif feature_evaluation != None:
                            # using semantic diversity when generating new features
                            y = feature_evaluation(new_features)
                            if not y in regressor.generated_features:
                                break
                        else:
                            break
                ind.gene[i] = new_features
            # regressor.generated_features.add(str(new_features))

    replaces_features(individual)
    return individual,


def map_elites_selection(data, scores, k, map_elites_configuration, bins=10):
    """
    Selects k elements from a Mx2 matrix using map-elites algorithm.

    Parameters:
    -----------
    data : numpy array of shape (M, 2)
        The input data to be selected from.
    scores : dictionary of length M
        The scores of the input data, indexed by element indices.
    k : int
        The number of elements to select.
    bins : int, optional (default=10)
        The number of bins to divide the space into.

    Returns:
    --------
    elite_indices : numpy array of shape (k,)
        The indices of the selected elite data.
    """
    # Create an empty grid of bins to store elites in each bin
    elite_bins = np.empty((bins, bins), dtype=np.object)
    for i in range(bins):
        for j in range(bins):
            elite_bins[i, j] = []

    # Calculate the range of each bin
    min_x, min_y = np.min(data, axis=0)
    max_x, max_y = np.max(data, axis=0)
    x_margin = 1e-6 * (max_x - min_x)
    y_margin = 1e-6 * (max_y - min_y)
    x_range = (max_x + x_margin - min_x) / bins
    y_range = (max_y + y_margin - min_y) / bins

    # Place each data point into the corresponding bin
    for idx, d in enumerate(data):
        # in very rare cases, x_range and y_range could be zero
        if x_range > 0:
            x_bin = int((d[0] - min_x) // x_range)
            x_bin = min(x_bin, bins - 1)
        else:
            x_bin = 0
        if y_range > 0:
            y_bin = int((d[1] - min_y) // y_range)
            y_bin = min(y_bin, bins - 1)
        else:
            y_bin = 0
        elite_bins[x_bin, y_bin].append((idx, scores[idx]))

    # Select the best elites from each bin
    elites = []
    for i in range(bins):
        for j in range(bins):
            bin_elites = sorted(elite_bins[i, j], key=lambda x: x[1], reverse=True)
            elites.extend(bin_elites[:1])

    if map_elites_configuration.map_elites_random_sample:
        # Randomly select k elites from the overall set
        if len(elites) > k:
            elites = random.sample(elites, k)
        else:
            elites = elites

        # Return the selected elite data
        return np.array([x[0] for x in elites])
    else:
        elites.sort(key=lambda x: x[1], reverse=True)

        # Return the selected elite data
        return np.array([x[0] for x in elites])


def mapElitesCrossover(ind1: MultipleGeneGP, ind2: MultipleGeneGP, target: np.ndarray,
                       map_elites_configuration: MAPElitesConfiguration):
    importance_result, semantic_result = get_semantic_results(ind1, ind2, target)
    kpca = Pipeline(
        [
            ('Standardization', StandardScaler(with_mean=False)),
            ('KPCA', KernelPCA(kernel='cosine', n_components=2))
        ]
    )

    kpca_semantics = kpca.fit_transform(semantic_result)
    assert not np.any(np.isnan(kpca_semantics)), "No KPCA semantics should be nan!"
    selected = []
    while len(selected) < len(ind1.gene):
        remains = len(ind1.gene) - len(selected)
        result = map_elites_selection(kpca_semantics, importance_result, len(ind1.gene),
                                      map_elites_configuration)
        selected.extend(result[:remains])
    ind1.gene = [(ind1.gene + ind2.gene)[i] for i in selected]
    return [ind1]


def semanticFeatureCrossover(ind1: MultipleGeneGP, ind2: MultipleGeneGP, target: np.ndarray):
    importance_result, semantic_result = get_semantic_results(ind1, ind2, target)
    distances = 1 - pairwise_distances(semantic_result, metric='cosine')

    # remove duplicate features
    to_remove = []
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[i][j] >= 1:
                if i < j:
                    to_remove.append(j)
                else:
                    to_remove.append(i)

    n = importance_result.shape[0]
    indices = np.arange(n)
    # remove some duplicate items
    indices = np.setdiff1d(indices, to_remove)
    for i in to_remove:
        distances[i, :] = np.inf
        distances[:, i] = np.inf
    selected = []
    for i in range(0, ind1.gene_num, 2):
        # get probability
        probs = importance_result[indices]
        if len(probs) == 0 or np.sum(probs) == 0:
            return []
        probs /= np.sum(probs)

        # sampling
        i = np.random.choice(indices, p=probs)
        distances[i, i] = np.inf
        # negative distance
        j = np.argmin(distances[i])

        # save i,j
        if len(selected) + 1 == ind1.gene_num:
            # only need to select one gene
            selected.extend([i])
        else:
            selected.extend([i, j])

        # delete i,j
        distances[i, :] = np.inf
        distances[:, i] = np.inf
        distances[j, :] = np.inf
        distances[:, j] = np.inf
        indices = np.setdiff1d(indices, [i, j])
    ind1.gene = [(ind1.gene + ind2.gene)[i] for i in selected]
    # the number of selected genes should not be changed
    assert len(ind1.gene) == len(ind2.gene)
    return [ind1]


def get_semantic_results(ind1, ind2, target):
    # get similarities
    matrix1, matrix2 = ind1.semantics.T, ind2.semantics.T
    matrix1, matrix2 = matrix1 - target[np.newaxis, :], matrix2 - target[np.newaxis, :]
    semantic_result = np.vstack((matrix1, matrix2))
    # get coefficients
    vector1, vector2 = np.abs(ind1.coef), np.abs(ind2.coef)
    importance_result = np.hstack((vector1, vector2))
    return importance_result, semantic_result


def feature_crossover(ind1, ind2, positive, threshold_ratio):
    pop = [ind1, ind2]
    good_features, threshold = construct_feature_pools(pop, positive, threshold_ratio=threshold_ratio)
    new_pop = []
    for ind in pop:
        ind = cxOnePoint_multiple_gene_weight_plus(ind, good_features, threshold, positive)
        new_pop.append(ind)
    return new_pop


def cxOnePoint_multiple_gene_weight_plus(ind: MultipleGeneGP, good_features, threshold, positive):
    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            if (positive and c >= threshold) or (not positive and c < threshold):
                new_features = cxOnePoint(copy.deepcopy(random.choice(good_features)),
                                          copy.deepcopy(random.choice(good_features)))
                ind.gene[i] = random.choice(new_features)

    replaces_features(ind)
    return ind


def mutUniform_multiple_gene_weighted(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.weighted_selection(), expr, pset)
    return individual,


def cxOnePoint_multiple_gene_deterministic(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.deterministic_select(), ind2.deterministic_select())
    return ind1, ind2


def mutUniform_multiple_gene_deterministic(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.deterministic_select(), expr, pset)
    return individual,


def staticLimit(key, max_value, min_value):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value or key(ind) < min_value:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds

        return wrapper

    return decorator


def staticLimit_multiple_gene(key, max_value, min_value=0, random_fix=True, failure_counter: FailureCounter = None):
    """
    :param random_fix: Random fix is a parameter to determine whether we to randomly select a gene from two parents
    if there is an unsatisfied gene was generated.
    There is no specific reason for using random fix operation. When there are two parents,
    this operation may perform like a random crossover, which maybe helpful for improving the population diversity.
    :return:
    """
    if failure_counter == None:
        failure_counter = FailureCounter()
    failure_counter.value = 0

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind.gene) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                ind: MultipleGeneGP
                for j, x in enumerate(ind.gene):
                    if callable(max_value):
                        height_limitation = max_value()
                    else:
                        height_limitation = max_value

                    if key(x) > height_limitation or key(x) < min_value:
                        failure_counter.value += 1
                        # replace an unreasonable gene with a parent gene
                        if random_fix:
                            parent = random.choice(keep_inds)
                        else:
                            parent = keep_inds[i]
                        gene = copy.deepcopy(parent[j])
                        ind.gene[j] = gene
                    assert key(ind.gene[j]) <= height_limitation
                    assert key(ind.gene[j]) >= min_value
            return new_inds

        return wrapper

    return decorator


def result_calculation(func, data, original_features):
    result = multiple_gene_evaluation(func, data.T)
    result = result_post_process(result, data, original_features)
    return result


def result_post_process(result, data, original_features):
    # some post process step
    result = quick_fill(result, data)
    if original_features:
        result = np.concatenate([np.array(result).T, data], axis=1)
    else:
        result = np.array(result).T
    return result


def quick_fill(result, data):
    for i in range(len(result)):
        yp = result[i]
        if not isinstance(yp, np.ndarray):
            yp = np.full(len(data), 0)
        elif yp.size == 1:
            yp = np.full(len(data), yp)
        result[i] = yp
    result = np.nan_to_num(result, posinf=0, neginf=0)
    return result


def genFull_with_prob(pset, min_, max_, model,
                      type_=None, sample_type=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    terminal_probs = model.terminal_prob
    primitive_probs = model.primitive_prob
    return generate_with_prob(pset, min_, max_, condition, terminal_probs, primitive_probs, type_, sample_type)


def generate_with_prob(pset, min_, max_, condition, terminal_probs=None, primitive_probs=None,
                       type_=None, sample_type=None):
    if terminal_probs is not None and hasattr(pset, 'pset_id'):
        if isinstance(terminal_probs, dict):
            # shared terminal variables
            terminal_probs = [terminal_probs[name] for name in
                              [t.__name__ if isclass(t) else t.name for t in pset.terminals[object]]]
            terminal_probs = np.array(terminal_probs)
            terminal_probs /= terminal_probs.sum()
        else:
            # in default, there is a matrix of terminal probabilities
            terminal_probs = terminal_probs[pset.pset_id]
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_, None)]
    while len(stack) != 0:
        depth, type_, prim_name = stack.pop()
        if condition(height, depth):
            try:
                if sample_type == 'Dirichlet':
                    # sample from a dirichlet distribution
                    cat_prob = np.random.dirichlet(terminal_probs.flatten())
                    term = pset.terminals[type_][np.argmax(cat_prob)]
                else:
                    if isinstance(terminal_probs, StringDecisionTreeClassifier):
                        if prim_name is None:
                            label = terminal_probs.predict_proba_sample(np.array(['None']).reshape(-1, 1))
                        else:
                            label = terminal_probs.predict_proba_sample(np.array([prim_name]).reshape(-1, 1))
                        if is_number(label):
                            # sample a constant
                            t = [t for t in pset.terminals[type_] if not isinstance(t, Terminal)][0]
                            id = [t for t in pset.terminals[type_]].index(t)
                        else:
                            id = [t.name for t in pset.terminals[type_]].index(label)
                        term = pset.terminals[type_][id]
                    elif isinstance(terminal_probs, np.ndarray):
                        assert not np.any(np.isnan(terminal_probs)), "No value should be nan!"
                        probability = terminal_probs.flatten()
                        probability = probability[:len(pset.terminals[type_])]
                        if np.sum(probability) == 0:
                            # problematic probability values
                            probability = np.full_like(probability, fill_value=1 / len(probability))
                        else:
                            probability = probability / np.sum(probability)
                        term = np.random.choice(pset.terminals[type_], p=probability)
                    else:
                        term = np.random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add " \
                                 "a terminal of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                if primitive_probs is not None and len(primitive_probs) > 0:
                    prim = np.random.choice(pset.primitives[type_], p=primitive_probs.flatten())
                else:
                    prim = np.random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add " \
                                 "a primitive of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg, prim_name))
    return expr


class MultiplePrimitiveSet(PrimitiveSet):

    def __init__(self, name, arity, prefix="ARG"):
        super().__init__(name, arity, prefix)
        self.pset_list: List[PrimitiveSet] = []
        layer_mgp: bool = False
        mgp_mode: bool = False
        mgp_scope: int = 0
