import operator
import warnings
from functools import partial
from multiprocessing import Pool
from typing import List

import dill
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap.algorithms import varAnd
from deap.tools import selRandom, selLexicase, selNSGA2
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau, rankdata
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from sklearn.compose.tests.test_target import DummyTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.metrics import r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors._kd_tree import KDTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, BaseDecisionTree

from evolutionary_forest.component.archive import *
from evolutionary_forest.component.primitives import *
from evolutionary_forest.component.selection import batch_tournament_selection, selAutomaticEpsilonLexicase, \
    selAutomaticEpsilonLexicaseK, selTournamentPlus
from evolutionary_forest.multigene_gp import *
from evolutionary_forest.pruning import oob_pruning
from evolutionary_forest.sklearn_utils import cross_val_predict
from utils.common_utils import timeit


class EnsembleClassifier(ClassifierMixin, BaseEstimator):
    """
    Combining several models
    """

    def __init__(self, trees):
        self.trees = trees

    def fit(self, X, y):
        pass

    def predict(self, X):
        predictions = []
        t: DecisionTreeClassifier
        for t in self.trees:
            predictions.append(t.predict(X))
        return stats.mode(predictions, axis=0)[0].flatten()


def similar(a, b):
    return cosine_similarity(a.case_values.reshape(1, -1), b.case_values.reshape(1, -1))[0][0] > 0.9


def spearman(ya, yb):
    return spearmanr(ya, yb)[0]


def kendall(ya, yb):
    return kendalltau(ya, yb)[0]


class TestFunction():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.regr = None

    def predict_loss(self):
        if len(self.x) > 0:
            y_p = self.regr.predict(self.x)
            return np.mean((self.y - y_p) ** 2)
        else:
            return 0

    def __deepcopy__(self, memodict={}):
        return copy.deepcopy(self)


class EvolutionaryForestRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_pop=50, n_gen=20, verbose=False, max_height=8, basic_primitives=True, normalize=True,
                 select='AutomaticLexicase', gene_num=5, mutation_scheme='uniform', automated_hof_size=False,
                 boost_size=1, external_archive=False, original_features=False, diversity_search='None',
                 cross_pb=0.5, mutation_pb=0.1, second_layer=None, semantic_diversity=None, test_fun=None,
                 bootstrap_training=False, mean_model=False, early_stop=-1, min_samples_leaf=1,
                 base_learner='Random-DT', score_func='R2', max_tree_depth=None,
                 environmental_selection=None, pre_selection=None, eager_training=False,
                 n_process=1, **param):
        self.pre_selection = pre_selection
        self.max_tree_depth = max_tree_depth
        self.score_func = score_func
        self.min_samples_leaf = min_samples_leaf
        self.mean_model = mean_model
        self.base_learner = base_learner
        self.bootstrap_training = bootstrap_training
        self.semantic_diversity = semantic_diversity
        self.original_features = original_features
        self.external_archive = external_archive
        self.boost_size = boost_size
        self.automated_hof_size = automated_hof_size
        self.mutation_scheme = mutation_scheme
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.verbose = verbose
        self.max_height = max_height
        self.initialized = False
        self.evaluated_pop = set()
        self.generated_features = set()
        self.pop = []
        self.basic_primitives = basic_primitives
        self.select = select
        self.gene_num = gene_num
        self.param = param
        self.diversity_search = diversity_search
        self.second_layer = second_layer
        self.test_fun: List[TestFunction] = test_fun
        self.train_data_history = []
        self.test_data_history = []
        self.fitness_history = []
        self.diversity_history = []
        self.early_stop = early_stop
        self.environmental_selection = environmental_selection
        self.eager_training = eager_training
        self.current_height = 1

        self.cross_pb = cross_pb
        self.mutation_pb = mutation_pb

        self.normalize = normalize
        if normalize:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        self.n_process = n_process

    def calculate_diversity(self, population):
        inds = []
        for p in population:
            inds.append(p.case_values.flatten())
        inds = np.array(inds)
        tree = KDTree(inds)
        neighbors = tree.query(inds, k=3 + 1)[0]
        mean_diversity = np.mean(neighbors, axis=1)
        if self.verbose:
            print('mean_diversity', np.mean(mean_diversity))
        return mean_diversity

    def fitness_evaluation(self, individual):
        X, Y = self.X, self.y
        Y = Y.flatten()

        func = self.toolbox.compile(individual)
        pipe = self.get_base_model()

        y_pred, estimators = yield (pipe, dill.dumps(func, protocol=-1))

        individual.predicted_values = y_pred

        # minimize
        self.calculate_case_values(individual, Y, y_pred)

        if 'CV' in self.score_func:
            assert len(individual.case_values) == 5
        else:
            assert len(individual.case_values) == len(Y)
        individual.pipe = pipe

        if self.bootstrap_training or self.eager_training:
            func = self.toolbox.compile(individual)
            Yp = result_calculation(func, self.X, self.original_features)
            out_of_bag = self.train_final_model(individual, Yp, Y)

        individual.coef = np.mean([x['Ridge'].feature_importances_[:self.gene_num]
                                   for x in estimators], axis=0)

        if self.base_learner == 'Random-DT-Plus':
            individual.pipe = EnsembleClassifier(estimators)

        # final score
        if self.bootstrap_training:
            yield self.calculate_fitness_value(individual, Y[out_of_bag], individual.pipe.predict(Yp[out_of_bag]))
        else:
            yield self.calculate_fitness_value(individual, Y, y_pred)

    def calculate_fitness_value(self, individual, Y, y_pred):
        if self.score_func == 'R2' or self.score_func == 'NoveltySearch':
            return -1 * r2_score(Y, y_pred),
        elif self.score_func == 'Spearman':
            return -1 * spearman(Y, y_pred),
        elif 'CV' in self.score_func:
            return -1 * np.mean(y_pred),
        else:
            raise Exception

    def calculate_case_values(self, individual, Y, y_pred):
        if self.score_func == 'R2':
            individual.case_values = ((y_pred - Y.flatten()).flatten()) ** 2
        elif self.score_func == 'Spearman':
            individual.case_values = np.abs(rankdata(y_pred) - rankdata(Y.flatten())).flatten()
        elif self.score_func == 'CV-R2':
            individual.case_values = -1 * y_pred
        elif self.score_func == 'NoveltySearch':
            base_values = (y_pred.flatten() - Y.flatten()) ** 2
            if len(self.hof) == 0:
                # first generation
                individual.case_values = base_values
            else:
                # maximize cross entropy
                ensemble_value = np.mean([x.predicted_values for x in self.hof], axis=0)
                ambiguity = (y_pred.flatten() - ensemble_value) ** 2
                assert len(ambiguity) == len(y_pred.flatten())
                individual.case_values = base_values - ambiguity
        else:
            raise Exception

    def train_final_model(self, individual, Yp, Y):
        # avoid re-training
        model = individual.pipe
        try:
            if self.original_features:
                model.predict(np.ones((1, self.gene_num + self.X.shape[1])))
            else:
                model.predict(np.ones((1, self.gene_num)))
            return None
        except NotFittedError:
            pass

        # ensure ensemble base leaner will not be retrained
        assert self.base_learner != 'Random-DT-Plus'

        # train final model
        if self.bootstrap_training:
            sample = np.random.randint(0, len(Y), size=(len(Y)))

            chosen = np.zeros_like(Y)
            chosen[sample] = 1
            out_of_bag = np.where(chosen == 0)[0]
            individual.out_of_bag = out_of_bag

            individual.pipe.fit(Yp[sample], Y[sample])
        else:
            if self.early_stop > 0:
                individual.pipe.fit(Yp[self.train_index], Y[self.train_index])
            else:
                individual.pipe.fit(Yp, Y)
            out_of_bag = None

        # feature importance generation
        base_model = model
        if hasattr(base_model, 'feature_importances_'):
            individual.coef = base_model.feature_importances_[:self.gene_num]
        elif hasattr(base_model, 'coef_'):
            individual.coef = np.abs(base_model.coef_[:self.gene_num])
        assert len(individual.coef) == self.gene_num
        return out_of_bag

    def entropy_calculation(self):
        pass

    def get_base_model(self):
        if self.base_learner == 'DT':
            ridge_model = DecisionTreeRegressor(max_depth=self.max_tree_depth,
                                                min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'Random-DT':
            ridge_model = DecisionTreeRegressor(splitter='random', max_depth=self.max_tree_depth,
                                                min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'RidgeCV':
            ridge_model = RidgeCV()
        elif self.base_learner == 'LR':
            ridge_model = LinearRegression()
        elif self.base_learner == 'ET':
            ridge_model = ExtraTreesRegressor(n_estimators=100)
        elif self.base_learner == 'GBDT':
            ridge_model = GradientBoostingRegressor(learning_rate=0.8, n_estimators=5)
        elif self.base_learner == 'Hybrid':
            splitter = random.choice(['random', 'best'])
            ridge_model = DecisionTreeRegressor(splitter=splitter,
                                                min_samples_leaf=self.min_samples_leaf)
        elif isinstance(self.base_learner, RegressorMixin):
            ridge_model = self.base_learner
        else:
            raise Exception
        pipe = Pipeline([
            ("Scaler", StandardScaler()),
            ("Ridge", ridge_model),
        ])
        if isinstance(pipe['Ridge'], BaseDecisionTree) and self.max_tree_depth != None:
            assert pipe['Ridge'].max_depth == self.max_tree_depth
        return pipe

    def lazy_init(self, x):
        class GeneralFeature:
            pass

        class CategoricalFeature:
            pass

        class BooleanFeature:
            pass

        class NumericalFeature:
            pass

        def type_detection(X):
            input_arr = []
            for x in range(X.shape[1]):
                v = X[:, x]
                if len(np.unique(v)) == 2:
                    input_arr.append(BooleanFeature)
                elif len(np.unique(v)) <= 10:
                    input_arr.append(CategoricalFeature)
                else:
                    input_arr.append(NumericalFeature)
            return input_arr

        if self.basic_primitives == False:
            feature_types = type_detection(x)
            pset = gp.PrimitiveSetTyped("MAIN", feature_types, GeneralFeature)
            has_numerical_feature = False
            for x in feature_types:
                if x == NumericalFeature:
                    has_numerical_feature = True
            has_categorical_feature = False
            for x in feature_types:
                if x == CategoricalFeature:
                    has_categorical_feature = True
            has_boolean_feature = False
            for x in feature_types:
                if x == BooleanFeature:
                    has_boolean_feature = True
            if has_numerical_feature:
                pset.addPrimitive(np.add, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.subtract, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.multiply, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(analytical_quotient, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(protect_sqrt, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.sin, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.cos, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(np_max, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(np_min, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(np_wrapper(np.greater), [NumericalFeature, NumericalFeature], BooleanFeature)
                pset.addPrimitive(np_wrapper(np.less), [NumericalFeature, NumericalFeature], BooleanFeature)
                pset.addPrimitive(identical_numerical, [NumericalFeature], GeneralFeature)
                pset.addPrimitive(same_numerical, [NumericalFeature], NumericalFeature)
            if has_categorical_feature:
                pset.addPrimitive(group_sum, [CategoricalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(group_mean, [CategoricalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(group_min, [CategoricalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(group_max, [CategoricalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(group_mode, [CategoricalFeature, CategoricalFeature], CategoricalFeature)
                pset.addPrimitive(group_count, [CategoricalFeature], NumericalFeature)
                pset.addPrimitive(identical_categorical, [CategoricalFeature], GeneralFeature)
                pset.addPrimitive(same_categorical, [CategoricalFeature], CategoricalFeature)
            if has_boolean_feature:
                pset.addPrimitive(np.logical_and, [BooleanFeature, BooleanFeature], BooleanFeature)
                pset.addPrimitive(np.logical_or, [BooleanFeature, BooleanFeature], BooleanFeature)
                pset.addPrimitive(np.logical_xor, [BooleanFeature, BooleanFeature], BooleanFeature)
                pset.addPrimitive(identical_boolean, [BooleanFeature], GeneralFeature)
                pset.addPrimitive(same_boolean, [BooleanFeature], BooleanFeature)
        elif self.basic_primitives == 'extend':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_logical_operators(pset)
            add_relation_operators(pset)
            pset.addPrimitive(protect_sqrt, 1)
            pset.addPrimitive(np.abs, 1)
            pset.addPrimitive(np.sin, 1)
            pset.addPrimitive(np.cos, 1)
            pset.addPrimitive(np_max, 2)
            pset.addPrimitive(np_min, 2)
        elif self.basic_primitives == 'logical':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_logical_operators(pset)
        elif self.basic_primitives == 'relation':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_relation_operators(pset)
        else:
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
        self.pset = pset

        if not hasattr(gp, 'rand101'):
            if self.basic_primitives == False:
                pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1), NumericalFeature)
            else:
                pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", MultipleGeneGP, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        self.toolbox = toolbox
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", multiple_gene_initialization, creator.Individual, toolbox.expr,
                         gene_num=self.gene_num)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", multiple_gene_compile, pset=pset)

        toolbox.register("evaluate", self.fitness_evaluation)
        if self.select == 'Tournament':
            toolbox.register("select", tools.selTournament, tournsize=self.param['tournament_size'])
        elif 'Tournament-' in self.select:
            toolbox.register("select", tools.selTournament, tournsize=int(self.select.split('-')[1]))
        elif 'TournamentPlus-' in self.select:
            toolbox.register("select", selTournamentPlus, tournsize=int(self.select.split('-')[1]))
        elif self.select == 'BatchTournament':
            toolbox.register("select", batch_tournament_selection, tournsize=self.param['tournament_size'],
                             batch_size=128)
        elif self.select == 'AutomaticLexicase':
            toolbox.register("select", selAutomaticEpsilonLexicase)
        elif self.select == 'AutomaticLexicaseK':
            toolbox.register("select", selAutomaticEpsilonLexicaseK)
        elif self.select == 'Lexicase':
            toolbox.register("select", selLexicase)
        elif self.select == 'Random':
            toolbox.register("select", selRandom)
        else:
            raise Exception

        if 'uniform' in self.mutation_scheme or 'weight-plus' in self.mutation_scheme \
            or 'all_gene' in self.mutation_scheme:
            def convert_to_int(s):
                if s.replace('.', '').isdecimal():
                    return float(s)
                else:
                    return None

            # extract mutation operator
            if '|' in self.mutation_scheme:
                mutation_operator = self.mutation_scheme.split('|')[1]
                crossover_operator = self.mutation_scheme.split('|')[0]
            else:
                mutation_operator = None
                crossover_operator = self.mutation_scheme

            threshold_ratio = convert_to_int(crossover_operator.split('-')[-1])
            if threshold_ratio is None:
                threshold_ratio = 0.2
            else:
                crossover_operator = '-'.join(crossover_operator.split('-')[:-1])
            self.cx_threshold_ratio = threshold_ratio

            if 'weight-plus' in crossover_operator:
                if crossover_operator == 'weight-plus-positive':
                    toolbox.register("mate", feature_crossover, positive=True, threshold_ratio=threshold_ratio)
                elif crossover_operator == 'weight-plus-negative':
                    toolbox.register("mate", feature_crossover, positive=False, threshold_ratio=threshold_ratio)
                elif crossover_operator == 'weight-plus-cross':
                    toolbox.register("mate", feature_crossover_cross, threshold_ratio=threshold_ratio)
                elif crossover_operator == 'weight-plus-cross-global-mean':
                    toolbox.register("mate", feature_crossover_cross_global, regressor=self)
                    self.good_features_threshold = 'mean'
                elif crossover_operator == 'weight-plus-cross-global-inverse':
                    toolbox.register("mate", feature_crossover_cross_global, regressor=self)
                    self.good_features_threshold = 1 - self.cx_threshold_ratio
                elif crossover_operator == 'weight-plus-cross-global':
                    toolbox.register("mate", feature_crossover_cross_global, regressor=self)
                    self.good_features_threshold = self.cx_threshold_ratio
                else:
                    raise Exception
            elif crossover_operator == 'all_gene':
                toolbox.register("mate", cxOnePoint_multiple_all_gene)
            elif crossover_operator == 'all_gene_permutation':
                toolbox.register("mate", partial(cxOnePoint_multiple_all_gene, permutation=True))
            else:
                assert crossover_operator == 'uniform'
                toolbox.register("mate", cxOnePoint_multiple_gene)

            if self.basic_primitives == False:
                toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
            else:
                toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

            if mutation_operator == None:
                toolbox.register("mutate", mutUniform_multiple_gene, expr=toolbox.expr_mut, pset=pset)
            elif mutation_operator == 'weight-mutation':
                toolbox.register("mutate", mutWeight_multiple_gene, expr=toolbox.expr_mut, pset=pset)
            elif mutation_operator == 'weight-mutation-global':
                toolbox.register("mutate", feature_mutation_global, expr=toolbox.expr_mut, pset=pset,
                                 regressor=self)
            else:
                raise Exception
        elif self.mutation_scheme == 'weight':
            toolbox.register("mate", cxOnePoint_multiple_gene_weight)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", mutUniform_multiple_gene_weight, expr=toolbox.expr_mut, pset=pset)
        else:
            raise Exception

        def dynamic_height():
            if len(self.generated_features) >= \
                np.log(len(self.pset.primitives) * (2 ** self.current_height - 1) * \
                       len(self.pset.terminals) * (2 ** self.current_height)) / \
                np.log(1.01):
                self.current_height += 1
            return self.current_height

        if self.max_height == 'dynamic':
            toolbox.decorate("mate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                               max_value=dynamic_height))
            toolbox.decorate("mutate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                 max_value=dynamic_height))
        else:
            toolbox.decorate("mate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                               max_value=self.max_height))
            toolbox.decorate("mutate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                 max_value=self.max_height))

        self.pop = toolbox.population(n=self.n_pop)

        # archive initialization
        if self.boost_size == 'auto':
            self.hof = LexicaseHOF()
        else:
            if self.semantic_diversity == 'equal':
                def similar(a, b):
                    return a.fitness.wvalues == b.fitness.wvalues

                self.hof = HallOfFame(self.boost_size, similar=similar)
            elif self.semantic_diversity == 'similar':
                def similar(a, b):
                    return cosine(a.case_values, b.case_values) <= 0.05

                self.hof = HallOfFame(self.boost_size, similar=similar)
            else:
                self.hof = HallOfFame(self.boost_size)
        if self.environmental_selection != None:
            self.hof = None

    def construct_global_feature_pool(self, pop):
        good_features, threshold = construct_feature_pools(pop, True, threshold_ratio=self.cx_threshold_ratio,
                                                           good_features_threshold=self.good_features_threshold)
        self.good_features = good_features
        self.cx_threshold = threshold

    def fit(self, X, y):
        if self.normalize:
            X = self.x_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(np.array(y).reshape(-1, 1))
        if self.early_stop > 0:
            indices = np.arange(len(X))
            _, valid_x, _, valid_y, idx1, idx2 = train_test_split(X, y, indices)
            self.train_index = idx1
            self.valid_x = valid_x
            self.valid_y = valid_y

        self.X, self.y = X, y
        if self.environmental_selection == 'NSGA2-100':
            self.random_objectives = np.random.uniform(0, 1, size=(len(y), 100))
        if self.environmental_selection == 'NSGA2-100-Normal':
            self.random_objectives = truncated_normal(sample=(len(y), 100))
        if self.environmental_selection == 'NSGA2-100-LHS':
            from smt.sampling_methods import LHS
            xlimits = np.repeat(np.array([[0.0, 1.0]]), len(y), axis=0)
            sampling = LHS(xlimits=xlimits)
            num = 100
            self.random_objectives = sampling(num).T

        self.lazy_init(X)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = self.eaSimple(self.pop, self.toolbox, self.cross_pb, self.mutation_pb, self.n_gen,
                                 stats=mstats, halloffame=self.hof, verbose=self.verbose)
        self.pop = pop

        self.second_layer_generation(X, y)
        self.final_model_lazy_training(self.hof)
        return self

    def second_layer_generation(self, X, y):
        if self.second_layer == 'None' or self.second_layer == None:
            return
        y_data = self.y
        predictions = []
        for individual in self.hof:
            predictions.append(individual.predicted_values)
        predictions = np.array(predictions)

        if self.second_layer == 'DREP':
            # forward selection
            y_sample = y_data.flatten()
            current_prediction = np.zeros_like(y_sample)
            remain_ind = set([i for i in range(len(self.hof))])
            min_index = 0
            for i in remain_ind:
                error = np.mean((predictions[i] - y_sample) ** 2)
                if error < np.mean((current_prediction - y_sample) ** 2):
                    current_prediction = predictions[i]
                    min_index = i
            remain_ind.remove(min_index)

            ensemble_list = np.zeros(len(self.hof))
            ensemble_list[min_index] = 1
            while True:
                div_list = []
                for i in remain_ind:
                    diversity = np.mean(((current_prediction - predictions[i]) ** 2))
                    loss = np.mean(((y_sample - predictions[i]) ** 2))
                    div_list.append((diversity, loss, i))
                div_list = list(sorted(div_list, key=lambda x: -x[0]))[:int(round(len(div_list) * 0.5))]
                div_list = list(sorted(div_list, key=lambda x: x[1]))
                index = div_list[0][2]
                ensemble_size = np.sum(ensemble_list)
                trial_prediction = ensemble_size / (ensemble_size + 1) * current_prediction + \
                                   1 / (ensemble_size + 1) * predictions[index]
                if np.mean(((trial_prediction - y_sample) ** 2)) > np.mean(((current_prediction - y_sample) ** 2)):
                    break
                current_prediction = trial_prediction
                ensemble_list[index] = 1
                remain_ind.remove(index)
            ensemble_list /= np.sum(ensemble_list)
            self.tree_weight = ensemble_list
        elif self.second_layer == 'Ridge':
            self.ridge = Ridge(alpha=1e-3, normalize=True, fit_intercept=False)
            self.ridge.fit(predictions.T, y_data)
            x = self.ridge.coef_
            x[x < 0] = 0
            x[x > 0] = 1
            x /= np.sum(x)
            x = x.flatten()
            self.tree_weight = x
        elif self.second_layer == 'TreeBaseline':
            base_line_score = np.mean(cross_val_score(DecisionTreeRegressor(), X, y))
            score = np.array(list(map(lambda x: x.fitness.wvalues[0], self.hof.items)))
            x = np.zeros_like(score)
            x[score > base_line_score] = 1
            if np.sum(x) == 0:
                x[0] = 1
            x /= np.sum(x)
            x = x.flatten()
            self.tree_weight = x
        elif self.second_layer == 'DiversityPrune':
            sample_len = 500
            predictions = []
            for individual in self.hof:
                func = self.toolbox.compile(individual)
                x = np.random.randn(sample_len, X.shape[1])
                Yp = result_calculation(func, x, self.original_features)
                predicted = individual.pipe.predict(Yp)
                predictions.append(predicted)
            predictions = np.array(predictions)

            # forward selection
            current_prediction = np.zeros(sample_len)
            remain_ind = set([i for i in range(len(self.hof))])

            # select first regressor
            min_index = 0
            remain_ind.remove(min_index)

            ensemble_list = np.zeros(len(self.hof))
            ensemble_list[min_index] = 1
            while True:
                div_list = []
                for i in remain_ind:
                    diversity = np.mean(((current_prediction - predictions[i]) ** 2))
                    div_list.append((diversity, i))
                div_list = list(sorted(div_list, key=lambda x: -x[0]))
                index = div_list[0][1]
                ensemble_size = np.sum(ensemble_list)
                trial_prediction = ensemble_size / (ensemble_size + 1) * current_prediction + \
                                   1 / (ensemble_size + 1) * predictions[index]
                if np.mean(((current_prediction - trial_prediction) ** 2)) < 0.05:
                    break
                current_prediction = trial_prediction
                ensemble_list[index] = 1
                remain_ind.remove(index)
            ensemble_list /= np.sum(ensemble_list)
            self.tree_weight = ensemble_list
        elif self.second_layer == 'GA':
            # oob calculation
            pop = self.hof
            x_train = self.X
            oob = np.zeros((len(pop), len(x_train)))
            for i, ind in enumerate(pop):
                sample = ind.out_of_bag
                chosen = np.zeros(len(x_train))
                chosen[sample] = 1
                out_of_bag = np.where(chosen == 0)[0]
                func = self.toolbox.compile(individual)
                Yp = result_calculation(func, x_train[out_of_bag], self.original_features)
                oob[i][out_of_bag] = ind.pipe.predict(Yp)

            weight = oob_pruning(oob, self.y)
            weight = weight / np.sum(weight)
            self.tree_weight = weight

    def bootstrap_fitness(self, Yp, Y_true):
        num = len(Y_true)
        sum = []
        for i in range(20):
            index = np.random.randint(0, num, num)
            sum.append(np.mean((Yp[index] - Y_true[index]) ** 2))
        return np.mean(sum)

    def final_model_lazy_training(self, pop):
        for p in pop:
            func = self.toolbox.compile(p)
            Yp = result_calculation(func, self.X, self.original_features)
            self.train_final_model(p, Yp, self.y)

    def predict(self, X, return_std=False):
        if self.normalize:
            X = self.x_scaler.transform(X)
        self.final_model_lazy_training(self.hof)

        predictions = []
        for individual in self.hof:
            if len(individual.gene) == 0:
                continue
            func = self.toolbox.compile(individual)
            Yp = result_calculation(func, X, self.original_features)
            # Yp = np.nan_to_num(Yp)
            predicted = individual.pipe.predict(Yp)

            if self.normalize:
                predicted = self.y_scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()
            predictions.append(predicted)
        if self.second_layer != 'None' and self.second_layer != None:
            predictions = np.array(predictions).T
            return predictions @ self.tree_weight
        else:
            return np.mean(predictions, axis=0)

    def get_hof(self):
        if self.hof != None:
            return [x for x in self.hof]
        else:
            return None

    def get_features(self, sample, individuals, flatten=False):
        all_Yp = []
        for individual in individuals:
            func = self.toolbox.compile(individual)
            Yp = result_calculation(func, self.X[sample], self.original_features)
            if flatten:
                all_Yp.append(Yp.flatten())
            else:
                all_Yp.append(Yp)
        return np.array(all_Yp)

    def pre_selection_individuals(self, parents, offspring):
        predicted_values = self.pre_selection_score(offspring, parents)
        final_offspring = []
        for i in np.argsort(-1 * predicted_values)[:len(parents)]:
            final_offspring.append(offspring[i])
        return final_offspring

    def pre_selection_score(self, offspring, parents):
        if self.pre_selection == 'surrogate-model':
            predicted_values = self.calculate_score_by_surrogate_model(offspring, parents)
        elif self.pre_selection == 'simple-task':
            predicted_values = self.calculate_score_by_statistical_methods(offspring, parents)
        else:
            raise Exception
        return predicted_values

    def calculate_score_by_statistical_methods(self, offspring, parents):
        all_values = np.array([p.case_values for p in parents])
        sample = np.sum(all_values, axis=0).argsort()[-50:]
        offspring_features = self.get_features(sample, offspring, False)
        all_score = []
        for f in offspring_features:
            pipe = self.get_base_model()
            y_pred = cross_val_predict(pipe, f, self.y[sample], cv=3).flatten()
            score = r2_score(self.y[sample], y_pred)
            all_score.append(score)
        return np.array(all_score)

    def calculate_score_by_surrogate_model(self, offspring, parents):
        def spearman(ya, yb):
            correlation = spearmanr(ya, yb)[0]
            return correlation

        sample = np.random.randint(0, len(self.y), size=5)
        # select individuals based on a surrogate model
        parent_features = self.get_features(sample, parents)
        target = np.array([p.fitness.wvalues[0] for p in parents])
        surrogate_model = ExtraTreesRegressor()
        # surrogate_model = XGBRegressor(n_jobs=1)
        surrogate_model.fit(parent_features, target)
        offspring_features = self.get_features(sample, offspring)
        predicted_values = surrogate_model.predict(offspring_features)
        scoring = make_scorer(spearman)
        print('CV', np.mean(cross_val_score(surrogate_model, parent_features, target, scoring=scoring)))
        return predicted_values

    def append_evaluated_features(self, pop):
        for ind in pop:
            for gene in ind.gene:
                self.generated_features.add(str(gene))

    def eaSimple(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
        arg = (self.X, self.y, self.original_features, self.score_func)
        if self.n_process > 1:
            self.pool = Pool(self.n_process, initializer=init_worker, initargs=(calculate_score, arg))
        else:
            init_worker(calculate_score, arg)
        history_hof = HallOfFame(len(population))
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = self.population_evaluation(toolbox, population)
        self.append_evaluated_features(population)
        for o in population:
            self.evaluated_pop.add(individual_to_tuple(o))

        if halloffame is not None:
            halloffame.update(population)

        if self.diversity_search != 'None':
            self.diversity_assignment(population)

        if 'global' in self.mutation_scheme:
            self.construct_global_feature_pool(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        error_list = []
        if self.pre_selection != None:
            surrogate_coefficient = 5
            pop_size = len(population) * surrogate_coefficient
        else:
            pop_size = len(population)

        best_hof = self.get_hof()
        # Begin the generational process
        for gen in range(1, ngen + 1):
            self.entropy_calculation()
            count = 0
            new_offspring = []
            while (len(new_offspring) < pop_size):
                if count > pop_size * 100:
                    raise Exception("Error!")
                count += 1
                # Select the next generation individuals
                offspring = toolbox.select(population + list(history_hof), 2)
                offspring = offspring[:]

                # Vary the pool of individuals
                offspring = varAnd(offspring, toolbox, cxpb, mutpb)
                for o in offspring:
                    if len(new_offspring) < pop_size:
                        if not individual_to_tuple(o) in self.evaluated_pop:
                            self.evaluated_pop.add(individual_to_tuple(o))
                            new_offspring.append(o)
            # delete some inherited information
            for ind in new_offspring:
                delattr(ind, 'pipe')
                delattr(ind, 'predicted_values')
                delattr(ind, 'case_values')
                delattr(ind, 'coef')

            if self.pre_selection != None:
                offspring = self.pre_selection_individuals(population, new_offspring)
                assert len(offspring) == pop_size / surrogate_coefficient
            else:
                offspring = new_offspring
                assert len(offspring) == pop_size
            self.append_evaluated_features(offspring)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = self.population_evaluation(toolbox, offspring)
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
            if 'global' in self.mutation_scheme:
                self.construct_global_feature_pool(population)

            self.fitness_history.append(np.mean([ind.fitness.wvalues[0] for ind in offspring]))
            if self.diversity_search != 'None':
                self.diversity_assignment(offspring)
            if self.external_archive:
                history_hof.update(offspring)

            # Replace the current population by the offspring
            if self.environmental_selection != None and 'NSGA2-100' in self.environmental_selection:
                for ind in offspring + population:
                    setattr(ind, 'original_fitness', ind.fitness.values)
                    fitness = ind.case_values @ self.random_objectives
                    ind.fitness.weights = (-1,) * len(self.random_objectives)
                    ind.fitness.values = list(fitness)
                population[:] = selNSGA2(offspring + population, len(population))
                for ind in population:
                    ind.fitness.weights = (-1,)
                    ind.fitness.values = getattr(ind, 'original_fitness')
                self.hof = population
            elif self.environmental_selection == 'NSGA2':
                mean_values = np.mean(np.array([ind.case_values for ind in offspring + population]), axis=0)
                assert len(mean_values) == len(self.y)
                mean_values = mean_values.reshape(1, -1)
                for ind in offspring + population:
                    setattr(ind, 'original_fitness', ind.fitness.values)
                    ind.fitness.weights = (-1, -1)
                    distances = pairwise_distances(ind.case_values.reshape(1, -1), mean_values)
                    ind.fitness.values = (ind.fitness.values[0], -distances[0][0])
                population[:] = selNSGA2(offspring + population, len(population))
                for ind in population:
                    ind.fitness.weights = (-1,)
                    ind.fitness.values = getattr(ind, 'original_fitness')
                self.hof = population
            else:
                population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            if self.early_stop > 0:
                self.second_layer_generation(self.X, self.y)
                # mean_pv = np.mean([ind.last_predicted_values[self.early_stop_index] for ind in self.hof], axis=0)
                # mean_error = np.mean((mean_pv - np.squeeze(self.y[self.early_stop_index], axis=2)) ** 2)
                y_p = self.predict(self.valid_x)
                mean_error = np.mean((y_p - self.valid_y) ** 2)
                if gen == 1 or mean_error < min(error_list):
                    best_hof = self.get_hof()
                error_list.append(mean_error)
                if gen > self.early_stop:
                    # diff_error = np.array(error_list[1:]) - np.array(error_list[:-1])
                    if np.min(error_list[-(self.early_stop):]) > np.min(error_list[:-(self.early_stop)]):
                        break

            if self.test_fun != None:
                self.second_layer_generation(self.X, self.y)
                self.train_data_history.append(self.test_fun[0].predict_loss())
                self.test_data_history.append(self.test_fun[1].predict_loss())
                self.diversity_history.append(self.diversity_summarization())

        if self.early_stop > 0:
            print('final generation', len(error_list))
            self.hof.clear()
            self.hof.update(best_hof)
        if self.n_process > 0:
            self.pool.close()
        return population, logbook

    @timeit
    def population_evaluation(self, toolbox, population):
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

        if self.n_process > 1:
            data = [next(f) for f in fitnesses]
            results = list(self.pool.map(calculate_score, data))
        else:
            results = list(map(lambda f: calculate_score(next(f)), fitnesses))
        for ind, fit, r in zip(invalid_ind, fitnesses, results):
            value = fit.send(r)
            ind.fitness.values = value
        return invalid_ind

    def diversity_summarization(self):
        # distance calculation
        if self.second_layer == 'None' or self.second_layer == None:
            all_ind = self.hof
        elif self.second_layer in ['DiversityPrune', 'TreeBaseline', 'GA']:
            if not hasattr(self, 'tree_weight'):
                return 0
            all_ind = list(map(lambda x: x[1], filter(lambda x: self.tree_weight[x[0]] > 0, enumerate(self.hof))))
        else:
            raise Exception
        inds = self.get_diversity_matrix(all_ind)
        dis = pairwise_distances(inds)
        return np.mean(dis)

    def get_diversity_matrix(self, all_ind):
        inds = []
        for p in all_ind:
            inds.append(p.predicted_values.flatten())
        inds = np.array(inds)
        return inds

    def diversity_assignment(self, population):
        # distance calculation
        inds = self.get_diversity_matrix(population)

        hof_inds = []
        pop = {
            'Pop': self.pop,
            'Archive': self.hof,
            'Pop+Archive': list(self.pop) + list(self.hof),
        }[self.diversity_search]
        for p in pop:
            hof_inds.append(p.predicted_values.flatten())
        hof_inds = np.array(hof_inds)
        for index, ind in enumerate(population):
            distance = -np.sqrt(np.sum((inds[index] - hof_inds) ** 2, axis=0))
            if len(ind.case_values) == len(self.y):
                ind.case_values = np.concatenate([ind.case_values, distance], axis=0)
            else:
                ind.case_values[len(self.y):] = distance


class EvolutionaryForestClassifier(EvolutionaryForestRegressor):
    def __init__(self, score_func='ZeroOne', **param):
        super().__init__(score_func=score_func, **param)
        self.y_scaler = DummyTransformer()

    def predict_proba(self, X):
        predictions = []
        for individual in self.hof:
            if self.verbose:
                print(individual.fitness.wvalues)
            func = self.toolbox.compile(individual)
            Yp = result_calculation(func, X, self.original_features)
            predicted = individual.pipe.predict_proba(Yp)
            predictions.append(predicted)
        return np.mean(predictions, axis=0)

    def lazy_init(self, x):
        self.label_encoder = OneHotEncoder(sparse=False)
        self.label_encoder.fit(self.y)
        return super().lazy_init(x)

    def entropy_calculation(self):
        if self.score_func == 'NoveltySearch':
            encoder = self.label_encoder
            ensemble_value = np.mean([encoder.transform(x.predicted_values.reshape(-1, 1)) for x in self.hof],
                                     axis=0)
            self.ensemble_value = ensemble_value
            return self.ensemble_value

    def calculate_case_values(self, individual, Y, y_pred):
        if self.score_func == 'ZeroOne':
            individual.case_values = -1 * (y_pred.flatten() == Y.flatten())
        elif self.score_func == 'CrossEntropy':
            one_hot_targets = OneHotEncoder(sparse=False).fit_transform(self.y)
            eps = np.finfo(float).eps
            individual.case_values = np.sum(one_hot_targets * np.log(y_pred + eps), axis=1)
        elif self.score_func == 'CV-Accuracy':
            individual.case_values = -1 * y_pred
        elif self.score_func == 'NoveltySearch':
            base_values = -1 * (y_pred.flatten() == Y.flatten())
            if len(self.hof) == 0:
                # first generation
                individual.case_values = base_values
            else:
                # maximize cross entropy
                encoder = self.label_encoder
                ensemble_value = self.ensemble_value
                y_pred_one_hot = encoder.transform(y_pred.reshape(-1, 1))
                individual.case_values = base_values + np.mean(np.abs(y_pred_one_hot - ensemble_value), axis=1)
        else:
            raise Exception

    def get_diversity_matrix(self, all_ind):
        inds = []
        for p in all_ind:
            encoder = self.label_encoder
            y_pred_one_hot = encoder.transform(p.predicted_values.reshape(-1, 1))
            inds.append(y_pred_one_hot.flatten())
        inds = np.array(inds)
        return inds

    def calculate_fitness_value(self, individual, Y, y_pred):
        if self.score_func == 'ZeroOne':
            return np.sum(-1 * (y_pred.flatten() == Y.flatten())),
        elif self.score_func == 'NoveltySearch':
            return np.sum(-1 * (y_pred.flatten() == Y.flatten())),
        elif self.score_func == 'CrossEntropy':
            return -1 * np.sum(individual.case_values),
        elif 'CV' in self.score_func:
            return -1 * np.mean(y_pred),
        else:
            raise Exception

    def predict(self, X, return_std=False):
        if self.normalize:
            X = self.x_scaler.transform(X)
        self.final_model_lazy_training(self.hof)
        predictions = []
        for individual in self.hof:
            if self.verbose:
                print(individual.fitness.wvalues)
            func = self.toolbox.compile(individual)
            Yp = result_calculation(func, X, self.original_features)
            predicted = individual.pipe.predict(Yp)
            predictions.append(predicted)
        return stats.mode(predictions, axis=0)[0].flatten()

    def get_base_model(self):
        if self.base_learner == 'DT':
            ridge_model = DecisionTreeClassifier(max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'Random-DT' or self.base_learner == 'Random-DT-Plus':
            ridge_model = DecisionTreeClassifier(splitter='random', max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'Random-DT-SQRT':
            ridge_model = DecisionTreeClassifier(splitter='random', max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 max_features='sqrt')
        elif isinstance(self.base_learner, ClassifierMixin):
            ridge_model = self.base_learner
        else:
            raise Exception
        pipe = Pipeline([
            ("Scaler", StandardScaler()),
            ("Ridge", ridge_model),
        ])
        return pipe


def truncated_normal(lower=0, upper=1, mu=0.5, sigma=0.1, sample=(100, 100)):
    # instantiate an object X using the above four parameters,
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    # generate 1000 sample data
    samples = X.rvs(sample)
    return samples


def calculate_score(args):
    (X, Y, original_features, score_func) = calculate_score.data
    pipe, func = args
    func = dill.loads(func)

    Yp = result_calculation(func, X, original_features)
    assert isinstance(Yp, np.ndarray)
    assert np.all(Yp != np.nan), f"{Yp}"
    assert not np.any(np.isinf(Yp)), f"{np.any(np.isinf(Yp))},{Yp}"

    if score_func in ['CV-R2', 'CV-Accuracy']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_validate(pipe, Yp, Y, return_estimator=True)
            y_pred = result['test_score']
            estimators = result['estimator']
    elif score_func == 'CrossEntropy':
        y_pred, estimators = cross_val_predict(pipe, Yp, Y, method='predict_proba')
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred, estimators = cross_val_predict(pipe, Yp, Y)
    return y_pred, estimators


def init_worker(function, data):
    function.data = data
