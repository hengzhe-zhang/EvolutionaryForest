import itertools
from collections import Counter
from multiprocessing import Pool
from typing import Union, List

import dill
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap.algorithms import varAnd
from deap.base import Toolbox
from deap.tools import selLexicase, selNSGA2, History, selBest
from gplearn.functions import _protected_sqrt
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel
from lineartree import LinearTreeRegressor
from mlxtend.evaluate import feature_importance_permutation
from numpy.linalg import norm
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau, rankdata
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose.tests.test_target import DummyTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.model_selection import cross_val_score, ParameterGrid, train_test_split
from sklearn.neighbors._kd_tree import KDTree
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
from sklearn.utils import compute_sample_weight
from sklearn2pmml.ensemble import GBDTLRClassifier
from sympy import parse_expr
from tpot import TPOTClassifier, TPOTRegressor

from evolutionary_forest.component.archive import *
from evolutionary_forest.component.archive import DREPHallOfFame, NoveltyHallOfFame, OOBHallOfFame, BootstrapHallOfFame
from evolutionary_forest.component.evaluation import calculate_score, get_cv_splitter
from evolutionary_forest.component.primitives import *
from evolutionary_forest.component.primitives import np_mean, add_extend_operators
from evolutionary_forest.component.selection import batch_tournament_selection, selAutomaticEpsilonLexicaseK, \
    selTournamentPlus, selAutomaticEpsilonLexicaseFast, selDoubleRound, selRandomPlus, selBagging, selTournamentNovelty, \
    selHybrid, selGPED, selMAPElite, selMAPEliteClustering, selKnockout
from evolutionary_forest.model.PLTree import SoftPLTreeRegressor, SoftPLTreeRegressorEM, PLTreeRegressor, RidgeDT, \
    LRDTClassifier, RidgeDTPlus
from evolutionary_forest.model.RBFN import RBFN
from evolutionary_forest.model.SafetyLR import SafetyLogisticRegression
from evolutionary_forest.model.SafetyScaler import SafetyScaler
from evolutionary_forest.multigene_gp import *
from evolutionary_forest.preprocess_utils import GeneralFeature, CategoricalFeature, BooleanFeature, \
    NumericalFeature, type_detection, FeatureTransformer
from evolutionary_forest.probability_gp import genHalfAndHalf
from evolutionary_forest.pruning import oob_pruning
from evolutionary_forest.strategies.estimation_of_distribution import EstimationOfDistribution
from evolutionary_forest.strategies.space_partition import SpacePartition, partition_scheme_varAnd, \
    partition_scheme_toolbox
from evolutionary_forest.strategies.surrogate_model import SurrogateModel
from evolutionary_forest.utils import get_feature_importance, feature_append, select_top_features, efficient_deepcopy, \
    gene_to_string, get_activations, reset_random, weighted_avg_and_std, save_array

eda_operators = ['probability-TS', 'EDA-Primitive', 'EDA-Terminal', 'EDA-PM',
                 'EDA-Terminal-PM', 'EDA-Terminal-Balanced', 'EDA-Terminal-SameWeight', 'EDA-Terminal-PMI',
                 'EDA-Terminal-PM-Biased', 'EDA-Terminal-PM-Population', 'EDA-PM-Population',
                 'EDA-Terminal-PM-Tournament', 'EDA-Terminal-PM-SC', 'EDA-Terminal-PM-SameIndex']
map_elite_series = ['MAP-Elite-Lexicase', 'MAP-Elite-Tournament', 'MAP-Elite-Tournament-3', 'MAP-Elite-Tournament-7',
                    'MAP-Elite-Random', 'MAP-Elite-Knockout', 'MAP-Elite-Knockout-S','MAP-Elite-Knockout-SA',
                    'Auto', 'Auto-MCTS']


class GBDTLRClassifierX(GBDTLRClassifier):

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        self.classes_ = self.gbdt_.classes_
        return self


class StaticRandomProjection(TransformerMixin, BaseEstimator):

    def __init__(self, components_: int):
        self.components_ = components_

    def _make_random_matrix(self, n_components, n_features):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X @ self.components_


class EnsembleRegressor(RegressorMixin, BaseEstimator):
    """
    Combining several models generated by 5-fold CV to form a final ensemble model
    """

    def __init__(self, trees):
        self.trees = trees

    def fit(self, X, y):
        pass

    def predict(self, X):
        predictions = []
        t: DecisionTreeRegressor
        for t in self.trees:
            predictions.append(t.predict(X))
        return np.mean(predictions, axis=0)


class EnsembleClassifier(ClassifierMixin, BaseEstimator):
    """
    Combining several models generated by 5-fold CV to form a final ensemble model
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


class EvolutionaryForestRegressor(RegressorMixin, TransformerMixin, BaseEstimator,
                                  SurrogateModel, SpacePartition, EstimationOfDistribution):
    def __init__(self, n_pop=50, n_gen=20, verbose=False, max_height=8, basic_primitives=True, normalize=True,
                 select='AutomaticLexicaseFast', gene_num=5, mutation_scheme='uniform', ensemble_size=1,
                 external_archive=False, original_features=False, diversity_search='None', cross_pb=0.5,
                 mutation_pb=0.1, second_layer=None, ensemble_selection=None, test_fun=None, bootstrap_training=False,
                 mean_model=False, early_stop=-1, min_samples_leaf=1, base_learner='Random-DT', score_func='R2',
                 max_tree_depth=None, environmental_selection=None, pre_selection=None, eager_training=False,
                 n_process=1, useless_feature_ratio=None, weighted_coef=False, feature_selection=False,
                 allow_repetitive=False, cv=5, elitism=0,
                 # Soft-PS-Tree
                 partition_number=4, ps_tree_local_model='RidgeCV',
                 dynamic_partition='Self-Adaptive', max_leaf_nodes=None, ps_tree_cv_label=True,
                 ps_tree_partition_model='DecisionTree', ps_tree_ratio=0.1,
                 only_original_features=True, shared_partition_scheme=False,
                 # More Parameters
                 initial_tree_size=None, decision_tree_count=2, basic_gene_num=0,
                 clearing_cluster_size=0, reduction_ratio=0,
                 random_state=0, class_weight=None, map_elite_parameter=None,
                 # Deprecated parameters
                 boost_size=None, validation_size=0, mab_parameter=None, semantic_diversity=None,
                 interleaving_period=0, **param):
        """
        Basic GP Parameters:
        n_pop: The size of the population
        n_gen: The number of generations
        verbose: Whether to print verbose information
        max_leaf_nodes: The maximum height of each GP tree
        basic_primitives: The primitive set used in GP
        normalize: Normalization before fitting a model
        select: The name of the selection operator, supporting "Tournament-(Tournament Size)", "Lexicase"
        gene_num: The number of genes in each GP individual
        ensemble_size: The size of the ensemble model
        external_archive: Using an external archive to store historical best results
        original_features: Whether to use original features in the model or not
        diversity_search: The name strategy to directly enhance the diversity
        cross_pb: The probability of crossover
        mutation_pb: The probability of mutation
        second_layer: The strategy to induce a second layer to assign different weights for the member in the ensemble
        semantic_diversity: The strategy to maintain the diversity of the ensemble model

        Soft PS-Tree Parameters:
        shared_partition_scheme: All individuals use a shared partition scheme

        PL-Tree Parameters:
        decision_tree_count: Number of piecewise trees in a PL-Tree

        EA Parameters:
        reduction_ratio: Pre-selection based on filters

        map_elite_parameter: Hyper-parameters for MAP-Elite
        """
        super(SurrogateModel, self).__init__()
        self.mab_parameter = {} if mab_parameter is None else mab_parameter
        self.validation_size = validation_size
        self.class_weight = class_weight
        reset_random(random_state)
        self.random_state = random_state
        self.reduction_ratio = reduction_ratio
        self.clearing_cluster_size = clearing_cluster_size
        self.basic_gene_num = basic_gene_num
        self.decision_tree_count = decision_tree_count
        self.shared_partition_scheme = shared_partition_scheme
        self.only_original_features = only_original_features
        self.initial_tree_size = initial_tree_size
        self.ps_tree_partition_model = ps_tree_partition_model
        self.ps_tree_cv_label = ps_tree_cv_label
        self.allow_repetitive = allow_repetitive
        self.pre_selection = pre_selection
        self.max_tree_depth = max_tree_depth
        self.elitism = elitism
        self.score_func = score_func
        self.min_samples_leaf = min_samples_leaf
        self.mean_model = mean_model
        self.base_learner = base_learner
        self.bootstrap_training = bootstrap_training
        self.ensemble_selection = ensemble_selection if semantic_diversity is None else semantic_diversity
        self.original_features = original_features
        self.external_archive = external_archive
        self.boost_size = boost_size
        self.ensemble_size = ensemble_size if boost_size is None else int(boost_size)
        self.mutation_scheme = mutation_scheme
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.verbose = verbose
        self.max_height: Union[str, int] = max_height
        self.initialized = False
        self.evaluated_pop = set()
        self.generated_features = set()
        self.pop: List[MultipleGeneGP] = []
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
        self.best_fitness_history = []
        self.operator_selection_history = []
        self.diversity_history = []
        self.early_stop = early_stop
        self.environmental_selection = environmental_selection
        self.eager_training = eager_training
        self.current_height = 1
        self.repetitive_feature_count = []
        self.useless_feature_ratio = useless_feature_ratio
        self.weighted_coef = weighted_coef
        self.feature_selection = feature_selection
        self.cv = cv
        self.stage_flag = False
        self.map_elite_parameter = {} if map_elite_parameter is None else map_elite_parameter

        self.cross_pb = cross_pb
        self.mutation_pb = mutation_pb
        self.current_gen = 0
        self.partition_number = partition_number
        self.ps_tree_local_model = ps_tree_local_model
        self.dynamic_partition = dynamic_partition
        self.max_leaf_nodes = max_leaf_nodes
        self.ps_tree_ratio = ps_tree_ratio
        if isinstance(self.ps_tree_ratio, str) and 'Interleave' in self.ps_tree_ratio:
            interleaving_period = int(np.round(n_gen / (n_gen * float(self.ps_tree_ratio.replace('Interleave-', '')))))
        self.interleaving_period = interleaving_period
        self.test_data_size = 0

        if param.get('record_training_data', False):
            self.test_fun[0].regr = self
            self.test_fun[1].regr = self

        self.normalize = normalize
        self.time_statistics = {
            'GP Evaluation': [],
            'ML Evaluation': [],
            'GP Generation': [],
        }
        if normalize is True:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        elif normalize == 'MinMax':
            self.x_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()
        elif normalize in [
            'BackwardDifferenceEncoder',
            'BinaryEncoder',
            'CatBoostEncoder',
            'CountEncoder',
            'HashingEncoder',
            'HelmertEncoder',
            'JamesSteinEncoder',
            'LeaveOneOutEncoder',
            'MEstimateEncoder',
            'OneHotEncoder',
            'OrdinalEncoder',
            'PolynomialEncoder',
            'QuantileEncoder',
            'SumEncoder',
            'SummaryEncoder',
            'TargetEncoder',
        ]:
            self.x_scaler = FeatureTransformer(encoding_scheme=normalize)
            self.y_scaler = StandardScaler()
        elif normalize is False:
            pass
        else:
            raise Exception

        statistical_feature = self.param.get('statistical_feature', False)
        if statistical_feature:
            from autox.autox_competition.feature_engineer import FeatureStat
            class FeatureStatFixed(FeatureStat):
                def fit_transform(self, df, target=None, df_feature_type=None, silence_group_cols=[],
                                  silence_agg_cols=[], select_all=True, max_num=None):
                    return super().fit_transform(df, target, df_feature_type, silence_group_cols, silence_agg_cols,
                                                 select_all, max_num)

            self.x_scaler = Pipeline([
                ("StatisticalFeatures", FeatureStatFixed()),
                ("CategoricalEncoder", self.x_scaler)
            ])

        self.n_process = n_process

        if self.early_stop > 0:
            assert self.bootstrap_training, "Only the bootstrap training mode supports early stopping"

        if self.base_learner == 'NN':
            # Wide&Deep
            self.hidden_layers = [16, 4]
            self.neural_network = MLPRegressor(self.hidden_layers, activation='logistic',
                                               max_iter=1000, early_stopping=True)

        if self.mutation_scheme == 'Transformer':
            self.transformer_switch = True
            self.mutation_scheme = 'uniform-plus'
        else:
            self.transformer_switch = False

        self.novelty_weight = 1
        self.dynamic_target = False
        self.ensemble_cooperation = False
        self.diversity_metric = 'CosineSimilarity'
        if isinstance(self.score_func, str) and '-KL' in self.score_func:
            self.score_func = self.score_func.replace('-KL', '')
            self.diversity_metric = 'KL-Divergence'

        if self.score_func == 'NoveltySearch-Dynamic':
            self.dynamic_target = True
            self.score_func = 'NoveltySearch'
        elif 'WeightedNoveltySearch-' in self.score_func:
            self.novelty_weight = float(self.score_func.split('-')[1])
            self.score_func = 'NoveltySearch'
        elif 'WeightedCooperationSearch-' in self.score_func:
            self.novelty_weight = float(self.score_func.split('-')[1])
            self.score_func = 'NoveltySearch'
            self.ensemble_cooperation = True

        if self.base_learner == 'Hybrid':
            self.tpot_model = TPOTRegressor()
            self.tpot_model._fit_init()
        else:
            self.tpot_model = None

        """
        Some parameters support Multi-task scheme
        However, some parameters do not consider this
        """
        self.base_model_dict = {}
        if isinstance(self.base_learner, BaseEstimator):
            self.base_model_list = None
            self.base_model_dict[self.base_learner.__class__.__name__] = self.base_learner
            self.base_learner = self.base_learner.__class__.__name__
        elif isinstance(self.base_learner, list):
            self.rmp_ratio = 0.5
            self.base_model_dict = {
                learner.__class__.__name__: learner for learner in self.base_learner
            }
            self.base_model_list = ','.join([learner.__class__.__name__ for learner in self.base_learner])
        elif self.base_learner.startswith('DT-LR-'):
            self.rmp_ratio = float(self.base_learner.split('-')[-1])
            self.base_learner = '-'.join(self.base_learner.split('-')[:-1])
            self.base_model_list = 'Random-DT,LogisticRegression'
        elif self.base_learner.startswith('Balanced-DT-LR-'):
            self.rmp_ratio = float(self.base_learner.split('-')[-1])
            self.base_learner = '-'.join(self.base_learner.split('-')[:-1])
            self.base_model_list = 'Balanced-Random-DT,Balanced-LogisticRegression'
        elif self.base_learner == 'DT-LR':
            self.rmp_ratio = 0.5
            if isinstance(self, EvolutionaryForestClassifier):
                self.base_model_list = 'DT,LogisticRegression'
            else:
                self.base_model_list = 'DT,Ridge'
        elif self.base_learner == 'RDT-LR':
            self.rmp_ratio = 0.5
            self.base_model_list = 'Random-DT,LogisticRegression'
        elif self.base_learner == 'Balanced-RDT-LR':
            self.rmp_ratio = 0.5
            self.base_model_list = 'Balanced-Random-DT,Balanced-LogisticRegression'
        elif self.base_learner == 'Balanced-DT-LR':
            self.rmp_ratio = 0.5
            self.base_model_list = 'Balanced-DT,Balanced-LogisticRegression'
        elif self.base_learner == 'DT-LGBM':
            self.base_model_list = 'Random-DT,LightGBM-Stump,DT'
        else:
            self.base_model_list = None

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

    def oob_error(self, pop):
        prediction = np.full((len(pop), len(self.y)), np.nan, dtype=np.float)
        for i, x in enumerate(pop):
            index = x.out_of_bag
            prediction[i][index] = x.oob_prediction
        label = np.nanmean(prediction, axis=0)
        label = np.nan_to_num(label)
        accuracy = r2_score(self.y, label)
        if self.verbose:
            print('oob score', accuracy)
        return accuracy

    def feature_quick_evaluation(self, gene):
        # quickly evaluate semantic information of an individual based on first 20 data items
        # usage: surrogate model
        func = compile(gene, self.pset)
        Yp = result_calculation([func], self.X[:20], False)
        return tuple(Yp.flatten())

    def get_model_coefficient(self, x):
        if isinstance(x['Ridge'], Pipeline):
            base_learner = x['Ridge'][-1]
        else:
            base_learner = x['Ridge']

        if isinstance(base_learner, (LinearModel, LinearClassifierMixin)):
            if len(base_learner.coef_.shape) == 2:
                coef = np.max(np.abs(base_learner.coef_), axis=0)[:self.gene_num]
            else:
                coef = np.abs(base_learner.coef_)[:self.gene_num]
        elif isinstance(base_learner, (BaseDecisionTree, LGBMModel)):
            coef = base_learner.feature_importances_[:self.gene_num]
        elif isinstance(base_learner, (GBDTLRClassifier)):
            coef = base_learner.gbdt_.feature_importances_[:self.gene_num]
        elif isinstance(base_learner, SVR):
            coef = np.ones(self.X.shape[1])
        elif isinstance(base_learner, (RidgeDT, LRDTClassifier)):
            coef = base_learner.feature_importance
        else:
            raise Exception
        return coef

    def fitness_evaluation(self, individual: MultipleGeneGP):
        # single individual evaluation
        X, Y = self.X, self.y
        Y = Y.flatten()

        func = self.toolbox.compile(individual)
        if self.base_learner == 'Dynamic-DT':
            self.min_samples_leaf = individual.dynamic_leaf_size
            pipe = self.get_base_model()
            self.min_samples_leaf = 1
        elif self.base_learner == 'Hybrid':
            pipe = self.get_base_model(base_model=individual.base_model)
        elif self.base_learner in ['DT-LR', 'Balanced-DT-LR', 'Balanced-RDT-LR', 'DT-LGBM',
                                   'RDT~LightGBM-Stump'] or \
            isinstance(self.base_learner, list):
            pipe = self.get_base_model(base_model=individual.base_model)
        elif self.base_learner == 'Dynamic-LogisticRegression':
            pipe = self.get_base_model(regularization_ratio=individual.dynamic_regularization)
        else:
            pipe = self.get_base_model()

        if self.base_learner == 'Soft-PLTree':
            pipe.partition_scheme = individual.partition_scheme

        # compile basic genes
        if self.basic_gene_num > 0:
            basic_gene = self.toolbox.compile(individual, basic_gene=True)
        else:
            basic_gene = []

        # send task to the job pool and waiting results
        if self.n_process > 1:
            y_pred, estimators, information = yield (pipe, dill.dumps(func, protocol=-1),
                                                     dill.dumps(basic_gene, protocol=-1))
        else:
            y_pred, estimators, information = yield (pipe, func, basic_gene)

        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if not 'CV' in self.score_func:
            assert len(y_pred) == len(Y), (len(y_pred), len(Y))
        self.time_statistics['GP Evaluation'].append(information['gp_evaluation_time'])
        self.time_statistics['ML Evaluation'].append(information['ml_evaluation_time'])
        if self.dynamic_partition == 'Self-Adaptive' and self.base_learner == 'Soft-PLTree':
            individual.partition_scheme = information['best_label']
            # del pipe.partition_scheme
            if not isinstance(self.partition_model['K-Means'], DBSCAN):
                assert np.all(individual.partition_scheme < self.partition_number)

        # calculate terminal importance based on the permutation importance method
        if self.mutation_scheme == 'EDA-Terminal-PMI':
            kcv = get_cv_splitter(pipe, self.cv)
            all_importance_values = []
            for id, index in enumerate(kcv.split(X, Y)):
                def prediction_function(X):
                    Yp = result_calculation(func, X, self.original_features)
                    return estimators[id].predict(Yp)

                train_index, test_index = index
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                importance_value = feature_importance_permutation(X_test, y_test, prediction_function, 'r2')[0]
                all_importance_values.append(importance_value)
            all_importance_values = np.mean(all_importance_values, axis=0)
            individual.terminal_importance_values = all_importance_values

        if self.base_learner == 'DT-RandomDT':
            self.base_learner = 'Random-DT'
            individual.pipe = self.get_base_model()
            self.base_learner = 'DT-RandomDT'
        elif self.base_learner == 'RandomDT-DT':
            self.base_learner = 'DT'
            individual.pipe = self.get_base_model()
            self.base_learner = 'RandomDT-DT'
        elif self.base_learner == 'SimpleDT-RandomDT':
            self.base_learner = 'Random-DT'
            individual.pipe = self.get_base_model()
            self.base_learner = 'SimpleDT-RandomDT'
        else:
            if self.cv == 1:
                individual.pipe = estimators[0]
            else:
                individual.pipe = pipe

        if self.bootstrap_training or self.eager_training:
            Yp = result_calculation(func, self.X, self.original_features)
            self.train_final_model(individual, Yp, Y)

        if self.bootstrap_training:
            # calculate prediction
            Yp_oob = Yp[individual.out_of_bag]
            if isinstance(self, ClassifierMixin):
                individual.oob_prediction = individual.pipe.predict_proba(Yp_oob)
            else:
                individual.oob_prediction = individual.pipe.predict(Yp_oob)

        individual.predicted_values = y_pred

        # minimize objective function
        self.calculate_case_values(individual, estimators, Y, y_pred)

        if 'CV' in self.score_func:
            assert len(individual.case_values) % 5 == 0
        elif self.score_func == 'CDFC':
            assert len(individual.case_values) == len(np.unique(Y))
        else:
            assert len(individual.case_values) == len(Y), len(individual.case_values)

        # individual.coef = np.mean([self.get_model_coefficient(x) for x in estimators], axis=0)
        individual.coef = np.max([self.get_model_coefficient(x) for x in estimators], axis=0)
        individual.coef = individual.coef / np.sum(individual.coef)
        individual.coef = np.nan_to_num(individual.coef, posinf=0, neginf=0)

        if self.verbose:
            pass

        if self.base_learner == 'Random-DT-Plus':
            if isinstance(self, ClassifierMixin):
                individual.pipe = EnsembleClassifier(estimators)
            elif isinstance(self, RegressorMixin):
                individual.pipe = EnsembleRegressor(estimators)
            else:
                raise Exception

        # count the number of nodes in decision tree
        if hasattr(estimators[0]['Ridge'], 'tree_'):
            individual.node_count = [estimators[i]['Ridge'].tree_.node_count for i in range(len(estimators))]

        # final score
        yield self.calculate_fitness_value(individual, estimators, Y, y_pred)

    def calculate_fitness_value(self, individual, estimators, Y, y_pred):
        """
        Smaller is better because the weight is -1.
        """
        if self.score_func == 'R2' or self.score_func == 'NoveltySearch' or self.score_func == 'MAE':
            score = r2_score(Y, y_pred)
            if self.weighted_coef:
                individual.coef = np.array(individual.coef) * score
            # print('r2_score',score)
            return -1 * score,
        elif self.score_func == 'MSE-Variance':
            error = mean_squared_error(Y, y_pred)
            return np.mean(error) + 0.01 * np.std(error),
        elif self.score_func == 'Lower-Bound':
            return np.max(mean_squared_error(Y, y_pred)),
        elif self.score_func == 'Spearman':
            return -1 * spearman(Y, y_pred),
        elif self.score_func == 'CV-NodeCount':
            return [estimators[i]['Ridge'].tree_.node_count for i in range(len(estimators))]
        elif 'CV' in self.score_func:
            return -1 * np.mean(y_pred),
        else:
            raise Exception

    def calculate_case_values(self, individual, estimators, Y, y_pred):
        # Minimize fitness values
        if self.score_func == 'R2' or self.score_func == 'MSE-Variance' or self.score_func == 'Lower-Bound':
            individual.case_values = ((y_pred - Y.flatten()).flatten()) ** 2
        elif self.score_func == 'MAE':
            individual.case_values = np.abs(((y_pred - Y.flatten()).flatten()))
        elif self.score_func == 'Spearman':
            individual.case_values = np.abs(rankdata(y_pred) - rankdata(Y.flatten())).flatten()
        elif 'CV' in self.score_func:
            individual.case_values = -1 * y_pred
        elif self.score_func == 'NoveltySearch':
            base_values = (y_pred.flatten() - Y.flatten()) ** 2
            if len(self.hof) == 0:
                # first generation
                # individual.case_values = base_values
                individual.case_values = np.concatenate([base_values, np.full_like(base_values, 0)])
            else:
                # maximize cross entropy
                ensemble_value = np.mean([x.predicted_values for x in self.hof], axis=0)
                ambiguity = (y_pred.flatten() - ensemble_value) ** 2
                ambiguity *= self.novelty_weight
                assert len(ambiguity) == len(y_pred.flatten())
                # individual.case_values = base_values - ambiguity
                individual.case_values = np.concatenate([base_values, -1 * ambiguity])
                # individual.case_values = np.concatenate([base_values, base_values])
        else:
            raise Exception

    def train_final_model(self, individual, Yp, Y, force_training=False):
        if self.base_learner == 'Fast-Soft-PLTree-EM':
            self.base_learner = 'Soft-PLTree-EM'
            individual.pipe = self.get_base_model()
        # avoid re-training
        model = individual.pipe
        if not force_training:
            # check the necessity of training
            try:
                if hasattr(individual, 'active_gene_num') and individual.active_gene_num > 0:
                    input_size = individual.active_gene_num
                else:
                    input_size = len(individual.gene)
                if self.base_learner == 'NN':
                    input_size += self.hidden_layers[-1]
                if hasattr(model, 'partition_scheme'):
                    input_size += 1
                if self.original_features:
                    input_size += self.X.shape[1]
                model.predict(np.ones((1, input_size)))
                return None
            except NotFittedError:
                pass

        # ensure ensemble base leaner will not be retrained
        assert self.base_learner != 'Random-DT-Plus'

        if self.base_learner == 'NN':
            nn_prediction = get_activations(self.neural_network, self.X)[-2]
            Yp = np.concatenate([Yp, nn_prediction], axis=1)

        if hasattr(model, 'partition_scheme'):
            # partition_scheme = individual.partition_scheme
            partition_scheme = model.partition_scheme
            Yp = np.concatenate([Yp, np.reshape(partition_scheme, (-1, 1))], axis=1)

        # train final model
        if self.bootstrap_training and not force_training:
            sample = np.random.randint(0, len(Y), size=(len(Y)))

            # get out-of-bag indices
            chosen = np.zeros_like(Y)
            chosen[sample] = 1
            out_of_bag = np.where(chosen == 0)[0]
            individual.out_of_bag = out_of_bag

            # convert sample index to weight
            n_samples = len(Y)
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
            sample_counts = np.bincount(sample, minlength=n_samples)
            curr_sample_weight *= sample_counts

            individual.pipe.fit(Yp, Y, Ridge__sample_weight=curr_sample_weight)
        else:
            individual.pipe.fit(Yp, Y)
            out_of_bag = None

        # feature importance generation
        base_model = model['Ridge']
        if hasattr(base_model, 'feature_importances_'):
            individual.coef = base_model.feature_importances_[:self.gene_num]
        elif hasattr(base_model, 'coef_'):
            individual.coef = np.abs(base_model.coef_[:self.gene_num]).flatten()
        # assert len(individual.coef) == self.gene_num
        return out_of_bag

    def entropy_calculation(self):
        pass

    def get_base_model(self, regularization_ratio=1, base_model=None):
        # Get the base model based on the hyperparameter
        if self.base_learner in ['DT', 'DT-RandomDT', 'PCA-DT'] or \
            base_model in ['DT', 'DT-RandomDT', 'PCA-DT']:
            ridge_model = DecisionTreeRegressor(max_depth=self.max_tree_depth,
                                                min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'SimpleDT-RandomDT':
            ridge_model = DecisionTreeRegressor(max_depth=3)
        elif self.base_learner in ['Random-DT', 'RandomDT-DT', 'Random-DT-Plus'] or \
            base_model in ['Random-DT', 'RandomDT-DT', 'Random-DT-Plus']:
            ridge_model = DecisionTreeRegressor(splitter='random', max_depth=self.max_tree_depth,
                                                min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'PL-Tree':
            ridge_model = LinearTreeRegressor(base_estimator=LinearRegression())
        elif self.base_learner == 'RidgeCV':
            ridge_model = RidgeCV(alphas=[0.1, 1, 10], store_cv_values=True,
                                  scoring=make_scorer(r2_score))
        elif self.base_learner == 'LR':
            ridge_model = LinearRegression()
        elif self.base_learner == 'Ridge' or base_model == 'Ridge' \
            or 'Fast-Soft-PLTree' in self.base_learner \
            or self.base_learner in ['Fast-PLTree', 'Fast-RidgeDT',
                                     'Fast-Simple-RidgeDT', 'Fast-RidgeDT-Plus'] or \
            self.base_learner == 'NN':
            ridge_model = Ridge()
        elif self.base_learner == 'RidgeDT':
            ridge_model = RidgeDT(decision_tree_count=self.decision_tree_count,
                                  max_leaf_nodes=self.max_leaf_nodes)
        elif self.base_learner == 'Simple-RidgeDT':
            ridge_model = RidgeDTPlus(decision_tree_count=0,
                                      max_leaf_nodes=self.max_leaf_nodes)
        elif self.base_learner == 'RidgeDT-Plus':
            ridge_model = RidgeDTPlus(decision_tree_count=self.decision_tree_count,
                                      max_leaf_nodes=self.max_leaf_nodes)
        elif self.base_learner == 'ET':
            ridge_model = ExtraTreesRegressor(n_estimators=100)
        elif self.base_learner == 'GBDT':
            ridge_model = GradientBoostingRegressor(learning_rate=0.8, n_estimators=5)
        elif self.base_learner == 'LightGBM':
            ridge_model = LGBMRegressor(n_estimators=10, learning_rate=1, n_jobs=1)
        elif self.base_learner == 'LightGBM-Stump' or base_model == 'LightGBM-Stump':
            ridge_model = LGBMRegressor(max_depth=1, learning_rate=1,
                                        n_estimators=math.ceil(np.log2(self.X.shape[0])),
                                        n_jobs=1)
        elif self.base_learner == 'Hybrid':
            ridge_model = base_model
        elif self.base_learner == 'RBF':
            ridge_model = RBFN(min(self.X.shape[0], 64))
        elif self.base_learner == 'SVR':
            ridge_model = SVR()
        elif self.base_learner == 'PLTree':
            min_samples_leaf = int(self.partition_number.split('-')[1])
            partition_number = int(self.partition_number.split('-')[2])
            ridge_model = PLTreeRegressor(min_samples_leaf=min_samples_leaf,
                                          max_leaf_nodes=partition_number)
        elif self.base_learner == 'Soft-PLTree' or self.base_learner == 'Soft-PLTree-EM':
            if self.base_learner == 'Soft-PLTree':
                model = SoftPLTreeRegressor
            else:
                model = SoftPLTreeRegressorEM
            feature_num = self.gene_num
            if self.original_features:
                feature_num += self.X.shape[1]
            ridge_model = model(
                min_samples_leaf=self.min_samples_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                feature_num=feature_num,
                gene_num=self.gene_num,
                base_model=self.ps_tree_local_model,
                partition_model=self.ps_tree_partition_model,
                partition_number=self.partition_number,
                only_original_features=self.only_original_features)
        elif isinstance(self.base_learner, RegressorMixin):
            ridge_model = self.base_learner
        else:
            raise Exception
        if self.base_learner == 'PCA-DT':
            # projection_matrix = np.random.random((self.gene_num, self.gene_num))
            # random_projection = StaticRandomProjection(projection_matrix)
            pipe = Pipeline([
                ("Scaler", StandardScaler()),
                ("Rotation", PCA(n_components=5)),
                ("Ridge", ridge_model),
            ])
        else:
            pipe = Pipeline([
                ("Scaler", StandardScaler()),
                ("Ridge", ridge_model),
            ])
        if isinstance(pipe['Ridge'], BaseDecisionTree) and self.max_tree_depth != None:
            assert pipe['Ridge'].max_depth == self.max_tree_depth
        return pipe

    def transform(self, X, ratio=0.5):
        ratio = 1 - ratio
        if self.normalize:
            X = self.x_scaler.transform(X)
        code_importance_dict = get_feature_importance(self, simple_version=False, fitness_weighted=False)
        if self.ensemble_size == 1:
            top_features = list(code_importance_dict.keys())
        else:
            top_features = select_top_features(code_importance_dict, ratio)
        transformed_features = feature_append(self, X, top_features, only_new_features=True)
        return transformed_features

    def lazy_init(self, x):
        if self.gene_num == 'Adaptive':
            self.gene_num = min(x.shape[1], 20)
        if isinstance(self.gene_num, str):
            self.gene_num = min(int(self.gene_num.replace('X', '')) * self.X.shape[1], 30)
        pset = self.primitive_initialization(x)

        if hasattr(gp, 'rand101'):
            delattr(gp, 'rand101')
        if self.basic_primitives == False:
            pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1), NumericalFeature)
        elif self.basic_primitives == 'float':
            pset.addEphemeralConstant("rand101", lambda: random.random())
        elif self.basic_primitives == 'no-constant':
            pass
        else:
            pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("FeatureIndividual", MultipleGeneGP, fitness=creator.FitnessMin)

        self.archive_initialization()
        if self.transformer_switch:
            self.transformer_tool = TransformerTool(self.X, self.y, self.hof, self.pset)

        # toolbox initialization
        toolbox = base.Toolbox()
        self.toolbox = toolbox
        if self.feature_selection:
            terminal_prob = self.get_terminal_probability()
            toolbox.register("expr", genHalfAndHalf, pset=pset, min_=1, max_=2, terminal_prob=terminal_prob)
        else:
            # generate initial population
            if self.initial_tree_size is None:
                toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
            else:
                min_height, max_height = self.initial_tree_size.split('-')
                min_height = int(min_height)
                max_height = int(max_height)
                toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_height, max_=max_height)

        # individual initialization
        partition_scheme = self.partition_scheme_initialization()
        toolbox.register("individual", multiple_gene_initialization, creator.FeatureIndividual, toolbox.expr,
                         gene_num=self.gene_num, basic_gene_num=self.basic_gene_num,
                         pset=self.pset, tpot_model=self.tpot_model, partition_scheme=partition_scheme,
                         base_model_list=self.base_model_list)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", multiple_gene_compile, pset=pset)

        toolbox.register('clone', efficient_deepcopy)

        toolbox.register("evaluate", self.fitness_evaluation)
        if self.select == 'Tournament':
            toolbox.register("select", tools.selTournament, tournsize=self.param['tournament_size'])
        elif self.select.startswith('Tournament-'):
            toolbox.register("select", tools.selTournament, tournsize=int(self.select.split('-')[1]))
        elif self.select.startswith('TournamentNovelty'):
            toolbox.register("select", selTournamentNovelty, tournsize=int(self.select.split('-')[1]))
        elif self.select.startswith('TournamentPlus-'):
            toolbox.register("select", selTournamentPlus, tournsize=int(self.select.split('-')[1]))
        elif self.select == 'BatchTournament':
            toolbox.register("select", batch_tournament_selection)
        elif self.select in ['AutomaticLexicase'] + map_elite_series:
            toolbox.register("select", selAutomaticEpsilonLexicaseFast)
        elif self.select == 'AutomaticLexicaseFast':
            toolbox.register("select", selAutomaticEpsilonLexicaseFast)
        elif self.select == 'DoubleRound':
            toolbox.register("select", selDoubleRound)
        elif self.select == 'DoubleRound-Random':
            toolbox.register("select", selDoubleRound, count=self.param['double_round_count'],
                             base_operator='Random')
        elif self.select == 'DoubleRound-Tournament':
            toolbox.register("select", selDoubleRound, count=self.param['double_round_count'],
                             base_operator='Tournament',
                             tournsize=self.param['double_round_tournament_size'])
        elif self.select == 'Parsimonious-DoubleRound-Tournament':
            toolbox.register("select", selDoubleRound, count=self.param['double_round_count'],
                             base_operator='Tournament',
                             tournsize=self.param['double_round_tournament_size'],
                             parsimonious=True)
        elif self.select == 'RandomPlus':
            toolbox.register("select", selRandomPlus)
        elif self.select == 'Bagging':
            toolbox.register("select", selBagging)
        elif self.select == 'AutomaticLexicaseK':
            toolbox.register("select", selAutomaticEpsilonLexicaseK)
        elif self.select == 'Lexicase':
            toolbox.register("select", selLexicase)
        elif self.select == 'GPED':
            toolbox.register("select", selGPED)
        elif self.select in ['Random']:
            toolbox.register("select", selRandom)
        elif self.select == 'Hybrid':
            toolbox.register("select", selHybrid)
        else:
            raise Exception

        if 'uniform' in self.mutation_scheme or 'novelty' in self.mutation_scheme \
            or 'weighted-cross' in self.mutation_scheme or 'all_gene' in self.mutation_scheme \
            or 'biased' in self.mutation_scheme:
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

            # decision threshold for feature crossover and mutation
            threshold_ratio = convert_to_int(crossover_operator.split('-')[-1])
            if threshold_ratio is None:
                threshold_ratio = 0.2
            else:
                crossover_operator = '-'.join(crossover_operator.split('-')[:-1])
            # calculate the threshold for crossover
            self.cx_threshold_ratio = threshold_ratio

            if self.basic_primitives == False:
                toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
            else:
                toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

            # special crossover operators
            if 'weighted-cross' in crossover_operator:
                # crossover features using feature importance value as the probability
                if crossover_operator == 'weighted-cross-positive':
                    toolbox.register("mate", feature_crossover, positive=True, threshold_ratio=threshold_ratio)
                elif crossover_operator == 'weighted-cross-negative':
                    toolbox.register("mate", feature_crossover, positive=False, threshold_ratio=threshold_ratio)
                elif crossover_operator == 'weighted-cross-PBIC':
                    toolbox.register("mate", cxOnePoint_all_gene_with_importance_probability)
                elif crossover_operator == 'weighted-cross-cross':
                    toolbox.register("mate", cxOnePoint_multiple_gene_cross)
                elif 'weighted-cross-threshold' in crossover_operator:
                    # feature-importance based crossover
                    threshold = None
                    if '~' in crossover_operator:
                        # parse threshold
                        self.mutation_scheme, threshold = crossover_operator.split('~')
                        threshold = float(threshold)
                        crossover_operator = self.mutation_scheme
                    if crossover_operator == 'weighted-cross-threshold-EDA':
                        # weighted crossover with EDA terminals
                        self.terminal_prob = np.ones(self.pset.terms_count)
                        self.terminal_prob /= np.sum(self.terminal_prob)
                        self.primitive_prob = np.ones(self.pset.prims_count)
                        self.primitive_prob /= np.sum(self.primitive_prob)
                        partial_func = partial(genFull_with_prob, primitive_probs=self.primitive_prob,
                                               terminal_probs=self.terminal_prob, min_=0, max_=2)
                        mutation = partial(mutUniform, expr=partial_func, pset=pset)
                    else:
                        mutation = partial(mutUniform, expr=toolbox.expr_mut, pset=pset)
                    # replace all useless features with the crossover results of features
                    toolbox.register("mate", cxOnePoint_multiple_gene_threshold, mutation=mutation,
                                     cross_pb=self.cross_pb, threshold=threshold)
                    self.cross_pb = 1
                    self.mutation_pb = 1
                elif crossover_operator == 'weighted-cross-best':
                    toolbox.register("mate", cxOnePoint_multiple_gene_best)
                elif crossover_operator == 'weighted-cross-worst':
                    toolbox.register("mate", cxOnePoint_multiple_gene_worst)
                elif crossover_operator == 'weighted-cross-cross-global-mean':
                    toolbox.register("mate", feature_crossover_cross_global, regressor=self)
                    self.good_features_threshold = 'mean'
                elif crossover_operator == 'weighted-cross-cross-global-inverse':
                    toolbox.register("mate", feature_crossover_cross_global, regressor=self)
                    self.good_features_threshold = 1 - self.cx_threshold_ratio
                elif crossover_operator == 'weighted-cross-cross-global':
                    toolbox.register("mate", feature_crossover_cross_global, regressor=self)
                    self.good_features_threshold = self.cx_threshold_ratio
                else:
                    raise Exception
            elif crossover_operator == 'all_gene':
                toolbox.register("mate", cxOnePoint_all_gene)
            elif crossover_operator == 'all_gene_permutation':
                toolbox.register("mate", partial(cxOnePoint_all_gene, permutation=True))
            elif crossover_operator == 'biased':
                toolbox.register("mate", cxOnePoint_multiple_gene_biased)
            elif crossover_operator == 'novelty':
                # ensure crossover results are different from parent individuals
                def test_func(individual):
                    func = compile(individual, pset)
                    Yp = result_calculation([func], self.X[:20], self.original_features)
                    return Yp.flatten()

                toolbox.register("mate", cxOnePoint_multiple_gene_novelty, test_func=test_func)
            elif crossover_operator == 'uniform-plus-semantic':
                toolbox.register("mate", cxOnePoint_multiple_gene_diversity, visited_features=self.generated_features)
            elif crossover_operator == 'uniform-plus-same-index':
                toolbox.register("mate", cxOnePoint_multiple_gene_same_index)
            else:
                assert 'uniform' in crossover_operator
                toolbox.register("mate", cxOnePoint_multiple_gene)

            # special mutation operators
            if mutation_operator is None:
                if self.transformer_switch:
                    # using transformer to generate sub-trees
                    def condition_probability():
                        return self.current_gen / self.n_gen

                    toolbox.register("mutate", mutUniform_multiple_gene_transformer, expr=toolbox.expr_mut, pset=pset,
                                     condition_probability=condition_probability, transformer=self.transformer_tool)
                else:
                    toolbox.register("mutate", mutUniform_multiple_gene, expr=toolbox.expr_mut, pset=pset)
            elif mutation_operator == 'worst':
                toolbox.register("mutate", mutUniform_multiple_gene_worst, expr=toolbox.expr_mut, pset=pset)
            elif mutation_operator == 'threshold':
                toolbox.register("mutate", mutUniform_multiple_gene_threshold, expr=toolbox.expr_mut, pset=pset)
            elif mutation_operator == 'weighted-mutation':
                toolbox.register("mutate", mutWeight_multiple_gene, expr=toolbox.expr_mut, pset=pset)
            elif mutation_operator == 'pool-mutation':
                toolbox.register("mutate", pool_based_mutation, expr=toolbox.expr_mut, pset=pset,
                                 regressor=self)
                # threshold for good features
                self.good_features_threshold = 1 - self.cx_threshold_ratio
            elif 'pool-mutation-tournament' in mutation_operator:
                tournament_size = int(mutation_operator.split('-')[-1])
                toolbox.register("mutate", pool_based_mutation, expr=toolbox.expr_mut, pset=pset,
                                 regressor=self, tournament_size=tournament_size)
                self.good_features_threshold = self.cx_threshold_ratio
            elif mutation_operator == 'pool-mutation-pearson':
                toolbox.register("mutate", pool_based_mutation, expr=toolbox.expr_mut, pset=pset,
                                 regressor=self, pearson_selection=True)
                self.good_features_threshold = self.cx_threshold_ratio
            elif mutation_operator == 'pool-mutation-semantic':
                toolbox.register("mutate", pool_based_mutation, expr=toolbox.expr_mut, pset=pset,
                                 regressor=self, feature_evaluation=self.feature_quick_evaluation)
                self.good_features_threshold = self.cx_threshold_ratio
            elif mutation_operator == 'weighted-mutation-global':
                toolbox.register("mutate", feature_mutation_global, expr=toolbox.expr_mut, pset=pset,
                                 regressor=self)
            else:
                raise Exception
        elif self.mutation_scheme == 'weighted':
            toolbox.register("mate", cxOnePoint_multiple_gene_weighted)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", mutUniform_multiple_gene_weighted, expr=toolbox.expr_mut, pset=pset)
        elif self.mutation_scheme in eda_operators:
            # Define the crossover operator combined the EDA operator
            if 'Biased' in self.mutation_scheme:
                toolbox.register("mate", cxOnePoint_multiple_gene_biased)
            elif 'SameWeight' in self.mutation_scheme:
                toolbox.register("mate", cxOnePoint_multiple_gene_same_weight)
            elif 'Tournament' in self.mutation_scheme:
                toolbox.register("mate", cxOnePoint_multiple_gene_tournament)
            elif 'SC' in self.mutation_scheme:
                toolbox.register("mate", cxOnePoint_multiple_gene_SC)
            elif 'SameIndex' in self.mutation_scheme:
                toolbox.register("mate", cxOnePoint_multiple_gene_same_index)
            else:
                toolbox.register("mate", cxOnePoint_multiple_gene)
            self.primitive_prob = np.zeros(self.pset.prims_count)
            self.terminal_prob = np.zeros(self.pset.terms_count)
            if 'Terminal' in self.mutation_scheme:
                self.primitive_prob = np.ones(self.pset.prims_count)
            if self.mutation_scheme == 'EDA-Terminal':
                # Sample terminals from a dirichlet distribution
                # The advantage is that this method supports prior distribution
                partial_func = partial(genFull_with_prob, primitive_probs=self.primitive_prob,
                                       terminal_probs=self.terminal_prob, sample_type='Dirichlet')
            else:
                partial_func = partial(genFull_with_prob, primitive_probs=self.primitive_prob,
                                       terminal_probs=self.terminal_prob)
            toolbox.register("expr_mut", partial_func,
                             min_=0, max_=2)
            toolbox.register("mutate", mutUniform_multiple_gene, expr=toolbox.expr_mut, pset=pset)
            # shrink_mutation = False
            # if shrink_mutation:
            #     toolbox.register("mutate", mutShrink_multiple_gene, expr=toolbox.expr_mut, pset=pset)
            # else:
            #     toolbox.register("mutate", mutUniform_multiple_gene, expr=toolbox.expr_mut, pset=pset)
        elif self.mutation_scheme in ['probability']:
            toolbox.register("mate", cxOnePoint_all_gene_with_probability, probability=self.cross_pb)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", mutUniform_multiple_gene_with_probability, expr=toolbox.expr_mut, pset=pset,
                             probability=self.mutation_pb)
            self.cross_pb = 1
            self.mutation_pb = 1
        else:
            raise Exception

        def dynamic_height():
            if len(self.generated_features) >= \
                np.log(len(self.pset.primitives) * (2 ** self.current_height - 1) * \
                       len(self.pset.terminals) * (2 ** self.current_height)) / \
                np.log(1.01):
                self.current_height += 1
            return self.current_height

        if self.verbose:
            history = History()
            self.history = history
            # Decorate the variation operators
            toolbox.decorate("mate", history.decorator)
            toolbox.decorate("mutate", history.decorator)

        if self.multi_gene_mutation():
            pass
        elif self.max_height == 'dynamic':
            toolbox.decorate("mate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                               max_value=dynamic_height))
            toolbox.decorate("mutate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                 max_value=dynamic_height))
        elif isinstance(self.max_height, str) and 'size-' in self.max_height:
            size = int(self.max_height.split('-')[1])
            toolbox.decorate("mate", staticLimit_multiple_gene(key=lambda x: len(x),
                                                               max_value=size))
            toolbox.decorate("mutate", staticLimit_multiple_gene(key=lambda x: len(x),
                                                                 max_value=size))
        else:
            toolbox.decorate("mate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                               max_value=self.max_height))
            toolbox.decorate("mutate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                 max_value=self.max_height))

        self.pop = toolbox.population(n=self.n_pop)

        self.random_index = np.random.randint(0, len(self.X), 5)

        def shuffle_along_axis(a, axis):
            idx = np.random.rand(*a.shape).argsort(axis=axis)
            return np.take_along_axis(a, idx, axis=axis)

        self.prediction_x = shuffle_along_axis(np.copy(self.X), axis=0)

        if self.base_learner == 'NN':
            cv = get_cv_splitter(self.neural_network, self.cv)
            self.nn_prediction = np.zeros((len(self.y), self.hidden_layers[-1]))

            X, y = self.X, self.y
            cv_loss = []
            for train_index, test_index in cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.neural_network.fit(X_train, y_train)
                cv_loss.append(r2_score(y_test, self.neural_network.predict(X_test)))
                self.nn_prediction[test_index] = get_activations(self.neural_network, X_test)[-2]
            # print('Mean CV loss', np.mean(cv_loss))

            self.neural_network.fit(self.X, self.y)
        else:
            self.nn_prediction = None

    def primitive_initialization(self, x):
        # initialize the primitive set
        if self.basic_primitives == False:
            feature_types = self.feature_types
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
                pset.addPrimitive(np.maximum, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.minimum, [NumericalFeature, NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.sin, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.cos, [NumericalFeature], NumericalFeature)
                # pset.addPrimitive(np.square, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(_protected_sqrt, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(identical_numerical, [NumericalFeature], GeneralFeature)
            if has_categorical_feature:
                pset.addPrimitive(np_bit_wrapper(np.bitwise_and), [CategoricalFeature, CategoricalFeature],
                                  CategoricalFeature)
                pset.addPrimitive(np_bit_wrapper(np.bitwise_or), [CategoricalFeature, CategoricalFeature],
                                  CategoricalFeature)
                pset.addPrimitive(np_bit_wrapper(np.bitwise_xor), [CategoricalFeature, CategoricalFeature],
                                  CategoricalFeature)
                pset.addPrimitive(identical_categorical, [CategoricalFeature], GeneralFeature)
            if has_boolean_feature:
                if has_numerical_feature:
                    pset.addPrimitive(np.greater, [NumericalFeature, NumericalFeature], BooleanFeature)
                    pset.addPrimitive(np.less, [NumericalFeature, NumericalFeature], BooleanFeature)
                pset.addPrimitive(np.logical_and, [BooleanFeature, BooleanFeature], BooleanFeature)
                pset.addPrimitive(np.logical_or, [BooleanFeature, BooleanFeature], BooleanFeature)
                pset.addPrimitive(np.logical_xor, [BooleanFeature, BooleanFeature], BooleanFeature)
                # pset.addPrimitive(np.equal, [BooleanFeature, BooleanFeature], BooleanFeature)
                # pset.addPrimitive(np.logical_not, [BooleanFeature], BooleanFeature)
                pset.addPrimitive(identical_boolean, [BooleanFeature], GeneralFeature)
        elif self.basic_primitives == 'extend':
            pset = MultiplePrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_extend_operators(pset)
        elif self.basic_primitives == 'extend-triple':
            pset = MultiplePrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_extend_operators(pset, triple_groupby_operators=True)
        elif self.basic_primitives == 'extend-simple':
            pset = MultiplePrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_extend_operators(pset, groupby_operators=False)
        elif isinstance(self.basic_primitives, str) and 'optimal' in self.basic_primitives:
            pset = MultiplePrimitiveSet("MAIN", x.shape[1])
            # support multivariate functions
            count = 0
            if '-' in self.basic_primitives:
                count = int(self.basic_primitives.split('-')[1])

                def np_add_multiple(*arg):
                    return np.sum(arg, axis=0)

                def np_max_multiple(*arg):
                    return np.max(arg, axis=0)

                def np_min_multiple(*arg):
                    return np.min(arg, axis=0)

                for i in range(3, count + 1):
                    np_add_multiple.__name__ = f'np_add_multiple_{i}'
                    np_max_multiple.__name__ = f'np_max_multiple_{i}'
                    np_min_multiple.__name__ = f'np_min_multiple_{i}'
                    pset.addPrimitive(np_add_multiple, i)
                    pset.addPrimitive(np_max_multiple, i)
                    pset.addPrimitive(np_min_multiple, i)

            self.basic_primitives = ','.join([
                'add', 'subtract', 'multiply', 'analytical_quotient',
                'protect_sqrt', 'sin', 'cos', 'maximum', 'minimum', 'negative',
            ])
            self.add_primitives_to_pset(pset)
            if count != 0:
                self.basic_primitives = f'optimal-{count}'
        elif self.basic_primitives == 'DIGEN':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            self.basic_primitives = ','.join([
                'add', 'subtract', 'multiply', 'analytical_quotient',
                'protect_sqrt', 'sin', 'cos', 'maximum', 'minimum', 'negative',
                'greater_or_equal_than', 'less_or_equal_than'
            ])
            self.add_primitives_to_pset(pset)
        elif self.basic_primitives == 'extend-plus':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_extend_operators(pset, addtional=True)
        elif self.basic_primitives == 'CDFC':
            # primitives used in CDFC
            # "Genetic programming for multiple-feature construction on high-dimensional classification"
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(np.maximum, 2)
            pset.addPrimitive(if_function, 3)
        elif self.basic_primitives == 'S-GP':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(np.maximum, 2)
            pset.addPrimitive(np.minimum, 2)
        elif self.basic_primitives == 'logical':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            # add_basic_operators(pset)
            add_logical_operators(pset)
        elif self.basic_primitives == 'relation':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            add_relation_operators(pset)
        elif self.basic_primitives == 'normalized':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])

            def tanh_wrapper(func):
                def simple_func(x1, x2):
                    return np.tanh(func(x1, x2))

                simple_func.__name__ = f'tanh_{func.__name__}'
                return simple_func

            pset.addPrimitive(tanh_wrapper(np.add), 2)
            pset.addPrimitive(tanh_wrapper(np.subtract), 2)
            pset.addPrimitive(tanh_wrapper(np.multiply), 2)
            pset.addPrimitive(tanh_wrapper(analytical_quotient), 2)
        elif self.basic_primitives == 'sin':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(np.sin, 1)
        elif self.basic_primitives == 'sin-cos':
            pset = MultiplePrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(np.sin, 1)
            pset.addPrimitive(np.cos, 1)
        elif self.basic_primitives == 'sin-tanh':
            pset = MultiplePrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(np.sin, 1)
            pset.addPrimitive(np.tanh, 1)
        elif self.basic_primitives == 'sigmoid':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(sigmoid, 1)
        elif self.basic_primitives == 'leaky_relu':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(leaky_relu, 1)
        elif self.basic_primitives == 'interpretable':
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            pset.addPrimitive(np.add, 2)
            pset.addPrimitive(np.subtract, 2)
            pset.addPrimitive(np.multiply, 2)
            pset.addPrimitive(protect_division, 2)
        elif isinstance(self.basic_primitives, str) and ',' in self.basic_primitives:
            # an array of basic primitives
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            self.add_primitives_to_pset(pset)
        else:
            pset = MultiplePrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
        if isinstance(pset, MultiplePrimitiveSet) and self.basic_gene_num > 0:
            pset.terminal_backup()
            # pset.terminals[object].clear()
            # pset.arguments = []
            for i in range(self.X.shape[1], self.X.shape[1] + self.basic_gene_num):
                pset.addTerminal(f'ARG{i}', f'ARG{i}')
                pset.arguments.append(f'ARG{i}')
        self.pset = pset
        return pset

    def add_primitives_to_pset(self, pset):
        for p in self.basic_primitives.split(','):
            p = p.strip()
            primitive = {
                'add': (np.add, 2),
                'subtract': (np.subtract, 2),
                'multiply': (np.multiply, 2),
                'analytical_quotient': (analytical_quotient, 2),
                'protect_sqrt': (protect_sqrt, 1),
                'analytical_loge': (analytical_loge, 1),
                'sin': (np.sin, 1),
                'cos': (np.cos, 1),
                'maximum': (np.maximum, 2),
                'minimum': (np.minimum, 2),
                'mean': (np_mean, 2),
                'arctan': (np.arctan, 1),
                'tanh': (np.tanh, 1),
                'cbrt': (np.cbrt, 1),
                'square': (np.square, 1),
                'negative': (np.negative, 1),
                'sigmoid': (sigmoid, 1),
                'leaky_relu': (leaky_relu, 1),
                'greater_or_equal_than': (greater_or_equal_than, 2),
                'less_or_equal_than': (less_or_equal_than, 2),
            }[p]
            pset.addPrimitive(primitive[0], primitive[1])

    def archive_initialization(self):
        # archive initialization
        if self.ensemble_size == 'auto':
            # Automatically determine the ensemble size
            self.hof = LexicaseHOF()
        elif 'Similar' in self.ensemble_selection:
            ratio = 0.95
            if '-' in self.ensemble_selection:
                ratio = float(self.ensemble_selection.split('-')[1])

            def similar(a, b):
                return np.dot(a.predicted_values, b.predicted_values) / \
                       (norm(a.predicted_values) * norm(b.predicted_values)) > ratio

            self.hof = HallOfFame(self.ensemble_size, similar=similar)
        elif self.ensemble_selection == 'Equal':
            def similar(a, b):
                return np.all(np.equal(a.predicted_values, b.predicted_values))

            self.hof = HallOfFame(self.ensemble_size, similar=similar)
        elif self.ensemble_selection == 'Bootstrap':
            self.hof = BootstrapHallOfFame(self.X, self.ensemble_size)
        elif self.ensemble_selection == 'OOB':
            self.hof = OOBHallOfFame(self.X, self.y, self.toolbox, self.ensemble_size)
        elif self.ensemble_selection == 'GreedySelection':
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y)
            self.hof.novelty_weight = self.novelty_weight
        elif self.ensemble_selection == 'GreedySelection-Resampling':
            if isinstance(self, ClassifierMixin):
                self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y,
                                                     unique=False, bagging_iteration=20, loss_function='MSE',
                                                     inner_sampling=0.5, outer_sampling=0.25)
            else:
                self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y,
                                                     unique=False, bagging_iteration=20, loss_function='MSE',
                                                     inner_sampling=0.1, outer_sampling=0.25)
        elif self.ensemble_selection == 'GreedySelection-Resampling-MSE':
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y,
                                                 unique=False, bagging_iteration=20, loss_function='MSE')
        elif self.ensemble_selection == 'GreedySelection-Resampling-Diverse':
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y, diversity_ratio=0.5,
                                                 unique=False, bagging_iteration=20, loss_function='MSE',
                                                 inner_sampling=0.5, outer_sampling=0.25)
        elif self.ensemble_selection == 'GreedySelection-Resampling-CrossEntropy':
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y,
                                                 unique=False, bagging_iteration=20, loss_function='CrossEntropy')
        elif self.ensemble_selection == 'GreedySelection-Resampling-Hinge':
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y,
                                                 unique=False, bagging_iteration=20, loss_function='Hinge')
        elif self.ensemble_selection == 'GreedySelection-Resampling-ZeroOne':
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y,
                                                 unique=False, bagging_iteration=20, loss_function='ZeroOne')
        elif isinstance(self.ensemble_selection, str) and \
            self.ensemble_selection.startswith('GreedySelection-Resampling-MSE-Custom-'):
            parameter = self.ensemble_selection.replace('GreedySelection-Resampling-MSE-Custom-', '')
            inner_sampling, outer_sampling, bagging_iteration = parameter.split('-')
            inner_sampling = float(inner_sampling)
            outer_sampling = float(outer_sampling)
            bagging_iteration = int(bagging_iteration)
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y, diversity_ratio=0,
                                                 unique=False, bagging_iteration=bagging_iteration, loss_function='MSE',
                                                 inner_sampling=inner_sampling, outer_sampling=outer_sampling)
        elif isinstance(self.ensemble_selection, str) and 'GreedySelection-Resampling~' in self.ensemble_selection:
            self.ensemble_selection, initial_size = self.ensemble_selection.split('~')
            initial_size = int(initial_size)
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y)
            self.hof.unique = False
            self.hof.bagging_iteration = 20
            self.hof.initial_size = initial_size
        elif self.ensemble_selection == 'NoveltySelection':
            self.hof = NoveltyHallOfFame(self.ensemble_size, self.y)
            self.hof.novelty_weight = self.novelty_weight
        elif isinstance(self.ensemble_selection, str) and ('similar' in self.ensemble_selection):
            # Using similarity metric to filter out individuals
            if '-' in self.ensemble_selection:
                ratio = float(self.ensemble_selection.split('-')[1])
            else:
                ratio = 0.95

            def similar(a, b):
                return cosine(a.case_values, b.case_values) >= ratio

            self.hof = HallOfFame(self.ensemble_size, similar=similar)
        elif self.score_func == 'NoveltySearch':
            if self.ensemble_selection == 'DREP':
                self.hof = DREPHallOfFame(self.ensemble_size, self.y)
            elif self.ensemble_selection == 'GreedySelection':
                self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y)
            elif self.ensemble_selection == 'Traditional':
                self.hof = HallOfFame(self.ensemble_size)
            else:
                self.hof = NoveltyHallOfFame(self.ensemble_size, self.y)
            self.hof.novelty_weight = self.novelty_weight
        elif self.ensemble_selection == None or self.ensemble_selection in ['None', 'none',
                                                                            'MAP-Elite']:
            self.hof = HallOfFame(self.ensemble_size)
        else:
            raise Exception
        if self.environmental_selection != None and 'Best' not in self.environmental_selection and \
            self.environmental_selection not in ['NSGA2-Mixup']:
            self.hof = None
        if isinstance(self.hof, EnsembleSelectionHallOfFame):
            self.hof.verbose = self.verbose

    def get_terminal_probability(self):
        """
        Using feature importance at initialization
        """
        # get a probability distribution based on the importance of original features in a random forest
        if isinstance(self, EvolutionaryForestClassifier):
            r = RandomForestClassifier(n_estimators=5)
        else:
            r = RandomForestRegressor(n_estimators=5)
        r.fit(self.X, self.y)
        terminal_prob = np.append(r.feature_importances_, 0.1)
        terminal_prob = terminal_prob / np.sum(terminal_prob)
        return terminal_prob

    def construct_global_feature_pool(self, pop):
        good_features, threshold = construct_feature_pools(pop, True, threshold_ratio=self.cx_threshold_ratio,
                                                           good_features_threshold=self.good_features_threshold)
        self.good_features = good_features
        self.cx_threshold = threshold

    def fit(self, X, y, test_X=None):
        self.y_shape = y.shape
        # self.feature_types = type_detection(X)
        if self.normalize:
            X = self.x_scaler.fit_transform(X, y)
            y = self.y_scaler.fit_transform(np.array(y).reshape(-1, 1))
        if test_X is not None and self.environmental_selection != 'NSGA2-Mixup':
            self.test_data_size = len(test_X)
            if self.normalize:
                test_X = self.x_scaler.transform(test_X)
            self.trainX = X
            X = np.concatenate(([X, test_X]), axis=0)
        if self.validation_size > 0:
            X, self.valid_x, y, self.valid_y = train_test_split(X, y, test_size=self.validation_size)
        self.X, self.y = X, y.flatten()
        if self.environmental_selection == 'NSGA2-Mixup':
            # indices = np.random.randint(0, len(self.X), (len(self.X), 2))
            # alpha = 1
            # lam = np.random.beta(alpha, alpha, len(self.X)).reshape(-1, 1)
            # self.mixup_x = lam * self.X[indices[:, 0]] + (1 - lam) * self.X[indices[:, 1]]

            # def shuffle_along_axis(a, axis):
            #     idx = np.random.rand(*a.shape).argsort(axis=axis)
            #     return np.take_along_axis(a, idx, axis=axis)
            # self.mixup_x = shuffle_along_axis(self.X, 0)
            regularization_model = self.param['regularization_model']
            if regularization_model == 'LGBM':
                lgbm = LGBMRegressor()
            elif regularization_model == 'SVR':
                lgbm = SVR()
            else:
                raise Exception
            lgbm.fit(self.X, self.y)
            self.test_X = test_X
            self.pseudo_label = lgbm.predict(self.x_scaler.transform(test_X))
            # self.mixup_y = lam.flatten() * self.y[indices[:, 0]] + (1 - lam.flatten()) * self.y[indices[:, 1]]
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

        stats_fit = tools.Statistics(lambda ind: ind.fitness.wvalues)
        stats_size = tools.Statistics(lambda x: len(x) // len(x.gene))
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std, axis=0)
        mstats.register("25%", partial(np.quantile, q=0.25), axis=0)
        mstats.register("75%", partial(np.quantile, q=0.75), axis=0)
        mstats.register("median", np.median, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)

        pop, log = self.eaSimple(self.pop, self.toolbox, self.cross_pb, self.mutation_pb, self.n_gen,
                                 stats=mstats, halloffame=self.hof, verbose=self.verbose)
        self.pop = pop

        self.final_model_lazy_training(self.hof)
        self.second_layer_generation(X, y)
        return self

    def second_layer_generation(self, X, y):
        # an interesting question is how to combine these base learners to achieve a good enough performance
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
        elif self.second_layer == 'Ridge-Prediction':
            predictions = []
            for individual in self.hof:
                if len(individual.gene) == 0:
                    continue
                func = self.toolbox.compile(individual)
                Yp = result_calculation(func, X, self.original_features)
                predicted = individual.pipe.predict(Yp)
                predictions.append(predicted)
            predictions = np.array(predictions)
            # fitting predicted values
            self.ridge = RidgeCV(normalize=True, fit_intercept=False)
            self.ridge.fit(predictions.T, y_data)
            self.tree_weight = self.ridge.coef_.flatten()
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
        elif self.second_layer == 'CAWPE':
            pop = self.hof
            weight = np.ones(len(pop))
            for i, ind in enumerate(pop):
                fitness = ind.fitness.wvalues[0]
                if isinstance(self, ClassifierMixin):
                    # using MSE as the default fitness criterion
                    weight[i] = (1 / fitness) ** 4
                else:
                    # using R^2 as the default fitness criterion
                    if fitness > 0:
                        weight[i] = (fitness) ** 4
                    else:
                        weight[i] = 0
            self.tree_weight = weight / np.sum(weight)

    def bootstrap_fitness(self, Yp, Y_true):
        num = len(Y_true)
        sum = []
        for i in range(20):
            index = np.random.randint(0, num, num)
            sum.append(np.mean((Yp[index] - Y_true[index]) ** 2))
        return np.mean(sum)

    def final_model_lazy_training(self, pop, force_training=False):
        for p in pop:
            X = self.X
            if self.test_data_size > 0:
                X = X[:-self.test_data_size]
            Yp = self.feature_generation(X, p)
            self.train_final_model(p, Yp, self.y, force_training=force_training)

    def feature_generation(self, X, individual):
        func = self.toolbox.compile(individual)
        if self.basic_gene_num > 0:
            basic_gene = self.toolbox.compile(individual, basic_gene=True)
            # synthesize some basic features
            x_basic = result_calculation(basic_gene, X, self.original_features)
            X = np.concatenate([X, x_basic], axis=1)
            # X = x_basic
        Yp = result_calculation(func, X, self.original_features)
        return Yp

    def predict(self, X, return_std=False):
        if self.normalize:
            X = self.x_scaler.transform(X)
        prediction_data_size = X.shape[0]
        if self.test_data_size > 0:
            X = np.concatenate([self.X, X])
        self.final_model_lazy_training(self.hof)

        predictions = []
        weight_list = []
        for individual in self.hof:
            if len(individual.gene) == 0:
                continue
            Yp = self.feature_generation(X, individual)
            if self.test_data_size > 0:
                Yp = Yp[-prediction_data_size:]
            if self.base_learner == 'NN':
                nn_prediction = get_activations(self.neural_network, X)[-2]
                Yp = np.concatenate([Yp, nn_prediction], axis=1)
            if isinstance(individual.pipe['Ridge'], SoftPLTreeRegressor) and \
                not isinstance(individual.pipe['Ridge'], SoftPLTreeRegressorEM):
                Yp = np.concatenate([Yp, np.zeros((len(Yp), 1))], axis=1)
            # Yp = np.nan_to_num(Yp)
            predicted = individual.pipe.predict(Yp)

            if self.normalize:
                predicted = self.y_scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()
            predictions.append(predicted)
            if hasattr(self.hof, 'ensemble_weight') and len(self.hof.ensemble_weight) > 0:
                weight_list.append(self.hof.ensemble_weight[individual_to_tuple(individual)])
        if self.second_layer != 'None' and self.second_layer != None:
            predictions = np.array(predictions).T
            final_prediction = predictions @ self.tree_weight
        elif len(weight_list) > 0:
            predictions = np.array(predictions).T
            weight_list = np.array(weight_list)
            if return_std:
                final_prediction = weighted_avg_and_std(predictions.T, weight_list)
            else:
                final_prediction = predictions @ weight_list / weight_list.sum()
        else:
            if return_std:
                final_prediction = np.mean(predictions, axis=0), np.std(predictions, axis=0)
            else:
                final_prediction = np.mean(predictions, axis=0)
        if len(self.y_shape) == 2:
            final_prediction = final_prediction.reshape(-1, 1)
        return final_prediction

    def get_hof(self):
        if self.hof != None:
            return [x for x in self.hof]
        else:
            return None

    def append_evaluated_features(self, pop):
        if self.mutation_scheme in eda_operators or 'uniform' in self.mutation_scheme:
            return
        # append evaluated features to an archive
        # this archive can be used to eliminate repetitive features
        if self.useless_feature_ratio != None:
            mean_importance = np.quantile([ind.coef for ind in pop], self.useless_feature_ratio)
        else:
            mean_importance = np.array([ind.coef for ind in pop]).mean()
        # mean_importance = np.mean([ind.fitness.wvalues[0] for ind in pop])
        for ind in pop:
            # if ind.fitness.wvalues[0] <= mean_importance:
            for gene, c in zip(ind.gene, ind.coef):
                if c <= mean_importance:
                    self.generated_features.add(str(gene))
                    if 'semantic' in self.mutation_scheme:
                        self.generated_features.add(self.feature_quick_evaluation(gene))

    def validity_check(self, ind, semantic_check=True):
        if semantic_check:
            # check through the semantic information
            data = self.X[self.random_index]
            all_string = []
            for g in ind.gene:
                func = gp.compile(g, self.pset)
                Y = func(*data.T)
                if len(np.unique(Y)) == 1:
                    return False
                all_string.append(Y)
            all_string = tuple(np.sort(all_string, axis=1).flatten())
        else:
            # check through the sympy package
            all_string = []
            for g in ind.gene:
                g = parse_expr(gene_to_string(g))
                all_string.append(str(g))
                if g == 0:
                    return False
            all_string = ','.join(sorted(all_string))
        if all_string not in self.evaluated_pop:
            self.evaluated_pop.add(all_string)
            return True
        else:
            return False

    def callback(self):
        # callback function after each generation
        if self.verbose:
            pop = self.pop
            dt = defaultdict(int)
            # parameters = np.zeros(2)
            for p in pop:
                # dt[str(p.base_learner)] += 1
                # dt[str(p.dynamic_leaf_size)] += 1
                dt[str(p.dynamic_regularization)] += 1
                # parameters += p.parameters / len(pop)
            print(dt)
            # print(parameters)

    def multi_gene_mutation(self):
        return self.mutation_scheme in ['uniform-plus', 'uniform-plus-semantic'] or \
               self.mutation_scheme in eda_operators

    def eaSimple(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
        """
        This is the main function of the genetic programming algorithm.
        :param population:  The list of GP individuals.
        :param toolbox: The toolbox includes all genetic operators.
        :param cxpb: The probability of crossover.
        :param mutpb: The probability of mutation.
        :param ngen: The number of generations.
        :param stats:
        :param halloffame:
        :param verbose:
        :return:
        """
        if self.verbose:
            print('data shape', self.X.shape, self.y.shape)
        other_parameters = {
            'nn_prediction': self.nn_prediction,
            'dynamic_target': self.dynamic_target,
            'cv_label': self.ps_tree_cv_label,
            'original_features': self.original_features,
            'test_data_size': self.test_data_size
        }
        arg = (self.X, self.y, self.score_func, self.cv, other_parameters)

        if self.n_process > 1:
            self.pool = Pool(self.n_process, initializer=init_worker, initargs=(calculate_score, arg))
        else:
            init_worker(calculate_score, arg)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        if self.reduction_ratio >= 1:
            self.population_reduction(population)

        # Evaluate the individuals with an invalid fitness
        if self.base_learner in ['RDT~LightGBM-Stump']:
            invalid_ind = self.multiobjective_evaluation(toolbox, population)
        else:
            invalid_ind = self.population_evaluation(toolbox, population)
        if self.environmental_selection == 'NSGA2-Mixup':
            self.mixup_evaluation(self.toolbox, population)

        self.append_evaluated_features(population)
        for o in population:
            self.evaluated_pop.add(individual_to_tuple(o))
        self.callback()

        if halloffame is not None:
            halloffame.update(population)

        if self.diversity_search != 'None':
            self.diversity_assignment(population)

        if 'global' in self.mutation_scheme or 'extreme' in self.mutation_scheme:
            self.construct_global_feature_pool(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            # self.pearson_correlation_analysis(population)
            # self.pearson_correlation_analysis_pairs(population)
            print(logbook.stream)

        error_list = []
        if self.pre_selection != None:
            surrogate_coefficient = 3
            pop_size = int(len(population) * surrogate_coefficient)
        else:
            pop_size = len(population)

        elite_map = {}
        pop_pool = []
        elite_map, pop_pool = self.map_elite_generation(population, elite_map, pop_pool)
        if self.select == 'Auto':
            selection_operators = self.mab_parameter.get('selection_operators',
                                                         'MAP-Elite-Lexicase,Tournament-7,Tournament-15').split(',')
            selection_data = np.ones((2, len(selection_operators)))
        else:
            selection_operators = None
            selection_data = None

        fitness_improvement = 0
        worse_iterations = 0
        comparison_criterion = self.mab_parameter.get('comparison_criterion', 'Case')
        """
        Fitness: Using the fitness improvement as the criterion of a success trial
        Fitness-Case: Using the fitness improvement as the criterion of a success trial
                      and the fitness improvement on a single case as the criterion of a neutral trial
        Case: Using the fitness improvement on a single case as the criterion of a success trial
        Case-Plus: Using real improvement as the score of a trail
        """
        if comparison_criterion in ['Fitness', 'Fitness-Case']:
            best_value = np.max([p.fitness.wvalues[0] for p in population])
        elif comparison_criterion in ['Case', 'Case-Plus', 'Case-Simple'] or isinstance(comparison_criterion, int):
            best_value = np.min([p.case_values for p in population], axis=0)
            worst_value = np.max([p.case_values for p in population], axis=0)
        else:
            raise Exception

        best_hof = self.get_hof()

        # MCTS
        mcts_dict = {}
        candidate_selection_operators = 'MAP-Elite-Lexicase,Tournament-7,Tournament-15,Lexicase'.split(',')
        candidate_survival_opearators = 'AFP,Best,None'.split(',')
        # candidate_surrogate_models = 'None'.split(',')
        mcts_dict['Survival Operators'] = np.ones((2, len(candidate_survival_opearators)))
        mcts_dict['Selection Operators'] = np.ones((2, len(candidate_selection_operators)))
        # for id, selection in enumerate(candidate_surrogate_models):
        #     mcts_dict[f'Surrogate_{id}'] = np.ones((2, len(candidate_selection_operators)))

        # archive initialization
        nsga_archive = population
        afp_archive = population
        best_archive = population

        # Begin the generational process
        for gen in range(1, ngen + 1):
            self.current_gen = gen
            self.repetitive_feature_count.append(0)
            self.entropy_calculation()
            if self.mutation_scheme in eda_operators or 'EDA' in self.mutation_scheme:
                self.frequency_counting()
                self.probability_sampling()
            if self.external_archive and self.hof != None and self.score_func == 'NoveltySearch':
                # recalculate the diversity metric for a fair comparison
                ensemble_value = np.mean([x.predicted_values for x in self.hof], axis=0)
                for x in self.hof:
                    ambiguity = (x.predicted_values - ensemble_value) ** 2
                    x.case_values[len(x.predicted_values):] = -1 * ambiguity

            if self.transformer_switch:
                self.transformer_tool.train()

            if self.cross_pb == 'linear':
                cxpb = np.interp(np.arange(0, self.n_gen), [0, self.n_gen - 1], [0.1, 0.8])[gen - 1]
            if self.mutation_pb == 'linear':
                mutpb = np.interp(np.arange(0, self.n_gen), [0, self.n_gen - 1], [0.8, 0.1])[gen - 1]

            start_time = time.time()
            count = 0
            new_offspring = []

            # Clearing strategy: only the best one in each cluster will survive
            if self.clearing_cluster_size > 1:
                all_case_values = np.array([p.case_values for p in population])
                sum_fitness = np.array([p.fitness.wvalues[0] for p in population])
                key = np.arange(0, len(population))
                label = KMeans(len(population) // self.clearing_cluster_size).fit_predict(all_case_values)
                df = pd.DataFrame(np.array([key, label, sum_fitness]).T, columns=['key', 'label', 'fitness'])
                df = df.sort_values('fitness', ascending=False).drop_duplicates(['label'])
                population = [population[int(k)] for k in list(df['key'])]

            # phi, rho = get_semantic_matrix(np.array([ind.case_values for ind in population]))

            if self.select == 'Auto-MCTS':
                # only useful when need to select the best operator
                xgb_model = self.surrogate_model_construction(population)
            else:
                xgb_model = None

            while (len(new_offspring) < pop_size):
                if count > pop_size * 100:
                    raise Exception("Error!")
                count += 1
                # Using the external archive
                if self.external_archive and self.hof != None:
                    parent = population + list(self.hof)
                else:
                    parent = population

                # Select the best survival operator based on the Thompson sampling
                if self.select == 'Auto-MCTS':
                    candidates = mcts_dict[f'Root']
                    survival_operator_id = np.argmax(np.random.beta(candidates[0], candidates[1]))
                    self.param['parent_archive'] = candidate_survival_opearators[survival_operator_id]
                    for o in parent:
                        o.survival_operator_id = survival_operator_id
                parent_archive = self.param.get('parent_archive', None)
                if parent_archive == 'Fitness-Size':
                    parent = nsga_archive
                if parent_archive == 'AFP':
                    parent = afp_archive
                if parent_archive == 'Best':
                    parent = best_archive

                if self.select == 'Auto' or self.select == 'Auto-MCTS':
                    if self.select == 'Auto':
                        selection_operator_id = np.argmax(np.random.beta(selection_data[0], selection_data[1]))
                        selection_operator = selection_operators[selection_operator_id]
                    else:
                        candidates = mcts_dict['Selection Operators']
                        selection_operator_id = np.argmax(np.random.beta(candidates[0], candidates[1]))
                        selection_operator = candidate_selection_operators[selection_operator_id]
                    if selection_operator == 'MAP-Elite-Lexicase':
                        if len(elite_map) > 0:
                            parent = list(elite_map.values())
                        offspring = selAutomaticEpsilonLexicaseFast(parent, 2)
                    elif selection_operator == 'Tournament-3':
                        offspring = selTournament(parent, 2, tournsize=3)
                    elif selection_operator == 'Tournament-7':
                        offspring = selTournament(parent, 2, tournsize=7)
                    elif selection_operator == 'Tournament-15':
                        offspring = selTournament(parent, 2, tournsize=15)
                    elif selection_operator == 'Tournament-30':
                        offspring = selTournament(parent, 2, tournsize=30)
                    elif selection_operator == 'Lexicase':
                        offspring = selAutomaticEpsilonLexicaseFast(parent, 2)
                    elif selection_operator == 'MAP-Elite-Tournament-3':
                        if len(elite_map) > 0:
                            parent = list(elite_map.values())
                        offspring = selTournament(parent, 2, tournsize=3)
                    elif selection_operator == 'MAP-Elite-Tournament-7':
                        if len(elite_map) > 0:
                            parent = list(elite_map.values())
                        offspring = selTournament(parent, 2, tournsize=7)
                    elif selection_operator == 'MAP-Elite-Random':
                        if len(elite_map) > 0:
                            parent = list(elite_map.values())
                        offspring = selRandom(parent, 2)
                    else:
                        raise Exception

                    for o in offspring:
                        o.selection_operator = selection_operator_id
                elif self.select in map_elite_series and \
                    ((self.map_elite_parameter['trigger_time'] == 'All' and len(elite_map) > 0) or
                     (self.map_elite_parameter['trigger_time'] == 'Improvement' and
                      len(elite_map.values()) > 0 and fitness_improvement >= 0)):
                    if isinstance(elite_map, list):
                        parent = elite_map
                    elif isinstance(elite_map, dict):
                        parent = list(elite_map.values())
                    else:
                        raise TypeError

                    if self.select == 'MAP-Elite-Random':
                        offspring = selRandom(parent, 2)
                    elif self.select == 'MAP-Elite-Knockout':
                        offspring = selKnockout(parent, 2)
                    elif self.select == 'MAP-Elite-Knockout-S':
                        offspring = selKnockout(parent, 2, version='S')
                    elif self.select == 'MAP-Elite-Knockout-SA':
                        offspring = selKnockout(parent, 2, version='S',auto_case=True)
                    elif self.select == 'MAP-Elite-Tournament-3':
                        offspring = selTournament(parent, 2, tournsize=3)
                    elif self.select == 'MAP-Elite-Tournament-7':
                        offspring = selTournament(parent, 2, tournsize=7)
                    elif self.select == 'MAP-Elite-Lexicase':
                        offspring = selAutomaticEpsilonLexicaseFast(parent, 2)
                    else:
                        raise Exception
                else:
                    if self.base_learner in ['DT-LR', 'Balanced-DT-LR', 'Balanced-RDT-LR']:
                        if random.random() < self.rmp_ratio:
                            # cross between different models
                            parent_a = self.sample_model_name(parent)
                            offspring_a = toolbox.select(parent_a, 1)
                            parent_b = self.sample_model_name(parent)
                            offspring_b = toolbox.select(parent_b, 1)
                            offspring = [offspring_a[0], offspring_b[0]]
                        else:
                            parent_a = self.sample_model_name(parent)
                            offspring = toolbox.select(parent_a, 2)
                    else:
                        offspring = toolbox.select(parent, 2)

                surrogate_model_type = 'None'
                offspring: List[MultipleGeneGP] = offspring[:]
                parent_fitness = [o.fitness.wvalues[0] for o in offspring]

                # Vary the pool of individuals
                if self.multi_gene_mutation():
                    # This flag is used to determine whether we use original information
                    # flag = (self.basic_gene_num > 0) and (random.random() > self.bi_level_probability)
                    limitation_check = staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                 max_value=self.max_height)
                    if surrogate_model_type == 'XGBoost':
                        ratio = 3
                        offspring = list(chain.from_iterable([
                            varAndPlus(offspring, toolbox, cxpb, mutpb, self.gene_num, limitation_check)
                            for _ in range(ratio)]))
                        final_ids = np.argsort(xgb_model.predict(self.get_genotype_features(offspring)))[-2:]
                        offspring = list(itertools.compress(offspring, final_ids))
                    else:
                        offspring = varAndPlus(offspring, toolbox, cxpb, mutpb, self.gene_num, limitation_check)

                    # mutation under-level gene
                    if self.basic_gene_num > 0:
                        self.pset.change_state()
                        assert len(self.pset.terminals) <= self.X.shape[1] + 1
                        for gene in offspring:
                            gene.basic_gene_mode = True
                        offspring = varAndPlus(offspring, toolbox, cxpb, mutpb,
                                               self.basic_gene_num, limitation_check)
                        for gene in offspring:
                            gene.basic_gene_mode = False
                        self.pset.change_state()
                else:
                    offspring = varAnd(offspring, toolbox, cxpb, mutpb)

                if self.base_learner == 'Hybrid':
                    # Update base learners
                    base_models = [o.base_model for o in offspring]
                    base_models = varAnd(base_models, self.tpot_model._toolbox, 0.5, 0.1)
                    for o, b in zip(offspring, base_models):
                        o.base_model = b

                if self.base_learner == 'Soft-PLTree':
                    # Update partition scheme
                    partition_schemes = [o.partition_scheme for o in offspring]
                    # Read hyperparameters from parameter dict
                    ps_tree_cross_pb = self.param.get('ps_tree_cross_pb', 0)
                    ps_tree_mutate_pb = self.param.get('ps_tree_mutate_pb', 0)
                    scheme_toolbox: Toolbox = partition_scheme_toolbox(self.partition_number,
                                                                       ps_tree_cross_pb, ps_tree_mutate_pb)
                    partition_schemes = partition_scheme_varAnd(partition_schemes,
                                                                scheme_toolbox,
                                                                cxpb=1, mutpb=1)
                    for o, b in zip(offspring, partition_schemes):
                        o.partition_scheme = b

                for o in offspring:
                    o.parent_fitness = parent_fitness
                    if self.base_learner == 'Dynamic-DT':
                        """
                        Dynamically tune the leaf size of decision tree
                        """
                        # base learner mutation
                        base_learner_mutation_rate = 0.1
                        if random.random() < base_learner_mutation_rate:
                            o.dynamic_leaf_size = np.clip(o.dynamic_leaf_size + random.choice([-1, 1]),
                                                          1, 10)

                    if self.base_learner == 'Dynamic-LogisticRegression':
                        """
                        Dynamically tune the regularization term of logistic regression
                        """
                        base_learner_mutation_rate = 0.1
                        if random.random() < base_learner_mutation_rate:
                            o.dynamic_regularization = np.clip(o.dynamic_regularization * random.choice([0.1, 10]),
                                                               1e-4, 1e4)

                    if len(new_offspring) < pop_size:
                        if self.mutation_scheme == 'uniform-plus' or self.mutation_scheme in eda_operators:
                            if self.allow_repetitive or (not individual_to_tuple(o) in self.evaluated_pop):
                                self.evaluated_pop.add(individual_to_tuple(o))
                                new_offspring.append(o)
                        else:
                            # old version of redundant checking
                            if self.allow_repetitive or (not individual_to_tuple(o) in self.evaluated_pop):
                                self.evaluated_pop.add(individual_to_tuple(o))
                                new_offspring.append(o)

            self.time_statistics['GP Generation'].append(time.time() - start_time)
            # delete some inherited information
            for ind in new_offspring:
                # delete fitness values
                if ind.fitness.valid:
                    del ind.fitness.values
                for attr in ('predicted_values', 'case_values', 'pipe', 'coef'):
                    if hasattr(ind, attr):
                        delattr(ind, attr)

            if self.pre_selection != None:
                offspring, predicted_values = self.pre_selection_individuals(population, new_offspring,
                                                                             self.n_pop)
                assert len(offspring) == self.n_pop, \
                    f"{len(offspring), self.n_pop}"
            else:
                offspring = new_offspring
                assert len(offspring) == pop_size

            self.partition_scheme_updating()
            if self.reduction_ratio >= 1:
                self.population_reduction(population)

            if self.stage_flag:
                print('Start Evaluation')
            # Evaluate the individuals with an invalid fitness
            if self.base_learner in ['RDT~LightGBM-Stump']:
                invalid_ind = self.multiobjective_evaluation(toolbox, offspring)
            else:
                invalid_ind = self.population_evaluation(toolbox, offspring)
            if self.environmental_selection == 'NSGA2-Mixup':
                self.mixup_evaluation(self.toolbox, offspring)

            # Record the fitness improvement
            fitness_improvement = np.max([ind.fitness.wvalues[0] for ind in offspring]) - \
                                  np.max([ind.fitness.wvalues[0] for ind in population])

            if self.validation_size > 0:
                best_parent: MultipleGeneGP = population[np.argmax([ind.fitness.wvalues[0] for ind in population])]
                best_offspring: MultipleGeneGP = offspring[np.argmax([ind.fitness.wvalues[0] for ind in offspring])]
                self.final_model_lazy_training([best_parent, best_offspring])
                parent_input = self.feature_generation(self.valid_x, best_parent)
                offspring_input = self.feature_generation(self.valid_x, best_offspring)
                parent_score = r2_score(self.valid_y, best_parent.pipe.predict(parent_input))
                offspring_score = r2_score(self.valid_y, best_offspring.pipe.predict(offspring_input))
                if parent_score > offspring_score:
                    worse_iterations += 1
                else:
                    worse_iterations = 0
                if worse_iterations > 10:
                    break

            # print('Fitness Improvement', fitness_improvement,
            #       np.mean([ind.fitness.wvalues[0] for ind in offspring]),
            #       np.mean([ind.fitness.wvalues[0] for ind in population]))
            if self.stage_flag:
                print('Stop Evaluation')

            if self.score_func == 'NoveltySearch':
                fitness_list = [x.fitness.wvalues for x in offspring]
                q75, q25 = np.percentile(fitness_list, [75, 25])
                iqr = q75 - q25
                median = np.median(fitness_list)
                # Avoid meaningless diversity individuals
                offspring = list(filter(lambda x:
                                        x.fitness.wvalues[0] >= median - 1.5 * iqr
                                        , offspring))
                # print('Valid individuals', len(offspring))
                assert len(offspring) > 0, f'{median, iqr}'

            if self.dynamic_target:
                # if dynamic target, then re-evaluate all individuals
                for x in self.hof:
                    del x.fitness.values
                self.population_evaluation(toolbox, self.hof)

            elite_map, pop_pool = self.map_elite_generation(offspring, elite_map, pop_pool)

            # record historical best values
            if comparison_criterion == 'Case' or comparison_criterion == 'Case-Plus':
                best_value = np.min([np.min([p.case_values for p in population], axis=0), best_value], axis=0)
                worst_value = np.max([np.max([p.case_values for p in population], axis=0), worst_value], axis=0)
            if comparison_criterion == 'Fitness' or comparison_criterion == 'Fitness-Case':
                # historical best fitness values
                best_value = max(*[p.fitness.wvalues[0] for p in population], best_value)
                parent_case_values = np.min([p.case_values for p in population], axis=0)
            if self.select == 'Auto':
                mode = self.mab_parameter.get('mode', 'Decay')
                cnt = Counter({
                    id: 0
                    for id in range(0, len(selection_data[0]))
                })
                if mode == 'Decay':
                    selection_data[0] *= self.mab_parameter['decay_ratio']
                    selection_data[1] *= self.mab_parameter['decay_ratio']
                C = self.mab_parameter.get('threshold', 100)
                if comparison_criterion == 'Case-Simple':
                    best_value = np.min([p.case_values for p in population], axis=0)
                for o in offspring:
                    cnt[o.selection_operator] += 1
                    if (comparison_criterion in ['Fitness', 'Fitness-Case'] and o.fitness.wvalues[0] > best_value) or \
                        (comparison_criterion in ['Case', 'Case-Simple'] and np.any(o.case_values < best_value)) or \
                        (isinstance(comparison_criterion, int) and
                         np.sum(o.case_values < best_value) > comparison_criterion):
                        selection_data[0][o.selection_operator] += 1
                    elif comparison_criterion == 'Fitness-Case' and np.any(o.case_values < parent_case_values):
                        # don't consider this trial as success or failure
                        pass
                    elif comparison_criterion == 'Case-Plus':
                        if np.any(o.case_values < best_value):
                            selection_data[0][o.selection_operator] += np.sum(o.case_values < best_value)
                        if np.any(o.case_values > worst_value):
                            selection_data[1][o.selection_operator] += np.sum(o.case_values > worst_value)
                    else:
                        selection_data[1][o.selection_operator] += 1
                self.operator_selection_history.append(tuple(cnt.values()))
                if self.verbose:
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

            if self.select == 'Auto-MCTS':
                selection_counter = defaultdict(int)
                gene_counter = defaultdict(int)
                C = self.mab_parameter.get('threshold', 100)
                for o in offspring:
                    if (comparison_criterion in ['Fitness', 'Fitness-Case'] and o.fitness.wvalues[0] > best_value) or \
                        (comparison_criterion in ['Case', 'Case-Simple'] and np.any(o.case_values < best_value)) or \
                        (isinstance(comparison_criterion, int) and
                         np.sum(o.case_values < best_value) > comparison_criterion):
                        mcts_dict['Survival Operators'][0][o.survival_operator_id] += 1
                        mcts_dict['Selection Operators'][0][o.selection_operator] += 1
                    elif comparison_criterion == 'Fitness-Case' and np.any(o.case_values < parent_case_values):
                        # don't consider this trial as success or failure
                        pass
                    elif comparison_criterion == 'Case-Plus':
                        if np.any(o.case_values < best_value):
                            mcts_dict['Survival Operators'][0][o.survival_operator_id] += np.sum(
                                o.case_values < best_value)
                            mcts_dict['Selection Operators'][0][o.selection_operator] += np.sum(
                                o.case_values < best_value)
                        if np.any(o.case_values < worst_value):
                            mcts_dict['Survival Operators'][1][o.survival_operator_id] += np.sum(
                                o.case_values < worst_value)
                            mcts_dict['Selection Operators'][1][o.selection_operator] += np.sum(
                                o.case_values < worst_value)
                    else:
                        mcts_dict['Survival Operators'][1][o.survival_operator_id] += 1
                        mcts_dict['Selection Operators'][1][o.selection_operator] += 1
                    selection_counter[o.selection_operator] += 1
                    gene_counter[o.survival_operator_id] += 1

                if self.verbose:
                    print(selection_counter, gene_counter)

                mode = self.mab_parameter.get('mode', 'Decay')
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
                        data *= self.mab_parameter['decay_ratio']
                    # avoid trivial solutions
                    data = np.clip(data, 1e-2, None)
                    mcts_dict[k] = data
                if self.verbose:
                    print('MCTS Result', mcts_dict)

            # Update the hall of fame with the generated individuals
            if self.stage_flag:
                print('Start HOF updating')
            if halloffame is not None and worse_iterations == 0:
                # if len(elite_map) > 0:
                #     hof_elite_map, _ = selMAPEliteClustering(offspring, pop_pool, self.map_elite_parameter,
                #                                              halloffame.maxsize)
                #     halloffame.clear()
                #     halloffame.update(list(hof_elite_map.values()))
                # else:
                halloffame.update(offspring)
            if self.stage_flag:
                print('Stop HOF updating')

            if 'global' in self.mutation_scheme or 'extreme' in self.mutation_scheme:
                self.construct_global_feature_pool(population)
            self.append_evaluated_features(offspring)

            self.fitness_history.append(np.mean([ind.fitness.wvalues[0] for ind in offspring]))
            best_fitness = np.max([ind.fitness.wvalues[0] for ind in offspring])
            if self.param.get('record_training_data', False):
                self.best_fitness_history.append(best_fitness)
            if self.diversity_search != 'None':
                self.diversity_assignment(offspring)

            # multiobjective GP based on fitness-size
            if self.select == 'Auto-MCTS':
                for ind in offspring + nsga_archive + afp_archive + best_archive:
                    setattr(ind, 'original_fitness', ind.fitness.values)
                    ind.fitness.weights = (-1, -1)
                    ind.fitness.values = [ind.fitness.values[0], sum([len(x) for x in ind.gene])]
                nsga_archive = selNSGA2(offspring + nsga_archive, len(population))
                best_archive = selBest(offspring + best_archive, len(population))
                for ind in offspring:
                    ind.age = 0
                    ind.fitness.values = [ind.fitness.values[0], ind.age]
                # multiobjective GP based on age-fitness
                for ind in nsga_archive + afp_archive + best_archive:
                    if hasattr(ind, 'age'):
                        ind.age += 1
                    else:
                        ind.age = 0
                    ind.fitness.values = [ind.fitness.values[0], ind.age]
                afp_archive = selNSGA2(offspring + afp_archive, len(population))
                for ind in offspring + nsga_archive + afp_archive + best_archive:
                    ind.fitness.weights = (-1,)
                    ind.fitness.values = getattr(ind, 'original_fitness')

            # Replace the current population by the offspring
            self.survival_selection(gen, population, offspring)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                # self.pearson_correlation_analysis(population)
                # self.pearson_correlation_analysis_pairs(population)
                features = set(chain.from_iterable(list(map(lambda x: [str(y) for y in x.gene], population))))
                # features = set(chain.from_iterable(list(map(lambda x: [str(y) for y in x.gene], self.hof))))
                print('number of features', len(features))
                print('archive size', len(self.hof))
                # print('\n'.join(map(lambda x: str(x), population)))
                print(logbook.stream)
                if self.base_model_list != None:
                    model_dict = defaultdict(int)
                    for p in population:
                        model_dict[p.base_model] += 1
                    print('Population', model_dict)
                    model_dict = defaultdict(int)
                    for p in self.hof:
                        model_dict[p.base_model] += 1
                    print('Hall of fame', model_dict)

            if self.early_stop > 0:
                mean_error = self.oob_error(self.hof)
                # print('new score', gen, mean_error)
                if gen == 1 or mean_error > max(error_list):
                    best_hof = self.get_hof()
                error_list.append(mean_error)
                if gen > self.early_stop:
                    # there is no improvement over X generations
                    if np.max(error_list[-(self.early_stop):]) <= np.max(error_list[:-(self.early_stop)]):
                        break

            if self.test_fun != None:
                self.second_layer_generation(self.X, self.y)
                training_loss = self.test_fun[0].predict_loss()
                self.train_data_history.append(training_loss)
                testing_loss = self.test_fun[1].predict_loss()
                self.test_data_history.append(testing_loss)
                if verbose:
                    print('Training Loss', training_loss)
                    print('Testing Loss', testing_loss)
                self.diversity_history.append(self.diversity_summarization())

            self.callback()

        if self.early_stop > 0:
            print('final generation', len(error_list))
            self.hof.clear()
            if isinstance(self.hof, HallOfFame):
                self.hof.update(best_hof)
            elif isinstance(self.hof, list):
                self.hof.extend(best_hof)
            else:
                raise Exception("Unsupported hall of fame!")
        if self.bootstrap_training:
            pass
            ## at this stage, it is possible to re-train a model without bootstrap sampling
            # self.final_model_lazy_training(self.hof, force_training=True)
        if verbose:
            print('Final Ensemble Size', len(self.hof))
        if self.n_process > 1:
            self.pool.close()
        return population, logbook

    def map_elite_generation(self, population, elite_map, pop_pool):
        """
        :param population: Store individuals in the current generation
        :param elite_map: Store selected individuals
        :param pop_pool: Store k*pop_size individuals
        :return:
        """
        if self.select in map_elite_series:
            if self.map_elite_parameter['type'] == 'Grid':
                elite_map = selMAPElite(population, elite_map, self.map_elite_parameter)
            elif self.map_elite_parameter['type'] == 'Grid-Symmetric':
                elite_map = selMAPElite(population, elite_map, self.map_elite_parameter,self.y)
            elif self.map_elite_parameter['type'] == 'Grid-Auto':
                elite_maps = []
                for id, parameters in enumerate([{'fitness_ratio': x,
                                                  'map_size': y}
                                                 for x in [0.2, 0.5, 0.8] for y in [5, 10, 15]]):
                    self.map_elite_parameter.update(parameters)
                    if len(elite_map) > 0:
                        elite_maps.append(selMAPElite(population, elite_map[id], self.map_elite_parameter))
                    else:
                        elite_maps.append(selMAPElite(population, {}, self.map_elite_parameter))
            elif self.map_elite_parameter['type'] == 'Cluster':
                elite_map, pop_pool = selMAPEliteClustering(population, pop_pool, self.map_elite_parameter)
            elif self.map_elite_parameter['type'] == 'Cluster-Exploitation':
                self.map_elite_parameter['fitness_ratio'] = np.clip(0, 0.05, 0.95)
                elite_map, pop_pool = selMAPEliteClustering(population, pop_pool, self.map_elite_parameter)
            elif self.map_elite_parameter['type'] == 'Cluster-Exploration':
                self.map_elite_parameter['fitness_ratio'] = np.clip(1, 0.05, 0.95)
                elite_map, pop_pool = selMAPEliteClustering(population, pop_pool, self.map_elite_parameter)
            else:
                raise Exception
        if self.ensemble_selection == 'MAP-Elite':
            self.hof = list(elite_map.values())
        return elite_map, pop_pool

    def sample_model_name(self, parent):
        models = self.base_model_list.split(',')
        model = random.choice(models)
        parent = list(filter(lambda x: x.base_model == model, parent))
        return parent

    def survival_selection(self, gen, population, offspring):
        # Using NSGA-II or other operators to select parent individuals
        if self.environmental_selection in ['NSGA2'] and self.base_learner == 'RDT~LightGBM-Stump':
            population[:] = selNSGA2(offspring + population, len(population))
            self.hof = population
        elif self.environmental_selection == 'NSGA2-Mixup':
            population[:] = selNSGA2(offspring + population, len(population))
        elif self.environmental_selection != None and 'NSGA2-100' in self.environmental_selection:
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
        elif self.environmental_selection in ['NSGA2-CV', 'NSGA2-CV2']:
            for ind in offspring + population:
                setattr(ind, 'original_fitness', ind.fitness.values)
                fitness = ind.case_values
                if self.environmental_selection == 'NSGA2-CV2':
                    fitness = fitness @ np.random.uniform(0, 1, size=(5, 2))
                ind.fitness.weights = (-1,) * len(fitness)
                ind.fitness.values = list(fitness)
            population[:] = selNSGA2(offspring + population, len(population))
            for ind in population:
                ind.fitness.weights = (-1,)
                ind.fitness.values = getattr(ind, 'original_fitness')
            self.hof = population
        elif self.environmental_selection in ['NSGA2', 'NSGA2-Best', 'NSGA2-Best-Half']:
            # two objectives:
            # 1. accuracy
            # 2. node count
            if self.environmental_selection == 'NSGA2-Best-Half' and gen < (self.n_gen // 2):
                population[:] = offspring
            else:
                avg_size = np.mean([[len(y) for y in x.gene] for x in offspring + population])
                for ind in offspring + population:
                    setattr(ind, 'original_fitness', ind.fitness.values)
                    ind.fitness.weights = (-1, -1)
                    ind_size = np.mean([len(y) for y in ind.gene]) / avg_size
                    ind.fitness.values = (np.sum(ind.case_values), max(1, ind_size))
                    # ind_size = sum([len(y) for y in ind.gene])
                    # ind.fitness.values = (ind.fitness.wvalues[0], ind_size)
                population[:] = selNSGA2(offspring + population, len(population))
                for ind in population:
                    ind.fitness.weights = (-1,)
                    ind.fitness.values = getattr(ind, 'original_fitness')
                if self.environmental_selection != 'NSGA2-Best':
                    self.hof = population
        elif self.environmental_selection == 'MOEA/D':
            # MOEA/D with random decomposition
            def selMOEAD(individuals, k):
                inds = []
                while True:
                    if len(inds) >= k:
                        break
                    weight = np.random.uniform(0, 1, size=(len(individuals[0].case_values)))
                    fun = lambda x: np.sum(weight * np.array(x.case_values))
                    ind = sorted(individuals, key=lambda x: float(fun(x)))[0]
                    individuals.remove(ind)
                    inds.append(ind)
                return inds

            population[:] = selMOEAD(offspring + population, len(population))
            self.hof = population
        elif self.ensemble_selection=='Best':
            population[:] = selBest(offspring + population, len(population))
        else:
                population[:] = offspring

    # @timeit
    def population_reduction(self, population):
        # reduce the population size through feature selection
        ratio = self.reduction_ratio
        new_population = []
        # while len(new_population) < (len(population) // ratio):
        for p in population:
            # inds = random.sample(population, ratio)
            inds = [p]
            genes = list(chain.from_iterable([ind.gene for ind in inds]))

            # syntactic feature selection
            gene_str = set()
            final_genes = []
            for g in genes:
                # remove syntactic equivalent  features and constant features
                if str(g) not in gene_str and not isinstance(g[0], gp.rand101):
                    gene_str.add(str(g))
                    final_genes.append(g)

            # semantic feature selection
            gene_compiled = []
            for gene in final_genes:
                gene_compiled.append(compile(gene, self.pset))
            # random sample a subset to perform feature selection
            x = self.X[np.random.randint(low=0, size=20, high=self.y.shape[0])]
            features = result_calculation(gene_compiled, x, False)
            features, selected_index = np.unique(features, return_index=True, axis=1)
            genes = [final_genes[r] for r in selected_index]
            assert len(genes) == features.shape[1]

            # mrmr feature selection
            # if len(genes) < self.gene_num:
            #     continue
            # result = mrmr_regression(features, self.y, self.gene_num,
            #                          relevance=_f_regression, redundancy=_corr_pearson)
            # result, _, _ = MRMR.mrmr(features, self.y, n_selected_features=self.gene_num)
            # selected_features = [genes[r] for r in result]

            # another strategy: only eliminate redundant features
            # selected_features = [genes[r] for r in range(self.gene_num)]

            # relief selection
            # score = reliefF.reliefF(features,self.y)
            # selected_features = [genes[r] for r in np.argsort(score)[-self.gene_num:]]
            # ind = copy.deepcopy(inds[0])
            # ind.gene = selected_features

            p.gene = genes
            # new_population.append(ind)

    def pearson_correlation_analysis_pairs(self, population):
        # Population Analysis
        def pearson_calculation(ind: MultipleGeneGP):
            features = self.feature_generation(self.X, ind)
            corr = np.nan_to_num(np.abs(np.array(pd.DataFrame(features).corr())), nan=1)
            return np.mean(corr)

        good_pearson_list = np.abs([pearson_calculation(p) for p in population])
        print('Pearson Correlation', np.mean(good_pearson_list))

    def pearson_correlation_analysis(self, population):
        # Population Analysis
        poor_individuals = list(sorted(population, key=lambda x: x.fitness.wvalues))[:len(population) // 5]
        good_individuals = list(sorted(population, key=lambda x: x.fitness.wvalues))[-len(population) // 5:]

        def pearson_calculation(ind: MultipleGeneGP):
            features = self.feature_generation(self.X, ind)
            # return pearsonr(features[:, np.argmax(ind.coef)], self.y)[0]
            all_values = np.abs(np.nan_to_num([pearsonr(features[:, p], self.y)[0] for p in range(features.shape[1])]))
            return np.mean(all_values), np.min(all_values), np.max(all_values)

        good_pearson_list = np.abs([pearson_calculation(p) for p in good_individuals])
        bad_pearson_list = np.abs([pearson_calculation(p) for p in poor_individuals])
        for i in range(3):
            print(np.mean(good_pearson_list[:, i]), np.mean(bad_pearson_list[:, i]))

    def multiobjective_evaluation(self, toolbox, population):
        if self.base_learner == 'RDT~LightGBM-Stump':
            second_pop = copy.deepcopy(population)
            for o in population:
                o.base_model = 'Random-DT'
            for o in second_pop:
                o.base_model = 'LightGBM-Stump'

        self.population_evaluation(toolbox, population)
        self.population_evaluation(toolbox, second_pop)
        population.extend(second_pop)

        for oa, ob in zip(population, second_pop):
            oa.fitness.weights = (-1, -1)
            oa.fitness.values = (oa.fitness.values[0], ob.fitness.values[0])
            ob.fitness.weights = (-1, -1)
            ob.fitness.values = (oa.fitness.values[0], ob.fitness.values[0])
        return population

    def mixup_evaluation(self, toolbox, population):
        self.final_model_lazy_training(population)
        X = self.test_X
        if self.normalize:
            X = self.x_scaler.transform(X)
        for individual in population:
            Yp = self.feature_generation(X, individual)
            predicted = individual.pipe.predict(Yp)

            individual.fitness.weights = (-1, -1)
            individual.fitness.values = (individual.fitness.values[0], r2_score(self.pseudo_label, predicted))

    # @timeit
    def population_evaluation(self, toolbox, population):
        # individual evaluation tasks distribution
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

        # distribute tasks
        if self.n_process > 1:
            data = [next(f) for f in fitnesses]
            results = list(self.pool.map(calculate_score, data))
        else:
            results = list(map(lambda f: calculate_score(next(f)), fitnesses))

        # aggregate results
        for ind, fit, r in zip(invalid_ind, fitnesses, results):
            value = fit.send(r)
            ind.fitness.values = value

        # updating all partition scheme
        if self.base_learner == 'Soft-PLTree' and self.shared_partition_scheme:
            partition_scheme = np.zeros((self.X.shape[0], self.partition_number))
            for p in population:
                assert p.partition_scheme.shape[0] == self.X.shape[0]
                assert len(p.partition_scheme.shape) == 1
                p.partition_scheme = p.partition_scheme.astype(int)
                partition_scheme[np.arange(self.X.shape[0]), p.partition_scheme] += 1
            partition_scheme = partition_scheme.argmax(axis=1)
            for p in population:
                p.partition_scheme = partition_scheme
        return invalid_ind

    def diversity_summarization(self):
        """
        Calculate the diversity between individuals
        """
        if 'CV' in self.score_func:
            return 0
        # distance calculation
        if self.second_layer == 'None' or self.second_layer == None or \
            self.second_layer == 'CAWPE' or self.second_layer == 'Ridge-Prediction':
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

    def complexity(self):
        count = 0
        # count GP complexity
        for h in self.hof:
            h: MultipleGeneGP
            for x in h.gene:
                count += len(x)

            # count Base Model Complexity
            learner = h.pipe['Ridge']
            if isinstance(learner, DecisionTreeRegressor):
                count += learner.tree_.node_count
            if isinstance(learner, SoftPLTreeRegressor):
                count += learner.complexity()
        return count


class EvolutionaryForestClassifier(ClassifierMixin, EvolutionaryForestRegressor):
    def __init__(self, score_func='ZeroOne', **param):
        super().__init__(score_func=score_func, **param)
        self.y_scaler = DummyTransformer()
        if self.base_learner == 'Hybrid':
            config_dict = {
                'sklearn.linear_model.LogisticRegression': {
                    'penalty': ["l2"],
                    'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                    'solver': ['liblinear'],
                },
                'sklearn.tree.DecisionTreeClassifier': {
                    'criterion': ["gini", "entropy"],
                    'max_depth': range(1, 11),
                    'min_samples_split': range(2, 21),
                    'min_samples_leaf': range(1, 21)
                },
            }
            self.tpot_model = TPOTClassifier(config_dict=config_dict, template='Classifier')
            self.tpot_model._fit_init()
        else:
            self.tpot_model = None

    def get_params(self, deep=True):
        params = super().get_params(deep)
        # Hack to make get_params return base class params...
        cp = copy.copy(self)
        cp.__class__ = EvolutionaryForestRegressor
        params.update(EvolutionaryForestRegressor.get_params(cp, deep))
        return params

    def oob_error(self, pop):
        # how to calculate the prediction result?
        count = np.zeros(len(self.y))
        # prediction = np.zeros((len(self.y), len(np.unique(self.y))))
        prediction = np.full((len(self.y), len(np.unique(self.y))), 0, dtype=np.float)
        for i, x in enumerate(pop):
            index = x.out_of_bag
            count[index] += 1
            prediction[index] += x.oob_prediction
        # label = stats.mode(np.array(prediction), axis=0, nan_policy='omit')[0].flatten()
        classes = pop[0].pipe['Ridge'].classes_
        label = classes.take(np.argmax(prediction, axis=1), axis=0)
        accuracy = accuracy_score(self.y, label)
        if self.verbose:
            print('oob score', accuracy)
        return accuracy

    def predict_proba(self, X):
        if self.normalize:
            X = self.x_scaler.transform(X)
        prediction_data_size = X.shape[0]
        if self.test_data_size > 0:
            X = np.concatenate([self.trainX, X])
        self.final_model_lazy_training(self.hof)
        selection_flag = np.ones(len(self.hof), dtype=bool)
        predictions = []
        weight_list = []
        for i, individual in enumerate(self.hof):
            func = self.toolbox.compile(individual)
            Yp = result_calculation(func, X, self.original_features)
            if self.test_data_size > 0:
                Yp = Yp[-prediction_data_size:]
            predicted = individual.pipe.predict_proba(Yp)

            if hasattr(self.hof, 'loss_function') and self.hof.loss_function == 'ZeroOne':
                # zero-one loss
                argmax = np.argmax(predicted, axis=1)
                predicted[:] = 0
                predicted[np.arange(0, len(predicted)), argmax] = 1

            if not np.all(np.isfinite(predicted)):
                # skip prediction results containing NaN
                selection_flag[i] = False
                continue
            predictions.append(predicted)
            if hasattr(self.hof, 'ensemble_weight') and len(self.hof.ensemble_weight) > 0:
                weight_list.append(self.hof.ensemble_weight[individual_to_tuple(individual)])

        if self.second_layer != 'None' and self.second_layer != None:
            assert len(self.hof) == len(self.tree_weight)
            predictions = np.array(predictions).T
            return (predictions @ self.tree_weight[selection_flag]).T
        elif len(weight_list) > 0:
            predictions = np.array(predictions).T
            weight_list = np.array(weight_list)
            return (predictions @ (weight_list / weight_list.sum())).T
        else:
            return np.mean(predictions, axis=0)

    def lazy_init(self, x):
        # encoding target labels
        self.label_encoder = OneHotEncoder(sparse=False)
        self.label_encoder.fit(self.y.reshape(-1, 1))
        super().lazy_init(x)
        if self.class_weight == 'Balanced':
            self.class_weight = compute_sample_weight(class_weight='balanced', y=self.y)
            if hasattr(self.hof, 'class_weight'):
                self.hof.class_weight = np.reshape(self.class_weight, (-1, 1))
        if hasattr(self.hof, 'task_type'):
            self.hof.task_type = 'Classification'
        if isinstance(self.hof, EnsembleSelectionHallOfFame):
            self.hof.categories = len(np.unique(self.y))
            self.hof.label = self.label_encoder.transform(self.y.reshape(-1, 1))

    def entropy_calculation(self):
        if self.score_func == 'NoveltySearch':
            ensemble_value = np.mean([x.predicted_values for x in self.hof],
                                     axis=0)
            self.ensemble_value = ensemble_value
            return self.ensemble_value

    def calculate_case_values(self, individual, estimators, Y, y_pred):
        # Minimize fitness values
        if self.score_func == 'NoveltySearch':
            Y = self.label_encoder.transform(Y.reshape(-1, 1))

        # smaller is better
        if self.score_func == 'ZeroOne' or self.score_func == 'ZeroOne-NodeCount':
            individual.case_values = -1 * (y_pred.flatten() == Y.flatten())
        elif self.score_func == 'CV-NodeCount':
            individual.case_values = [estimators[i]['Ridge'].tree_.node_count for i in range(len(estimators))]
        elif self.score_func == 'CDFC':
            matrix = confusion_matrix(Y.flatten(), y_pred.flatten())
            score = matrix.diagonal() / matrix.sum(axis=1)
            individual.case_values = -1 * score
        elif self.score_func == 'CrossEntropy' or self.score_func == 'NoveltySearch':
            one_hot_targets = OneHotEncoder(sparse=False).fit_transform(self.y.reshape(-1, 1))
            eps = 1e-15
            # Cross entropy
            individual.case_values = -1 * np.sum(one_hot_targets * np.log(np.clip(y_pred, eps, 1 - eps)), axis=1)
            assert not np.any(np.isnan(individual.case_values)), save_array(individual.case_values)
            assert np.size(individual.case_values) == np.size(self.y)
        elif 'CV' in self.score_func:
            individual.case_values = -1 * y_pred
        else:
            raise Exception

        if self.score_func == 'NoveltySearch':
            individual.original_case_values = individual.case_values
            # KL-Divergence for regularization
            if len(self.hof) != 0:
                """
                Cooperation vs Diversity:
                "Diversity with Cooperation: Ensemble Methods for Few-Shot Classification" ICCV 2019
                """
                ensemble_value = self.ensemble_value
                if self.diversity_metric == 'CosineSimilarity':
                    # cosine similarity
                    # larger indicates similar
                    eps = 1e-15
                    ambiguity = np.sum((y_pred * ensemble_value), axis=1) / \
                                np.maximum(norm(y_pred, axis=1) * norm(ensemble_value, axis=1), eps)
                    assert not (np.isnan(ambiguity).any() or np.isinf(ambiguity).any()), save_array((ambiguity, y_pred,
                                                                                                     ensemble_value))
                elif self.diversity_metric == 'KL-Divergence':
                    # smaller indicates similar
                    kl_divergence = lambda a, b: np.sum(np.nan_to_num(a * np.log(a / b), posinf=0, neginf=0), axis=1)
                    ambiguity = (kl_divergence(y_pred, ensemble_value) + kl_divergence(ensemble_value, y_pred)) / 2
                    assert not (np.isnan(ambiguity).any() or np.isinf(ambiguity).any()), save_array((ambiguity, y_pred,
                                                                                                     ensemble_value))
                    ambiguity *= -1
                else:
                    raise Exception
                ambiguity *= self.novelty_weight
                if self.ensemble_cooperation:
                    individual.case_values = individual.case_values - ambiguity
                else:
                    individual.case_values = individual.case_values + ambiguity
                assert not (np.isnan(individual.case_values).any() or np.isinf(individual.case_values).any())

        if self.class_weight is not None:
            individual.case_values = individual.case_values * self.class_weight

    def get_diversity_matrix(self, all_ind):
        inds = []
        for p in all_ind:
            y_pred_one_hot = p.predicted_values
            inds.append(y_pred_one_hot.flatten())
        inds = np.array(inds)
        return inds

    def calculate_fitness_value(self, individual, estimators, Y, y_pred):
        # smaller is better, similar to negative R^2
        if self.score_func == 'ZeroOne' or self.score_func == 'CDFC':
            # larger is better
            score = np.sum(y_pred.flatten() == Y.flatten())
            if self.weighted_coef:
                individual.coef = np.array(individual.coef) * (score)
            return -1 * score,
        elif self.score_func == 'NoveltySearch':
            return np.mean(individual.original_case_values),
        elif self.score_func == 'CrossEntropy':
            return np.mean(individual.case_values),
        elif self.score_func == 'CV-NodeCount' or self.score_func == 'ZeroOne-NodeCount':
            score = -1 * np.sum(y_pred.flatten() == Y.flatten())
            return np.mean([estimators[i]['Ridge'].tree_.node_count for i in range(len(estimators))]),
        elif 'CV' in self.score_func:
            return -1 * np.mean(y_pred),
        else:
            raise Exception

    def predict(self, X, return_std=False):
        predictions = self.predict_proba(X)
        assert np.all(np.sum(predictions, axis=1))
        labels = np.sort(np.unique(self.y))
        return labels[np.argmax(predictions, axis=1)]

    def get_base_model(self, regularization_ratio=1, base_model=None):
        base_model_str = base_model if isinstance(base_model, str) else ''
        if self.base_learner == 'DT' or self.base_learner == 'PCA-DT' or \
            self.base_learner == 'Dynamic-DT':
            ridge_model = DecisionTreeClassifier(max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'Balanced-DT' or base_model == 'Balanced-DT':
            ridge_model = DecisionTreeClassifier(max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 class_weight='balanced')
        elif self.base_learner == 'DT-SQRT':
            ridge_model = DecisionTreeClassifier(max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 max_features='sqrt')
        elif self.base_learner == 'LogisticRegression' or self.base_learner == 'Fast-LRDT' \
            or base_model_str == 'LogisticRegression':
            ridge_model = SafetyLogisticRegression(max_iter=1000, solver='liblinear')
        elif self.base_learner == 'Balanced-LogisticRegression' or \
            base_model_str == 'Balanced-LogisticRegression':
            ridge_model = SafetyLogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
        elif self.base_learner == 'Dynamic-LogisticRegression':
            ridge_model = LogisticRegression(C=regularization_ratio, max_iter=1000, solver='liblinear')
        elif self.base_learner == 'GBDT-PL':
            ridge_model = LGBMClassifier(n_estimators=10, learning_rate=1, max_depth=3, linear_tree=True)
        elif self.base_learner == 'GBDT-LR':
            ridge_model = GBDTLRClassifierX(LGBMClassifier(n_estimators=10, learning_rate=1, max_depth=3),
                                            SafetyLogisticRegression(max_iter=1000, solver='liblinear'))
        elif self.base_learner == 'LightGBM':
            parameter_grid = {
                "learning_rate": [0.5],
                'n_estimators': [10],
            }
            parameter = random.choice(list(ParameterGrid(parameter_grid)))
            ridge_model = LGBMClassifier(**parameter, extra_trees=True, num_leaves=63, n_jobs=1)
        elif self.base_learner == 'DT-Criterion':
            ridge_model = DecisionTreeClassifier(criterion=random.choice(['entropy', 'gini']),
                                                 max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'Random-DT' or self.base_learner == 'Random-DT-Plus' \
            or base_model == 'Random-DT':
            ridge_model = DecisionTreeClassifier(splitter='random', max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'Balanced-Random-DT' or base_model == 'Balanced-Random-DT':
            ridge_model = DecisionTreeClassifier(splitter='random', max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 class_weight='balanced')
        elif self.base_learner == 'Random-DT-Criterion':
            ridge_model = DecisionTreeClassifier(criterion=random.choice(['entropy', 'gini']),
                                                 splitter='random',
                                                 max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf)
        elif self.base_learner == 'LogisticRegressionCV':
            ridge_model = LogisticRegressionCV(solver='liblinear')
        elif self.base_learner == 'Random-DT-SQRT':
            ridge_model = DecisionTreeClassifier(splitter='random', max_depth=self.max_tree_depth,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 max_features='sqrt')
        elif self.base_learner == 'LRDT':
            ridge_model = LRDTClassifier(decision_tree_count=self.decision_tree_count,
                                         max_leaf_nodes=self.max_leaf_nodes)
        elif self.base_learner == 'Hybrid':
            # extract base model
            ridge_model = self.tpot_model._compile_to_sklearn(base_model)[-1]
        elif isinstance(self.base_learner, ClassifierMixin):
            ridge_model = self.base_learner
        elif base_model_str != '' and base_model_str in self.base_model_dict:
            ridge_model = self.base_model_dict[base_model_str]
        elif self.base_learner != '' and self.base_learner in self.base_model_dict:
            ridge_model = self.base_model_dict[self.base_learner]
        else:
            raise Exception
        if self.base_learner == 'PCA-DT':
            pipe = Pipeline([
                ("Scaler", StandardScaler()),
                ('PCA', PCA()),
                ("Ridge", ridge_model),
            ])
        else:
            pipe = Pipeline([
                # ("Scaler", StandardScaler()),
                ("Scaler", SafetyScaler()),
                ("Ridge", ridge_model),
            ])
        return pipe


def truncated_normal(lower=0, upper=1, mu=0.5, sigma=0.1, sample=(100, 100)):
    # instantiate an object X using the above four parameters,
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    # generate 1000 sample data
    samples = X.rvs(sample)
    return samples


def init_worker(function, data):
    function.data = data
