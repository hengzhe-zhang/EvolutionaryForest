import gc
import inspect
from multiprocessing import Pool
from typing import Optional

import dill
import scipy
from category_encoders import TargetEncoder
from deap import gp
from deap import tools
from deap.algorithms import varAnd
from deap.tools import (
    selNSGA2,
    History,
    selBest,
    cxTwoPoint,
    mutFlipBit,
    selDoubleTournament,
    selSPEA2,
    selTournamentDCD,
)
from lightgbm import LGBMRegressor, LGBMModel
from lineartree import LinearTreeRegressor
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau, rankdata, wilcoxon
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, HuberRegressor, Lasso, LassoCV, ElasticNetCV
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KDTree
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import (
    MaxAbsScaler,
    QuantileTransformer,
    SplineTransformer,
)
from sklearn.svm import SVR
from sklearn.tree import BaseDecisionTree
from sklearn2pmml.ensemble import GBDTLRClassifier
from tpot import TPOTRegressor
from xgboost import XGBRegressor

from evolutionary_forest.component.archive import *
from evolutionary_forest.component.archive import (
    DREPHallOfFame,
    NoveltyHallOfFame,
    OOBHallOfFame,
    BootstrapHallOfFame,
)
from evolutionary_forest.component.archive_operators.cvt_map_elites_archive import (
    CVTMAPElitesHOF,
)
from evolutionary_forest.component.archive_operators.greedy_selection_archive import (
    GreedyHallOfFame,
)
from evolutionary_forest.component.archive_operators.grid_map_elites_archive import (
    GridMAPElites,
)
from evolutionary_forest.component.archive_operators.important_features import (
    construct_important_feature_archive,
)
from evolutionary_forest.component.archive_operators.meta_learner.meta_base import (
    MetaLearner,
)
from evolutionary_forest.component.archive_operators.shapley_hof import (
    ShapleyPrunedHallOfFame,
)
from evolutionary_forest.component.base_learner.in_context_learning import (
    InContextLearnerRegressor,
)
from evolutionary_forest.component.bloat_control.alpha_dominance import AlphaDominance
from evolutionary_forest.component.bloat_control.direct_semantic_approximation import (
    DSA,
)
from evolutionary_forest.component.bloat_control.double_lexicase import doubleLexicase
from evolutionary_forest.component.bloat_control.prune_and_plant import PAP
from evolutionary_forest.component.bloat_control.semantic_hoist import SHM
from evolutionary_forest.component.bloat_control.simplification import (
    Simplification,
    hash_based_simplification,
)
from evolutionary_forest.component.bloat_control.tarpeian import Tarpeian
from evolutionary_forest.component.configuration import (
    CrossoverMode,
    ArchiveConfiguration,
    ImbalancedConfiguration,
    EvaluationConfiguration,
    check_semantic_based_bc,
    BloatControlConfiguration,
    SelectionMode,
    BaseLearnerConfiguration,
    ExperimentalConfiguration,
    DepthLimitConfiguration,
)
from evolutionary_forest.component.constant_optimization.random_constant import (
    constant_controller,
)
from evolutionary_forest.component.crossover.crossover_controller import (
    perform_semantic_macro_crossover,
    handle_tpot_base_learner_mutation,
    check_redundancy_and_fix,
    norevisit_strategy_handler,
)
from evolutionary_forest.component.crossover.intron_based_crossover import (
    IntronPrimitive,
    IntronTerminal,
)
from evolutionary_forest.component.crossover.semantic_crossover import resxo, stagexo
from evolutionary_forest.component.crossover_mutation import (
    hoistMutation,
    individual_combination,
)
from evolutionary_forest.component.ensemble_learning.stacking_strategy import (
    StackingStrategy,
)
from evolutionary_forest.component.ensemble_learning.utils import (
    EnsembleRegressor,
    EnsembleClassifier,
)
from evolutionary_forest.component.ensemble_selection.DSE import (
    DynamicSelectionEnsemble,
)
from evolutionary_forest.component.ensemble_selection.RF_DSE import RFRoutingEnsemble
from evolutionary_forest.component.ensemble_selection.dynamic_ensemble_selection.deep_des import (
    DESMetaRegressor,
)
from evolutionary_forest.component.environmental_selection import (
    NSGA2,
    EnvironmentalSelection,
    SPEA2,
    Best,
    NSGA3,
)
from evolutionary_forest.component.evaluation import (
    calculate_score,
    single_tree_evaluation,
    EvaluationResults,
    select_from_array,
    get_sample_weight,
    split_and_combine_data_decorator,
)
from evolutionary_forest.component.external_archive.mutliobjective_archive import (
    ModelSizeArchive,
)
from evolutionary_forest.component.fitness import *
from evolutionary_forest.component.generalization.pac_bayesian import (
    PACBayesianConfiguration,
    SharpnessType,
)
from evolutionary_forest.component.generalization.pac_bayesian_tool import (
    automatic_perturbation_std,
    sharpness_based_dynamic_depth_limit,
)
from evolutionary_forest.component.generalization.wknn import R2WKNN
from evolutionary_forest.component.generation import varAndPlus
from evolutionary_forest.component.initialization import (
    initialize_crossover_operator,
    unique_initialization,
)
from evolutionary_forest.component.log_tool.semantic_lib_log import SemanticLibLog
from evolutionary_forest.component.mutation.common import MutationOperator
from evolutionary_forest.component.mutation.learning_based_mutation import (
    BuildingBlockLearning,
)
from evolutionary_forest.component.primitive_controller import (
    get_functions,
    get_differentiable_functions,
)
from evolutionary_forest.component.primitive_functions import *
from evolutionary_forest.component.racing_selection import RacingFunctionSelector
from evolutionary_forest.component.random_constant import *
from evolutionary_forest.component.selection import (
    batch_tournament_selection,
    selAutomaticEpsilonLexicaseK,
    selTournamentPlus,
    selAutomaticEpsilonLexicaseFast,
    selDoubleRound,
    selRandomPlus,
    selBagging,
    selTournamentNovelty,
    selHybrid,
    selGPED,
    selMAPElites,
    selMAPEliteClustering,
    selKnockout,
    selRoulette,
    selMaxAngleSelection,
    selAngleDrivenSelection,
    selStatisticsTournament,
    selLexicographicParsimonyPressure,
    SelectionConfiguration,
    Selection,
    TournamentLexicase,
    selLexicaseDCD,
    selAutomaticEpsilonLexicaseCLFast,
    selAutomaticEpsilonLexicaseInverseCLFast,
    selGroupALS,
    selLexicaseTournament,
    selLexicaseKNN,
    hybrid_lexicase_dcd,
    selHOFRandom,
)
from evolutionary_forest.component.selection_operators.lexicase_pareto_tournament import (
    sel_lexicase_pareto_tournament_random_subset,
    sel_lexicase_pareto_tournament_weighted_subset,
)
from evolutionary_forest.component.selection_operators.niche_base_selection import (
    niche_base_selection,
)
from evolutionary_forest.component.selection_operators.pareto_tournament import (
    sel_pareto_tournament,
    sel_subset_best,
    pareto_tournament_controller,
)
from evolutionary_forest.component.stateful_gp import make_class
from evolutionary_forest.component.stgp.constant_biased_tree_generation import (
    genHalfAndHalf_STGP_constant_biased,
)
from evolutionary_forest.component.stgp.smooth_scaler import NearestValueTransformer
from evolutionary_forest.component.stgp.strong_type_generation import (
    genHalfAndHalf_STGP,
)
from evolutionary_forest.component.stgp.strongly_type_gp_utility import (
    get_typed_pset,
)
from evolutionary_forest.component.strategy import Clearing
from evolutionary_forest.component.test_function import TestFunction, TestDiversity
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.component.verification.configuration_check import (
    consistency_check,
)
from evolutionary_forest.model.FeatureClipper import FeatureClipper, FeatureSmoother
from evolutionary_forest.model.MTL import MTLRidgeCV, MTLLassoCV
from evolutionary_forest.model.MixupPredictor import MixupRegressor
from evolutionary_forest.model.OptimalKNN import (
    WeightedKNNWithGP,
    WeightedKNNWithGPRidge,
)
from evolutionary_forest.model.PLTree import (
    SoftPLTreeRegressor,
    SoftPLTreeRegressorEM,
    PLTreeRegressor,
    RidgeDT,
    LRDTClassifier,
    RidgeDTPlus,
    RandomWeightRidge,
)
from evolutionary_forest.model.RBFN import RBFN
from evolutionary_forest.model.RidgeFeatureSelector import (
    RidgeForwardFeatureSelector,
    feature_selection,
)
from evolutionary_forest.model.SafeRidgeCV import (
    BoundedRidgeCVSimple,
    SplineRidgeCV,
    SmoothRidgeCV,
    BoundedRidgeCV,
)
from evolutionary_forest.model.SafetyScaler import SafetyScaler
from evolutionary_forest.model.WKNN import GaussianKNNRegressor
from evolutionary_forest.model.gp_tree_wrapper import GPWrapper
from evolutionary_forest.multigene_gp import *
from evolutionary_forest.preprocess_utils import (
    NumericalFeature,
    FeatureTransformer,
    StandardScalerWithMinMaxScaler,
    DummyScaler,
    StandardScalerWithMinMaxScalerAndBounds,
)
from evolutionary_forest.preprocessing.SigmoidTransformer import SigmoidTransformer
from evolutionary_forest.probability_gp import genHalfAndHalf, genFull
from evolutionary_forest.strategies.adaptive_operator_selection import (
    MultiArmBandit,
    MCTS,
)
from evolutionary_forest.strategies.auto_sam import auto_tune_sam, auto_sam_scaling
from evolutionary_forest.strategies.estimation_of_distribution import (
    EstimationOfDistribution,
    eda_operators,
)
from evolutionary_forest.strategies.hist_loss import discretize_and_replace
from evolutionary_forest.strategies.int_scaler import YIntScaler
from evolutionary_forest.strategies.multifidelity_evaluation import (
    MultiFidelityEvaluation,
)
from evolutionary_forest.strategies.surrogate_model import SurrogateModel
from evolutionary_forest.utility.evomal_loss import *
from evolutionary_forest.utility.feature_engineering_utils import combine_features
from evolutionary_forest.utility.feature_importance.aggregation import (
    aggregate_feature_importances,
)
from evolutionary_forest.utility.feature_importance_util import (
    feature_importance_process,
)
from evolutionary_forest.utility.feature_selection import remove_constant_variables
from evolutionary_forest.utility.larmark_constant import lamarck_constant
from evolutionary_forest.utility.metric.distance_metric import get_diversity_matrix
from evolutionary_forest.utility.metric.evaluation_metric import safe_r2_score
from evolutionary_forest.utility.metric.moving_average import MovingAverage
from evolutionary_forest.utility.multi_tree_utils import gene_addition
from evolutionary_forest.utility.population_analysis import (
    statistical_difference_between_populations,
)
from evolutionary_forest.utility.scaler.OneHotStandardScaler import OneHotStandardScaler
from evolutionary_forest.utility.scaler.StandardPCA import StandardScalerPCA
from evolutionary_forest.utility.skew_transformer import (
    SkewnessCorrector,
    CubeSkewnessCorrector,
)
from evolutionary_forest.utility.tree_pool import SemanticLibrary
from evolutionary_forest.utility.tree_size_counter import get_tree_size
from evolutionary_forest.utils import *
from evolutionary_forest.utils import model_to_string

multi_gene_operators = [
    "uniform-plus",
    "uniform-plus-SC",
    "uniform-plus-BSC",
    "uniform-plus-AdaptiveCrossover",
    "uniform-plus-semantic",
    "parsimonious_mutation",
]
map_elite_series = [
    "MAP-Elite-Lexicase",
    "MAP-Elite-Tournament",
    "MAP-Elite-Roulette",
    "MAP-Elite-Tournament-3",
    "MAP-Elite-Tournament-7",
    "MAP-Elite-Random",
    "MAP-Elite-Knockout",
    "MAP-Elite-Knockout-S",
    "MAP-Elite-Knockout-A",
    "MAP-Elite-Knockout-SA",
]
reset_random(0)


def transform_y(y_data, y_scaler):
    return y_scaler.inverse_transform(y_data.reshape(-1, 1)).flatten()


class StaticRandomProjection(TransformerMixin, BaseEstimator):
    def __init__(self, components_: int):
        self.components_ = components_

    def _make_random_matrix(self, n_components, n_features):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X @ self.components_


def similar(a, b):
    return (
        cosine_similarity(a.case_values.reshape(1, -1), b.case_values.reshape(1, -1))[
            0
        ][0]
        > 0.9
    )


def spearman(ya, yb):
    if np.var(ya) == 0 or np.var(yb) == 0:
        return 0
    return spearmanr(ya, yb)[0]


def kendall(ya, yb):
    return kendalltau(ya, yb)[0]


class EvolutionaryForestRegressor(RegressorMixin, TransformerMixin, BaseEstimator):
    """
    Support both TGP and MGP for evolutionary feature construction
    """

    def __init__(
        self,
        # Basic GP Parameters (Core)
        n_pop=200,  # Population size
        n_gen=50,  # Number of generations
        cross_pb=0.9,  # Probability of crossover
        mutation_pb=0.1,  # Probability of mutation
        gene_num=10,  # Number of genes in each GP individual
        mutation_scheme="uniform",  # Mutation scheme used in GP
        verbose=False,  # Whether to print verbose information
        basic_primitives=True,  # Primitive set used in GP
        normalize=True,  # Whether to normalize before fitting a model
        select="AutomaticLexicaseFast",  # Name of the selection operator
        elitism=0,  # Number of the best individuals to be directly passed to next generation
        # Basic GP Parameters (Not Recommend to Change)
        external_archive=None,  # External archive to store historical best results
        original_features=False,  # Whether to use original features in the model or not
        second_layer=None,  # Strategy to induce a second layer for assigning different weights in ensemble
        allow_revisit=False,  # Whether to allow repetitive individuals
        n_process=1,  # Number of processes for parallel execution
        constant_type="Int",  # Type of constants used in GP
        early_stop=-1,  # Early stopping criteria (number of generations)
        random_fix=True,  # Whether to use random fix when height limit is not satisfied in GP
        # Ensemble Learning Parameters
        base_learner="Random-DT",  # Base learner used in the ensemble
        min_samples_leaf=1,  # Minimum samples required to form a leaf node in a tree
        max_tree_depth=None,  # Maximum depth of a decision tree
        cv=5,  # Number of cross-validation folds
        score_func="R2",  # Scoring function for evaluation
        ensemble_prediction="Mean",  # Method of ensemble prediction
        ensemble_selection=None,  # Ensemble selection method
        ensemble_size=100,  # Size of the ensemble model
        # PS-Tree Parameters
        partition_number=4,  # Number of partitions in PS-Tree
        ps_tree_local_model="RidgeCV",  # Type of local model used in PS-Tree
        dynamic_partition="Self-Adaptive",  # Strategy to dynamically change partition scheme
        ps_tree_cv_label=True,  # Whether to use label information in CV in PS-Tree
        ps_tree_partition_model="DecisionTree",  # Type of model used for partitioning
        only_original_features=True,  # Whether to only use original features in PS-Tree
        shared_partition_scheme=False,  # Whether to use shared partition scheme for all individuals
        max_leaf_nodes=None,  # Maximum height of each decision tree
        # SR-Forest Parameters (TEVC 2023)
        ps_tree_ratio=0.1,  # Ratio of PS-Tree in multi-fidelity evaluation
        decision_tree_count=0,  # Number of piecewise trees in a SR-Tree
        # More Parameters
        initial_tree_size=None,  # Initial size of GP tree
        reduction_ratio=0,  # Ratio of samples removed in pre-selection based on filters
        random_state=None,  # Random state used for reproducibility
        validation_size=0,  # Size of the validation set for using in HOF
        mab_parameter=None,  # Parameters for the MAB
        interleaving_period=0,  # Period of interleaving (Multi-fidelity Evaluation)
        # MEGP Parameters (EuroGP 2023)
        map_elite_parameter=None,  # Hyper-parameters for MAP-Elite
        # Deprecated parameters
        boost_size=None,  # Alias of "ensemble_size"
        semantic_diversity=None,  # Alias of "ensemble_selection"
        # MGP hyperparameters
        mgp_mode=False,  # Whether to use MGP
        mgp_scope=None,  # Scope of MGP
        number_of_register=10,  # Number of registers in MGP
        intron_probability=0,  # Probability of intron gene in MGP
        register_mutation_probability=0.1,  # Probability of register mutation in MGP
        delete_irrelevant=False,  # Whether to delete irrelevant genes in MGP
        delete_redundant=False,  # Whether to delete redundant genes in MGP
        # MGP hyperparameters
        irrelevant_feature_ratio=0.01,  # Ratio of irrelevant features in MGP
        strict_layer_mgp=True,  # Whether to use strict layering in MGP
        number_of_parents=0,  # Number of parents in crossover
        # Strategies
        bloat_control=None,  # Bloat control method in GP
        # SHM-GP hyperparameters
        intron_gp=False,  # Whether to use intron GP
        # User Parameters
        custom_primitives=None,  # Custom primitives for GP
        # Debug Parameters
        test_fun=None,  # Test function for evaluation
        # Experimental Parameters (Maybe deprecated in any version)
        diversity_search="None",  # Strategy to assign diversity objective
        bootstrap_training=False,  # Whether to use bootstrap samples for training
        meta_learner=None,  # Whether to use the mean model for predictions
        environmental_selection=None,  # Environmental selection method
        pre_selection=None,  # Pre-selection method
        eager_training=False,  # Whether to train models eagerly
        useless_feature_ratio=None,  # Ratio of useless features to be removed
        weighted_coef=False,  # Whether to use weighted coefficients
        feature_selection=False,  # Whether to perform feature selection
        outlier_detection=False,  # Whether to perform outlier detection
        semantic_repair=0,  # Semantic repair method
        dynamic_reduction=0,  # Dynamic reduction strategy
        num_of_active_trees=0,  # Number of active genes in MGP
        intron_threshold=0,  # Threshold for identifying introns in MGP
        force_sr_tree=False,  # Whether to force to use SR-Tree
        gradient_boosting=False,  # Whether to use gradient boosting
        post_prune_threshold=0,  # Threshold for post-pruning of features in GP
        redundant_hof_size=0,  # Size of redundant Hall of Fame
        delete_low_similarity=False,  # Whether to delete low-similarity genes
        importance_propagation=False,  # Whether to use importance propagation in GP
        ridge_alphas=None,  # Alpha values for Ridge regression
        parsimonious_probability=1,  # Probability of GP with parsimonious terminal usage
        shared_eda=False,  # Whether to use shared estimation of distribution in GP
        rmp_ratio=0.5,  # Multi-task Optimization
        force_retrain=False,
        learner=None,
        constant_ratio=0,
        bounded_prediction=False,
        racing=False,
        simplification=False,
        validation_ratio=0,
        post_selection_method=None,
        stochastic_mode=False,
        log_item=None,
        feature_clipping=False,
        seed_with_linear_model=False,
        norevisit_strategy="",
        lamarck_constant=False,
        categorical_encoding=None,
        validation_based_ensemble_selection=0,
        remove_constant_features=True,
        precision="Float64",
        **params,
    ):
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

        mgp_mode: A modular GP system
        """
        self.precision = precision
        self.seed_with_linear_model = seed_with_linear_model
        self.init_some_logs()
        if log_item is None:
            log_item = ""
        self.log_item = log_item
        self.post_selection_method = post_selection_method
        self.validation_ratio = validation_ratio
        self.simplification = simplification
        self.racing: RacingFunctionSelector = racing
        self.bounded_prediction = bounded_prediction
        self.constant_ratio = constant_ratio
        self.learner = learner
        self.force_retrain = force_retrain
        self.base_learner_configuration = BaseLearnerConfiguration(**params)
        self.rmp_ratio = rmp_ratio
        self.columns = None
        self.custom_primitives = custom_primitives
        # EDA distribution is shared across different genes
        # This can alleviate the genetic drift on a specific gene
        self.shared_eda = shared_eda
        # feature boosting means generate data iteratively
        # only for parsimonious mutation
        self.ridge_alphas = ridge_alphas
        self.random_fix = random_fix
        self.importance_propagation = importance_propagation
        self.delete_low_similarity = delete_low_similarity
        self.redundant_hof_size = redundant_hof_size
        self.parsimonious_probability = parsimonious_probability
        self.post_prune_threshold = post_prune_threshold
        self.intron_gp = intron_gp
        self.gradient_boosting = gradient_boosting
        self.force_sr_tree = force_sr_tree
        self.bloat_control = bloat_control
        if self.bloat_control is None:
            self.bloat_control = {}
        if bloat_control is not None:
            self.bloat_control_configuration = BloatControlConfiguration(
                **params,
                **bloat_control,
            )
        else:
            self.bloat_control_configuration = BloatControlConfiguration(**params)
        # explicitly to make some genes as introns
        self.intron_threshold = intron_threshold
        # determine first k trees are active, used for preserving introns
        self.num_of_active_trees = num_of_active_trees
        self.dynamic_reduction = dynamic_reduction
        self.irrelevant_feature_ratio = irrelevant_feature_ratio
        self.delete_irrelevant = delete_irrelevant
        self.delete_redundant = delete_redundant
        self.register_mutation_probability = register_mutation_probability
        self.intron_probability = intron_probability
        self.strict_layer_mgp = strict_layer_mgp
        self.constant_type = constant_type
        self.number_of_parents = number_of_parents
        self.ensemble_prediction = ensemble_prediction
        self.layer_mgp = False
        if isinstance(mgp_scope, str):
            mgp_scope = int(mgp_scope.replace("Strict-", ""))
            # The whole GP program consists of n layers
            # Each layer can get inputs from previous layers
            self.layer_mgp = True
        self.mgp_scope = mgp_scope
        self.mab_parameter = mab_parameter
        self.validation_size = validation_size
        if random_state is not None:
            reset_random(random_state)
        self.random_state = random_state
        self.reduction_ratio = reduction_ratio
        self.mgp_mode = mgp_mode
        self.decision_tree_count = decision_tree_count
        self.shared_partition_scheme = shared_partition_scheme
        self.only_original_features = only_original_features
        self.initial_tree_size = initial_tree_size
        self.ps_tree_partition_model = ps_tree_partition_model
        self.ps_tree_cv_label = ps_tree_cv_label
        self.allow_revisit = allow_revisit
        self.pre_selection = pre_selection
        self.max_tree_depth = max_tree_depth
        self.elitism = elitism
        self.score_function_controller(params, score_func)
        self.min_samples_leaf = min_samples_leaf
        self.meta_learner = meta_learner
        self.base_learner = base_learner
        self.bootstrap_training = bootstrap_training
        self.semantic_diversity = semantic_diversity
        self.ensemble_selection = (
            ensemble_selection if semantic_diversity is None else semantic_diversity
        )
        self.original_features = original_features
        self.external_archive = external_archive
        self.boost_size = boost_size
        self.ensemble_size = ensemble_size if boost_size is None else int(boost_size)
        self.mutation_scheme = mutation_scheme
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.verbose = verbose
        self.success_rate = MovingAverage(window_size=100)
        self.fs_success_rate = MovingAverage(window_size=100)
        self.initialized = False
        self.pop: List[MultipleGeneGP] = []
        self.basic_primitives: str = basic_primitives
        self.select = select
        self.gene_num = gene_num
        self.param = params
        self.diversity_search = diversity_search
        self.second_layer = second_layer
        self.test_fun: List[TestFunction] = test_fun
        self.history_initialization()
        self.early_stop = early_stop
        self.eager_training = eager_training
        self.useless_feature_ratio = useless_feature_ratio
        self.weighted_coef = weighted_coef
        self.feature_selection = feature_selection
        self.cv = cv
        self.stage_flag = False
        self.map_elite_parameter = (
            {} if map_elite_parameter is None else map_elite_parameter
        )
        self.number_of_register = number_of_register
        self.outlier_detection = outlier_detection
        self.counter_initialization()

        self.cross_pb = cross_pb
        self.mutation_pb = mutation_pb
        self.partition_number = partition_number
        self.ps_tree_local_model = ps_tree_local_model
        self.dynamic_partition = dynamic_partition
        self.max_leaf_nodes = max_leaf_nodes
        self.ps_tree_ratio = ps_tree_ratio
        self.semantic_repair = semantic_repair
        self.stochastic_mode = stochastic_mode

        if isinstance(self.ps_tree_ratio, str) and "Interleave" in self.ps_tree_ratio:
            interleaving_period = int(
                np.round(
                    n_gen
                    / (n_gen * float(self.ps_tree_ratio.replace("Interleave-", "")))
                )
            )
        self.interleaving_period = interleaving_period
        self.test_data = None

        if params.get("record_training_data", False) and self.test_fun != None:
            self.test_fun[0].regr = self
            self.test_fun[1].regr = self

        self.normalize = normalize
        if normalize is True:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        elif normalize == "OneHotStandard":
            self.x_scaler = OneHotStandardScaler()
            self.y_scaler = StandardScaler()
        elif normalize == "TargetNormalization":
            self.x_scaler = DummyScaler()
            self.y_scaler = StandardScaler()
        elif normalize == "Skew":
            self.x_scaler = StandardScaler()
            self.y_scaler = SkewnessCorrector()
        elif normalize == "CubeSkew":
            self.x_scaler = StandardScaler()
            self.y_scaler = CubeSkewnessCorrector()
        elif normalize == "Spline":
            self.x_scaler = Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("Spline", SplineTransformer()),
                ]
            )
            self.y_scaler = StandardScaler()
        elif normalize == "Sigmoid":
            self.x_scaler = StandardScaler()
            self.y_scaler = SigmoidTransformer()
        elif normalize == "STD+MinMax":
            self.x_scaler = StandardScalerWithMinMaxScaler()
            self.y_scaler = StandardScaler()
        elif normalize == "STD+MinMax+Int":
            self.x_scaler = StandardScalerWithMinMaxScaler()
            self.y_scaler = YIntScaler(StandardScaler())
        elif normalize == "STD+MinMax+Bound":
            self.x_scaler = StandardScalerWithMinMaxScalerAndBounds()
            self.y_scaler = StandardScaler()
        elif normalize == "MinMax":
            self.x_scaler = MinMaxScaler(feature_range=(0, 1))
            self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        elif normalize == "Robust":
            self.x_scaler = RobustScaler()
            self.y_scaler = RobustScaler()
        elif normalize == "MaxAbs":
            self.x_scaler = MaxAbsScaler()
            self.y_scaler = MaxAbsScaler()
        elif normalize == "Quantile":
            self.x_scaler = QuantileTransformer(output_distribution="normal")
            self.y_scaler = StandardScaler()
        elif normalize in [
            "BackwardDifferenceEncoder",
            "BinaryEncoder",
            "CatBoostEncoder",
            "CountEncoder",
            "HashingEncoder",
            "HelmertEncoder",
            "JamesSteinEncoder",
            "LeaveOneOutEncoder",
            "MEstimateEncoder",
            "OneHotEncoder",
            "OrdinalEncoder",
            "PolynomialEncoder",
            "QuantileEncoder",
            "SumEncoder",
            "SummaryEncoder",
            "TargetEncoder",
        ]:
            self.x_scaler = FeatureTransformer(encoding_scheme=normalize)
            self.y_scaler = StandardScaler()
        elif normalize is False:
            self.x_scaler = None
            self.y_scaler = None
        else:
            raise Exception

        self.n_process = n_process
        self.novelty_weight = 1
        self.dynamic_target = False
        self.ensemble_cooperation = False
        self.diversity_metric = "CosineSimilarity"
        if isinstance(self.score_func, str) and "-KL" in self.score_func:
            self.score_func = self.score_func.replace("-KL", "")
            self.diversity_metric = "KL-Divergence"

        if self.score_func == "NoveltySearch-Dynamic":
            self.dynamic_target = True
            self.score_func = "NoveltySearch"
        elif (
            isinstance(self.score_func, str)
            and "WeightedNoveltySearch-" in self.score_func
        ):
            self.novelty_weight = float(self.score_func.split("-")[1])
            self.score_func = "NoveltySearch"
        elif (
            isinstance(self.score_func, str)
            and "WeightedCooperationSearch-" in self.score_func
        ):
            self.novelty_weight = float(self.score_func.split("-")[1])
            self.score_func = "NoveltySearch"
            self.ensemble_cooperation = True

        if self.base_learner == "Hybrid":
            self.tpot_model = TPOTRegressor()
            self.tpot_model._fit_init()
        else:
            self.tpot_model = None

        """
        Some parameters support Multi-task scheme
        However, some parameters do not consider this
        """
        self.base_model_dict = {}
        if isinstance(self.base_learner, list):
            self.base_model_dict = {
                learner.__class__.__name__: learner for learner in self.base_learner
            }
            self.base_model_list = ",".join(
                [learner.__class__.__name__ for learner in self.base_learner]
            )
        elif self.base_learner == "DT-LR":
            if isinstance(self, ClassifierMixin):
                self.base_model_list = "DT,LogisticRegression"
            else:
                self.base_model_list = "DT,RidgeCV"
        elif self.base_learner == "RDT-LR":
            self.base_model_list = "Random-DT,LogisticRegression"
        elif self.base_learner == "Balanced-RDT-LR":
            self.base_model_list = "Balanced-Random-DT,Balanced-LogisticRegression"
        elif self.base_learner == "Balanced-DT-LR":
            self.base_model_list = "Balanced-DT,Balanced-LogisticRegression"
        elif self.base_learner == "DT-LGBM":
            self.base_model_list = "Random-DT,LightGBM-Stump,DT"
        elif self.base_learner == "Spline-Ridge":
            self.base_model_list = "SplineRidgeCV,RidgeCV"
        elif self.base_learner.startswith("Ridge~"):
            self.base_model_list = ",".join(self.base_learner.split("~"))
        else:
            self.base_model_list = None

        delete_keys = []
        for k in params.keys():
            if k in vars(self):
                delete_keys.append(k)
        for k in delete_keys:
            del params[k]

        if environmental_selection == "NSGA2":
            self.environmental_selection: NSGA2 = NSGA2(
                self, None, **self.param, **vars(self)
            )
        elif environmental_selection == "AlphaNSGA2":
            self.environmental_selection: NSGA2 = NSGA2(
                self, None, **self.param, **vars(self), alpha_dominance_sam=True
            )
        elif environmental_selection == "NSGA3":
            self.environmental_selection = NSGA3(self, None, **self.param)
        elif environmental_selection == "SPEA2":
            self.environmental_selection = SPEA2(self, None, **self.param)
        elif environmental_selection == "Best":
            self.environmental_selection = Best()
        elif environmental_selection == "BestSAM":
            self.environmental_selection = Best("sam_loss")
        else:
            self.environmental_selection = environmental_selection
        # if isinstance(self.environmental_selection, R2PACBayesian):
        #     # These two objectives are meaningful, not normalize them
        #     self.environmental_selection.objective_normalization = False

        self.pac_bayesian = PACBayesianConfiguration(**params, **vars(self))
        self.crossover_configuration = CrossoverConfiguration(
            intron_parameters=bloat_control,
            **params,
            **vars(self),
        )
        self.mutation_configuration = MutationConfiguration(
            **params,
            **vars(self),
        )
        self.evaluation_configuration = EvaluationConfiguration(
            **params, **vars(self), classification=isinstance(self, ClassifierMixin)
        )
        if self.constant_type in ["GD", "GD+"]:
            # in this case, enable gradient descent
            self.evaluation_configuration.gradient_descent = True
        self.archive_configuration = ArchiveConfiguration(
            **params,
            **vars(self),
        )
        self.selection_configuration = SelectionConfiguration(
            **params,
            **vars(self),
        )
        self.imbalanced_configuration = ImbalancedConfiguration(
            **params,
            **vars(self),
        )
        self.multi_fidelity_evaluation = MultiFidelityEvaluation(**params, **vars(self))
        self.surrogate_model = SurrogateModel(self, **params)
        self.estimation_of_distribution = EstimationOfDistribution(
            algorithm=self, **params, **vars(self)
        )
        self.experimental_configuration = ExperimentalConfiguration(
            **params, **vars(self)
        )
        self.depth_limit_configuration = DepthLimitConfiguration(**params, **vars(self))
        self.elites_archive = None
        self.stacking_strategy = StackingStrategy(self)
        self.reference_lgbm = None
        if self.mutation_configuration.pool_based_addition:
            self.evaluation_configuration.save_semantics = True
        self.feature_clipping = feature_clipping
        self.norevisit_strategy = norevisit_strategy
        self.lamarck_constant = lamarck_constant

        self.automatic_operator_selection_initialization()
        self.automatic_local_search_initialization()
        self.categorical_encoding: Optional[str] = categorical_encoding
        self.remove_constant_features = remove_constant_features
        self.validation_based_ensemble_selection = validation_based_ensemble_selection
        self.semantic_lib_log = SemanticLibLog()

    def automatic_operator_selection_initialization(self):
        if self.select == "Auto":
            self.aos = MultiArmBandit(self, **self.param)
        elif self.select == "Auto-MCTS":
            self.aos = MCTS(self, **self.param)
        else:
            self.aos = None

    def automatic_local_search_initialization(self):
        if (
            self.mutation_configuration.pool_based_addition == True
            and self.mutation_configuration.semantic_local_search_pb == "Adaptive"
        ):
            self.aos = MultiArmBandit(self, **self.param)
            self.aos.mab_configuration.selection_operators = "LocalSearch,GlobalSearch"
            self.aos.selection_operators = "LocalSearch,GlobalSearch".split(",")
        else:
            self.aos = None

    def init_some_logs(self):
        self.duel_logs = []
        self.sharpness_logs = []
        self.loocv_logs = []
        self.training_r2_logs = []
        self.sharpness_ratio_logs = []
        self.generalization_gap_logs = []

    def history_initialization(self):
        self.train_data_history = []
        self.test_data_history = []
        # average fitness of the ensemble model
        self.archive_fitness_history = []
        # average diversity of the ensemble model
        self.archive_diversity_history = []
        # average diversity of the ensemble model
        self.archive_cos_distance_history = []
        # ambiguity of the ensemble model
        self.ambiguity_history = []
        self.average_loss_history = []
        # average fitness of the population
        self.pop_avg_fitness_history = []
        # average diversity of the population
        self.pop_diversity_history = []
        # average cosine diversity of the population
        self.pop_cos_distance_history = []
        self.tree_genotypic_diversity = []
        self.tree_phenotypic_diversity = []
        self.avg_tree_size_history = []
        # best fitness of the ensemble model
        self.best_fitness_history = []
        # average fitness of individuals in grid
        self.elite_grid_avg_fitness_history = []
        # average diversity of individuals in grid
        self.elite_grid_avg_diversity_history = []
        # successful rate of macro-crossover and micro-crossover
        self.macro_crossover_successful_rate = []
        self.micro_crossover_successful_rate = []
        # successful rate of crossover
        self.crossover_successful_rate = []
        self.mutation_successful_rate = []
        # final MAP-Elite grid
        self.final_elite_grid = []
        # modular GP
        self.redundant_features_history = []
        self.irrelevant_features_history = []
        self.repetitive_feature_count = []
        # history of adaptive crossover probability
        self.adaptive_probability_history = [[], []]
        self.evaluated_pop = set()
        self.generated_features = set()
        self.time_statistics = {
            "GP Evaluation": [],
            "ML Evaluation": [],
            "GP Generation": [],
        }
        self.safe_initialization_check()

        # Pareto front of training error and test error
        self.pareto_front = []
        self.noise_pareto_front_1 = []
        self.noise_pareto_front_10 = []
        self.adversarial_pareto_front_10 = []
        self.noise_sample_pareto_front_10 = []
        self.noise_sample_pareto_front_5 = []
        self.data_pareto_front_50 = []
        self.data_pareto_front_200 = []
        self.data_pareto_front_500 = []
        self.knn_pareto_front = []
        self.wknn_pareto_front = []
        self.dt_pareto_front = []
        # Pareto front of test error and sharpness
        self.test_pareto_front = []
        # Pareto front of test error and model size
        self.size_pareto_front = []
        # Pareto front of training error, model size and test error
        self.training_test_pareto_front = []

    def counter_initialization(self):
        self.current_gen = 0
        self.intron_nodes_counter = 0
        self.all_nodes_counter = 0
        self.successful_repair = 0
        self.failed_repair = 0
        self.current_height = 1
        self.redundant_features = 0
        self.irrelevant_features = 0

    def score_function_controller(self, params, score_func):
        if score_func in ["R2-VarianceReduction", "R2-VRM"]:
            score_func = "R2-PAC-Bayesian"
        if isinstance(score_func, str) and score_func == "R2-Rademacher-Complexity":
            self.score_func = RademacherComplexityR2(self, **params)
        elif (
            isinstance(score_func, str)
            and score_func == "R2-Local-Rademacher-Complexity"
        ):
            self.score_func = LocalRademacherComplexityR2(self, **params)
        elif (
            isinstance(score_func, str)
            and score_func == "R2-Rademacher-Complexity-Size"
        ):
            self.score_func = RademacherComplexitySizeR2(self, **params)
        elif (
            isinstance(score_func, str)
            and score_func == "R2-Rademacher-Complexity-FeatureCount"
        ):
            self.score_func = RademacherComplexityFeatureCountR2(self, **params)
        elif isinstance(score_func, str) and score_func == "R2-PAC-Bayesian":
            self.score_func = R2PACBayesian(self, **params)

        elif isinstance(score_func, str) and score_func == "R2-PAC-Bayesian-Scaler":
            self.score_func = PACBayesianR2Scaler(self, **params)
        elif (
            isinstance(score_func, str) and score_func == "R2-Rademacher-Complexity-ALl"
        ):
            self.score_func = RademacherComplexityAllR2(self, **params)
        elif (
            isinstance(score_func, str)
            and score_func == "R2-Rademacher-Complexity-Scaler"
        ):
            self.score_func = RademacherComplexityR2Scaler(self, **params)
        elif (
            isinstance(score_func, str)
            and score_func == "R2-Rademacher-Complexity-Size-Scaler"
        ):
            self.score_func = RademacherComplexityR2Scaler(self, **params)
        elif (
            isinstance(score_func, str)
            and score_func == "R2-Local-Rademacher-Complexity-Scaler"
        ):
            self.score_func = LocalRademacherComplexityR2Scaler(self, **params)
        elif isinstance(score_func, str) and score_func == "R2-Tikhonov":
            self.score_func = TikhonovR2()
        elif isinstance(score_func, str) and score_func == "R2-WCRV":
            self.score_func = R2WCRV(self)
        elif isinstance(score_func, str) and score_func == "R2-IODC":
            self.score_func = R2IODC(self)
        elif isinstance(score_func, str) and score_func == "R2-GrandComplexity":
            self.score_func = R2GrandComplexity()
        elif isinstance(score_func, str) and score_func == "R2-Size":
            self.score_func = R2Size()
        elif isinstance(score_func, str) and score_func == "R2-Smoothness":
            self.score_func = R2Smoothness(**params)
        elif isinstance(score_func, str) and score_func == "R2-Pearson":
            self.score_func = R2Pearson()
        elif isinstance(score_func, str) and score_func == "R2-Spearman":
            self.score_func = R2Spearman()
        elif isinstance(score_func, str) and score_func == "R2-BootstrapError":
            self.score_func = R2BootstrapError()
        elif isinstance(score_func, str) and (
            score_func.startswith("R2-FeatureCountScaler")
        ):
            weight = float(score_func.split("-")[-1])
            self.score_func = R2FeatureCountScaler(weight=weight)
        elif isinstance(score_func, str) and (
            score_func == "R2-FeatureCount" or score_func.startswith("R2-FeatureCount")
        ):
            weight = float(score_func.split("-")[-1])
            self.score_func = R2FeatureCount(weight=weight)
        elif isinstance(score_func, str) and (
            score_func == "R2-Size-Scaler" or score_func.startswith("R2-Size-Scaler")
        ):
            weight = float(score_func.split("-")[-1])
            self.score_func = R2SizeScaler(weight=weight)
        elif isinstance(score_func, str) and score_func == "R2-GAP":
            self.score_func = R2CVGap(self)
        elif isinstance(score_func, str) and score_func == "R2-WKNN":
            self.score_func = R2WKNN(self)
        else:
            self.score_func = score_func

    def calculate_diversity(self, population):
        inds = []
        for p in population:
            inds.append(p.case_values.flatten())
        inds = np.array(inds)
        tree = KDTree(inds)
        neighbors = tree.query(inds, k=3 + 1)[0]
        mean_diversity = np.mean(neighbors, axis=1)
        if self.verbose:
            print("mean_diversity", np.mean(mean_diversity))
        return mean_diversity

    def feature_quick_evaluation(self, gene):
        # quickly evaluate semantic information of an individual based on first 20 data items
        # usage: surrogate model
        func = compile(gene, self.pset)
        Yp = result_calculation([func], self.X[:20], False)
        return tuple(Yp.flatten())

    def get_model_coefficient(self, x):
        ridge_ = x["Ridge"]
        if isinstance(ridge_, Pipeline):
            base_learner = ridge_[-1]
        else:
            base_learner = ridge_

        if hasattr(base_learner, "shap_values"):
            coef = base_learner.shap_values
        elif hasattr(base_learner, "pi_values"):
            # permutation importance
            coef = base_learner.pi_values
        elif isinstance(base_learner, (LinearModel, LinearClassifierMixin)):
            if len(base_learner.coef_.shape) == 2:
                coef = np.max(np.abs(base_learner.coef_), axis=0)
            else:
                coef = np.abs(base_learner.coef_)
        elif isinstance(base_learner, (BaseDecisionTree, LGBMModel)):
            coef = base_learner.feature_importances_
        elif isinstance(base_learner, (GBDTLRClassifier)):
            coef = base_learner.gbdt_.feature_importances_
        elif isinstance(base_learner, SVR):
            coef = np.ones(self.X.shape[1])
        elif isinstance(base_learner, (RidgeDT, LRDTClassifier)):
            coef = base_learner.feature_importance
        elif hasattr(base_learner, "feature_importances_"):
            coef = base_learner.feature_importances_
        elif isinstance(
            base_learner,
            (KNeighborsRegressor, MeanRegressor, MedianRegressor, BaseEstimator),
        ):
            # Temporarily set all features to have equal importance values
            coef = np.ones(self.gene_num)
        else:
            raise Exception
        return coef

    def fitness_evaluation(self, individual: MultipleGeneGP):
        # single individual evaluation
        X, Y = self.X, self.y
        Y = Y.flatten()

        # func = self.toolbox.compile(individual)
        pipe: GPPipeline
        if self.base_learner == "Dynamic-DT":
            self.min_samples_leaf = individual.dynamic_leaf_size
            pipe = self.get_base_model()
            self.min_samples_leaf = 1
        elif self.base_learner == "Hybrid":
            pipe = self.get_base_model(base_model=individual.base_model)
        elif (
            self.base_learner
            in [
                "DT-LR",
                "Balanced-DT-LR",
                "Balanced-RDT-LR",
                "DT-LGBM",
            ]
            or self.base_learner.startswith("Ridge~")
            or isinstance(self.base_learner, list)
        ):
            pipe = self.get_base_model(base_model=individual.base_model)
        elif self.base_learner == "Dynamic-LogisticRegression":
            pipe = self.get_base_model(
                regularization_ratio=individual.dynamic_regularization
            )
        elif self.base_learner == "AdaptiveLasso":
            pipe = self.get_base_model(
                lasso_alpha=np.power(10, individual.parameters.lasso)
            )
        else:
            pipe = self.get_base_model()

        if self.base_learner == "Soft-PLTree":
            pipe.partition_scheme = individual.partition_scheme

        if self.mgp_mode == "Register":
            pipe.register = individual.parameters["Register"]

        if self.intron_probability > 0:
            pipe.active_gene = individual.active_gene

        # send task to the job pool and waiting results
        if individual.num_of_active_trees > 0:
            genes = individual.gene[: individual.num_of_active_trees]
        else:
            genes = individual.gene

        information: EvaluationResults
        y_pred: np.ndarray
        if self.n_process > 1:
            y_pred, estimators, information = (
                yield pipe,
                dill.dumps(genes, protocol=-1),
                individual.individual_configuration,
            )
        else:
            y_pred, estimators, information = (
                yield pipe,
                genes,
                individual.individual_configuration,
            )

        if self.cv == 1:
            (
                _,
                Y,
            ) = train_test_split(Y, test_size=0.2, random_state=0)
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if isinstance(self.score_func, Fitness) or not "CV" in self.score_func:
            if self.evaluation_configuration.mini_batch:
                assert len(y_pred) == self.evaluation_configuration.batch_size, (
                    len(y_pred),
                    self.evaluation_configuration.batch_size,
                )
            else:
                assert len(y_pred) == len(Y), (len(y_pred), len(Y))
        individual.evaluation_time = (
            information.gp_evaluation_time + information.ml_evaluation_time
        )
        self.time_statistics["GP Evaluation"].append(information.gp_evaluation_time)
        self.time_statistics["ML Evaluation"].append(information.ml_evaluation_time)

        # calculate terminal importance based on the permutation importance method
        if self.mutation_scheme == "EDA-Terminal-PMI":
            individual.estimators = estimators

        if self.base_learner == "DT-RandomDT":
            self.base_learner = "Random-DT"
            individual.pipe = self.get_base_model()
            self.base_learner = "DT-RandomDT"
        elif self.base_learner == "RandomDT-DT":
            self.base_learner = "DT"
            individual.pipe = self.get_base_model()
            self.base_learner = "RandomDT-DT"
        elif self.base_learner == "SimpleDT-RandomDT":
            self.base_learner = "Random-DT"
            individual.pipe = self.get_base_model()
            self.base_learner = "SimpleDT-RandomDT"
        else:
            individual.pipe = pipe

        if (
            (self.bootstrap_training or self.eager_training)
            and
            # save computational resources
            not isinstance(individual.pipe[1], RidgeCV)
        ):
            Yp = multi_tree_evaluation(
                genes,
                self.pset,
                X,
                self.original_features,
                register_array=individual.parameters["Register"]
                if self.mgp_mode == "Register"
                else None,
                configuration=self.evaluation_configuration,
                individual_configuration=individual.individual_configuration,
            )
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
        if self.evaluation_configuration.mini_batch:
            mini_batch_y = self.get_mini_batch_y()
            self.calculate_case_values(individual, mini_batch_y, y_pred)
        else:
            self.calculate_case_values(individual, Y, y_pred)

        if isinstance(self.score_func, str) and "CV" in self.score_func:
            assert len(individual.case_values) % 5 == 0
        elif self.score_func == "CDFC":
            assert len(individual.case_values) == len(np.unique(Y))
        else:
            if self.evaluation_configuration.mini_batch:
                assert (
                    len(individual.case_values)
                    == self.evaluation_configuration.batch_size
                ), len(individual.case_values)
            elif self.validation_ratio > 0:
                pass
            else:
                assert len(individual.case_values) == len(
                    Y
                ), f"{len(individual.case_values)},{len(Y)}"

        individual.hash_result = information.hash_result
        individual.semantics = information.semantic_results
        if isinstance(individual.semantics, torch.Tensor):
            individual.semantics = individual.semantics.detach().numpy()
        individual.correlation_results = information.correlation_results
        self.importance_post_process(individual, estimators)
        if information.feature_numbers is not None:
            individual.coef = aggregate_feature_importances(
                individual.coef, information.feature_numbers
            )

        if self.mgp_mode and self.importance_propagation:
            # Importance Propagation
            # This propagation will influence terminal variable selection as well as multi-parent crossover
            for i, gene in enumerate(reversed(individual.gene)):
                coef = individual.coef[i]
                used_id = set()

                # Iterate through terminals in current gene
                for terminal in gene:
                    if isinstance(terminal, Terminal) and terminal.name.startswith(
                        "ARG"
                    ):
                        # print("Feature Occurred", terminal.name)
                        terminal_id = int(terminal.name.replace("ARG", ""))
                        if (
                            terminal_id >= self.X.shape[1]
                            and terminal_id not in used_id
                        ):
                            # If terminal has not been counted yet, add its coefficient to the corresponding feature's coefficient
                            used_id.add(terminal_id)
                            terminal_id -= self.X.shape[1]
                            individual.coef[terminal_id] += coef

        if self.verbose:
            pass

        if self.base_learner == "Random-DT-Plus":
            if isinstance(self, ClassifierMixin):
                individual.pipe = EnsembleClassifier(estimators)
            elif isinstance(self, RegressorMixin):
                individual.pipe = EnsembleRegressor(estimators)
            else:
                raise Exception

        # count the number of nodes in decision tree
        if hasattr(estimators[0]["Ridge"], "tree_"):
            individual.node_count = [
                estimators[i]["Ridge"].tree_.node_count for i in range(len(estimators))
            ]
        if check_semantic_based_bc(self.bloat_control) or self.intron_gp:
            # only do this in intron mode
            similarity_information = information.introns_results
            if check_semantic_based_bc(self.bloat_control) or self.intron_gp:
                assert len(similarity_information) == len(individual.gene)
                # mark level to all genes
                for tree_information, gene in zip(
                    similarity_information, individual.gene
                ):
                    q = []
                    for id, coef in sorted(
                        tree_information.items(), key=lambda x: x[0]
                    ):
                        s = id
                        # Replace gene node with IntronPrimitive or IntronTerminal
                        if isinstance(gene[s], Primitive):
                            gene[s] = IntronPrimitive(
                                gene[s].name, gene[s].args, gene[s].ret
                            )
                            if isinstance(coef, tuple) and len(coef) == 3:
                                gene[s].equal_subtree = coef[1]
                                coef = coef[0]
                        elif isinstance(gene[s], Terminal):
                            gene[s] = IntronTerminal(
                                gene[s].value,
                                getattr(gene[s], "conv_fct") == str,
                                gene[s].ret,
                            )

                        # Set correlation and level for the new gene node
                        if isinstance(coef, tuple):
                            gene[s].corr = coef[0]
                            # locality sensitive hash
                            gene[s].hash_id = coef[1]
                        else:
                            gene[s].corr = coef
                        gene[s].level = len(q)
                        if isinstance(gene[s], Primitive):
                            if len(q) > 0:
                                q[-1] -= 1
                            q.append(gene[s].arity)
                        elif isinstance(gene[s], Terminal):
                            if len(q) > 0:
                                q[-1] -= 1
                            while len(q) > 0 and q[-1] == 0:
                                q.pop(-1)
                    assert len(q) == 0

        # final score
        if self.evaluation_configuration.mini_batch:
            mini_batch_y = self.get_mini_batch_y()
            yield self.calculate_fitness_value(
                individual, estimators, mini_batch_y, y_pred
            )
        else:
            yield self.calculate_fitness_value(individual, estimators, Y, y_pred)

    def get_mini_batch_y(self):
        configuration = self.evaluation_configuration
        mini_batch_y = select_from_array(
            self.y,
            configuration.current_generation * configuration.batch_size // 4,
            configuration.batch_size,
        )
        return mini_batch_y

    def importance_post_process(self, individual: MultipleGeneGP, estimators):
        # max is not an appropriate indicator, because it will highlight not robust features
        importance_values = [self.get_model_coefficient(x) for x in estimators]
        individual.coef = np.mean(importance_values, axis=0)

        if individual.intron_probability > 0:
            coef = np.zeros_like(individual.active_gene, dtype=np.float32)
            i = 0
            for id, a in zip(
                range(0, len(individual.active_gene)), individual.active_gene
            ):
                if a:
                    coef[id] = individual.coef[i]
                    i += 1
            individual.coef = coef
            assert i == int(sum(individual.active_gene))

        individual.coef = feature_importance_process(individual.coef)

    def calculate_fitness_value(
        self, individual: MultipleGeneGP, estimators, Y, y_pred
    ):
        """
        Calculates the fitness value of an individual based on the score function.

        Smaller values are better because the weight is -1.
        """
        if isinstance(self.score_func, Fitness):
            return self.score_func.fitness_value(individual, estimators, Y, y_pred)
        elif self.score_func.startswith("BoundedS"):
            if "-" in self.score_func:
                lower_bound = float(self.score_func.split("-")[1])
                upper_bound = float(self.score_func.split("-")[2])
            else:
                lower_bound, upper_bound = 0, 1
            original_mse = ((y_pred - Y.flatten()).flatten()) ** 2
            mse = np.mean(np.clip(original_mse, lower_bound, upper_bound))
            return (mse,)
        elif (
            self.score_func in ["R2", "ExponentialR2", "LogR2"]
            or self.score_func.startswith("EvoMAL")
            or self.score_func == "NoveltySearch"
            or self.score_func == "MAE"
            or (
                self.score_func.startswith("Bounded")
                and not self.score_func.startswith("BoundedS")
            )
        ):
            # Calculate R2 score
            if self.imbalanced_configuration.balanced_fitness:
                sample_weight = get_sample_weight(
                    self.X, self.test_X, self.y, self.imbalanced_configuration
                )
                score = r2_score(Y, y_pred, sample_weight=sample_weight)
            elif (
                self.evaluation_configuration.sample_weight == "Adaptive-Plus"
                and individual.individual_configuration.sample_weight is not None
            ):
                score = r2_score(
                    Y,
                    y_pred,
                    sample_weight=individual.individual_configuration.sample_weight,
                )
            else:
                score = r2_score(Y, y_pred)

            # Multiply coefficients by score if weighted_coef is True
            if self.weighted_coef:
                individual.coef = np.array(individual.coef) * score

            if self.validation_ratio > 0:
                number = len(Y) - round(len(Y) * self.validation_ratio)
                score = r2_score(Y[:number], y_pred[:number])
                validation_score = (Y[number:] - y_pred[number:]) ** 2
                individual.validation_score = validation_score
            # Return negative of R2 score
            return (-1 * score,)
        elif self.score_func == "R2-L2":
            X_features = self.feature_generation(self.X, individual)
            score = r2_score(Y, y_pred)
            coef_norm = np.linalg.norm(individual.coef) ** 2
            weights = [float(s) for s in self.pac_bayesian.objective.split(",")]
            feature_norm = (
                np.linalg.norm(StandardScaler().fit_transform(X_features).flatten())
                ** 2
            )
            individual.fitness_list = (
                (score, 1),
                (coef_norm, -1 * weights[0]),
                (feature_norm, -1 * weights[1]),
            )
            return (0,)
        elif self.score_func == "QuadError":
            # Return maximum mean squared error
            return (np.mean((Y - y_pred) ** 4),)
        elif self.score_func == "Lower-Bound":
            # Return maximum mean squared error
            return (np.max(mean_squared_error(Y, y_pred)),)
        elif self.score_func == "Spearman":
            # Calculate and return negative Spearman correlation coefficient
            return (-1 * spearman(Y, y_pred),)
        elif self.score_func == "CV-NodeCount":
            # Return number of nodes in each estimator's tree
            return [
                estimators[i]["Ridge"].tree_.node_count for i in range(len(estimators))
            ]
        elif "CV" in self.score_func:
            # Calculate and return negative mean of predicted values
            return (-1 * np.mean(y_pred),)
        else:
            # Raise exception if score function is not recognized
            raise Exception

    def calculate_case_values(self, individual, Y, y_pred):
        # Minimize fitness values
        if (
            self.score_func in ["R2", "R2-L2", "R2-Tikhonov"]
            or self.score_func == "MSE-Variance"
            or self.score_func == "Lower-Bound"
            or isinstance(self.score_func, Fitness)
        ):
            if self.validation_ratio:
                number = len(Y) - round(len(Y) * self.validation_ratio)
                individual.case_values = (
                    (y_pred - Y.flatten()).flatten()[:number]
                ) ** 2
            else:
                individual.case_values = ((y_pred - Y.flatten()).flatten()) ** 2

            if (
                self.evaluation_configuration.sample_weight == "Adaptive-Plus"
                and individual.individual_configuration.sample_weight is not None
            ):
                individual.case_values = individual.case_values * (
                    individual.individual_configuration.sample_weight
                )

            if self.evaluation_configuration.loss_discretization != None:
                bin, strategy = self.evaluation_configuration.loss_discretization.split(
                    "-"
                )
                bin = int(bin)
                individual.case_values = discretize_and_replace(
                    individual.case_values, bin, strategy
                )
        elif self.score_func == "ExponentialR2":
            individual.case_values = np.exp(
                np.clip(((y_pred - Y.flatten()).flatten()) ** 2, 0, 10)
            )
        elif self.score_func == "LogR2":
            individual.case_values = np.log(1 + ((y_pred - Y.flatten()).flatten()) ** 2)
        elif self.score_func == "MAE":
            individual.case_values = np.abs(((y_pred - Y.flatten()).flatten()))
        elif self.score_func == "Bounded" or self.score_func.startswith("Bounded"):
            if "-" in self.score_func:
                lower_bound = float(self.score_func.split("-")[1])
                upper_bound = float(self.score_func.split("-")[2])
            else:
                lower_bound, upper_bound = 0, 1
            original_mse = ((y_pred - Y.flatten()).flatten()) ** 2
            individual.case_values = np.clip(original_mse, lower_bound, upper_bound)
        elif self.score_func.startswith("EvoMAL"):
            loss_function = {
                "EvoMAL1": loss_function_1,
                "EvoMAL2": loss_function_2,
                "EvoMAL3": loss_function_3,
                "EvoMAL4": loss_function_4,
                "EvoMAL5": loss_function_5,
            }
            individual.case_values = loss_function[self.score_func](y_pred, Y.flatten())
        elif self.score_func == "Spearman":
            individual.case_values = np.abs(
                rankdata(y_pred) - rankdata(Y.flatten())
            ).flatten()
        elif self.score_func == "QuadError":
            individual.case_values = ((y_pred - Y.flatten()).flatten()) ** 4
        elif "CV" in self.score_func:
            individual.case_values = -1 * y_pred
        elif self.score_func == "NoveltySearch":
            base_values = (y_pred.flatten() - Y.flatten()) ** 2
            if len(self.hof) == 0:
                # first generation
                individual.case_values = np.concatenate(
                    [base_values, np.full_like(base_values, 0)]
                )
            else:
                # maximize cross entropy
                ensemble_value = np.mean([x.predicted_values for x in self.hof], axis=0)
                ambiguity = (y_pred.flatten() - ensemble_value) ** 2
                ambiguity *= self.novelty_weight
                assert len(ambiguity) == len(y_pred.flatten())
                individual.case_values = np.concatenate([base_values, -1 * ambiguity])
        else:
            raise Exception

    def train_final_model(
        self, individual: MultipleGeneGP, Yp, Y, force_training=False
    ):
        if self.base_learner == "Lasso-RidgeCV":
            self.base_learner = "RidgeCV"
            individual.pipe = self.get_base_model()
            self.base_learner = "Lasso-RidgeCV"
        if self.base_learner == "RidgeCV-FS":
            individual.pipe = self.build_pipeline(RidgeForwardFeatureSelector())
        if self.base_learner == "RidgeCV-FS+":
            individual, Yp = feature_selection(individual, Yp, Y)
            individual.pipe = self.get_base_model()
        if self.base_learner == "BoundedRidgeCV-ENet":
            self.base_learner = "ElasticNetCV"
            individual.pipe = self.get_base_model()
            self.base_learner = "BoundedRidgeCV-ENet"
        if self.base_learner == "RidgeCV-ENet":
            self.base_learner = "ElasticNetCV"
            individual.pipe = self.get_base_model()
            self.base_learner = "RidgeCV-ENet"
        if self.imbalanced_configuration.balanced_final_training:
            individual.pipe = self.get_base_model()
        # avoid re-training
        model = individual.pipe
        if not force_training:
            # check the necessity of training
            try:
                if (
                    hasattr(individual, "num_of_active_trees")
                    and individual.num_of_active_trees > 0
                ):
                    input_size = individual.num_of_active_trees
                else:
                    input_size = len(individual.gene)
                if hasattr(model, "partition_scheme"):
                    input_size += 1
                if self.original_features:
                    input_size += self.X.shape[1]
                if self.intron_probability > 0:
                    input_size = individual.active_gene.sum()
                if self.mgp_mode == "Register":
                    input_size = self.number_of_register
                input_size = len(individual.pipe["Scaler"].scale_)
                model.predict(np.ones((1, input_size)))
                return None
            except (NotFittedError, AttributeError):
                pass

        # ensure ensemble base leaner will not be retrained
        assert self.base_learner != "Random-DT-Plus"

        if hasattr(model, "partition_scheme"):
            partition_scheme = model.partition_scheme
            Yp = np.concatenate([Yp, np.reshape(partition_scheme, (-1, 1))], axis=1)

        if self.intron_probability > 0:
            Yp = Yp[:, individual.active_gene]

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
            curr_sample_weight = np.ones((n_samples,), dtype=np.float32)
            sample_counts = np.bincount(sample, minlength=n_samples)
            curr_sample_weight *= sample_counts

            individual.pipe.fit(Yp, Y, Ridge__sample_weight=curr_sample_weight)
        else:
            if self.imbalanced_configuration.balanced_final_training:
                sample_weight = get_sample_weight(
                    self.X, self.test_X, self.y, self.imbalanced_configuration
                )
                individual.pipe.fit(Yp, Y, Ridge__sample_weight=sample_weight)
            else:
                individual.pipe.fit(Yp, Y)
            out_of_bag = None

        # feature importance generation
        if model.steps[0][0] == "feature":
            base_model = model["model"]["Ridge"]
        else:
            base_model = model["Ridge"]

        # update feature importance after final training
        if hasattr(base_model, "feature_importances_"):
            individual.coef = base_model.feature_importances_[: self.gene_num]
        elif hasattr(base_model, "coef_"):
            individual.coef = np.abs(base_model.coef_[: self.gene_num]).flatten()
        # assert len(individual.coef) == self.gene_num
        return out_of_bag

    def entropy_calculation(self):
        pass

    def get_base_model(self, regularization_ratio=1, base_model=None, lasso_alpha=None):
        # Get the base model based on the hyperparameter
        if self.base_learner in [
            "DT",
            "DT-RandomDT",
            "PCA-DT",
            "Dynamic-DT",
        ] or base_model in ["DT", "DT-RandomDT", "PCA-DT"]:
            ridge_model = DecisionTreeRegressor(
                max_depth=self.max_tree_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif self.base_learner == "InContextLearner":
            ridge_model = InContextLearnerRegressor(**self.param)
        elif self.base_learner == "NCA":
            ridge_model = WeightedKNNWithGP(**self.param, distance="Softmax")
        elif self.base_learner == "NCA-Euclidean":
            ridge_model = WeightedKNNWithGP(**self.param, distance="Euclidean")
        elif self.base_learner == "NCA~2":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Softmax", reduced_dimension=2
            )
        elif self.base_learner == "NCA-Euclidean~2":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Euclidean", reduced_dimension=2
            )
        elif self.base_learner == "NCA~3":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Softmax", reduced_dimension=3
            )
        elif self.base_learner == "NCA-Euclidean~3":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Euclidean", reduced_dimension=3
            )
        elif self.base_learner == "NCA~1":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Softmax", reduced_dimension=3
            )
        elif self.base_learner == "NCA-Euclidean~1":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Euclidean", reduced_dimension=3
            )
        elif self.base_learner == "NCA~Weighted":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Softmax", weighted_instance=True
            )
        elif self.base_learner == "NCA-Euclidean~Weighted":
            ridge_model = WeightedKNNWithGP(
                **self.param, distance="Euclidean", weighted_instance=True
            )
        elif self.base_learner == "NCA+Ridge":
            ridge_model = WeightedKNNWithGPRidge(**self.param, distance="Softmax")
        elif self.base_learner == "NCA-Euclidean+Ridge":
            ridge_model = WeightedKNNWithGPRidge(**self.param, distance="Euclidean")
        elif self.base_learner == "NCA+Ridge~Split":
            ridge_model = WeightedKNNWithGPRidge(
                **self.param, distance="Softmax", mode="split"
            )
        elif self.base_learner == "NCA-Euclidean+Ridge~Split":
            ridge_model = WeightedKNNWithGPRidge(
                **self.param, distance="Euclidean", mode="split"
            )
        elif self.base_learner == "MTL-Ridge":
            if len(self.y.shape) == 1:
                ridge_model = RidgeCV(
                    store_cv_results=True, scoring=make_scorer(r2_score)
                )
            else:
                ridge_model = MTLRidgeCV()
        elif self.base_learner == "MTL-Lasso":
            ridge_model = MTLLassoCV()
        elif self.base_learner == "SimpleDT-RandomDT":
            ridge_model = DecisionTreeRegressor(max_depth=3)
        elif self.base_learner in [
            "Random-DT",
            "RandomDT-DT",
            "Random-DT-Plus",
        ] or base_model in ["Random-DT", "RandomDT-DT", "Random-DT-Plus"]:
            ridge_model = DecisionTreeRegressor(
                splitter="random",
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif isinstance(self.base_learner, str) and self.base_learner.startswith("MLP"):
            hidden_layer_sizes = int(self.base_learner.split("-")[1])
            ridge_model = MLPRegressor(
                hidden_layer_sizes=(hidden_layer_sizes,),
                max_iter=1000,
                learning_rate_init=0.1,
            )
        elif self.base_learner == "PL-Tree":
            ridge_model = LinearTreeRegressor(base_estimator=LinearRegression())
        elif (
            self.base_learner == "RidgeCV"
            or "PCA-RidgeCV" in self.base_learner
            or self.base_learner == "RidgeCV-ENet"
            or base_model == "RidgeCV"
            or self.base_learner in ["RidgeCV-FS", "RidgeCV-FS+"]
        ):
            # from sklearn.linear_model._coordinate_descent import _alpha_grid
            # alphas = _alpha_grid(self.X, self.y, normalize=True)
            if self.ridge_alphas is None:
                ridge = [0.1, 1, 10]
            elif self.ridge_alphas == "Auto":
                if self.X.shape[0] < 200:
                    ridge = [1e1, 1e2, 1e3]
                else:
                    ridge = [0.1, 1, 10]
            elif self.ridge_alphas == "Auto-Unique":
                if len(np.unique(self.y)) <= 50:
                    ridge = [1e1, 1e2, 1e3]
                else:
                    ridge = [0.1, 1, 10]
            else:
                if isinstance(self.ridge_alphas, (float, int)):
                    ridge = [self.ridge_alphas]
                else:
                    ridge = eval(self.ridge_alphas)
            ridge_model = RidgeGCV(
                alphas=ridge, store_cv_results=True, scoring=make_scorer(safe_r2_score)
            )
        elif self.base_learner == "RidgeCV-Mixup":
            ridge_model = MixupRegressor(
                RidgeGCV(
                    alphas=[0.1, 1, 10],
                    store_cv_results=True,
                    scoring=make_scorer(r2_score),
                )
            )
        elif self.base_learner == "SplineRidgeCV" or base_model == "SplineRidgeCV":
            ridge_model = SplineRidgeCV(
                store_cv_results=True, scoring=make_scorer(r2_score)
            )
        elif self.base_learner == "Bounded-RidgeCVSimple":
            ridge_model = BoundedRidgeCVSimple(
                store_cv_results=True, scoring=make_scorer(r2_score)
            )
        elif (
            self.base_learner == "Bounded-RidgeCV"
            or self.base_learner == "BoundedRidgeCV-ENet"
        ):
            ridge_model = BoundedRidgeCV(
                store_cv_results=True, scoring=make_scorer(r2_score)
            )
        elif self.base_learner == "Smooth-RidgeCV":
            ridge_model = SmoothRidgeCV(
                store_cv_results=True, scoring=make_scorer(r2_score)
            )
        elif self.base_learner == "ElasticNetCV":
            ridge_model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], n_alphas=10
            )
        elif self.base_learner == "LR":
            ridge_model = LinearRegression()
        elif isinstance(self.base_learner, str) and self.base_learner.startswith("DT"):
            ridge_model = DecisionTreeRegressor(
                min_samples_leaf=int(self.base_learner.split("-")[1])
            )
        elif isinstance(base_model, str) and base_model.startswith("DT"):
            ridge_model = DecisionTreeRegressor(
                min_samples_leaf=int(base_model.split("-")[1])
            )
        elif self.base_learner == "Mean":
            ridge_model = MeanRegressor()
        elif self.base_learner == "Median":
            ridge_model = MedianRegressor()
        elif self.base_learner == "Lasso":
            ridge_model = LassoCV(n_alphas=10, max_iter=1000)
        elif self.base_learner == "AdaptiveLasso":
            ridge_model = Lasso(lasso_alpha, precompute=True, max_iter=1000)
        elif self.base_learner == "RandomWeightRidge":
            ridge_model = RandomWeightRidge()
        elif (
            self.base_learner == "Ridge"
            or base_model == "Ridge"
            or (
                isinstance(self.base_learner, str)
                and self.base_learner.startswith("Fast-")
            )
        ):
            ridge_model = Ridge(self.base_learner_configuration.ridge_alpha)
        elif self.base_learner == "RidgeDT":
            ridge_model = RidgeDT(
                decision_tree_count=self.decision_tree_count,
                max_leaf_nodes=self.max_leaf_nodes,
            )
        elif self.base_learner == "Simple-RidgeDT":
            ridge_model = RidgeDTPlus(
                decision_tree_count=0, max_leaf_nodes=self.max_leaf_nodes
            )
        elif self.base_learner == "RidgeDT-Plus":
            ridge_model = RidgeDTPlus(
                decision_tree_count=self.decision_tree_count,
                max_leaf_nodes=self.max_leaf_nodes,
            )
        elif self.base_learner == "RidgeBDT-Plus":
            ridge_model = RidgeDTPlus(
                decision_tree_count=self.decision_tree_count,
                max_leaf_nodes=self.max_leaf_nodes,
                final_model_splitter="best",
                min_samples_leaf=self.min_samples_leaf,
            )
        elif self.base_learner == "ET":
            ridge_model = ExtraTreesRegressor(n_estimators=100)
        elif self.base_learner == "GBDT":
            ridge_model = GradientBoostingRegressor(learning_rate=0.8, n_estimators=5)
        elif self.base_learner == "LightGBM":
            ridge_model = LGBMRegressor(n_estimators=10, learning_rate=1, n_jobs=1)
        elif self.base_learner == "LightGBM-Stump" or base_model == "LightGBM-Stump":
            ridge_model = LGBMRegressor(
                max_depth=1,
                learning_rate=1,
                n_estimators=math.ceil(np.log2(self.X.shape[0])),
                n_jobs=1,
            )
        elif self.base_learner == "Hybrid":
            ridge_model = base_model
        elif self.base_learner == "RBF":
            ridge_model = RBFN(min(self.X.shape[0], 64))
        elif self.base_learner == "SVR":
            ridge_model = SVR()
        elif self.base_learner == "KNN" or base_model == "KNN":
            ridge_model = KNeighborsRegressor(weights="uniform")
        elif self.base_learner.startswith("KNN-"):
            n_neighbors = int(self.base_learner.split("-")[1])
            ridge_model = KNeighborsRegressor(
                n_neighbors=n_neighbors, weights="uniform"
            )
        elif isinstance(base_model, str) and base_model.startswith("KNN-"):
            n_neighbors = int(base_model.split("-")[1])
            ridge_model = KNeighborsRegressor(
                n_neighbors=n_neighbors, weights="uniform"
            )
        elif self.base_learner == "WKNN":
            ridge_model = KNeighborsRegressor(weights="distance")
        elif self.base_learner.startswith("GKNN"):
            k = int(self.base_learner.split("-")[1])
            ridge_model = GaussianKNNRegressor(k=k)
        elif self.base_learner == "HuberRegressor":
            ridge_model = HuberRegressor()
        elif self.base_learner == "PLTree":
            min_samples_leaf = int(self.partition_number.split("-")[1])
            partition_number = int(self.partition_number.split("-")[2])
            ridge_model = PLTreeRegressor(
                min_samples_leaf=min_samples_leaf, max_leaf_nodes=partition_number
            )
        elif (
            self.base_learner == "Soft-PLTree" or self.base_learner == "Soft-PLTree-EM"
        ):
            if self.base_learner == "Soft-PLTree":
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
                only_original_features=self.only_original_features,
            )
        elif isinstance(self.base_learner, RegressorMixin):
            ridge_model = efficient_deepcopy(self.base_learner)
        else:
            raise Exception
        pipe = self.build_pipeline(ridge_model)
        return pipe

    def build_pipeline(self, ridge_model):
        scaler = SafetyScaler()
        if "PCA" in self.base_learner:
            n_components = 0.99
            if "~" in self.base_learner:
                n_components = float(self.base_learner.split("~")[1])
            scaler = StandardScalerPCA(n_components=n_components)
        if self.feature_clipping:
            scaler = FeatureClipper()
        if self.evaluation_configuration.feature_smoothing:
            scaler = FeatureSmoother()
        components = [
            ("Scaler", scaler),
            ("Ridge", ridge_model),
        ]
        pipe = GPPipeline(components)
        if isinstance(pipe["Ridge"], BaseDecisionTree) and self.max_tree_depth != None:
            assert pipe["Ridge"].max_depth == self.max_tree_depth
        return pipe

    def transform(self, X, ratio=0.5, **params):
        ratio = 1 - ratio
        if self.normalize:
            X = self.x_scaler.transform(X)
        code_importance_dict = get_feature_importance(
            self, latex_version=False, **params
        )
        # code_importance_dict_latex = get_feature_importance(
        #     self, latex_version=True, mean_fitness=True
        # )
        # plot_feature_importance(code_importance_dict_latex)
        if self.ensemble_size == 1:
            top_features = list(code_importance_dict.keys())
        else:
            top_features = select_top_features(code_importance_dict, ratio)

        only_new_feature = not self.evaluation_configuration.original_features
        transformed_features = combine_features(
            self, X, top_features, only_new_features=only_new_feature
        )
        transformed_features = np.nan_to_num(
            transformed_features.astype(np.float32), posinf=0, neginf=0
        )
        return transformed_features

    def reference_copy(self):
        # ensure changing the algorithm reference
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name, None)
            if hasattr(attribute, "algorithm") and isinstance(
                attribute.algorithm, EvolutionaryForestRegressor
            ):
                attribute.algorithm = self

    def lazy_init(self, x):
        self.reference_copy()
        if self.gene_num == "Dynamic":
            if len(x) <= 100:
                self.gene_num = 2
            else:
                self.gene_num = 10
        if isinstance(self.score_func, str) and self.score_func == "MTL-R2":
            self.score_func = MTLR2(self.y.shape[1])
        elif isinstance(self.score_func, str) and self.score_func == "MTL-R2Size":
            if len(self.y.shape) == 1:
                self.score_func = R2Size()
            else:
                self.score_func = MTLR2Size(self.y.shape[1])

        if isinstance(self.gene_num, str) and "Max" in self.gene_num:
            self.gene_num = min(int(self.gene_num.replace("Max-", "")), x.shape[1])
        if isinstance(self.n_pop, str) and "N" in self.n_pop:
            # based on the shape before extension
            self.n_pop = extract_numbers(self.x_shape[1], self.n_pop)
        if self.semantic_repair > 0:
            # Sampling some data point for semantic repair
            id = np.random.randint(0, self.X.shape[0], 20)
            self.semantic_repair_input = self.X[id]
            self.semantic_repair_target = self.y[id]
        else:
            self.semantic_repair_input = None

        if self.gene_num == "Adaptive":
            self.gene_num = min(x.shape[1], 20)
        if isinstance(self.gene_num, str):
            self.gene_num = min(
                int(self.gene_num.replace("X", "")) * self.X.shape[1], 30
            )
        pset = self.primitive_initialization(x)

        if self.check_alpha_dominance_nsga2():
            self.alpha_dominance = AlphaDominance(self)

        # hall of fame initialization
        self.archive_initialization()

        # toolbox initialization
        toolbox = TypedToolbox()
        self.toolbox = toolbox
        if self.feature_selection:
            terminal_prob = self.get_terminal_probability()
            toolbox.expr = partial(
                genHalfAndHalf, pset=pset, min_=1, max_=2, terminal_prob=terminal_prob
            )
        else:
            # generate initial population
            self.tree_initialization_function(pset, toolbox)

        # individual initialization
        toolbox.individual = partial(
            multiple_gene_initialization,
            MultipleGeneGP,
            toolbox.expr,
            algorithm=self,
            **self.param,
            **vars(self),
        )
        if (
            self.mutation_configuration.gene_addition_rate > 0
            or self.mutation_configuration.gene_deletion_rate > 0
        ):
            toolbox.population = partial(
                unique_initialization, list, toolbox.individual
            )
        else:
            toolbox.population = partial(tools.initRepeat, list, toolbox.individual)
        toolbox.compile = partial(multiple_gene_compile, pset=pset)

        toolbox.register("clone", efficient_deepcopy)
        toolbox.evaluate = self.fitness_evaluation

        self.selection_operator_initialization(toolbox)

        # mutation operators
        if "EDA" in self.mutation_scheme:
            assert (
                self.mutation_scheme in eda_operators
            ), "EDA-based mutation operator must be listed!"
        if "LBM" in self.mutation_scheme:
            initialize_crossover_operator(self, toolbox)
            self.mutation_scheme = BuildingBlockLearning(pset)
            toolbox.mutate = self.mutation_scheme.mutate
        elif (
            "uniform" in self.mutation_scheme
            or self.mutation_scheme in multi_gene_operators
        ):
            # extract mutation operator
            if "|" in self.mutation_scheme:
                mutation_operator = self.mutation_scheme.split("|")[1]
            else:
                mutation_operator = None

            self.mutation_expression_function(toolbox)
            if self.basic_primitives.startswith("Pipeline"):
                toolbox.tree_generation = genHalfAndHalf_STGP
            else:
                toolbox.tree_generation = gp.genFull

            initialize_crossover_operator(self, toolbox)

            # special mutation operators
            if self.mutation_scheme == "parsimonious_mutation":
                toolbox.register(
                    "mutate",
                    mutProbability_multiple_gene,
                    pset=pset,
                    parsimonious_probability=self.parsimonious_probability,
                )
            elif mutation_operator is None:
                toolbox.register(
                    "mutate",
                    mutUniform_multiple_gene,
                    expr=toolbox.expr_mut,
                    pset=pset,
                    tree_generation=gp.genFull,
                    configuration=self.mutation_configuration,
                )
            elif mutation_operator == "worst":
                toolbox.register(
                    "mutate",
                    mutUniform_multiple_gene_worst,
                    expr=toolbox.expr_mut,
                    pset=pset,
                )
            else:
                raise Exception
        elif self.mutation_scheme in eda_operators:
            initialize_crossover_operator(self, toolbox)

            # If the mutation scheme is 'EDA-Terminal', use Dirichlet distribution to sample terminals
            if self.basic_primitives.startswith("Pipeline"):
                partial_func = genHalfAndHalf_STGP
            elif self.mutation_scheme == "EDA-Terminal":
                partial_func = partial(
                    genFull_with_prob, model=self, sample_type="Dirichlet"
                )
            else:
                partial_func = partial(genFull_with_prob, model=self)

            # Set tree_generation function using partial function
            toolbox.tree_generation = partial_func

            # Set expression mutation function based on mutation_expr_height value
            if self.mutation_configuration.mutation_expr_height is not None:
                min_, max_ = self.mutation_configuration.mutation_expr_height.split("-")
                min_, max_ = int(min_), int(max_)
                toolbox.expr_mut = partial(partial_func, min_=min_, max_=max_)
            else:
                toolbox.expr_mut = partial(partial_func, min_=0, max_=2)

            # Register mutation operator using random tree mutation
            toolbox.register(
                "mutate",
                mutUniform_multiple_gene,
                expr=toolbox.expr_mut,
                pset=pset,
                tree_generation=partial_func,
                configuration=self.mutation_configuration,
            )
        else:
            raise Exception

        if self.verbose:
            history = History()
            self.history = history

        self.size_failure_counter = FailureCounter()
        random_replace = (
            self.mutation_configuration.gene_addition_rate > 0
            or self.mutation_configuration.gene_deletion_rate > 0
        )

        random_fix = self.random_fix and (
            self.mutation_configuration.local_search_dropout == 0
        )
        self.static_limit_function = staticLimit_multiple_gene(
            key=operator.attrgetter("height"),
            max_value=self.depth_limit_configuration.max_height,
            min_value=self.depth_limit_configuration.min_height,
            random_fix=random_fix,
            failure_counter=self.size_failure_counter,
            random_replace=random_replace,
        )

        if not self.multi_gene_mutation():
            toolbox.decorate("mate", self.static_limit_function)
            toolbox.decorate("mutate", self.static_limit_function)
        else:
            # For multi-tree variation operators, height constraint is only checked once to save computational resources
            pass
        if self.intron_threshold > 0 and self.depth_limit_configuration.min_height > 0:
            raise Exception("Not supported in static limit")
        if self.intron_threshold > 0:
            simple_depth_limit = staticLimit(
                key=operator.attrgetter("height"),
                max_value=self.depth_limit_configuration.max_height,
                min_value=self.depth_limit_configuration.min_height,
            )
            self.neutral_mutation = simple_depth_limit(
                partial(mutUniform, expr=toolbox.expr_mut, pset=self.pset)
            )

        self.pop = toolbox.population(n=self.n_pop)

        seed_with_linear_model = self.seed_with_linear_model
        if seed_with_linear_model:
            tree = []
            for x in self.pset.terminals[object]:
                if not isinstance(x, Callable):
                    tree.append(PrimitiveTree([x]))
            self.pop[0].gene = tree

        self.random_index = np.random.randint(0, len(self.X), 5)

        def shuffle_along_axis(a, axis):
            idx = np.random.rand(*a.shape).argsort(axis=axis)
            return np.take_along_axis(a, idx, axis=axis)

        self.prediction_x = shuffle_along_axis(np.copy(self.X), axis=0)

        self.thread_pool_initialization()
        self.toolbox.root_crossover = self.crossover_configuration.root_crossover
        self.evaluation_configuration.pset = pset
        self.toolbox.pset = pset

        if self.imbalanced_configuration.balanced_evaluation:
            self.evaluation_configuration.sample_weight = get_sample_weight(
                self.X, self.test_X, self.y, self.imbalanced_configuration
            )

        if self.racing:
            self.racing = RacingFunctionSelector(
                self.pset,
                self.toolbox.expr,
                **self.param,
                algorithm=self,
                verbose=self.verbose,
            )

        if isinstance(self.score_func, R2PACBayesian):
            self.score_func.lazy_init()

        if (
            isinstance(self.pac_bayesian.perturbation_std, str)
            and "Auto" in self.pac_bayesian.perturbation_std
        ):
            self.pac_bayesian.perturbation_std = auto_tune_sam(
                self.X, self.y, self.pac_bayesian.perturbation_std
            )

        if (
            isinstance(self.pac_bayesian.perturbation_std, str)
            and "Scaling" in self.pac_bayesian.perturbation_std
        ):
            self.pac_bayesian.perturbation_std = auto_sam_scaling(
                self.X, self.y, self.pac_bayesian.perturbation_std
            )
        consistency_check(self)

        if self.bounded_prediction == "Smooth":
            self.smooth_model: NearestValueTransformer = NearestValueTransformer()
            self.smooth_model.fit(self.X, self.y)

        if self.mutation_configuration.pool_based_addition:
            self.tree_pool = SemanticLibrary(
                verbose=self.verbose,
                pset=pset,
                mutation_configuration=self.mutation_configuration,
                x_columns=self.X.shape[0],
                **self.param,
            )
            self.tree_pool.target_semantics = self.y
            interval = self.mutation_configuration.pool_hard_instance_interval
            clustering_mode = self.mutation_configuration.library_clustering_mode
            if interval == 0 and clustering_mode is not False:
                self.tree_pool.set_clustering_based_semantics(self.y, clustering_mode)
            if self.mutation_configuration.include_subtree_to_lib:
                self.evaluation_configuration.semantic_library = self.tree_pool
        # self.mutation_configuration.pool_addition_mode = pool_mode_controller(
        #     self.mutation_configuration.pool_addition_mode, self.X, self.y
        # )

    def tree_initialization_function(self, pset, toolbox: TypedToolbox):
        if self.initial_tree_size is None:
            toolbox.expr = partial(gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        else:
            min_height, max_height = self.initial_tree_size.split("-")
            min_height = int(min_height)
            max_height = int(max_height)
            if self.basic_primitives.startswith("Pipeline"):
                if self.constant_ratio > 0:
                    toolbox.expr = partial(
                        genHalfAndHalf_STGP_constant_biased,
                        pset=pset,
                        min_=min_height,
                        max_=max_height,
                        constant_ratio=self.constant_ratio,
                    )
                else:
                    toolbox.expr = partial(
                        genHalfAndHalf_STGP,
                        pset=pset,
                        min_=min_height,
                        max_=max_height,
                    )
            else:
                toolbox.expr = partial(
                    gp.genHalfAndHalf, pset=pset, min_=min_height, max_=max_height
                )

    def mutation_expression_function(self, toolbox: TypedToolbox):
        if self.mutation_configuration.mutation_expr_height is not None:
            # custom defined height
            min_, max_ = self.mutation_configuration.mutation_expr_height.split("-")
            min_, max_ = int(min_), int(max_)
            toolbox.expr_mut = partial(gp.genFull, min_=min_, max_=max_)
        elif self.basic_primitives == "StrongTyped":
            # STGP
            toolbox.expr_mut = partial(gp.genFull, min_=1, max_=3)
        elif self.basic_primitives.startswith("Pipeline"):
            toolbox.expr_mut = partial(genHalfAndHalf_STGP, min_=0, max_=2)
        else:
            toolbox.expr_mut = partial(gp.genFull, min_=0, max_=2)

    def get_crossover_configuration(self):
        return self.crossover_configuration

    def selection_operator_initialization(self, toolbox):
        if self.bloat_control is not None:
            if self.bloat_control.get("double_lexicase", False):
                self.select = "DoubleLexicase"
            if self.bloat_control.get("double_tournament", False):
                self.select = "DoubleTournament"
            if self.bloat_control.get("statistics_tournament", False):
                self.select = "StatisticsTournament"
            if self.bloat_control.get("lexicographic_pressure", False):
                self.select = "LPP"
        # selection operators
        if isinstance(self.select, Selection):
            toolbox.register("select", self.select.select)
        elif self.select == "LexicaseDCD":
            toolbox.register("select", selLexicaseDCD)
        elif self.select == "LexicaseTournament":
            toolbox.register("select", selLexicaseTournament)
        elif self.select == "HOFRandom":
            toolbox.register("select", selHOFRandom, hof=self.hof)
        elif self.select.startswith("KNN"):
            base_operator, neighbor, strategy = self.select.split("-")[1:]
            neighbor = int(neighbor)
            toolbox.register(
                "select",
                selLexicaseKNN,
                base_operator=base_operator,
                neighbor=neighbor,
                strategy=strategy,
                y=self.y,
            )
        elif self.select == "Tournament":
            toolbox.register(
                "select", tools.selTournament, tournsize=self.param["tournament_size"]
            )
        elif self.select == "Tournament-Lexicase":
            select = TournamentLexicase(self, **self.param)
            toolbox.select = partial(select.select)
        elif self.select.startswith("Tournament-"):
            toolbox.register(
                "select", tools.selTournament, tournsize=int(self.select.split("-")[1])
            )
        elif self.select.startswith("TournamentNovelty"):
            toolbox.register(
                "select", selTournamentNovelty, tournsize=int(self.select.split("-")[1])
            )
        elif self.select.startswith("TournamentPlus-"):
            toolbox.register(
                "select", selTournamentPlus, tournsize=int(self.select.split("-")[1])
            )
        elif self.select == "StatisticsTournament":
            toolbox.register(
                "select",
                selStatisticsTournament,
                tournsize=self.bloat_control.get("fitness_size", 7),
            )
        elif self.select == "TournamentDCD":
            toolbox.register("select", selTournamentDCD)
        elif self.select == "HybridLexicaseDCD":
            toolbox.register("select", hybrid_lexicase_dcd)
        elif self.select == "BatchTournament":
            toolbox.register("select", batch_tournament_selection)
        elif self.select == "LPP":
            toolbox.register("select", selLexicographicParsimonyPressure)
        elif self.select in ["AutomaticLexicase"] + map_elite_series:
            toolbox.register("select", selAutomaticEpsilonLexicaseFast)
        elif self.select == "CLLexicase+":
            toolbox.register("select", selAutomaticEpsilonLexicaseCLFast)
        elif self.select == "InverseCLLexicase+":
            toolbox.register("select", selAutomaticEpsilonLexicaseInverseCLFast)
        elif self.select == "AutomaticLexicaseFast":
            toolbox.register("select", selAutomaticEpsilonLexicaseFast)
        elif self.select == "GroupALS":
            toolbox.register("select", selGroupALS)
        elif self.select == "GroupALS-DCD":
            toolbox.register(
                "select", partial(selGroupALS, inner_selection="LexicaseDCD")
            )
        elif self.select == "GroupDCD-ALS":
            toolbox.register("select", partial(selGroupALS, inner_selection="DCD-ALS"))
        elif self.select == "Niching":
            toolbox.register("select", niche_base_selection)
        elif self.select == "Niching+":
            toolbox.register("select", partial(niche_base_selection, key_objective=1))
        elif self.select == "DoubleLexicase" or self.select == "SoftmaxDLS":
            lexicase_round = self.bloat_control_configuration.lexicase_round
            size_selection = self.bloat_control_configuration.size_selection
            if self.select == "SoftmaxDLS":
                size_selection = "Softmax"
            toolbox.register(
                "select",
                partial(
                    doubleLexicase,
                    lexicase_round=lexicase_round,
                    size_selection=size_selection,
                ),
            )
        elif self.select == "DoubleTournament":
            # select with double tournament
            toolbox.register(
                "select",
                selDoubleTournament,
                fitness_size=self.bloat_control.get("fitness_size", 7),
                parsimony_size=self.bloat_control.get("parsimony_size", 1.4),
                fitness_first=True,
            )
        elif self.select == "DoubleRound":
            toolbox.register("select", selDoubleRound)
        elif self.select.startswith("LexicaseParetoTournament"):
            if "-" in self.select:
                subset_size = int(self.select.split("-")[1])
            else:
                subset_size = 2
            if self.select.startswith("LexicaseParetoTournamentWeighted"):
                toolbox.register(
                    "select",
                    sel_lexicase_pareto_tournament_weighted_subset,
                    subset_size=subset_size,
                )
            else:
                toolbox.register(
                    "select",
                    sel_lexicase_pareto_tournament_random_subset,
                    subset_size=subset_size,
                )
        elif self.select.startswith("ParetoTournament"):
            if "-" in self.select:
                subset_ratio = float(self.select.split("-")[1])
            else:
                subset_ratio = 0.1
            # Number of models to randomly select in each subset
            subset_size = int(self.n_pop * subset_ratio)
            toolbox.register("select", sel_pareto_tournament, subset_size=subset_size)
        elif self.select == "SubsetBest":
            toolbox.register("select", sel_subset_best)
        elif self.select == "DoubleRound-Random":
            toolbox.register(
                "select",
                selDoubleRound,
                count=self.param["double_round_count"],
                base_operator="Random",
            )
        elif self.select == "DoubleRound-Tournament":
            toolbox.register(
                "select",
                selDoubleRound,
                count=self.param["double_round_count"],
                base_operator="Tournament",
                tournsize=self.param["double_round_tournament_size"],
            )
        elif self.select == "Parsimonious-DoubleRound-Tournament":
            toolbox.register(
                "select",
                selDoubleRound,
                count=self.param["double_round_count"],
                base_operator="Tournament",
                tournsize=self.param["double_round_tournament_size"],
                parsimonious=True,
            )
        elif self.select == "RandomPlus":
            toolbox.register("select", selRandomPlus)
        elif self.select == "Bagging":
            toolbox.register("select", selBagging)
        elif self.select == "AutomaticLexicaseK":
            toolbox.register("select", selAutomaticEpsilonLexicaseK)
        elif self.select == "GPED":
            toolbox.register("select", selGPED)
        elif self.select == "MaxAngleSelection":
            toolbox.register("select", selMaxAngleSelection, target=self.y)
        elif self.select == "AngleDrivenSelection":
            toolbox.register("select", selAngleDrivenSelection, target=self.y)
        elif self.select == "Knockout":
            toolbox.register("select", selKnockout)
        elif self.select == "Knockout-A":
            toolbox.register("select", selKnockout, auto_case=True)
        elif self.select == "Knockout-S":
            toolbox.register("select", selKnockout, version="S")
        elif self.select == "Knockout-SA":
            toolbox.register("select", selKnockout, version="S", auto_case=True)
        elif self.select in ["Random"]:
            toolbox.register("select", selRandom)
        elif self.select == "Hybrid":
            toolbox.register("select", selHybrid)
        elif self.select == "Auto":
            pass
        else:
            raise Exception

    def primitive_initialization(self, x):
        # Initialize the function set
        if isinstance(self.basic_primitives, str) and self.basic_primitives.startswith(
            "Pipeline"
        ):
            pset = get_typed_pset(
                self.X.shape[1],
                self.basic_primitives,
                categorical_features=self.categorical_features,
            )
        elif (
            isinstance(self.basic_primitives, str)
            and "optimal" in self.basic_primitives
        ):
            pset = PrimitiveSet("MAIN", x.shape[1])
            self.basic_primitives = ",".join(
                [
                    "Add",
                    "Sub",
                    "Mul",
                    "AQ",
                    "Sqrt",
                    "Sin",
                    "Cos",
                    "Max",
                    "Min",
                    "Neg",
                ]
            )
            self.evaluation_configuration.basic_primitives = self.basic_primitives
            self.mutation_configuration.basic_primitives = self.basic_primitives
            self.add_primitives_to_pset(pset)
        elif self.basic_primitives == "CDFC":
            # Primitives used in CDFC
            # "Genetic programming for multiple-feature construction on high-dimensional classification"
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            add_basic_operators(pset)
            pset.addPrimitive(np.maximum, 2)
            pset.addPrimitive(if_function, 3)
        elif isinstance(self.basic_primitives, str) and "," in self.basic_primitives:
            # an array of basic primitives
            pset = gp.PrimitiveSet("MAIN", x.shape[1])
            self.add_primitives_to_pset(pset)
        else:
            pset = PrimitiveSet("MAIN", x.shape[1])
            self.basic_primitives = ",".join(
                [
                    "Add",
                    "Sub",
                    "Mul",
                    "AQ",
                ]
            )
            self.add_primitives_to_pset(pset)

        if self.remove_constant_features:
            self.columns_without_constants = remove_constant_variables(pset, x)

        # add constant
        for constant in ["rand101", "pi", "e"]:
            if hasattr(gp, constant):
                delattr(gp, constant)
        attrs_to_remove = [
            attr
            for attr in dir(gp)
            if attr.startswith("rand") and is_number(attr.replace("rand", ""))
        ]
        for attr in attrs_to_remove:
            delattr(gp, attr)

        if self.constant_type is None:
            pass
        elif self.basic_primitives == False:
            pset.addEphemeralConstant(
                "rand101", lambda: random.randint(-1, 1), NumericalFeature
            )
        elif self.constant_type == "Float":
            if self.basic_primitives.startswith("Pipeline"):
                pset.addEphemeralConstant(
                    "rand101", lambda: random.uniform(-1, 1), float
                )
            else:
                pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
        elif self.constant_type in ["GD", "GD+", "GD-", "GD--"]:

            def random_variable():
                return torch.randn(1, requires_grad=True, dtype=torch.float32)

            pset.addEphemeralConstant("rand101", random_variable)
        elif self.constant_type == "SRC":
            biggest_val = np.max(np.abs(self.X))
            generator = scaled_random_constant(biggest_val)
            pset.addEphemeralConstant("rand101", generator)
        else:
            constant_generator = constant_controller(self.constant_type)
            if not isinstance(pset, gp.PrimitiveSet):
                pset.addEphemeralConstant("rand101", constant_generator, float)
            else:
                pset.addEphemeralConstant("rand101", constant_generator)
        # Check if MGP mode is enabled and create a new primitive set for each gene
        if isinstance(pset, PrimitiveSet) and self.mgp_mode is True:
            new_pset = MultiplePrimitiveSet("MAIN", self.X.shape[1])
            new_pset.mgp_scope = self.mgp_scope
            new_pset.layer_mgp = self.layer_mgp
            for x in range(self.gene_num):
                # Copy the base primitive set and modify it for this gene
                base_count = self.X.shape[1]
                base_pset = copy.deepcopy(pset)
                if x >= self.mgp_scope and self.strict_layer_mgp:
                    # If using strict layer MGP, remove input features for higher layers
                    base_pset.terminals[object] = []
                    base_pset.arguments = []
                if self.mgp_scope is not None:
                    if self.layer_mgp:
                        assert (
                            self.gene_num % self.mgp_scope == 0
                        ), "Number of genes must be a multiple of scope!"
                        # Calculate the start and end boundaries for this gene based on the MGP scope and layer setting
                        start_base = max(
                            base_count + ((x // self.mgp_scope) - 1) * self.mgp_scope,
                            base_count,
                        )
                        end_base = max(
                            base_count + (x // self.mgp_scope) * self.mgp_scope,
                            base_count,
                        )
                    else:
                        # Calculate the start and end boundaries for this gene based on a sliding window approach
                        start_base = max(base_count + x - self.mgp_scope, base_count)
                        end_base = base_count + x
                    assert start_base >= self.X.shape[1]
                else:
                    # If no MGP scope is specified, include all input features for this gene
                    start_base = base_count
                    end_base = base_count + x
                for i in range(start_base, end_base):
                    base_pset.addTerminal(f"ARG{i}", f"ARG{i}")
                    base_pset.arguments.append(f"ARG{i}")
                base_pset.pset_id = x
                new_pset.pset_list.append(base_pset)
            # Use the new primitive set for subsequent operations
            pset = new_pset
        # Check if register mode is enabled and add input registers to the primitive set
        elif isinstance(pset, PrimitiveSet) and self.mgp_mode == "Register":
            pset.number_of_register = self.number_of_register
            # Input register is easy to implement
            base_count = self.X.shape[1]
            for i in range(base_count, base_count + self.number_of_register):
                pset.addTerminal(f"ARG{i}", f"ARG{i}")
                pset.arguments.append(f"ARG{i}")

        if self.columns != None:
            rename_dict = dict({f"ARG{k}": v for k, v in enumerate(self.columns)})
            pset.renameArguments(**rename_dict)
        self.pset = pset
        return pset

    def add_primitives_to_pset(
        self, pset: PrimitiveSet, primitives=None, transformer_wrapper=False
    ):
        if primitives is None:
            primitives = self.basic_primitives
        if self.custom_primitives is None:
            custom_primitives = {}
        else:
            custom_primitives = self.custom_primitives
            primitives = primitives.split(",") + ",".join(custom_primitives.keys())
        constant_dict = {"pi": scipy.constants.pi}

        for p in primitives.split(","):
            p = p.strip()
            if p in constant_dict:
                pset.addTerminal(constant_dict[p], name=constant_dict[p])
                continue
            elif p in custom_primitives:
                primitive = custom_primitives[p]
                number_of_parameters = len(inspect.signature(primitive).parameters)
                primitive = (primitive, number_of_parameters)
            else:
                if self.constant_type in ["GD", "GD+", "GD-", "GD--"]:
                    primitive = get_differentiable_functions(p)
                else:
                    primitive = get_functions(p)
            if transformer_wrapper:
                pset.addPrimitive(make_class(primitive[0]), primitive[1], name=p)
            else:
                pset.addPrimitive(primitive[0], primitive[1], name=p)

    def archive_initialization(self):
        # archive initialization
        if self.ensemble_size == "auto":
            # Automatically determine the ensemble size
            self.hof = LexicaseHOF()
        elif self.ensemble_selection == "SAM":
            # Automatically determine the ensemble size
            def comparison(a, b):
                return a.sam_loss < b.sam_loss

            self.hof = CustomHOF(
                self.ensemble_size,
                comparison_function=comparison,
                key_metric=lambda x: -x.sam_loss,
            )
        elif self.ensemble_selection == "WeightedSum":
            # Automatically determine the ensemble size
            def comparison(a, b):
                return sum(a.fitness.wvalues) > sum(b.fitness.wvalues)

            self.hof = CustomHOF(
                self.ensemble_size,
                comparison_function=comparison,
                key_metric=lambda x: sum(x.fitness.wvalues),
            )
        elif self.ensemble_selection in ["Statistical", "Statistical-FewTree"]:
            # Automatically determine the ensemble size
            def comparison(a, b):
                return (
                    # A significantly better in fitness
                    (
                        a.fitness.wvalues[0] > b.fitness.wvalues[0]
                        and wilcoxon(a.case_values, b.case_values).pvalue <= 0.05
                    )
                    or (
                        # A significantly better in number of features
                        self.ensemble_selection == "Statistical-FewTree"
                        and len(a.gene) < len(b.gene)
                        and wilcoxon(a.case_values, b.case_values).pvalue > 0.05
                    )
                )

            self.hof = CustomHOF(
                self.ensemble_size,
                comparison_function=comparison,
                key_metric=lambda x: sum(x.fitness.wvalues),
            )
        elif self.validation_ratio > 0 or self.ensemble_selection == "VS":
            # Automatically determine the ensemble size
            def comparison(a, b):
                a_score = a.validation_score
                b_score = b.validation_score
                # a_score = a.case_values
                # b_score = b.optimal_case_values
                if np.all((a_score - b_score) == 0):
                    return False
                p_value = wilcoxon(a_score, b_score).pvalue
                if np.median(a_score) < np.median(b_score) and p_value < 0.05:
                    return True
                elif np.median(a_score) <= np.median(b_score) and sum(
                    [len(tree) for tree in a.gene]
                ) < sum([len(tree) for tree in b.gene]):
                    return True
                else:
                    return False

            class VHOF(CustomHOF):
                def insert(self, item):
                    # item.optimal_case_values = item.case_values
                    super().insert(item)

            self.hof = VHOF(
                self.ensemble_size,
                comparison_function=comparison,
                key_metric=lambda x: sum(x.fitness.wvalues),
            )

        elif self.ensemble_selection == "StrictlyImprovement":
            self.hof = StrictlyImprovementHOF(self.ensemble_size)
        elif self.ensemble_selection == "GeneralizationHOF":
            assert self.ensemble_size == 1
            self.hof = GeneralizationHOF(
                self.X, self.y, self.pset, verbose=self.verbose
            )
        elif self.ensemble_selection == "UniqueObjective":
            self.hof = HallOfFame(
                self.ensemble_size,
                similar=lambda a, b: a.fitness.wvalues == b.fitness.wvalues,
            )
        elif self.ensemble_selection == "UniqueSemantics":
            self.hof = HallOfFame(
                self.ensemble_size,
                similar=lambda a, b: np.all(a.predicted_values == b.predicted_values),
            )
        elif self.ensemble_selection == "ShapelyHOF":
            self.hof = ShapleyPrunedHallOfFame(self.ensemble_size, y=self.y)
        elif self.ensemble_selection == "GreedyHOF":
            self.hof = GreedyHallOfFame(self.ensemble_size, y=self.y, **self.param)
        elif self.ensemble_selection == "GridMAPElitesHOF":
            self.hof = GridMAPElites(self.ensemble_size, y=self.y, **self.param)
        elif self.ensemble_selection == "CVT-MAPElitesHOF":
            self.hof = CVTMAPElitesHOF(self.ensemble_size, y=self.y, **self.param)
        elif (
            isinstance(self.ensemble_selection, str)
            and "Similar" in self.ensemble_selection
        ):
            ratio = 0.95
            if "-" in self.ensemble_selection:
                ratio = float(self.ensemble_selection.split("-")[1])

            def similar(a, b):
                return (
                    np.dot(a.predicted_values, b.predicted_values)
                    / (norm(a.predicted_values) * norm(b.predicted_values))
                    > ratio
                )

            self.hof = HallOfFame(self.ensemble_size, similar=similar)
        elif self.ensemble_selection == "Equal":

            def similar(a, b):
                return np.all(np.equal(a.predicted_values, b.predicted_values))

            self.hof = HallOfFame(self.ensemble_size, similar=similar)
        elif self.ensemble_selection == "Bootstrap":
            self.hof = BootstrapHallOfFame(self.X, self.ensemble_size)
        elif self.ensemble_selection == "OOB":
            self.hof = OOBHallOfFame(self.X, self.y, self.toolbox, self.ensemble_size)
        elif self.ensemble_selection == "GreedySelection":
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y)
            self.hof.novelty_weight = self.novelty_weight
        elif self.ensemble_selection == "GreedySelection-Resampling":
            if isinstance(self, ClassifierMixin):
                self.hof = GreedySelectionHallOfFame(
                    self.ensemble_size,
                    self.y,
                    unique=False,
                    bagging_iteration=20,
                    loss_function="MSE",
                    inner_sampling=0.5,
                    outer_sampling=0.25,
                )
            else:
                self.hof = GreedySelectionHallOfFame(
                    self.ensemble_size,
                    self.y,
                    unique=False,
                    bagging_iteration=20,
                    loss_function="MSE",
                    inner_sampling=0.1,
                    outer_sampling=0.25,
                )
        elif self.ensemble_selection == "GreedySelection-Resampling-MSE":
            self.hof = GreedySelectionHallOfFame(
                self.ensemble_size,
                self.y,
                unique=False,
                bagging_iteration=20,
                loss_function="MSE",
            )
        elif self.ensemble_selection == "GreedySelection-Resampling-CrossEntropy":
            self.hof = GreedySelectionHallOfFame(
                self.ensemble_size,
                self.y,
                unique=False,
                bagging_iteration=20,
                loss_function="CrossEntropy",
            )
        elif self.ensemble_selection == "GreedySelection-Resampling-Hinge":
            self.hof = GreedySelectionHallOfFame(
                self.ensemble_size,
                self.y,
                unique=False,
                bagging_iteration=20,
                loss_function="Hinge",
            )
        elif self.ensemble_selection == "GreedySelection-Resampling-ZeroOne":
            self.hof = GreedySelectionHallOfFame(
                self.ensemble_size,
                self.y,
                unique=False,
                bagging_iteration=20,
                loss_function="ZeroOne",
            )
        elif isinstance(
            self.ensemble_selection, str
        ) and self.ensemble_selection.startswith(
            "GreedySelection-Resampling-MSE-Custom-"
        ):
            parameter = self.ensemble_selection.replace(
                "GreedySelection-Resampling-MSE-Custom-", ""
            )
            inner_sampling, outer_sampling, bagging_iteration = parameter.split("-")
            inner_sampling = float(inner_sampling)
            outer_sampling = float(outer_sampling)
            bagging_iteration = int(bagging_iteration)
            self.hof = GreedySelectionHallOfFame(
                self.ensemble_size,
                self.y,
                unique=False,
                bagging_iteration=bagging_iteration,
                loss_function="MSE",
                inner_sampling=inner_sampling,
                outer_sampling=outer_sampling,
            )
        elif (
            isinstance(self.ensemble_selection, str)
            and "GreedySelection-Resampling~" in self.ensemble_selection
        ):
            self.ensemble_selection, initial_size = self.ensemble_selection.split("~")
            initial_size = int(initial_size)
            self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y)
            self.hof.unique = False
            self.hof.bagging_iteration = 20
            self.hof.initial_size = initial_size
        elif self.ensemble_selection == "NoveltySelection":
            self.hof = NoveltyHallOfFame(self.ensemble_size, self.y)
            self.hof.novelty_weight = self.novelty_weight
        elif isinstance(self.ensemble_selection, str) and (
            "similar" in self.ensemble_selection
        ):
            # Using similarity metric to filter out individuals
            if "-" in self.ensemble_selection:
                ratio = float(self.ensemble_selection.split("-")[1])
            else:
                ratio = 0.95

            def similar(a, b):
                return cosine(a.case_values, b.case_values) >= ratio

            self.hof = HallOfFame(self.ensemble_size, similar=similar)
        elif self.score_func == "NoveltySearch":
            if self.ensemble_selection == "DREP":
                self.hof = DREPHallOfFame(self.ensemble_size, self.y)
            elif self.ensemble_selection == "GreedySelection":
                self.hof = GreedySelectionHallOfFame(self.ensemble_size, self.y)
            elif self.ensemble_selection == "Traditional":
                self.hof = HallOfFame(self.ensemble_size)
            else:
                self.hof = NoveltyHallOfFame(self.ensemble_size, self.y)
            self.hof.novelty_weight = self.novelty_weight
        elif self.ensemble_selection == "MultiTask":
            self.hof = MultiTaskHallOfFame(self.ensemble_size, self.base_model_list)
        elif self.validation_size > 0 and self.early_stop == 0:
            # in multi-objective case
            # don't need to save hof
            self.hof = ValidationHallOfFame(self.get_validation_score)
            assert (
                self.environmental_selection is not None
            ), "Need to specify Environmental Selection!"
        elif self.validation_size > 0 and self.early_stop > 0:
            self.hof = EarlyStoppingHallOfFame(self.get_validation_score)
        elif self.ensemble_selection == None or self.ensemble_selection in [
            "None",
            "none",
            "MAP-Elite",
            "Half-MAP-Elite",
            "AngleDrivenSelection",
            "MaxAngleSelection",
        ]:
            if self.redundant_hof_size > 0:
                self.hof = HallOfFame(self.redundant_hof_size)
            else:
                self.hof = HallOfFame(self.ensemble_size)
        else:
            raise Exception
        if isinstance(self.hof, EnsembleSelectionHallOfFame):
            self.hof.verbose = self.verbose
        self.model_size_archive = ModelSizeArchive(self.n_pop, self.score_func)
        if self.bloat_control is not None and self.bloat_control.get(
            "lasso_prune", False
        ):
            candidates = self.bloat_control.get("lasso_candidates", 1)
            self.lasso_hof = HallOfFame(candidates)
        else:
            self.lasso_hof = None

    def get_terminal_probability(self):
        """
        Using feature importance at initialization
        """
        # get a probability distribution based on the importance of original features in a random forest
        if isinstance(self, ClassifierMixin):
            r = RandomForestClassifier(n_estimators=5)
        else:
            r = RandomForestRegressor(n_estimators=5)
        r.fit(self.X, self.y)
        terminal_prob = np.append(r.feature_importances_, 0.1)
        terminal_prob = terminal_prob / np.sum(terminal_prob)
        return terminal_prob

    def construct_global_feature_pool(self, pop):
        good_features, threshold = construct_feature_pools(
            pop,
            True,
            threshold_ratio=self.cx_threshold_ratio,
            good_features_threshold=self.good_features_threshold,
        )
        self.good_features = good_features
        self.cx_threshold = threshold

    def fit(self, X, y, test_X=None, categorical_features=None):
        if self.precision == "Float32":
            X = X.astype(np.float32)
            y = y.astype(np.float32)
        self.counter_initialization()
        self.history_initialization()

        self.categorical_features = categorical_features
        if self.categorical_encoding is not None:
            categorical_features: list[bool]
            categorical_indices = [
                i
                for i, is_categorical in enumerate(categorical_features)
                if is_categorical
            ]
            assert (
                len(categorical_features) == X.shape[1]
            ), f"Mismatch between categorical indices and number of features, {len(categorical_features)} vs {X.shape[1]}"
            if self.categorical_encoding == "Target":
                self.categorical_encoder = TargetEncoder(cols=categorical_indices)
            else:
                raise Exception(
                    f"Unknown categorical encoding method {self.categorical_encoding}"
                )
            X = np.array(self.categorical_encoder.fit_transform(X, y))

        # whether input data is standardized
        # self.standardized_flag = is_standardized(X)

        self.x_shape = X.shape
        self.y_shape = y.shape
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()  # store column names
            X = X.to_numpy()
            y = y.to_numpy()

        # Normalize X and y if specified
        if self.normalize:
            X = self.x_scaler.fit_transform(X, y)
            if len(y.shape) == 1:
                y = np.array(y).reshape(-1, 1)
            y = self.y_scaler.fit_transform(y)
            X = self.add_noise_to_data(X)

        # Transductive Learning
        if test_X is not None:
            if self.normalize:
                test_X = self.x_scaler.transform(test_X)
            self.test_X = test_X
            self.evaluation_configuration.transductive_learning = True

        # Split into train and validation sets if validation size is greater than 0
        if self.validation_size > 0:
            X, self.valid_x, y, self.valid_y = train_test_split(
                X, y, test_size=self.validation_size
            )
            self.valid_y = self.valid_y.flatten()

        if self.validation_based_ensemble_selection > 0:
            X, self.des_valid_x, y, self.des_valid_y = train_test_split(
                X, y, test_size=self.validation_based_ensemble_selection
            )
            self.des_valid_y = self.des_valid_y.flatten()

        X = self.pretrain(X, y)

        if (
            isinstance(self.environmental_selection, (NSGA2, SPEA2))
            and self.environmental_selection.knee_point == "Validation"
        ):
            X, valid_x, y, valid_y = train_test_split(X, y, test_size=0.2)
            self.environmental_selection.validation_x = valid_x
            self.environmental_selection.validation_y = valid_y

        # Save X and y
        self.X: np.ndarray
        self.y: np.ndarray
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.flatten()
        self.X, self.y = X, y

        # Initialize population with lazy initialization
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

        if self.gradient_boosting:
            # Using gradient boosting mode
            assert self.boost_size == 1, "Only supports a single model"
            # Deep copy original y and gene number
            original_y = copy.deepcopy(self.y)
            original_gene_num = self.gene_num
            best_models = []

            # Set number of genes to 1
            self.gene_num = 1
            # Initialize population with new gene number
            self.lazy_init(X)
            for g in range(original_gene_num):
                # Clear fitness values
                for p in self.pop:
                    del p.fitness.values
                self.hof.clear()
                # Run genetic programming
                pop, log = self.eaSimple(
                    self.pop,
                    self.toolbox,
                    self.cross_pb,
                    self.mutation_pb,
                    self.n_gen,
                    stats=mstats,
                    halloffame=self.hof,
                    verbose=self.verbose,
                )
                self.pop = pop
                self.hof[0].pipe = self.get_base_model()
                self.final_model_lazy_training(self.hof)
                self.stacking_strategy.stacking_layer_generation(self.X, self.y)
                # If using the predict function, we need to consider a lot of details about normalization
                Yp = self.feature_generation(X, self.hof[0])
                y_pred = self.hof[0].pipe.predict(Yp)
                # Add the best GP tree to a list
                best_models.extend(copy.deepcopy(self.hof[0].gene))
                if self.verbose:
                    print("Iteration %d" % g, "Score: %f" % (r2_score(self.y, y_pred)))
                # Gradient boost for regression
                self.y = self.y - y_pred
            # Reset y to original value
            self.y = original_y
            assert len(self.hof) == 1
            # Retrain the best individual with original number of genes
            best_ind: MultipleGeneGP = self.hof[0]
            best_ind.gene = best_models
            best_ind.pipe = self.get_base_model()
            # clear hall of fame
            self.hof.clear()
            self.hof.update([best_ind])
            self.final_model_lazy_training(self.hof)
            # in fact, not useful
            self.stacking_strategy.stacking_layer_generation(self.X, self.y)
        else:
            # Not using gradient boosting mode
            pop, log = self.eaSimple(
                self.pop,
                self.toolbox,
                self.cross_pb,
                self.mutation_pb,
                self.n_gen,
                stats=mstats,
                halloffame=self.hof,
                verbose=self.verbose,
            )
            self.pop = pop

            self.final_model_lazy_training(self.hof)
            self.stacking_strategy.stacking_layer_generation(X, y)
        self.training_with_validation_set()
        return self

    def pretrain(self, X, y):
        if self.learner is not None:
            data = []
            models = []
            for learner in self.learner.split(","):
                if learner == "LR":
                    model = LinearRegression()
                elif learner == "LightGBM":
                    model = LGBMRegressor()
                elif learner == "KNN":
                    model = KNeighborsRegressor()
                else:
                    raise Exception
                model.fit(X, y)
                data.append(model.predict(X).flatten())
                models.append(model)
            X = np.concatenate([X, np.array(data).T], axis=1)
            self.pretrain_models = models

        self.pretrain_reference_model(X, y)
        return X

    def pretrain_reference_model(self, X, y):
        if (
            isinstance(self.score_func, R2PACBayesian)
            and self.score_func.sharpness_type == SharpnessType.DataLGBM
        ):
            if self.pac_bayesian.reference_model == "XGB":
                self.reference_lgbm = XGBRegressor(n_jobs=1)
            elif self.pac_bayesian.reference_model == "LR":
                self.reference_lgbm = LinearRegression()
            elif self.pac_bayesian.reference_model == "Ridge":
                self.reference_lgbm = RidgeGCV()
            elif self.pac_bayesian.reference_model == "KNN":
                self.reference_lgbm = KNeighborsRegressor(n_neighbors=3, n_jobs=1)
            elif self.pac_bayesian.reference_model == "D-KNN":
                self.reference_lgbm = KNeighborsRegressor(
                    n_neighbors=3, n_jobs=1, weights="distance"
                )
            elif self.pac_bayesian.reference_model == "DT":
                self.reference_lgbm = DecisionTreeRegressor()
            elif self.pac_bayesian.reference_model == "RF":
                self.reference_lgbm = RandomForestRegressor()
            elif self.pac_bayesian.reference_model == "ET":
                self.reference_lgbm = ExtraTreesRegressor()
            elif self.pac_bayesian.reference_model == "KR":
                self.reference_lgbm = GridSearchCV(
                    KernelRidge(),
                    {
                        "kernel": (
                            "linear",
                            "poly",
                            "rbf",
                            "sigmoid",
                        ),
                        "alpha": (
                            1e-4,
                            1e-2,
                            0.1,
                            1,
                        ),
                    },
                )
            self.reference_lgbm.fit(X, y)

    def add_noise_to_data(self, X):
        param = self.param
        if "noise_scale" in param and param["noise_scale"] > 0:
            # Generate Gaussian noise with the same shape as `X`
            noise = np.random.normal(loc=0, scale=param["noise_scale"], size=X.shape)
            # Add the noise to `X`
            X = X + noise
        return X

    def individual_prediction(self, X, individuals=None):
        if self.normalize:
            X = self.x_scaler.transform(X)

        # get detailed prediction results instead of an ensemble
        predictions = []
        if individuals is None:
            individuals = self.hof
        # try to fit all models
        self.final_model_lazy_training(individuals)
        for individual in individuals:
            if len(individual.gene) == 0:
                continue
            Yp = self.feature_construction(X, individual)
            if isinstance(Yp, torch.Tensor):
                Yp = Yp.detach().numpy()
            predicted = individual.pipe.predict(Yp)
            predictions.append(predicted)
        predictions = np.array(predictions)

        if self.normalize:
            predictions = self.y_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).reshape(len(individuals), -1)
        return predictions

    def feature_construction(self, X, individual):
        Yp = multi_tree_evaluation(
            individual.gene,
            self.pset,
            X,
            self.original_features,
            register_array=individual.parameters["Register"]
            if self.mgp_mode == "Register"
            else None,
            configuration=self.evaluation_configuration,
            noise_configuration=self.pac_bayesian.noise_configuration,
            individual_configuration=individual.individual_configuration,
        )
        return Yp

    def final_model_lazy_training(self, pop, X=None, y=None, force_training=False):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        for p in pop:
            if self.test_data is not None:
                X = X[: -len(self.test_data)]
            Yp = self.feature_generation(X, p)
            self.train_final_model(p, Yp, y, force_training=force_training)

    def feature_generation(
        self,
        X,
        individual: MultipleGeneGP,
        random_noise=0,
        random_seed=0,
        noise_configuration=None,
    ):
        self.pac_bayesian: PACBayesianConfiguration
        if self.pac_bayesian is not None and self.pac_bayesian.cached_sharpness:
            evaluation_cache = self.pac_bayesian.tree_sharpness_cache
        else:
            evaluation_cache = None

        if individual.num_of_active_trees > 0:
            genes = individual.gene[: individual.num_of_active_trees]
        else:
            genes = individual.gene
        Yp = multi_tree_evaluation(
            genes,
            self.pset,
            X,
            self.original_features,
            configuration=self.evaluation_configuration,
            register_array=individual.parameters["Register"]
            if self.mgp_mode == "Register"
            else None,
            random_noise=random_noise,
            random_seed=random_seed,
            noise_configuration=noise_configuration,
            reference_label=self.y,
            individual_configuration=individual.individual_configuration,
            evaluation_cache=evaluation_cache,
        )
        if isinstance(Yp, torch.Tensor):
            Yp = Yp.detach().numpy()
        return Yp

    @split_and_combine_data_decorator(data_arg_position=1, data_arg_name="X")
    def predict(self, X, return_std=False):
        if self.categorical_encoding is not None:
            X = np.array(self.categorical_encoder.transform(X))
            print("Target encoding for categorical features")

        if self.normalize:
            # Scale X data if normalize flag is set
            X = self.x_scaler.transform(X)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X = self.pretrain_predict(X)

        prediction_data_size = X.shape[0]
        if self.test_data is not None:
            # Concatenate new X data with existing X data in the transductive learning setting
            X = np.concatenate([self.X, X])

        # Train the final model using lazy training
        self.final_model_lazy_training(self.hof, force_training=self.force_retrain)

        if self.meta_learner is not None and self.meta_learner != "DeepDES":
            return self.final_meta_learner.predict(X)

        predictions = []
        weight_list = []

        # Generate features for each individual in the hall of fame
        for individual in self.hof:
            individual: MultipleGeneGP
            if len(individual.gene) == 0:
                continue
            # Generate features using the individual's genes
            Yp = self.feature_generation(X, individual)
            if self.test_data is not None:
                # In transductive learning setting
                Yp = Yp[-prediction_data_size:]
            if isinstance(
                individual.pipe["Ridge"], SoftPLTreeRegressor
            ) and not isinstance(individual.pipe["Ridge"], SoftPLTreeRegressorEM):
                Yp = np.concatenate([Yp, np.zeros((len(Yp), 1))], axis=1)
            if self.intron_probability > 0:
                # Apply intron probability mask to Yp if intron_probability is greater than 0
                Yp = Yp[:, individual.active_gene]
            predicted = individual.pipe.predict(Yp)

            if self.bounded_prediction == True:
                predicted = np.clip(predicted, self.y.min(), self.y.max())
            if self.bounded_prediction == "Smooth":
                self.smooth_model: NearestValueTransformer
                predicted = self.smooth_model.transform(predicted)

            if self.normalize:
                # Unscale predicted values if normalize flag is set
                if len(predicted.shape) == 1:
                    predicted = predicted.reshape(-1, 1)

                predicted = self.y_scaler.inverse_transform(predicted)

                if len(predicted.shape) == 2 and predicted.shape[1] == 1:
                    predicted = predicted.flatten()
            if (
                hasattr(self.hof, "ensemble_weight")
                and len(self.hof.ensemble_weight) > 0
            ):
                # Append ensemble weights if they exist
                weight_list.append(
                    self.hof.ensemble_weight[individual_to_tuple(individual)]
                )

            if len(weight_list) > 0 or return_std or self.meta_learner is not None:
                # need to store all predictions
                predictions.append(predicted)
            else:
                # save memory
                if len(predictions) == 0:
                    predictions = [predicted]
                else:
                    predictions[0] += predicted
                del predicted

        if len(predictions) != len(self.hof):
            predictions[0] = predictions[0] / len(self.hof)
            assert len(predictions) == 1

        if isinstance(self.hof, MetaLearner):
            return self.hof.predict(predictions)
        if self.meta_learner is not None:
            if isinstance(self.final_meta_learner, DESMetaRegressor):
                if self.verbose:
                    self.final_meta_learner.plot_sample_weights(
                        X[:20], np.array(predictions).T[:20]
                    )
                final_prediction = self.final_meta_learner.predict(
                    X, np.array(predictions).T
                )
            else:
                raise Exception
        elif self.second_layer != "None" and self.second_layer != None:
            predictions = np.array(predictions).T
            final_prediction = predictions @ self.tree_weight
        elif len(weight_list) > 0:
            final_prediction = self.weighted_ensemble_prediction(
                predictions, weight_list, return_std
            )
        else:
            if return_std:
                final_prediction = np.mean(predictions, axis=0), np.std(
                    predictions, axis=0
                )
            else:
                final_prediction = self.make_ensemble_prediction(predictions)
        final_prediction = self.multi_task_reshape(final_prediction)
        if len(self.y_shape) == 2 and np.any(self.y_shape == 1):
            final_prediction = final_prediction.reshape(-1, 1)
        return final_prediction

    def construct_meta_learner(self):
        models = [
            GPWrapper(ind, self.feature_generation, self.y_scaler) for ind in self.hof
        ]
        if self.meta_learner == "DES":
            self.final_meta_learner = DynamicSelectionEnsemble(models)
        elif self.meta_learner == "RF-DES":
            self.final_meta_learner = RFRoutingEnsemble(models)
        elif self.meta_learner == "DeepDES":
            self.final_meta_learner = DESMetaRegressor(verbose=False, **self.param)

        if self.meta_learner == "DeepDES":
            # learn on an original y-scale
            if self.validation_based_ensemble_selection > 0:
                y = transform_y(self.des_valid_y, self.y_scaler)
                valid_predictions = self.individual_prediction(
                    self.des_valid_x, self.hof
                )
                valid_predictions = np.array(
                    [transform_y(p, self.y_scaler) for p in valid_predictions]
                ).T
                self.final_meta_learner.fit(self.des_valid_x, valid_predictions, y)
            else:
                cross_val_predict = np.array(
                    [
                        transform_y(ind.predicted_values, self.y_scaler)
                        for ind in self.hof
                    ]
                ).T
                y = transform_y(self.y, self.y_scaler)
                # self.final_meta_learner.fit(self.X, cross_val_predict, y)
                # usage = self.final_meta_learner.count_base_learner_usage(
                #     self.X, cross_val_predict
                # )
                # indices = np.argsort(usage)[:30]
                # hall_of_fame = [self.hof[i] for i in indices]
                # self.hof.clear()
                # self.hof.update(hall_of_fame)
                # cross_val_predict = np.array(
                #     [
                #         transform_y(ind.predicted_values, self.y_scaler)
                #         for ind in self.hof
                #     ]
                # ).T
                self.final_meta_learner.fit(self.X, cross_val_predict, y)
        else:
            self.final_meta_learner.fit(self.X, self.y)

    def weighted_ensemble_prediction(self, predictions, weight_list, return_std):
        predictions = np.array(predictions).T
        weight_list = np.array(weight_list)
        if return_std:
            final_prediction = weighted_avg_and_std(predictions.T, weight_list)
        else:
            final_prediction = predictions @ weight_list / weight_list.sum()
        return final_prediction

    def multi_task_reshape(self, final_prediction):
        if len(self.X) != len(self.y):
            number_of_tasks = len(self.y) // len(self.X)
            final_prediction = final_prediction.reshape(-1, number_of_tasks)
        return final_prediction

    def make_ensemble_prediction(self, predictions):
        if self.ensemble_prediction == "Mean":
            final_prediction = np.mean(predictions, axis=0)
        elif self.ensemble_prediction == "Median":
            final_prediction = np.median(predictions, axis=0)
        else:
            raise Exception
        return final_prediction

    def get_hof(self):
        if self.hof != None:
            return [x for x in self.hof]
        else:
            return None

    def append_evaluated_features(self, pop):
        if (
            isinstance(self.mutation_scheme, MutationOperator)
            or self.mutation_scheme in eda_operators
            or "uniform" in self.mutation_scheme
        ):
            return
        # append evaluated features to an archive
        # this archive can be used to eliminate repetitive features
        if self.useless_feature_ratio != None:
            mean_importance = np.quantile(
                [ind.coef for ind in pop], self.useless_feature_ratio
            )
        else:
            mean_importance = np.array([ind.coef for ind in pop]).mean()
        # mean_importance = np.mean([ind.fitness.wvalues[0] for ind in pop])
        for ind in pop:
            # if ind.fitness.wvalues[0] <= mean_importance:
            for gene, c in zip(ind.gene, ind.coef):
                if c <= mean_importance:
                    self.generated_features.add(str(gene))
                    if "semantic" in self.mutation_scheme:
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
            all_string = ",".join(sorted(all_string))
        if all_string not in self.evaluated_pop:
            self.evaluated_pop.add(all_string)
            return True
        else:
            return False

    def callback(self):
        if self.mutation_configuration.pool_based_addition:
            self.tree_pool: SemanticLibrary
            interval = self.mutation_configuration.pool_hard_instance_interval
            mode = self.mutation_configuration.library_clustering_mode
            if (
                self.current_gen == 0
                and self.mutation_configuration.lib_feature_selection is not None
            ):
                transformed_X = StandardScaler().fit_transform(self.X)
                self.pearson_matrix = np.corrcoef(transformed_X, rowvar=False)

            if "SemanticLibSize" in self.log_item:
                self.semantic_lib_log.semantic_lib_size_history.append(
                    len(self.tree_pool.trees)
                )
                self.semantic_lib_log.success_rate_update()
            if "SemanticLibOperator" in self.log_item:
                self.semantic_lib_log.best_individual_update(self.hof)

            if interval > 0 and self.current_gen % interval == 0:
                if (
                    self.current_gen > 0
                    and self.mutation_configuration.lib_feature_selection is not None
                    # not the final generation
                    and self.current_gen != self.n_gen
                ):
                    self.tree_pool.update_forbidden_list(
                        self.pop,
                        self.pearson_matrix,
                        self.mutation_configuration.lib_feature_selection,
                        self.mutation_configuration.lib_feature_selection_mode,
                        self,
                    )

                if self.current_gen > 0 and mode in [
                    "Label-K-Means",
                    "Feature-K-Means",
                ]:
                    pass
                else:
                    case_fitness = np.array([ind.case_values for ind in self.pop])
                    predicted_values = np.array(
                        [ind.predicted_values for ind in self.pop]
                    )
                    self.tree_pool.update_hard_instance(
                        case_fitness,
                        predicted_values,
                        mode,
                        self.X,
                        self.y,
                        self.current_gen,
                        self.n_gen,
                    )
            self.tree_pool.append_full_tree(self.pop, self.y)
            self.tree_pool.train_nn()

            if self.verbose:
                print("Library Size", len(self.tree_pool.trees))
        # self.model_size_archive.update(self.pop)
        self.validation_set_generation()
        gc.collect()
        if isinstance(self.mutation_scheme, MutationOperator):
            self.mutation_scheme.callback(self.pop)
        # callback function after each generation
        if self.verbose:
            primitive_usage = defaultdict(int)
            for p in self.pop:
                for x in p.gene:
                    for s in x:
                        primitive_usage[s.name] += 1
            # print(primitive_usage)

            # if self.current_gen > 0 and self.current_gen % 3 == 0:
            #     self.tree_pool.plot_top_frequencies()
            pop = self.pop
            if self.log_item != "":
                best_ind: MultipleGeneGP = sorted(
                    pop, key=lambda x: x.fitness.wvalues[0]
                )[-1]

                if "LOOCV" in self.log_item:
                    self.loocv_logs.append(best_ind.fitness.wvalues[0])

                if "SharpnessRatio" in self.log_item:
                    self.sharpness_ratio_logs.append(
                        best_ind.fitness.values[1] / best_ind.sam_loss
                    )

                if "SAM" in self.log_item:
                    self.sharpness_logs.append(best_ind.fitness.values[1])

                if "Duel" in self.log_item:
                    if not np.all(
                        np.equal(best_ind.case_values, self.hof[0].case_values)
                    ):
                        p_value = wilcoxon(
                            best_ind.case_values / self.hof[0].case_values,
                            np.ones(len(best_ind.case_values)),
                            alternative="less",
                        )[1]
                        self.duel_logs.append(p_value)
                    else:
                        self.duel_logs.append(0)

            dt = defaultdict(int)
            # parameters = np.zeros(2)
            for p in pop:
                # dt[str(p.dynamic_leaf_size)] += 1
                dt[str(p.dynamic_regularization)] += 1
            print("Information", dt)
            if self.all_nodes_counter > 0:
                print(
                    "Ratio of Introns",
                    self.intron_nodes_counter / self.all_nodes_counter,
                )
            if (
                "dynamic_standardization" in self.param
                and self.param["dynamic_standardization"]
            ):
                # Define a dictionary to store counts
                scaler_counts = defaultdict(int)

                # Iterate through each element in the pop list
                for p in pop:
                    # Get the type of dynamic_standardization attribute
                    scaler_type = type(
                        p.individual_configuration.dynamic_standardization
                    ).__name__
                    # Increment the count for the respective scaler type
                    scaler_counts[scaler_type] += 1

                # Print the counts for each scaler type
                for scaler_type, count in scaler_counts.items():
                    print(f"{scaler_type} count: {count}")
            # print(parameters)

        if self.pac_bayesian.automatic_std and (
            self.current_gen % 20 == 0 and self.current_gen != 0
        ):
            automatic_perturbation_std(self, self.pop)

    def update_instance_weights(self):
        if isinstance(
            self.evaluation_configuration.sample_weight, str
        ) and self.evaluation_configuration.sample_weight.startswith("Adaptive"):
            for o in self.pop:
                # top 50% of samples are assigned with weight 1, the rest are assigned with weight 0
                weights = np.full_like(
                    o.case_values, self.evaluation_configuration.minor_sample_weight
                )
                weights[np.argsort(o.case_values)[: len(o.case_values) // 2]] = 1
                o.individual_configuration.sample_weight = weights

    def validation_set_generation(self):
        if self.archive_configuration.dynamic_validation:
            X, y = np.concatenate([self.X, self.valid_x], axis=0), np.concatenate(
                [self.y, self.valid_y], axis=0
            )
            self.X, self.valid_x, self.y, self.valid_y = train_test_split(
                X, y, test_size=self.validation_size
            )

    def multi_gene_mutation(self):
        return (
            self.mutation_scheme in multi_gene_operators
            or self.mutation_scheme in eda_operators
            or isinstance(self.mutation_scheme, MutationOperator)
        )

    def safe_initialization_check(self):
        attributes = self.__dict__.items()
        lists_and_sets = [
            (key, attr) for (key, attr) in attributes if isinstance(attr, (list, set))
        ]
        # Iterate through all lists and sets to check for numbers.
        for key, lst_or_set in lists_and_sets:
            for item in lst_or_set:
                if isinstance(item, (int, float)):
                    raise Exception(
                        f"Safe initialization is not possible with numbers in lists or sets "
                        f"{key, lst_or_set}."
                    )

        # Return True if no numbers are found.
        return True

    def eaSimple(
        self,
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        verbose=__debug__,
    ):
        """
        This is the main function of the genetic programming algorithm.
        :param population:  The list of GP individuals.
        :param toolbox: The toolbox includes all genetic operators.
        :param cxpb: The probability of crossover.
        :param mutpb: The probability of mutation.
        :param ngen: The number of generations.
        :param stats:
        :param halloffame: An archive of the best individuals.
        :param verbose:
        :return:
        """

        target = self.y
        if self.verbose:
            print("data shape", self.X.shape, target.shape)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        if self.reduction_ratio >= 1:
            self.population_reduction(population)

        # Evaluate all individuals
        evaluated_inds = self.population_evaluation(toolbox, population)

        if self.pac_bayesian.automatic_std:
            automatic_perturbation_std(self, population)

        self.post_processing_after_evaluation(None, population)

        self.append_evaluated_features(population)
        for o in population:
            self.evaluated_pop.add(individual_to_tuple(o))

        if halloffame is not None:
            halloffame.update(population)

        if self.lasso_hof is not None:
            self.lasso_hof.update(population)

        if self.diversity_search != "None":
            self.diversity_assignment(population)

        if isinstance(self.mutation_scheme, str) and (
            "global" in self.mutation_scheme or "extreme" in self.mutation_scheme
        ):
            self.construct_global_feature_pool(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(evaluated_inds), **record)
        if verbose:
            print(logbook.stream)

        if isinstance(self.environmental_selection, EnvironmentalSelection):
            self.environmental_selection.select(population, [])

        self.redundant_features_calculation(population)
        self.statistical_result_update(population, verbose)
        self.callback()

        # MAP-ELites Initialization
        elite_map = {}
        pop_pool = []
        elite_map, pop_pool = self.map_elite_generation(population, elite_map, pop_pool)

        fitness_improvement = 0
        # iterations of no fitness improvement on the training set
        # used for early stopping
        no_improvement_iteration = 0
        adaptive_hoist_probability = None
        historical_best_fitness = self.get_current_best_fitness()

        # archive initialization
        if self.select == "Auto-MCTS":
            self.aos: MCTS
            self.aos.archive_initialization(population)
        # self.update_semantic_repair_input_matrix(population)
        elites_archive = self.update_external_archive(population, None)

        # Begin the generational process
        number_of_evaluations = self.n_pop
        total_evaluations = (self.n_gen + 1) * self.n_pop
        gen = 0
        # fix at the start of evolution
        pop_size = self.n_pop
        discarded_individuals = 0

        while number_of_evaluations < total_evaluations:
            gen += 1
            self.current_gen = gen
            self.evaluation_configuration.current_generation = self.current_gen
            if (
                self.dynamic_reduction > 0
                and (gen > 1)
                and (
                    (number_of_evaluations - self.n_pop)
                    % ((total_evaluations - self.n_pop) // self.dynamic_reduction)
                    == 0
                )
            ):
                pop_size //= 2
                assert self.pre_selection == None
            pop_size = min(pop_size, total_evaluations - number_of_evaluations)
            number_of_evaluations += pop_size

            self.repetitive_feature_count.append(0)
            self.entropy_calculation()
            if isinstance(self.mutation_scheme, str) and (
                self.mutation_scheme in eda_operators or "EDA" in self.mutation_scheme
            ):
                if self.mutation_scheme == "EDA-Terminal-PM-Frequency":
                    self.estimation_of_distribution.frequency_counting(
                        importance_weight=False
                    )
                else:
                    self.estimation_of_distribution.frequency_counting()
                self.estimation_of_distribution.probability_sampling()
            if (
                self.external_archive == "HallOfFame"
                and (self.hof is not None)
                and self.score_func == "NoveltySearch"
            ):
                # recalculate the diversity metric for a fair comparison
                ensemble_value = np.mean([x.predicted_values for x in self.hof], axis=0)
                for x in self.hof:
                    ambiguity = (x.predicted_values - ensemble_value) ** 2
                    x.case_values[len(x.predicted_values) :] = -1 * ambiguity

            cxpb, mutpb = self.linear_adaptive_rate(gen, cxpb, mutpb)
            cxpb, mutpb = self.get_adaptive_mutation_rate(cxpb, mutpb)

            start_time = time.time()
            count = 0
            new_offspring = []

            population = Clearing(**self.param).do(population)
            if self.dynamic_target:
                best_ind = toolbox.clone(
                    population[np.argmax([x.fitness.wvalues[0] for x in population])]
                )
                del best_ind.fitness.values
                new_offspring.append(best_ind)

            self.hoist_one_layer(population)

            adaptive_hoist_probability = self.hoist_mutation(
                population, adaptive_hoist_probability, no_improvement_iteration
            )

            # determine the number of individuals to generate
            individuals_to_generate = pop_size
            if self.pre_selection != None:
                individuals_to_generate *= self.surrogate_model.brood_generation_ratio

            if self.norevisit_strategy == "Crossover+Mutation":
                offspring = []
                while len(offspring) < pop_size:
                    a, b = self.select_pair_of_parents(
                        population,
                        toolbox,
                        elite_map,
                        elites_archive,
                        fitness_improvement,
                    )
                    offspring.append(a)
                    if len(offspring) < pop_size:
                        offspring.append(b)
            else:
                offspring = None

            # offspring generation
            while len(new_offspring) < individuals_to_generate:
                if count > pop_size * 100:
                    raise Exception("Error!")
                count += 1
                if self.norevisit_strategy == "Crossover+Mutation":
                    offspring = offspring
                else:
                    offspring = self.select_pair_of_parents(
                        population,
                        toolbox,
                        elite_map,
                        elites_archive,
                        fitness_improvement,
                    )

                    offspring: List[MultipleGeneGP] = offspring[:]

                if not self.bloat_control_configuration.hoist_before_selection:
                    shm = SHM(self, **self.bloat_control)
                    # deep copy and then hoist
                    offspring = pickle_deepcopy(offspring)
                    shm.semantic_hoist_mutation(offspring, adaptive_hoist_probability)

                offspring = self.semantic_prune_and_plant(offspring)

                # Vary the pool of individuals
                if self.multi_gene_mutation():
                    limitation_check = self.static_limit_function
                    if (
                        self.bloat_control is not None
                        and random.random()
                        < self.bloat_control.get("tree_combination", 0)
                    ):
                        offspring = individual_combination(
                            offspring, toolbox, self.pset, limitation_check
                        )
                    else:
                        offspring: List[MultipleGeneGP]
                        crossover_configuration = self.crossover_configuration
                        if (
                            random.random()
                            < crossover_configuration.semantic_crossover_probability
                            and crossover_configuration.semantic_crossover_mode
                            == CrossoverMode.Independent
                        ):
                            offspring, parent = perform_semantic_macro_crossover(
                                offspring, self.crossover_configuration, toolbox, self.y
                            )
                            # mark as macro crossover
                            self.record_parent_fitness(
                                parent, offspring, crossover_type="Macro"
                            )
                        else:
                            if (
                                crossover_configuration.semantic_crossover_mode
                                == CrossoverMode.Independent
                            ):
                                # only mark this in parallel mode
                                for o in offspring:
                                    o.crossover_type = "Micro"
                            # these original individuals will not change,
                            # because var function will copy these individuals internally
                            offspring = varAndPlus(
                                offspring,
                                toolbox,
                                cxpb,
                                mutpb,
                                limitation_check,
                                crossover_configuration=crossover_configuration,
                                mutation_configuration=self.mutation_configuration,
                                algorithm=self,
                            )
                else:
                    offspring: MultipleGeneGP = varAnd(offspring, toolbox, cxpb, mutpb)

                self.torch_variable_clone(offspring)
                # default disabled
                self.fix_very_trivial_trees_mode(offspring)

                # Handle mutation and initialization checks
                offspring = check_redundancy_and_fix(
                    offspring,
                    self.bloat_control,
                    toolbox,
                    self.pset,
                    self.new_tree_generation,
                )

                offspring = self.prune_and_plant(offspring)

                # very strong mutation
                if self.intron_threshold > 0:
                    for o in offspring:
                        for id, gene in enumerate(o.gene):
                            if o.coef[id] < self.intron_threshold:
                                (o.gene[id],) = self.neutral_mutation(gene)

                self.semantic_repair_features(offspring, elites_archive)

                offspring = handle_tpot_base_learner_mutation(
                    offspring, self.base_learner, self.tpot_model
                )

                self.self_adaptive_evolution(offspring)
                offspring = self.post_selection(offspring)

                norevisit_strategy_handler(
                    offspring,
                    toolbox,
                    self.norevisit_strategy,
                    self.evaluated_pop,
                    partial(gene_addition, algorithm=self),
                )

                for o in offspring:
                    self.self_adaptive_mutation(o, population)

                    if len(new_offspring) < individuals_to_generate:
                        # checking redundant individuals
                        if (
                            self.allow_revisit
                            or (not individual_to_tuple(o) in self.evaluated_pop)
                            # for single-tree, may have many chances to repeat,
                            # so less restrictive
                            or (self.gene_num == 1 and count > pop_size * 10)
                        ):
                            # sometime, when gene num is very small, it is hard to generate a unique individual
                            self.evaluated_pop.add(individual_to_tuple(o))
                            new_offspring.append(o)
                        else:
                            discarded_individuals += 1

            if self.verbose:
                print("Discarded Individuals", discarded_individuals)

            if self.pre_selection != None:
                new_offspring = self.surrogate_model.pre_selection_individuals(
                    population, new_offspring, self.n_pop
                )
                assert len(new_offspring) == self.n_pop, f"{len(offspring), self.n_pop}"
            else:
                new_offspring = new_offspring

            assert len(new_offspring) == pop_size, f"{len(new_offspring), pop_size}"
            new_offspring = self.semantic_approximation(new_offspring)
            new_offspring = self.tarpeian(new_offspring)

            self.time_statistics["GP Generation"].append(time.time() - start_time)
            # delete some inherited information
            for ind in new_offspring:
                # delete fitness values
                if ind.fitness.valid:
                    del ind.fitness.values
                for attr in ("predicted_values", "case_values", "pipe", "coef"):
                    if hasattr(ind, attr):
                        delattr(ind, attr)

            offspring = new_offspring

            self.multi_fidelity_evaluation.update_evaluation_mode(self.current_gen)
            if self.reduction_ratio >= 1:
                self.population_reduction(population)

            if self.stage_flag:
                print("Start Evaluation")
            # Evaluate all individuals
            evaluated_inds = self.population_evaluation(toolbox, offspring)

            self.post_processing_after_evaluation(population, offspring)

            if self.verbose:
                p_value = statistical_difference_between_populations(
                    offspring, population
                )
                print("P value of two consecutive population", p_value)

            self.gp_simplification(offspring)

            # Record the fitness improvement
            fitness_improvement = np.max(
                [ind.fitness.wvalues[0] for ind in offspring]
            ) - np.max([ind.fitness.wvalues[0] for ind in population])

            if self.stage_flag:
                print("Stop Evaluation")

            if self.score_func == "NoveltySearch":
                fitness_list = [x.fitness.wvalues for x in offspring]
                q75, q25 = np.percentile(fitness_list, [75, 25])
                iqr = q75 - q25
                median = np.median(fitness_list)
                # Avoid meaningless diversity of individuals
                offspring = list(
                    filter(
                        lambda x: x.fitness.wvalues[0] >= median - 1.5 * iqr, offspring
                    )
                )
                assert len(offspring) > 0, f"{median, iqr}"

            if self.dynamic_target:
                # if dynamic target, then re-evaluate all individuals
                for x in self.hof:
                    del x.fitness.values
                self.population_evaluation(toolbox, self.hof)

            elite_map, pop_pool = self.map_elite_generation(
                offspring, elite_map, pop_pool
            )

            self.adaptive_operator_selection_update(offspring, population)

            # Update the hall of fame with the generated individuals
            if self.stage_flag:
                print("Start HOF updating")
            if self.dynamic_target:
                halloffame.clear()
            if self.lasso_hof is not None:
                self.lasso_hof.update(population)

            if halloffame is not None:
                # update the hall of fame with the generated individuals
                halloffame.update(offspring)

            if self.verbose:
                if self.semantic_repair > 0:
                    print(
                        "Successful Repair",
                        self.successful_repair
                        / (self.successful_repair + self.failed_repair),
                    )
                print(
                    "Average number of features",
                    np.mean([o.gene_num for o in offspring]),
                )
                if len(self.pop_diversity_history) > 0:
                    print("Diversity", self.pop_diversity_history[-1])
                if self.num_of_active_trees == 0:
                    num_of_trees = self.gene_num
                else:
                    num_of_trees = self.num_of_active_trees
                # check_number_of_unique_tree_semantics(offspring, num_of_trees)
                if self.base_learner == "AdaptiveLasso":
                    lasso_parameter = [o.parameters.lasso for o in offspring]
                    print(
                        "Average Lasso",
                        np.mean(lasso_parameter),
                        np.max(lasso_parameter),
                        np.min(lasso_parameter),
                    )
                    non_zero_features = [np.sum(o.coef != 0) for o in offspring]
                    print(
                        "Average Selected Features",
                        np.mean(non_zero_features),
                        np.max(non_zero_features),
                        np.min(non_zero_features),
                    )

            if self.stage_flag:
                print("Stop HOF updating")

            if isinstance(self.mutation_scheme, str) and (
                "global" in self.mutation_scheme or "extreme" in self.mutation_scheme
            ):
                self.construct_global_feature_pool(population)
            self.append_evaluated_features(offspring)

            best_fitness = np.max([ind.fitness.wvalues[0] for ind in offspring])
            if self.param.get("record_training_data", False):
                self.best_fitness_history.append(best_fitness)
            if self.diversity_search != "None":
                self.diversity_assignment(offspring)

            # multi-objective GP based on fitness-size
            if self.select == "Auto-MCTS":
                self.aos.store_in_archive(offspring)

            self.redundant_features_calculation(offspring)
            # Replace the current population by the offspring
            self.survival_selection(gen, population, offspring)
            # self.update_semantic_repair_input_matrix(population)
            elites_archive = self.update_external_archive(population, elites_archive)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(evaluated_inds), **record)

            if self.verbose and self.hof is not None:
                print("Print model:")

                def replace(ss):
                    # MMTGP
                    for s in range(
                        self.gene_num + self.X.shape[1], self.X.shape[1] - 1, -1
                    ):
                        ss = ss.replace(f"ARG{s}", f"INS{s - self.X.shape[1]}")
                    return ss

                best_model = self.hof[0]

                print(
                    "\n".join([replace(str(g)) for g in best_model.gene]),
                )

                # trees = self.feature_generation(self.X, best_model)
                # prediction = best_model.pipe.predict(trees)
                # prediction_error = (prediction - self.y) ** 2
                # cv_error = np.mean(best_model.case_values)
                # print(
                #     "\n".join([replace(str(g)) for g in best_model.gene]),
                #     "Fitness",
                #     cv_error,
                #     "Prediction Error",
                #     np.mean(prediction_error),
                #     "Difference",
                #     np.mean(prediction_error - cv_error),
                # )

            if verbose:
                features = set(
                    chain.from_iterable(
                        list(map(lambda x: [str(y) for y in x.gene], population))
                    )
                )
                print("number of features", len(features))
                print("archive size", len(self.hof))
                print(logbook.stream)
                if self.base_model_list != None:
                    model_dict = defaultdict(int)
                    for p in population:
                        model_dict[p.base_model] += 1
                    print("Population", model_dict)
                    model_dict = defaultdict(int)
                    for p in self.hof:
                        model_dict[p.base_model] += 1
                    print("Hall of fame", model_dict)

            # statistical information for adaptive GP
            current_best_fitness = self.get_current_best_fitness()

            # check iterations of non-fitness improvement
            if historical_best_fitness < current_best_fitness:
                historical_best_fitness = current_best_fitness
                no_improvement_iteration = 0
            else:
                no_improvement_iteration += 1
            if self.early_stop > 0:
                if no_improvement_iteration > self.early_stop:
                    break

            self.statistical_result_update(population, verbose)
            self.callback()
        if self.early_stop > 0:
            required_evaluations = (self.current_gen + 1) * self.n_pop
            assert number_of_evaluations == required_evaluations
        else:
            assert number_of_evaluations == total_evaluations

        if verbose:
            if self.early_stop > 0:
                print("Final Generation", self.current_gen)
            print("Final Ensemble Size", len(self.hof))

        if self.n_process > 1:
            self.pool.close()
        if self.force_sr_tree:
            self.base_learner = self.base_learner.replace("Fast-", "")
            for g in self.hof:
                # reset base learners
                g.pipe = self.get_base_model()
        else:
            # change base learner, but not change all existing models
            if isinstance(self.base_learner, str):
                self.base_learner = self.base_learner.replace("Fast-", "")
        if isinstance(self.base_learner, str):
            assert not self.base_learner.startswith("Fast-")
        self.post_prune(self.hof)
        return population, logbook

    def adaptive_operator_selection_update(self, offspring, population):
        # if self.select in ["Auto", "Auto-MCTS"]:
        #     self.aos.update(population, offspring)
        if self.aos is not None:
            self.aos.update(population, offspring)

    def fix_very_trivial_trees_mode(self, offspring):
        """
        Address and rectify trivial trees (specifically zero trees) in the provided offspring list.

        A 'zero tree' is considered trivial and is defined as a tree that contains only a single node
        of type Terminal with a value of 0. If the bloat control configuration has the "fix_zero_tree"
        parameter set to True, this method will regenerate such trees using the `new_tree_generation` method.
        """

        # try to fix zero tree
        fix_zero_tree = self.bloat_control is not None and self.bloat_control.get(
            "fix_zero_tree", False
        )
        if fix_zero_tree:
            for o in offspring:
                for gid, g in enumerate(o.gene):
                    if len(g) == 1 and isinstance(g[0], Terminal) and g[0].value == 0:
                        # It is a zero tree
                        o.gene[gid] = self.new_tree_generation()

    def select_pair_of_parents(
        self, population, toolbox, elite_map, elites_archive, fitness_improvement
    ):
        # Using the external archive
        if self.external_archive == "HallOfFame" and self.hof is not None:
            parent = population + list(self.hof)
        else:
            parent = population
        if self.select == "Auto" or self.select == "Auto-MCTS":
            offspring = self.aos.select(parent)
        elif (self.select in map_elite_series) and (
            (self.map_elite_parameter["trigger_time"] == "All")
            or (
                self.map_elite_parameter["trigger_time"] == "Improvement"
                and fitness_improvement >= 0
            )
        ):
            if isinstance(elite_map, list):
                parent = elite_map
            elif isinstance(elite_map, dict):
                parent = list(elite_map.values())
            else:
                raise TypeError

            offspring = self.custom_selection(parent, self.select, elite_map)
        else:
            if (
                self.crossover_configuration.semantic_crossover_mode
                == CrossoverMode.Sequential
                and random.random()
                < self.crossover_configuration.semantic_crossover_probability
            ):
                # select two parents using macro-crossover operator
                # then, apply traditional operators on these selected parents
                parents_a, pa = self.semantic_crossover_for_parent(
                    toolbox, parent, elites_archive, self.crossover_configuration
                )
                parents_b, pb = self.semantic_crossover_for_parent(
                    toolbox, parent, elites_archive, self.crossover_configuration
                )
                offspring = [pa, pb]
                # No matter whether apply macro-crossover or not, always mark it as macro-crossover
                self.record_parent_fitness(parents_a, [pa], crossover_type="Macro")
                self.record_parent_fitness(parents_b, [pb], crossover_type="Macro")
            else:
                offspring = self.traditional_parent_selection(
                    toolbox, parent, elites_archive
                )
                if (
                    self.crossover_configuration.semantic_crossover_mode
                    == CrossoverMode.Sequential
                ):
                    # If this is a sequential model, mark as micro-crossover
                    for o in offspring:
                        o.crossover_type = "Micro"
        return offspring

    def hoist_mutation(
        self, population, adaptive_hoist_probability, no_improvement_iteration
    ):
        if self.bloat_control is not None and self.bloat_control.get(
            "hoist_mutation", False
        ):
            if self.bloat_control.get("hash_simplification", False):
                hash_based_simplification(population, population)

            if self.bloat_control.get("hof_simplification", False):
                # Maybe useless when using hoist mutation.
                # Hoist mutation is almost able to hoist the most important part.
                hash_based_simplification(population, self.hof)

            adaptive_hoist_probability = self.tune_hoist_probability(
                adaptive_hoist_probability, no_improvement_iteration
            )

            if self.bloat_control_configuration.hoist_before_selection:
                shm = SHM(self, **self.bloat_control)
                shm.semantic_hoist_mutation(population, adaptive_hoist_probability)
        return adaptive_hoist_probability

    def hoist_one_layer(self, population):
        if self.bloat_control is not None and self.bloat_control.get(
            "hoist_one_layer", False
        ):
            for o in population:
                for gene in o.gene:
                    number_of_layer = self.bloat_control.get("number_of_layer", 1)
                    levle_gene = list(
                        filter(lambda g: g.level <= number_of_layer, gene)
                    )
                    best_id = max(
                        [(k, getattr(g, "corr", 0)) for k, g in enumerate(levle_gene)],
                        key=lambda x: (x[1], x[0]),
                    )[0]
                    hoistMutation(gene, best_id)

    def get_current_best_fitness(self):
        if self.validation_size > 0:
            current_best_fitness = self.hof[0].validation_score
        elif isinstance(self.score_func, R2PACBayesian):
            current_best_fitness = np.max([-1 * ind.sam_loss for ind in self.hof])
        else:
            current_best_fitness = np.max([ind.fitness.wvalues[0] for ind in self.hof])
        return current_best_fitness

    def semantic_approximation(self, new_offspring):
        if self.bloat_control is not None and self.bloat_control.get(
            "subtree_approximation", False
        ):
            static_limit_function = staticLimit_multiple_gene(
                key=operator.attrgetter("height"),
                max_value=self.depth_limit_configuration.max_height,
                min_value=self.depth_limit_configuration.min_height,
                random_fix=False,
            )
            dsa = DSA(self)
            subtree_approximation = static_limit_function(
                dsa.subtree_semantic_approximation
            )
            new_offspring = subtree_approximation(*new_offspring)
        return new_offspring

    def tune_hoist_probability(
        self, adaptive_hoist_probability, no_improvement_iteration
    ):
        if self.bloat_control.get("hoist_probability") == "adaptive":
            if adaptive_hoist_probability is None:
                adaptive_hoist_probability = 1
            if no_improvement_iteration > 0:
                adaptive_hoist_probability -= 0.1
            else:
                adaptive_hoist_probability += 0.1
            adaptive_hoist_probability = np.clip(adaptive_hoist_probability, 0, 1)
            print(adaptive_hoist_probability)
        return adaptive_hoist_probability

    def gp_simplification(self, offspring):
        # prune after evaluation
        simplification = Simplification(self, **self.bloat_control)
        for o in offspring:
            for gene in o.gene:
                simplification.gene_prune(gene)

    def tarpeian(self, new_offspring):
        # Tarpeian
        if self.bloat_control and self.bloat_control.get("tarpeian", False):
            # If lexicase selection, individuals can be directly deleted
            assert "Lexicase" in self.select
            tpe = Tarpeian(self, **self.bloat_control)
            new_offspring = tpe.tarpeian(new_offspring)
        return new_offspring

    def prune_and_plant(self, offspring):
        if self.bloat_control is not None and self.bloat_control.get(
            "prune_and_plant", False
        ):
            # Traditional Prune-and-Plant (After Mutation)
            if random.random() < self.bloat_control.get("prune_and_plant_pb", 1):
                pap = PAP(self)
                offspring = pap.prune_and_plant(offspring, best=False)
        return offspring

    def semantic_prune_and_plant(self, offspring):
        if self.bloat_control is not None and self.bloat_control.get(
            "semantic_prune_and_plant", False
        ):
            # Semantic Prune-and-Plant (Before Mutation)
            if random.random() < self.bloat_control.get("prune_and_plant_pb", 1):
                pap = PAP(self)
                offspring = pap.prune_and_plant(offspring, best=True)
        return offspring

    def torch_variable_clone(self, offspring):
        if self.evaluation_configuration.gradient_descent:
            # clone all variables to non-tensor
            for ind in offspring:
                for tree in ind.gene:
                    for id, f in enumerate(tree):
                        if isinstance(f, Terminal) and isinstance(
                            f.value, torch.Tensor
                        ):
                            node = torch.tensor(
                                [f.value.item()], dtype=torch.float32
                            ).requires_grad_(True)
                            tree[id] = Terminal(node, False, f.ret)

    def post_processing_after_evaluation(self, parent, population):
        # re-assign fitness for all individuals if using PAC-Bayesian
        if isinstance(self.score_func, Fitness):
            self.score_func.post_processing(
                parent, population, self.hof, self.elites_archive
            )

        if self.pac_bayesian.adaptive_depth:
            self.depth_limit_configuration.max_height = (
                sharpness_based_dynamic_depth_limit(
                    population, self.depth_limit_configuration.max_height
                )
            )

        if self.lamarck_constant:
            for ind in population:
                ind: MultipleGeneGP
                coef = ind.pipe["Ridge"].coef_
                ind.pipe["Scaler"].mean_ = ind.pipe["Scaler"].mean_ * coef
                ind.gene = lamarck_constant(ind.gene, self.pset, coef)
                ind.pipe["Ridge"].coef_ = np.ones_like(coef)

    def semantic_crossover_for_parent(
        self,
        toolbox,
        parent,
        external_archive,
        crossover_configuration: CrossoverConfiguration,
    ):
        available_parent = []
        parents = []
        while len(available_parent) == 0:
            parents = self.traditional_parent_selection(
                toolbox, parent, external_archive
            )
            # copy individuals before any modifications
            offspring = [efficient_deepcopy(parents[0]), efficient_deepcopy(parents[1])]
            target = self.y
            if (
                crossover_configuration.semantic_selection_mode
                == SelectionMode.MAPElites
            ):
                available_parent = mapElitesCrossover(
                    offspring[0],
                    offspring[1],
                    target,
                    crossover_configuration.map_elites_configuration,
                )
            elif (
                crossover_configuration.semantic_selection_mode
                == SelectionMode.AngleDriven
            ):
                available_parent = semanticFeatureCrossover(
                    offspring[0], offspring[1], target
                )
            elif crossover_configuration.semantic_selection_mode == SelectionMode.ResXO:
                coef = offspring[0]["Ridge"].coef
                _, d, best_feature_idx = resxo(
                    offspring[0].semantics, offspring[1].semantics, coef, self.y
                )
                offspring[0].gene[d] = offspring[1].gene[best_feature_idx]
                available_parent = [offspring[0]]
            elif (
                crossover_configuration.semantic_selection_mode == SelectionMode.StageXO
            ):
                _, selected_indices_p1, selected_indices_p2 = stagexo(
                    offspring[0].semantics, offspring[1].semantics, self.y
                )
                offspring[0].gene = [
                    offspring[0].gene[idx] for idx in selected_indices_p1
                ], [offspring[1].gene[idx] for idx in selected_indices_p2]
                available_parent = [offspring[0]]
            else:
                raise Exception(
                    "Unsupported semantic selection mode:",
                    crossover_configuration.semantic_selection_mode,
                )

        return parents, available_parent[0]

    def record_parent_fitness(self, parents, offspring, crossover_type):
        if self.crossover_configuration.semantic_crossover_mode is not None:
            assert len(parents) <= 2
            # record parent fitness for both sequential mode and parallel mode
            parent_fitness = [parent.fitness.wvalues[0] for parent in parents]
            for o in offspring:
                o.parent_fitness = parent_fitness
                o.crossover_type = crossover_type

    def get_validation_score(self, best_individual, force_training=False):
        """
        Evaluate the performance on validation set
        """
        # if not trained, train the model
        self.final_model_lazy_training([best_individual], force_training=force_training)
        parent_input = self.feature_generation(self.valid_x, best_individual)
        # then, evaluate the performance
        score = r2_score(self.valid_y, best_individual.pipe.predict(parent_input))
        return score

    def linear_adaptive_rate(self, gen, cxpb, mutpb):
        if self.cross_pb == "Linear":
            cxpb = np.interp(np.arange(0, self.n_gen), [0, self.n_gen - 1], [0.9, 0.5])[
                gen - 1
            ]
        if self.mutation_pb == "Linear":
            mutpb = np.interp(
                np.arange(0, self.n_gen), [0, self.n_gen - 1], [0.1, 0.5]
            )[gen - 1]
        return cxpb, mutpb

    def get_adaptive_pb_array(self, probability, inverse=False):
        probability = float(
            probability.replace("InverseAdaptive-", "")
            .replace("InverseDynamicAdaptive-", "")
            .replace("DynamicAdaptive-", "")
            .replace("Adaptive-", "")
        )
        unique_hash = [
            len(np.unique([o.hash_result[i] for o in self.pop]))
            for i in range(self.gene_num)
        ]
        probability = cross_scale(
            unique_hash,
            probability,
            inverse=inverse,
            power=self.crossover_configuration.adaptive_power,
        )
        return probability

    def get_adaptive_mutation_rate(self, cxpb, mutpb):
        if self.mgp_mode:
            if isinstance(self.cross_pb, str) and self.cross_pb.startswith("Adaptive"):
                cxpb = self.get_adaptive_pb_array(self.cross_pb)
            if isinstance(self.mutation_pb, str) and self.mutation_pb.startswith(
                "Adaptive"
            ):
                mutpb = self.get_adaptive_pb_array(self.mutation_pb)

            if isinstance(self.cross_pb, str) and self.cross_pb.startswith(
                "InverseAdaptive"
            ):
                cxpb = self.get_adaptive_pb_array(self.cross_pb, inverse=True)
            if isinstance(self.mutation_pb, str) and self.mutation_pb.startswith(
                "InverseAdaptive"
            ):
                mutpb = self.get_adaptive_pb_array(self.mutation_pb, inverse=True)

            if isinstance(self.cross_pb, str) and self.cross_pb.startswith(
                "DynamicAdaptive"
            ):
                if (
                    self.current_gen
                    > self.n_gen * self.crossover_configuration.inverse_point
                ):
                    cxpb = self.get_adaptive_pb_array(self.cross_pb, inverse=True)
                else:
                    cxpb = self.get_adaptive_pb_array(self.cross_pb)
            if isinstance(self.mutation_pb, str) and self.mutation_pb.startswith(
                "DynamicAdaptive"
            ):
                if (
                    self.current_gen
                    > self.n_gen * self.crossover_configuration.inverse_point
                ):
                    mutpb = self.get_adaptive_pb_array(self.mutation_pb, inverse=True)
                else:
                    mutpb = self.get_adaptive_pb_array(self.mutation_pb)

            if isinstance(self.cross_pb, str) and self.cross_pb.startswith(
                "InverseDynamicAdaptive"
            ):
                if (
                    self.current_gen
                    > self.n_gen * self.crossover_configuration.inverse_point
                ):
                    cxpb = self.get_adaptive_pb_array(self.cross_pb)
                else:
                    cxpb = self.get_adaptive_pb_array(self.cross_pb, inverse=True)
            if isinstance(self.mutation_pb, str) and self.mutation_pb.startswith(
                "InverseDynamicAdaptive"
            ):
                if (
                    self.current_gen
                    > self.n_gen * self.crossover_configuration.inverse_point
                ):
                    mutpb = self.get_adaptive_pb_array(self.mutation_pb)
                else:
                    mutpb = self.get_adaptive_pb_array(self.mutation_pb, inverse=True)

            if self.param.get("record_training_data", False) and isinstance(
                cxpb, (list, np.ndarray)
            ):
                # if probability is a list or an array
                self.adaptive_probability_history[0].append(cxpb[0])
                self.adaptive_probability_history[1].append(cxpb[-1])
        else:
            if isinstance(self.cross_pb, str):
                cxpb = float(
                    self.cross_pb.replace("InverseAdaptive-", "").replace(
                        "Adaptive-", ""
                    )
                )
            if isinstance(self.mutation_pb, str):
                mutpb = float(
                    self.mutation_pb.replace("InverseAdaptive-", "").replace(
                        "Adaptive-", ""
                    )
                )
        return cxpb, mutpb

    def post_prune(self, hof: List[MultipleGeneGP]):
        if self.redundant_hof_size > 0:
            # select the smallest one from redundant individuals
            best_ind = min(self.hof, key=lambda x: len(x))
            self.hof.clear()
            self.hof.insert(best_ind)

        if self.bloat_control is not None and self.bloat_control.get(
            "lasso_prune", False
        ):
            lasso = LassoCV()
            best_ind = None
            best_score = 0
            for ind in self.lasso_hof:
                all_genes = []
                Yp = multi_tree_evaluation(ind.gene, self.pset, self.X)
                lasso.fit(Yp, self.y)
                min_score = np.min(np.mean(lasso.mse_path_, axis=1))
                for g, ok in zip(ind.gene, np.abs(lasso.coef_) > 0):
                    if ok:
                        all_genes.append(g)
                ind.gene = all_genes
                ind.pipe = self.get_base_model()
                if best_ind is None or min_score < best_score:
                    best_ind = ind
                    best_score = min_score
            self.hof.clear()
            self.hof.insert(best_ind)

        if self.post_prune_threshold > 0 or (
            self.bloat_control is not None
            and self.bloat_control.get("post_prune_threshold", 0) > 0
        ):
            if self.post_prune_threshold > 0:
                post_prune_threshold = self.post_prune_threshold
            else:
                post_prune_threshold = self.bloat_control.get("post_prune_threshold", 0)
            co_linearity_deletion = self.bloat_control.get(
                "co_linearity_deletion", True
            )
            trivial_feature_deletion = self.bloat_control.get(
                "trivial_feature_deletion", True
            )
            for ind in hof:
                # prune the individual based on multi-co-linearity and zero correlation
                all_genes = []
                previous_results = []
                for gene in list(sorted(ind.gene, key=len)):
                    quick_result = single_tree_evaluation(gene, self.pset, self.X)[0]
                    y = quick_fill([quick_result], self.X)[0]
                    if (
                        trivial_feature_deletion
                        and np.abs(pearsonr(y, self.y)[0]) < post_prune_threshold
                    ):
                        continue
                    for y_p in previous_results:
                        if (
                            co_linearity_deletion
                            and np.abs(pearsonr(y_p, y)[0]) > 1 - post_prune_threshold
                        ):
                            break
                    else:
                        all_genes.append(gene)
                        previous_results.append(y)
                ind.gene = all_genes
                # re-initialize the final pipeline
                ind.pipe = self.get_base_model()

    """
    Another Idea:
    If cannot hoist, then mark it as intron.
    """

    def get_params(self, deep=True):
        out = super().get_params(deep)
        for k, v in self.param.items():
            out[k] = v
        return out

    def new_tree_generation(self):
        # randomly generate a new tree
        # Return: a new gene
        tree_size = None
        if self.bloat_control is not None:
            tree_size = self.bloat_control.get("reinitialization_tree_size", None)

        if tree_size is None:
            a, b = self.initial_tree_size.split("-")
        else:
            a, b = tree_size.split("-")
        a, b = int(a), int(b)
        # generate trees without correlation information
        if self.mutation_scheme in eda_operators:
            new_tree = genFull_with_prob(self.pset, a, b, self)[:]
        else:
            new_tree = genFull(self.pset, a, b)
        new_tree = PrimitiveTree(new_tree)
        return new_tree

    def thread_pool_initialization(self):
        arg = (self.X, self.y, self.score_func, self.cv, self.evaluation_configuration)
        # pool initialization
        if self.n_process > 1:
            self.pool = Pool(
                self.n_process, initializer=init_worker, initargs=(calculate_score, arg)
            )
        else:
            init_worker(calculate_score, arg)

    def semantic_repair_features(self, offspring, external_archive):
        # semantic repair means to replace one gene with another better gene
        cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if self.semantic_repair > 0:
            for o in offspring:
                data = self.semantic_repair_input
                for id, gene in enumerate(o.gene):
                    # evaluate the current gene
                    final_result = single_tree_evaluation(
                        gene, self.pset.pset_list[id], data
                    )
                    final_result = quick_fill([final_result], data)[0]
                    current_sim = np.abs(
                        cos_sim(final_result, self.semantic_repair_target)
                    )
                    if current_sim < self.semantic_repair:
                        # try to remove the current gene
                        new_genes_a = sorted(
                            selRandom(external_archive[id], 7), key=lambda x: x[0]
                        )[-1]
                        new_genes_b = sorted(
                            selRandom(external_archive[id], 7), key=lambda x: x[0]
                        )[-1]
                        decorator = staticLimit(
                            key=operator.attrgetter("height"),
                            max_value=self.depth_limit_configuration.max_height,
                            min_value=self.depth_limit_configuration.min_height,
                        )(cxOnePoint)
                        test_genes = decorator(
                            copy.deepcopy(new_genes_a[1]), copy.deepcopy(new_genes_b[1])
                        )

                        best_gene_score = 0
                        for new_gene in test_genes:
                            trial_result = single_tree_evaluation(
                                new_gene, self.pset.pset_list[id], data
                            )
                            trial_result = quick_fill([trial_result], data)[0]
                            trail_sim = np.abs(
                                cos_sim(trial_result, self.semantic_repair_target)
                            )
                            replace_threshold = trail_sim / current_sim

                            if replace_threshold > 10 and trail_sim > best_gene_score:
                                best_gene_score = trail_sim
                                final_result = trial_result
                                o.gene[id] = new_gene
                        if best_gene_score > 0:
                            self.successful_repair += 1
                        else:
                            self.failed_repair += 1
                    data = np.concatenate(
                        [data, np.reshape(final_result, (-1, 1))], axis=1
                    )

    def update_semantic_repair_input_matrix(self, population):
        best_ind = sorted(population, key=lambda x: x.fitness.wvalues)[-1]
        id = np.argsort(best_ind.case_values)[-20:]
        self.semantic_repair_input = self.X[id]
        self.semantic_repair_target = self.y[id]

    def statistical_result_update(self, population, verbose):
        if self.meta_learner is not None:
            self.construct_meta_learner()

        if self.gene_num == 1 and self.crossover_configuration.var_or:
            all_crossover = 0
            successful_crossover = 0
            for p in population:
                if p.parent_fitness is not None and len(p.parent_fitness) == 2:
                    if p.fitness.wvalues[0] > p.parent_fitness[0]:
                        successful_crossover += 1
                    all_crossover += 1
            if all_crossover > 0:
                # ignore the first generation
                self.crossover_successful_rate.append(
                    successful_crossover / all_crossover
                )

            all_mutation = 0
            successful_mutation = 0
            for p in population:
                # sometimes, an offspring is generated by the mutation operator
                if p.parent_fitness is not None and len(p.parent_fitness) == 1:
                    if p.fitness.wvalues[0] > p.parent_fitness[0]:
                        successful_mutation += 1
                    all_mutation += 1
            if all_mutation > 0:
                # ignore the first generation
                self.mutation_successful_rate.append(successful_mutation / all_mutation)

        if (
            self.crossover_configuration.semantic_crossover_mode is not None
            and "SuccessRate" in self.log_item
        ):
            # if an individual is better than either of its parent, it is considered as a successful crossover
            successful_crossover = 0
            all_crossover = 0
            for p in population:
                if p.crossover_type == "Macro":
                    if np.all(p.fitness.wvalues[0] > np.array(p.parent_fitness)):
                        successful_crossover += 1
                    all_crossover += 1
            # Generally speaking, it is impossible to ==0, unless all of them do not use Macro Crossover
            if all_crossover > 0:
                self.macro_crossover_successful_rate.append(
                    successful_crossover / all_crossover
                )
            else:
                self.macro_crossover_successful_rate.append(0)

            successful_crossover = 0
            all_crossover = 0
            for p in population:
                if p.crossover_type == "Micro":
                    if np.all(p.fitness.wvalues[0] > np.array(p.parent_fitness)):
                        successful_crossover += 1
                    all_crossover += 1
            if all_crossover > 0:
                self.micro_crossover_successful_rate.append(
                    successful_crossover / all_crossover
                )
            else:
                self.micro_crossover_successful_rate.append(0)

        if self.check_alpha_dominance_nsga2():
            self.alpha_dominance.update_best(population)

        if self.test_fun != None:
            self.training_with_validation_set()
            self.stacking_strategy.stacking_layer_generation(self.X, self.y)

            if len(self.test_fun) > 0:
                training_loss = self.test_fun[0].predict_loss()
                self.train_data_history.append(training_loss)
                if verbose:
                    print("Training Loss", training_loss)
            if len(self.test_fun) > 1:
                testing_loss = self.test_fun[1].predict_loss()
                self.test_data_history.append(testing_loss)
                if verbose:
                    print("Test Loss", testing_loss)

            if "PopulationAverageDiversity" in self.log_item:
                self.pop_diversity_history.append(
                    self.euclidian_diversity_calculation(population)
                )
            if not isinstance(self, ClassifierMixin):
                if "PopulationAverageCosineDistance" in self.log_item:
                    self.pop_cos_distance_history.append(
                        self.cos_distance_calculation(population)
                    )
                if len(self.hof) > 0:
                    if "Ambiguity" in self.log_item or "AverageLoss" in self.log_item:
                        diversity = TestDiversity(self.test_fun[1], self)
                        average_loss, ambiguity = diversity.calculate_diversity(
                            self.hof
                        )
                        self.ambiguity_history.append(ambiguity)
                        self.average_loss_history.append(average_loss)
                    if "ArchiveAverageCosineDistance" in self.log_item:
                        self.archive_cos_distance_history.append(
                            self.cos_distance_calculation(self.hof)
                        )
            if (
                "GenotypicDiversity" in self.log_item
                or "PhenotypicDiversity" in self.log_item
            ):
                (
                    genotype_sum_entropy,
                    phenotype_sum_entropy,
                ) = self.gp_tree_entropy_calculation(population)
                if "GenotypicDiversity" in self.log_item:
                    self.tree_genotypic_diversity.append(genotype_sum_entropy)
                if "PhenotypicDiversity" in self.log_item:
                    self.tree_phenotypic_diversity.append(phenotype_sum_entropy)
            self.avg_tree_size_history.append(
                np.mean(
                    [np.mean([get_tree_size(g) for g in p.gene]) for p in population]
                )
            )
            if len(self.hof) > 0:
                if "ArchiveAverageFitness" in self.log_item:
                    # average fitness of archive
                    self.archive_fitness_history.append(
                        np.mean([ind.fitness.wvalues[0] for ind in self.hof])
                    )
                if "ArchiveAverageDiversity" in self.log_item:
                    # average diversity of archive
                    self.archive_diversity_history.append(
                        self.euclidian_diversity_calculation()
                    )

    def training_with_validation_set(self):
        """
        This is also applicable to multi-objective.
        The only difference is in multi-objective case, the best model is selected from the Pareto front.
        """
        if self.validation_size <= 0:
            # In the case of not using a validation set, no need to do anything
            return
        # Train the final model with the validation set if data combination is enabled and validation set is provided
        if self.archive_configuration.data_combination and self.validation_size > 0:
            # Combine the training and validation sets
            X = np.concatenate((self.X, self.valid_x), axis=0)
            y = np.concatenate((self.y, self.valid_y), axis=0)
            # Train the final model using the combined set
            hof = self.hof
            self.final_model_lazy_training(hof, X, y, force_training=True)

    def gp_tree_entropy_calculation(self, population):
        # Calculate the entropy of the genotype and phenotype distributions in the population
        entropy = lambda p: -p * np.log(p)
        genotype = defaultdict(int)
        phenotype = defaultdict(int)
        # Count the frequency of each genotype and phenotype in the population
        for p in population:
            for t, h in zip(p.gene, p.hash_result):
                genotype[str(t)] += 1
                phenotype[h] += 1
        # Calculate the entropy of the phenotype distribution
        phenotype_sum_value = sum(phenotype.values())
        phenotype_sum_entropy = 0
        for k, v in phenotype.items():
            phenotype_sum_entropy += entropy(v / phenotype_sum_value)
        # Calculate the entropy of the genotype distribution
        genotype_sum_value = sum(genotype.values())
        genotype_sum_entropy = 0
        for k, v in genotype.items():
            genotype_sum_entropy += entropy(v / genotype_sum_value)
        return genotype_sum_entropy, phenotype_sum_entropy

    def self_adaptive_evolution(self, offspring: List[MultipleGeneGP]):
        if isinstance(self.cross_pb, str) and self.cross_pb.startswith("Adaptive-"):
            cross_pb = float(self.cross_pb.replace("Adaptive-", ""))
        else:
            cross_pb = self.cross_pb

        if isinstance(self.mutation_pb, str) and self.mutation_pb.startswith(
            "Adaptive-"
        ):
            mutation_pb = float(self.mutation_pb.replace("Adaptive-", ""))
        else:
            mutation_pb = self.mutation_pb

        # Check if the intron gene mode is active
        if self.intron_probability > 0:
            for i in range(0, len(offspring), 2):
                if random.random() < cross_pb:
                    # Get the active genes of two offspring and perform crossover
                    a, b = offspring[i].active_gene, offspring[i + 1].active_gene
                    a, b = cxTwoPoint(a, b)
                    # Update the active genes of the offspring
                    offspring[i].active_gene, offspring[i + 1].active_gene = a, b

            # For each offspring, perform mutation with a certain probability
            for i in range(0, len(offspring)):
                if random.random() < mutation_pb:
                    # Get the active genes of an offspring and perform mutation
                    a = offspring[i].active_gene
                    (a,) = mutFlipBit(a, indpb=1 / self.gene_num)
                    # Update the active genes of the offspring
                    offspring[i].active_gene = a

            # For each offspring, force at least one active gene to be 1
            for i in range(0, len(offspring)):
                if np.sum(offspring[i].active_gene) == 0:
                    index = np.random.randint(0, len(offspring[i].active_gene))
                    offspring[i].active_gene[index] = 1

    def self_adaptive_mutation(self, o, population):
        if self.base_learner == "AdaptiveLasso":
            factor = np.random.uniform(0.5, 1, 1)[0]
            a, b = selRandom(population, 2)
            o.parameters.lasso = o.parameters.lasso + factor * (
                a.parameters.lasso - b.parameters.lasso
            )
            o.parameters.lasso = np.clip(o.parameters.lasso, -5, 0)
        if self.mgp_mode == "Register":
            new_register = np.random.randint(0, self.number_of_register, self.gene_num)
            flip_bit = (
                np.random.rand(self.gene_num) < self.register_mutation_probability
            )
            o.parameters["Register"][flip_bit] = new_register[flip_bit]
        if self.base_learner == "Dynamic-DT":
            """
            Dynamically tune the leaf size of decision tree
            """
            # base learner mutation
            base_learner_mutation_rate = 0.1
            if random.random() < base_learner_mutation_rate:
                o.dynamic_leaf_size = np.clip(
                    o.dynamic_leaf_size + random.choice([-1, 1]), 1, 10
                )
        if self.base_learner == "Dynamic-LogisticRegression":
            """
            Dynamically tune the regularization term of logistic regression
            """
            base_learner_mutation_rate = 0.1
            if random.random() < base_learner_mutation_rate:
                o.dynamic_regularization = np.clip(
                    o.dynamic_regularization * random.choice([0.1, 10]), 1e-4, 1e4
                )

    def check_multi_task_optimization(self):
        return self.base_learner in ["DT-LR", "Balanced-DT-LR", "Balanced-RDT-LR"]

    def traditional_parent_selection(self, toolbox, parent, external_archive):
        # The traditional parent selection process
        if self.check_multi_task_optimization():
            if isinstance(self.external_archive, int) and self.external_archive >= 1:
                parent_pool = parent + external_archive
            else:
                parent_pool = parent

                # multi-base learner selection
            if random.random() < self.rmp_ratio:
                # cross between different models
                # 0.5-> AA BB
                # 0.5-> AB BA
                model_name_a, parent_a = self.sample_model_name(parent_pool)
                offspring_a = toolbox.select(parent_a, 1)
                model_name_b, parent_b = self.sample_model_name(parent_pool)
                offspring_b = toolbox.select(parent_b, 1)
                # plot_pareto_front(parent_a)
                if self.select.startswith("ParetoTournament"):
                    if model_name_a == model_name_b:
                        if len(offspring_a) > 2:
                            offspring_a = pareto_tournament_controller(
                                offspring_a, self.select.split("~")[1]
                            )
                        if len(offspring_b) > 2:
                            offspring_b = pareto_tournament_controller(
                                offspring_b, self.select.split("~")[1]
                            )
                        offspring = offspring_a + offspring_b
                    else:
                        combinations = list(itertools.product(offspring_a, offspring_b))
                        offspring = [
                            item for sublist in combinations for item in sublist
                        ]
                    offspring = offspring[: len(offspring) // 2 * 2]
                else:
                    offspring = [offspring_a[0], offspring_b[0]]
            else:
                model_name, parent_a = self.sample_model_name(parent_pool)
                offspring = toolbox.select(parent_a, 2)
                if len(offspring) > 2:
                    offspring = pareto_tournament_controller(
                        offspring, self.select.split("~")[1]
                    )
        else:
            if self.number_of_parents > 0:
                # multi-parent generation, which is particularly useful for modular multi-tree GP
                offspring = [
                    self.offspring_generation_with_repair(
                        toolbox, parent, self.number_of_parents, external_archive
                    )[0]
                    for _ in range(2)
                ]
            else:
                if (
                    isinstance(self.external_archive, int)
                    and self.external_archive >= 1
                ):
                    if self.select.startswith("ParetoTournament"):
                        offspring = toolbox.select(
                            parent + external_archive, self.n_pop
                        )
                    else:
                        offspring = toolbox.select(parent + external_archive, 2)
                else:
                    offspring = toolbox.select(parent, 2)
        return offspring

    def redundant_features_calculation(self, offspring):
        if self.mgp_mode:
            current_redundant_features = 0
            current_irrelevant_features = 0
            for o in offspring:
                hash_set = set()
                for g in o.hash_result:
                    if g in hash_set:
                        current_redundant_features += 1
                        self.redundant_features += 1
                    else:
                        hash_set.add(g)
                for g in o.coef:
                    if g < self.irrelevant_feature_ratio:
                        current_irrelevant_features += 1
                        self.irrelevant_features += 1
            self.redundant_features_history.append(current_redundant_features)
            self.irrelevant_features_history.append(current_irrelevant_features)

    def update_external_archive(self, population, external_archive):
        if isinstance(self.external_archive, int):
            if self.check_multi_task_optimization():
                # multi-task optimization, need to store
                models = self.base_model_list.split(",")
                if external_archive is not None:
                    candidate = list(external_archive) + list(population)
                else:
                    candidate = list(population)

                external_archive = []
                for model in models:
                    parent = list(filter(lambda x: x.base_model == model, candidate))
                    external_archive.extend(selBest(parent, self.external_archive))
            else:
                # traditional way
                if external_archive is not None:
                    external_archive = selBest(
                        list(external_archive) + list(population), self.external_archive
                    )
                else:
                    external_archive = selBest(list(population), self.external_archive)
        elif (
            self.external_archive is not None
            and self.external_archive != False
            and self.external_archive.startswith("Size-")
        ):
            size = int(self.external_archive.replace("Size-", ""))
            if external_archive is not None:
                external_archive = selBest(
                    list(external_archive) + list(population), size * self.n_pop
                )
            else:
                external_archive = selBest(list(population), size * self.n_pop)
        elif self.external_archive == "ImportantFeatures" or self.semantic_repair > 0:
            external_archive = construct_important_feature_archive(
                population, external_archive
            )
        else:
            external_archive = None
        self.elites_archive = external_archive
        return external_archive

    def offspring_generation_with_repair(
        self, toolbox, parent, count, external_archive=None
    ):
        # Generate offspring based on multiple parents
        if self.external_archive == False or self.external_archive is None:
            offspring = toolbox.select(parent, count)
        elif self.external_archive == "ImportantFeatures":
            offspring = toolbox.select(parent, 1)
        else:
            # reuse external archive to fix an individual
            # don't select external archive independently, because external archive might be very small
            offspring = toolbox.select(parent + external_archive, count)
        # don't modify parent individuals
        offspring = [efficient_deepcopy(o) for o in offspring]
        if self.delete_low_similarity:
            for o in offspring:
                o.correlation_results, o.coef = o.coef, o.correlation_results
        assert abs(np.sum(offspring[0].coef) - 1) < 1e-5

        all_hash = set()
        for g in range(self.gene_num):
            need_to_change = False
            if offspring[0].hash_result[g] in all_hash:
                # if hash value re-appear, then change it with something
                need_to_change = True
            else:
                all_hash.add(offspring[0].hash_result[g])
            # replace negligible features by other features
            start_position = 0
            id = np.argmax([o.coef[g] for o in offspring[start_position:]])
            # must consider the start position, otherwise there will be an error
            id += start_position
            ratio = self.irrelevant_feature_ratio
            if ratio == "Dynamic":
                ratio = 0.1 * offspring[id].coef[g]

            def irrelevant_criterion():
                if offspring[0].coef[g] < ratio or np.isnan(offspring[0].coef[g]):
                    # smaller than threshold
                    return True
                else:
                    # larger than threshold
                    return False

            # offspring[id].coef[g]>offspring[0].coef[g]
            if (self.delete_irrelevant and irrelevant_criterion()) or (
                self.delete_redundant and need_to_change
            ):
                if self.external_archive == "ImportantFeatures":
                    # features rather than individuals in an archive,
                    # thus randomly select some potential useful features
                    candidates = selRandom(external_archive[g], count)
                    offspring[0].gene[g] = candidates[
                        np.argmax([o[0] for o in candidates])
                    ][1]
                else:
                    offspring[0].coef[g] = offspring[id].coef[g]
                    offspring[0].gene[g] = offspring[id].gene[g]
        if self.delete_low_similarity:
            # restore
            for o in offspring:
                o.correlation_results, o.coef = o.coef, o.correlation_results
        return offspring

    def custom_selection(self, parent, selection_operator, elite_map=None):
        if selection_operator.startswith("MAP-Elite") and len(elite_map) > 0:
            parent = list(elite_map.values())

        if selection_operator == "MAP-Elite-Lexicase":
            offspring = selAutomaticEpsilonLexicaseFast(parent, 2)
        elif selection_operator.startswith("Tournament"):
            tournsize = int(selection_operator.split("-")[1])
            offspring = selTournament(parent, 2, tournsize=tournsize)
        elif selection_operator == "AutomaticLexicase":
            offspring = selAutomaticEpsilonLexicaseFast(parent, 2)
        elif selection_operator == "MaxAngleSelection":
            offspring = selMaxAngleSelection(parent, 2, self.y)
        elif selection_operator == "AngleDrivenSelection":
            offspring = selAngleDrivenSelection(parent, 2, self.y)
        elif self.select == "MAP-Elite-Knockout-A":
            offspring = selKnockout(parent, 2, auto_case=True)
        elif self.select == "MAP-Elite-Knockout-S":
            offspring = selKnockout(parent, 2, version="S")
        elif self.select == "MAP-Elite-Knockout-SA":
            offspring = selKnockout(parent, 2, version="S", auto_case=True)
        elif selection_operator == "MAP-Elite-Tournament-3":
            offspring = selTournament(parent, 2, tournsize=3)
        elif selection_operator == "MAP-Elite-Tournament-7":
            offspring = selTournament(parent, 2, tournsize=7)
        elif selection_operator == "MAP-Elite-Random":
            offspring = selRandom(parent, 2)
        elif selection_operator == "MAP-Elite-Roulette":
            parent = list(filter(lambda x: x.fitness.wvalues[0] > 0, parent))
            offspring = selRoulette(parent, 2)
            assert len(offspring) == 2
        else:
            raise Exception
        return offspring

    def map_elite_generation(self, population, elite_map, pop_pool):
        """
        :param population: Store individuals in the current generation
        :param elite_map: Store selected individuals
        :param pop_pool: Store k*pop_size individuals
        :return:
        """
        if self.select in map_elite_series:
            if self.map_elite_parameter["type"] == "Grid":
                elite_map = selMAPElites(
                    population, elite_map, self.map_elite_parameter, self.y
                )
            elif self.map_elite_parameter["type"].startswith("Grid-Synthetic"):
                ratio = float(
                    self.map_elite_parameter["type"].replace("Grid-Synthetic-", "")
                )
                elite_map = selMAPElites(
                    population,
                    elite_map,
                    self.map_elite_parameter,
                    self.y,
                    data_augmentation=ratio,
                )
            elif self.map_elite_parameter["type"] == "Cluster":
                elite_map, pop_pool = selMAPEliteClustering(
                    population, pop_pool, self.map_elite_parameter
                )
            elif self.map_elite_parameter["type"] == "Cluster-Exploitation":
                self.map_elite_parameter["fitness_ratio"] = np.clip(0, 0.05, 0.95)
                elite_map, pop_pool = selMAPEliteClustering(
                    population, pop_pool, self.map_elite_parameter
                )
            elif self.map_elite_parameter["type"] == "Cluster-Exploration":
                self.map_elite_parameter["fitness_ratio"] = np.clip(1, 0.05, 0.95)
                elite_map, pop_pool = selMAPEliteClustering(
                    population, pop_pool, self.map_elite_parameter
                )
            else:
                raise Exception
            if self.ensemble_selection == "MAP-Elite":
                self.hof = list(elite_map.values())
            if self.ensemble_selection == "Half-MAP-Elite":
                self.hof = list(
                    sorted(
                        elite_map.values(),
                        key=lambda x: x.fitness.wvalues[0],
                        reverse=True,
                    )
                )[: len(elite_map) // 2]
            if self.param.get("record_training_data", False):
                self.elite_grid_avg_diversity_history.append(
                    self.euclidian_diversity_calculation(elite_map.values())
                )
                self.elite_grid_avg_fitness_history.append(
                    np.mean([ind.fitness.wvalues[0] for ind in elite_map.values()])
                )
                if self.current_gen == self.n_gen:
                    data = np.zeros(
                        (
                            self.map_elite_parameter["map_size"],
                            self.map_elite_parameter["map_size"],
                        )
                    )
                    for k, v in elite_map.items():
                        data[(k[0] - 1, k[1] - 1)] = v.fitness.wvalues[0]
                    self.final_elite_grid = data.tolist()
        if self.ensemble_selection == "MaxAngleSelection":
            self.hof = selMaxAngleSelection(
                population + list(self.hof), self.ensemble_size, self.y, unique=True
            )
        if self.ensemble_selection == "AngleDrivenSelection":
            self.hof = selAngleDrivenSelection(
                population + list(self.hof), self.ensemble_size, self.y, unique=True
            )
        return elite_map, pop_pool

    def sample_model_name(self, parent):
        models = self.base_model_list.split(",")
        # Randomly select either one population
        model = random.choice(models)
        parent = list(filter(lambda x: x.base_model == model, parent))
        return model, parent

    def survival_selection(self, gen, population, offspring):
        # Using NSGA-II or other operators to select parent individuals
        if isinstance(self.racing, RacingFunctionSelector):
            self.racing.update(offspring)
            population[:] = self.racing.environmental_selection(
                population, offspring, self.n_pop
            )
            return
        nsga2 = self.bloat_control is not None and self.bloat_control.get(
            "NSGA2", False
        )
        spea2 = self.bloat_control is not None and self.bloat_control.get(
            "SPEA2", False
        )
        if self.environmental_selection == "NSGA2":
            nsga2 = True
        if self.environmental_selection == "SPEA2":
            spea2 = True
        if isinstance(self.environmental_selection, EnvironmentalSelection):
            population[:] = self.environmental_selection.select(population, offspring)
        elif self.environmental_selection in ["NSGA2"]:
            population[:] = selNSGA2(offspring + population, len(population))
            self.hof = population
        elif self.check_alpha_dominance_nsga2():
            alpha = self.bloat_control["alpha_value"]
            alpha = self.alpha_dominance.selection(population, offspring, alpha)
            self.bloat_control["alpha_value"] = alpha
        elif nsga2 or spea2:
            max_size = max([len(x) for x in offspring + population])
            for ind in offspring + population:
                setattr(ind, "original_fitness", ind.fitness.values)
                ind.fitness.weights = (-1, -1)
                ind.fitness.values = (ind.fitness.values[0], len(ind) / max_size)
            if nsga2:
                population[:] = selNSGA2(offspring + population, len(population))
            elif spea2:
                population[:] = selSPEA2(offspring + population, len(population))
            self.hof = [max(population, key=lambda x: x.fitness.wvalues[0])]
            for ind in population:
                ind.fitness.weights = (-1,)
                ind.fitness.values = getattr(ind, "original_fitness")
        elif self.environmental_selection in ["NSGA2-CV"]:
            for ind in offspring + population:
                setattr(ind, "original_fitness", ind.fitness.values)
                fitness = ind.case_values
                ind.fitness.weights = (-1,) * len(fitness)
                ind.fitness.values = list(fitness)
            population[:] = selNSGA2(offspring + population, len(population))
            for ind in population:
                ind.fitness.weights = (-1,)
                ind.fitness.values = getattr(ind, "original_fitness")
            self.hof = population
        elif self.environmental_selection == "MOEA/D":
            # MOEA/D with random decomposition
            def selMOEAD(individuals, k):
                inds = []
                while True:
                    if len(inds) >= k:
                        break
                    weight = np.random.uniform(
                        0, 1, size=(len(individuals[0].case_values))
                    )
                    fun = lambda x: np.sum(weight * np.array(x.case_values))
                    ind = sorted(individuals, key=lambda x: float(fun(x)))[0]
                    individuals.remove(ind)
                    inds.append(ind)
                return inds

            population[:] = selMOEAD(offspring + population, len(population))
            self.hof = population
        elif self.environmental_selection == "Best":
            population[:] = selBest(offspring + population, len(population))
        else:
            population[:] = offspring

    def check_alpha_dominance_nsga2(self):
        return self.environmental_selection == "Alpha-Dominance-NSGA2" or (
            self.bloat_control is not None
            and self.bloat_control.get("alpha_dominance_NSGA2", False)
        )

    # @timeit
    def population_reduction(self, population):
        for p in population:
            final_trees = self.remove_identical_and_constant_trees(p)
            trees = self.remove_semantically_identical_trees(final_trees)
            p.gene = trees

    def remove_identical_and_constant_trees(self, p):
        trees = p.gene
        # syntactic feature selection
        tree_str = set()
        final_trees = []
        for tree in trees:
            # remove syntactic equivalent features and constant features
            if str(tree) not in tree_str and not isinstance(tree[0], gp.rand101):
                tree_str.add(str(tree))
                final_trees.append(tree)
        return final_trees

    def remove_semantically_identical_trees(self, final_trees):
        # semantic feature selection
        tree_compiled = []
        for tree in final_trees:
            tree_compiled.append(compile(tree, self.pset))
        # random sample a subset to perform feature selection
        x = self.X[np.random.randint(low=0, size=20, high=self.y.shape[0])]
        features = result_calculation(tree_compiled, x, False)
        features, selected_index = np.unique(features, return_index=True, axis=1)
        trees = [final_trees[r] for r in selected_index]
        assert len(trees) == features.shape[1]
        return trees

    def population_evaluation(self, toolbox, population):
        """
        :param population: a population of GP individuals
        :return: evaluated individuals
        """
        # individual evaluation tasks distribution
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitness_values = list(toolbox.map(toolbox.evaluate, invalid_ind))

        # distribute tasks
        if self.n_process > 1:
            data = [next(f) for f in fitness_values]
            results = list(self.pool.map(calculate_score, data))
        else:
            results = list(map(lambda f: calculate_score(next(f)), fitness_values))

        # aggregate results
        for ind, fitness_function, result in zip(invalid_ind, fitness_values, results):
            value = fitness_function.send(result)
            # automatically determine the weight length
            ind.fitness.weights = tuple(-1 for _ in range(len(value)))
            ind.fitness.values = value

        self.update_instance_weights()
        return invalid_ind

    def cos_distance_calculation(self, population=None):
        inds = get_diversity_matrix(population)
        if self.evaluation_configuration.mini_batch:
            inds -= self.get_mini_batch_y()
        else:
            inds -= self.y
        dis = pairwise_distances(inds, metric="cosine")
        return np.mean(dis)

    def number_of_used_features(self):
        possible_terminals = set([t.name for t in self.pset.terminals[object]])
        used_terminals = set()
        for h in self.hof:
            for g in h.gene:
                for x in g:
                    if isinstance(x, Terminal) and x.name in possible_terminals:
                        used_terminals.add(x.name)
        return len(used_terminals)

    def number_of_non_zero_features(self):
        top_1 = self.hof[0]
        if isinstance(top_1.pipe["Ridge"], LinearModel):
            # only for linear model
            coef = top_1.pipe["Ridge"].coef_
            non_zero = 0
            for g, c in zip(top_1.gene, coef):
                if c != 0:
                    non_zero += 1
        else:
            non_zero = len(top_1.gene)
        return non_zero

    def euclidian_diversity_calculation(self, individuals=None):
        """
        Calculate the diversity between individuals
        """
        if isinstance(self.score_func, str) and "CV" in self.score_func:
            return 0
        # distance calculation
        if individuals is None:
            individuals = self.hof
        if self.second_layer in [
            "None",
            None,
            "CAWPE",
            "Ridge-Prediction",
        ]:
            all_ind = individuals
        elif self.second_layer in ["DiversityPrune", "TreeBaseline"]:
            # with some prune
            if not hasattr(self, "tree_weight"):
                return 0
            all_ind = list(
                map(
                    lambda x: x[1],
                    filter(
                        lambda x: self.tree_weight[x[0]] > 0, enumerate(individuals)
                    ),
                )
            )
        else:
            raise Exception
        inds = get_diversity_matrix(all_ind)
        dis = pairwise_distances(inds)
        return np.mean(dis)

    def diversity_assignment(self, population):
        # distance calculation
        inds = get_diversity_matrix(population)

        hof_inds = []
        pop = {
            "Pop": self.pop,
            "Archive": self.hof,
            "Pop+Archive": list(self.pop) + list(self.hof),
        }[self.diversity_search]
        for p in pop:
            hof_inds.append(p.predicted_values.flatten())
        hof_inds = np.array(hof_inds)
        for index, ind in enumerate(population):
            distance = -np.sqrt(np.sum((inds[index] - hof_inds) ** 2, axis=0))
            if len(ind.case_values) == len(self.y):
                ind.case_values = np.concatenate([ind.case_values, distance], axis=0)
            else:
                ind.case_values[len(self.y) :] = distance

    def complexity(self):
        count = 0
        # count GP complexity
        for h in self.hof:
            h: MultipleGeneGP
            for x in h.gene:
                count += len(x)

            # count Base Model Complexity
            learner = h.pipe["Ridge"]
            if isinstance(learner, DecisionTreeRegressor):
                count += learner.tree_.node_count
            elif isinstance(learner, SoftPLTreeRegressor):
                count += learner.complexity()
            elif isinstance(learner, LinearModel):
                count += len(learner.coef_)
                if learner.intercept_ != 0:
                    count += 1
            else:
                raise Exception("Unknown Learner")
        return count

    def model(self, mtl_id=None):
        # assert self.standardized_flag, "Data is not standardized."
        if len(self.hof) == 1:
            assert len(self.hof) == 1
            best_ind = self.hof[0]
            final_model = self.single_model(best_ind, mtl_id)
            if isinstance(self.y_scaler, StandardScaler):
                mean, std = self.y_scaler.mean_[0], self.y_scaler.scale_[0]
                final_model = f"((({final_model})*{std})+{mean})"
            if isinstance(self.y_scaler, MinMaxScaler):
                min_val, max_val = (
                    self.y_scaler.data_min_[0],
                    self.y_scaler.data_max_[0],
                )
                final_model = f"(({final_model}) * ({max_val} - {min_val})) + {min_val}"
            if isinstance(self.y_scaler, YIntScaler):
                if self.y_scaler.is_integer:
                    final_model = f"Round({final_model})"
            return final_model
        else:
            final_model = ""
            for ind in self.hof:
                if final_model == "":
                    final_model += "(" + self.single_model(ind, mtl_id) + ")"
                else:
                    final_model += "+(" + self.single_model(ind, mtl_id) + ")"
            if self.y_scaler is not None:
                mean, std = self.y_scaler.mean_[0], self.y_scaler.scale_[0]
                final_model = f"((({final_model})/{len(self.hof)}*{std})+{mean})"
            return final_model

    def single_model(self, best_ind, mtl_id=None):
        if isinstance(best_ind.pipe, Pipeline):
            if "Scaler" in best_ind.pipe.named_steps:
                scaler: StandardScaler = best_ind.pipe["Scaler"]
            else:
                scaler: StandardScaler = None

            if mtl_id != None:
                learner: MTLRidgeCV = best_ind.pipe["Ridge"]
                learner = learner.mtl_ridge.estimators_[mtl_id]
            else:
                learner: LinearModel = best_ind.pipe["Ridge"]
        else:
            scaler = None
            learner = best_ind.pipe
        assert isinstance(learner, LinearModel)
        genes = best_ind.gene
        final_model = model_to_string(genes, learner, scaler)
        return final_model

    def pretrain_predict(self, X):
        if self.learner is not None:
            data = []
            for model in self.pretrain_models:
                data.append(model.predict(X).flatten())
            X = np.concatenate([X, np.array(data).T], axis=1)
        return X

    def post_selection(self, offspring):
        if self.post_selection_method == None:
            pass
        elif self.post_selection_method == "Dominance":
            assert len(offspring) == 2
            fitness_a = np.array(offspring[0].fitness_list)
            fitness_a = fitness_a[:, 0] * fitness_a[:, 1]
            fitness_b = np.array(offspring[1].fitness_list)
            fitness_b = fitness_b[:, 0] * fitness_b[:, 1]
            if np.all(fitness_a > fitness_b):
                return [offspring[0]]
            if np.all(fitness_b > fitness_a):
                return [offspring[1]]

            for a, b in zip(fitness_a, fitness_b):
                if a > b:
                    return [offspring[0]]
                elif b > a:
                    return [offspring[1]]
            return [random.choice(offspring)]
        elif self.post_selection_method == "Sharpness":
            assert len(offspring) == 2
            fitness_a = np.array(offspring[0].fitness_list)
            fitness_a = fitness_a[:, 0] * fitness_a[:, 1]
            fitness_b = np.array(offspring[1].fitness_list)
            fitness_b = fitness_b[:, 0] * fitness_b[:, 1]
            for a, b in zip(fitness_a[::-1], fitness_b[::-1]):
                if a > b:
                    return [offspring[0]]
                elif b > a:
                    return [offspring[1]]
            return [random.choice(offspring)]
        else:
            raise Exception
        return offspring


def init_worker(function, data):
    function.data = data
