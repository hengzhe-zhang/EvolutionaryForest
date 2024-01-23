import gc
import inspect
from multiprocessing import Pool

import dill
import numpy as np
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
from gplearn.functions import _protected_sqrt
from lightgbm import LGBMRegressor, LGBMModel
from lineartree import LinearTreeRegressor
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau, rankdata, ranksums, wilcoxon
from sklearn.base import ClassifierMixin, TransformerMixin
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
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
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
from evolutionary_forest.component.crossover.intron_based_crossover import (
    IntronPrimitive,
    IntronTerminal,
)
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
from evolutionary_forest.component.environmental_selection import (
    NSGA2,
    EnvironmentalSelection,
    SPEA2,
    Best,
    NSGA3,
)
from evolutionary_forest.component.evaluation import (
    calculate_score,
    pipe_combine,
    single_tree_evaluation,
    EvaluationResults,
    select_from_array,
    get_sample_weight,
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
from evolutionary_forest.component.generation import varAndPlus
from evolutionary_forest.component.initialization import (
    initialize_crossover_operator,
    unique_initialization,
)
from evolutionary_forest.component.mutation.common import MutationOperator
from evolutionary_forest.component.mutation.learning_based_mutation import (
    BuildingBlockLearning,
)
from evolutionary_forest.component.normalizer import TargetEncoder
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
)
from evolutionary_forest.component.selection_operators.niche_base_selection import (
    niche_base_selection,
)
from evolutionary_forest.component.stateful_gp import make_class, TargetEncoderNumpy
from evolutionary_forest.component.strategy import Clearing
from evolutionary_forest.component.test_function import TestFunction
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.model.MTL import MTLRidgeCV
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
from evolutionary_forest.model.SafeRidgeCV import BoundedRidgeCV, SplineRidgeCV
from evolutionary_forest.model.SafetyScaler import SafetyScaler
from evolutionary_forest.multigene_gp import *
from evolutionary_forest.preprocess_utils import (
    GeneralFeature,
    CategoricalFeature,
    BooleanFeature,
    NumericalFeature,
    FeatureTransformer,
    StandardScalerWithMinMaxScaler,
    DummyScaler,
)
from evolutionary_forest.preprocessing.SigmoidTransformer import SigmoidTransformer
from evolutionary_forest.probability_gp import genHalfAndHalf, genFull
from evolutionary_forest.strategies.adaptive_operator_selection import (
    MultiArmBandit,
    MCTS,
)
from evolutionary_forest.strategies.estimation_of_distribution import (
    EstimationOfDistribution,
    eda_operators,
)
from evolutionary_forest.strategies.multifidelity_evaluation import (
    MultiFidelityEvaluation,
)
from evolutionary_forest.strategies.surrogate_model import SurrogateModel
from evolutionary_forest.utility.evomal_loss import *
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
        basic_gene_num=0,  # Number of basic genes in a MGP
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
        mean_model=False,  # Whether to use the mean model for predictions
        environmental_selection=None,  # Environmental selection method
        pre_selection=None,  # Pre-selection method
        eager_training=False,  # Whether to train models eagerly
        useless_feature_ratio=None,  # Ratio of useless features to be removed
        weighted_coef=False,  # Whether to use weighted coefficients
        feature_selection=False,  # Whether to perform feature selection
        outlier_detection=False,  # Whether to perform outlier detection
        semantic_repair=0,  # Semantic repair method
        dynamic_reduction=0,  # Dynamic reduction strategy
        active_gene_num=0,  # Number of active genes in MGP
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
        self.validation_ratio = validation_ratio
        self.simplification = simplification
        self.racing = racing
        self.bounded_prediction = bounded_prediction
        self.constant_ratio = constant_ratio
        self.learner = learner
        self.force_retrain = force_retrain
        self.base_learner_configuration = BaseLearnerConfiguration(**params)
        self.pac_bayesian = PACBayesianConfiguration(**params)
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
        self.active_gene_num = active_gene_num
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
        self.basic_gene_num = basic_gene_num
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
        self.mean_model = mean_model
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
        self.initialized = False
        self.pop: List[MultipleGeneGP] = []
        self.basic_primitives = basic_primitives
        self.select = select
        self.gene_num = gene_num
        self.param = params
        self.diversity_search = diversity_search
        self.second_layer = second_layer
        self.test_fun: List[TestFunction] = test_fun
        self.history_initialization()
        self.early_stop = early_stop
        if environmental_selection == "NSGA2":
            self.environmental_selection: NSGA2 = NSGA2(self, None, **self.param)
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

        if isinstance(self.ps_tree_ratio, str) and "Interleave" in self.ps_tree_ratio:
            interleaving_period = int(
                np.round(
                    n_gen
                    / (n_gen * float(self.ps_tree_ratio.replace("Interleave-", "")))
                )
            )
        self.interleaving_period = interleaving_period
        self.test_data = None

        if params.get("record_training_data", False):
            self.test_fun[0].regr = self
            self.test_fun[1].regr = self

        self.normalize = normalize
        if normalize is True:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
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
            self.y_scaler = DummyScaler()
        elif normalize == "LN":
            self.x_scaler = Pipeline(
                [
                    ("TE", TargetEncoder()),
                    ("SC", StandardScaler()),
                ]
            )
            self.y_scaler = StandardScaler()
        elif normalize == "MinMax":
            self.x_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()
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

        if self.mutation_scheme == "Transformer":
            self.transformer_switch = True
            self.mutation_scheme = "uniform-plus"
        else:
            self.transformer_switch = False

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
                self.base_model_list = "DT,Ridge"
        elif self.base_learner == "RDT-LR":
            self.base_model_list = "Random-DT,LogisticRegression"
        elif self.base_learner == "Balanced-RDT-LR":
            self.base_model_list = "Balanced-Random-DT,Balanced-LogisticRegression"
        elif self.base_learner == "Balanced-DT-LR":
            self.base_model_list = "Balanced-DT,Balanced-LogisticRegression"
        elif self.base_learner == "DT-LGBM":
            self.base_model_list = "Random-DT,LightGBM-Stump,DT"
        else:
            self.base_model_list = None

        if self.select == "Auto":
            self.aos = MultiArmBandit(self, **self.param)
        elif self.select == "Auto-MCTS":
            self.aos = MCTS(self, **self.param)
        else:
            self.aos = None

        delete_keys = []
        for k in params.keys():
            if k in vars(self):
                delete_keys.append(k)
        for k in delete_keys:
            del params[k]

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
        if self.constant_type == "GD":
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
        self.surrogate_model = SurrogateModel(self)
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

    def history_initialization(self):
        self.train_data_history = []
        self.test_data_history = []
        # average fitness of the ensemble model
        self.archive_fitness_history = []
        # average diversity of the ensemble model
        self.archive_diversity_history = []
        # average diversity of the ensemble model
        self.archive_cos_distance_history = []
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
        elif isinstance(score_func, str) and score_func == "R2-BootstrapError":
            self.score_func = R2BootstrapError()
        elif isinstance(score_func, str) and score_func == "R2-FeatureCount":
            self.score_func = R2FeatureCount()
        elif isinstance(score_func, str) and score_func == "R2-Size-Scaler":
            self.score_func = R2SizeScaler(self, **params)
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
        if self.basic_primitives == "ML":
            ridge_ = x["model"]["Ridge"]
        else:
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
                coef = np.max(np.abs(base_learner.coef_), axis=0)[: self.gene_num]
            else:
                coef = np.abs(base_learner.coef_)[: self.gene_num]
        elif isinstance(base_learner, (BaseDecisionTree, LGBMModel)):
            coef = base_learner.feature_importances_[: self.gene_num]
        elif isinstance(base_learner, (GBDTLRClassifier)):
            coef = base_learner.gbdt_.feature_importances_[: self.gene_num]
        elif isinstance(base_learner, SVR):
            coef = np.ones(self.X.shape[1])
        elif isinstance(base_learner, (RidgeDT, LRDTClassifier)):
            coef = base_learner.feature_importance
        elif hasattr(base_learner, "feature_importances_"):
            coef = base_learner.feature_importances_
        elif isinstance(
            base_learner, (KNeighborsRegressor, MeanRegressor, MedianRegressor)
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
        elif self.base_learner in [
            "DT-LR",
            "Balanced-DT-LR",
            "Balanced-RDT-LR",
            "DT-LGBM",
            "RDT~LightGBM-Stump",
        ] or isinstance(self.base_learner, list):
            pipe = self.get_base_model(base_model=individual.base_model)
        elif self.base_learner == "Dynamic-LogisticRegression":
            pipe = self.get_base_model(
                regularization_ratio=individual.dynamic_regularization
            )
        elif self.base_learner == "AdaptiveLasso":
            pipe = self.get_base_model(
                lasso_alpha=np.power(10, individual.parameters["Lasso"])
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
        if individual.active_gene_num > 0:
            genes = individual.gene[: individual.active_gene_num]
        else:
            genes = individual.gene

        information: EvaluationResults
        if self.n_process > 1:
            y_pred, estimators, information = yield pipe, dill.dumps(genes, protocol=-1)
        else:
            y_pred, estimators, information = yield pipe, genes

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
            if self.cv == 1:
                individual.pipe = estimators[0]
            else:
                individual.pipe = pipe

        if self.bootstrap_training or self.eager_training:
            Yp = multi_tree_evaluation(
                genes,
                self.pset,
                X,
                self.original_features,
                sklearn_format=self.basic_primitives == "ML",
                register_array=individual.parameters["Register"]
                if self.mgp_mode == "Register"
                else None,
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
        individual.correlation_results = information.correlation_results
        self.importance_post_process(individual, estimators)

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
        if self.basic_primitives != "ML" and hasattr(estimators[0]["Ridge"], "tree_"):
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

        def coefficient_process(coefficient):
            if sum(coefficient) == 0:
                coefficient = np.ones_like(coefficient)
            coefficient = coefficient / np.sum(coefficient)
            coefficient = np.nan_to_num(coefficient, posinf=0, neginf=0)
            return coefficient

        individual.coef = coefficient_process(individual.coef)

    def calculate_fitness_value(self, individual, estimators, Y, y_pred):
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
            self.score_func == "R2"
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
        if self.base_learner == "RidgeCV-ENet":
            self.base_learner = "ElasticNetCV"
            individual.pipe = self.get_base_model()
            self.base_learner = "RidgeCV-ENet"
        if self.imbalanced_configuration.balanced_final_training:
            individual.pipe = self.get_base_model()
        # avoid re-training
        model = individual.pipe
        if self.basic_primitives == "ML" and model.steps[0][0] != "feature":
            individual.pipe = pipe_combine(Yp, model)
            model = individual.pipe
            Yp = self.X
        if not force_training:
            # check the necessity of training
            try:
                if (
                    hasattr(individual, "active_gene_num")
                    and individual.active_gene_num > 0
                ):
                    input_size = individual.active_gene_num
                else:
                    input_size = len(individual.gene)
                if hasattr(model, "partition_scheme"):
                    input_size += 1
                if self.original_features:
                    input_size += self.X.shape[1]
                if self.intron_probability > 0:
                    input_size = individual.active_gene.sum()
                if self.basic_primitives == "ML":
                    input_size = self.X.shape[1]
                if self.mgp_mode == "Register":
                    input_size = self.number_of_register
                model.predict(np.ones((1, input_size)))
                return None
            except NotFittedError:
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
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
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
        elif self.base_learner == "PL-Tree":
            ridge_model = LinearTreeRegressor(base_estimator=LinearRegression())
        elif self.base_learner == "RidgeCV" or self.base_learner == "RidgeCV-ENet":
            # from sklearn.linear_model._coordinate_descent import _alpha_grid
            # alphas = _alpha_grid(self.X, self.y, normalize=True)
            if self.ridge_alphas is None:
                ridge = [0.1, 1, 10]
            elif self.ridge_alphas == "Auto":
                if self.X.shape[0] < 200:
                    ridge = [1e1, 1e2, 1e3]
                else:
                    ridge = [0.1, 1, 10]
            elif self.ridge_alphas == "Auto-Fine":
                if self.X.shape[0] < 200:
                    ridge = [1e1, 3e1, 1e2, 3e2, 1e3]
                else:
                    ridge = [1e-1, 3e-3, 1e0, 3e0, 1e1]
            else:
                ridge = eval(self.ridge_alphas)
            ridge_model = RidgeCV(
                alphas=ridge, store_cv_values=True, scoring=make_scorer(r2_score)
            )
        elif self.base_learner == "SplineRidgeCV":
            ridge_model = SplineRidgeCV(
                store_cv_values=True, scoring=make_scorer(r2_score)
            )
        elif self.base_learner == "Bounded-RidgeCV":
            ridge_model = BoundedRidgeCV(
                store_cv_values=True, scoring=make_scorer(r2_score)
            )
        elif self.base_learner == "ElasticNetCV":
            ridge_model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], n_alphas=10
            )
        elif self.base_learner == "LR":
            ridge_model = LinearRegression()
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
        elif self.base_learner == "KNN":
            ridge_model = KNeighborsRegressor(weights="uniform")
        elif self.base_learner == "KNN-3":
            ridge_model = KNeighborsRegressor(n_neighbors=3, weights="uniform")
        elif self.base_learner == "KNN-D":
            ridge_model = KNeighborsRegressor(weights="distance")
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
        pipe = GPPipeline(
            [
                ("Scaler", SafetyScaler()),
                ("Ridge", ridge_model),
            ]
        )
        if isinstance(pipe["Ridge"], BaseDecisionTree) and self.max_tree_depth != None:
            assert pipe["Ridge"].max_depth == self.max_tree_depth
        return pipe

    def transform(self, X, ratio=0.5):
        ratio = 1 - ratio
        if self.normalize:
            X = self.x_scaler.transform(X)
        code_importance_dict = get_feature_importance(
            self, latex_version=False, fitness_weighted=False
        )
        if self.ensemble_size == 1:
            top_features = list(code_importance_dict.keys())
        else:
            top_features = select_top_features(code_importance_dict, ratio)
        transformed_features = feature_append(
            self, X, top_features, only_new_features=True
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
        if isinstance(self.gene_num, str) and "Max" in self.gene_num:
            self.gene_num = min(int(self.gene_num.replace("Max-", "")), x.shape[1])
        if isinstance(self.n_pop, str) and "N" in self.n_pop:
            self.n_pop = extract_numbers(x.shape[1], self.n_pop)
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
        if self.transformer_switch:
            self.transformer_tool = TransformerTool(self.X, self.y, self.hof, self.pset)

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
            toolbox.tree_generation = gp.genFull

            initialize_crossover_operator(self, toolbox)

            # special mutation operators
            if self.transformer_switch:
                # using transformer to generate subtrees
                def condition_probability():
                    return self.current_gen / self.n_gen

                toolbox.register(
                    "mutate",
                    mutUniform_multiple_gene_transformer,
                    expr=toolbox.expr_mut,
                    pset=pset,
                    condition_probability=condition_probability,
                    transformer=self.transformer_tool,
                )
            elif self.mutation_scheme == "parsimonious_mutation":
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
            if self.mutation_scheme == "EDA-Terminal":
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
        self.static_limit_function = staticLimit_multiple_gene(
            key=operator.attrgetter("height"),
            max_value=self.depth_limit_configuration.max_height,
            min_value=self.depth_limit_configuration.min_height,
            random_fix=self.random_fix,
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

    def tree_initialization_function(self, pset, toolbox: TypedToolbox):
        if self.initial_tree_size is None:
            toolbox.expr = partial(gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        else:
            min_height, max_height = self.initial_tree_size.split("-")
            min_height = int(min_height)
            max_height = int(max_height)
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
        # initialize the primitive set
        if self.basic_primitives == "StrongTyped":
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
                pset.addPrimitive(
                    np.add, [NumericalFeature, NumericalFeature], NumericalFeature
                )
                pset.addPrimitive(
                    np.subtract, [NumericalFeature, NumericalFeature], NumericalFeature
                )
                pset.addPrimitive(
                    np.multiply, [NumericalFeature, NumericalFeature], NumericalFeature
                )
                pset.addPrimitive(
                    analytical_quotient,
                    [NumericalFeature, NumericalFeature],
                    NumericalFeature,
                )
                pset.addPrimitive(
                    np.maximum, [NumericalFeature, NumericalFeature], NumericalFeature
                )
                pset.addPrimitive(
                    np.minimum, [NumericalFeature, NumericalFeature], NumericalFeature
                )
                pset.addPrimitive(np.sin, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(np.cos, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(_protected_sqrt, [NumericalFeature], NumericalFeature)
                pset.addPrimitive(
                    identical_numerical, [NumericalFeature], GeneralFeature
                )
            if has_categorical_feature:
                pset.addPrimitive(
                    np_bit_wrapper(np.bitwise_and),
                    [CategoricalFeature, CategoricalFeature],
                    CategoricalFeature,
                )
                pset.addPrimitive(
                    np_bit_wrapper(np.bitwise_or),
                    [CategoricalFeature, CategoricalFeature],
                    CategoricalFeature,
                )
                pset.addPrimitive(
                    np_bit_wrapper(np.bitwise_xor),
                    [CategoricalFeature, CategoricalFeature],
                    CategoricalFeature,
                )
                pset.addPrimitive(
                    identical_categorical, [CategoricalFeature], GeneralFeature
                )
            if has_boolean_feature:
                if has_numerical_feature:
                    pset.addPrimitive(
                        np.greater, [NumericalFeature, NumericalFeature], BooleanFeature
                    )
                    pset.addPrimitive(
                        np.less, [NumericalFeature, NumericalFeature], BooleanFeature
                    )
                pset.addPrimitive(
                    np.logical_and, [BooleanFeature, BooleanFeature], BooleanFeature
                )
                pset.addPrimitive(
                    np.logical_or, [BooleanFeature, BooleanFeature], BooleanFeature
                )
                pset.addPrimitive(
                    np.logical_xor, [BooleanFeature, BooleanFeature], BooleanFeature
                )
                pset.addPrimitive(identical_boolean, [BooleanFeature], GeneralFeature)
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
            self.add_primitives_to_pset(pset)
        elif isinstance(
            self.basic_primitives, str
        ) and self.basic_primitives.startswith("ML"):
            pset = PrimitiveSet("MAIN", x.shape[1])
            primitives = ",".join(
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
            models = self.basic_primitives.replace("ML-", "").split(",")
            for m in models:
                if m == "TargetEncoder":
                    pset.addPrimitive(
                        make_class(
                            TargetEncoderNumpy,
                            2,
                            parameters=dict(cols=[0], return_df=False),
                        ),
                        2,
                    )
                if m == "KNN-2":
                    pset.addPrimitive(make_class(KNeighborsRegressor, 2), 2)
                if m == "DT-2":
                    pset.addPrimitive(make_class(DecisionTreeRegressor, 2), 2)
                if m == "Ridge-2":
                    pset.addPrimitive(make_class(Ridge, 2), 2)
            self.add_primitives_to_pset(pset, primitives, transformer_wrapper=True)
            self.basic_primitives = "ML"
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

        if self.constant_type == "Normal":
            pset.addEphemeralConstant("rand101", lambda: np.random.normal())
        elif self.constant_type is None:
            pass
        elif self.basic_primitives == False:
            pset.addEphemeralConstant(
                "rand101", lambda: random.randint(-1, 1), NumericalFeature
            )
        elif self.constant_type == "Float":
            pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
        elif self.constant_type == "GD":

            def random_variable():
                return torch.randn(1, requires_grad=True, dtype=torch.float32)

            if self.constant_ratio == 0:
                pset.addEphemeralConstant("rand101", random_variable)
            else:
                for i in range(max(int(self.X.shape[1] * self.constant_ratio), 1)):
                    pset.addEphemeralConstant(f"rand{i}", random_variable)
        elif self.constant_type == "SRC":
            biggest_val = np.max(np.abs(self.X))
            generator = scaled_random_constant(biggest_val)
            pset.addEphemeralConstant("rand101", generator)
        else:
            assert self.constant_type == "Int"
            pset.addEphemeralConstant("rand101", random_int)
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

    def add_primitives_to_pset(self, pset, primitives=None, transformer_wrapper=False):
        if primitives is None:
            primitives = self.basic_primitives
        if self.custom_primitives is None:
            custom_primitives = {}
        else:
            custom_primitives = self.custom_primitives
            primitives = primitives.split(",") + ",".join(custom_primitives.keys())

        for p in primitives.split(","):
            p = p.strip()
            if p in custom_primitives:
                primitive = custom_primitives[p]
                number_of_parameters = len(inspect.signature(primitive).parameters)
                primitive = (primitive, number_of_parameters)
            else:
                if self.constant_type == "GD":
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

            pac_bayesian = R2PACBayesian(self, **self.param)
            self.hof = CustomHOF(
                self.ensemble_size,
                comparison_function=comparison,
                key_metric=lambda x: -x.sam_loss,
                # preprocess=lambda pop: pac_bayesian.assign_complexity_pop(pop),
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
        elif self.validation_size > 0:
            self.hof = ValidationHallOfFame(
                self.ensemble_size,
                self.get_validation_score,
                self.archive_configuration,
            )
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

    def fit(self, X, y, test_X=None):
        self.counter_initialization()
        self.history_initialization()

        self.y_shape = y.shape
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()  # store column names

        # Normalize X and y if specified
        if self.normalize:
            X = self.x_scaler.fit_transform(X, y)
            y = self.y_scaler.fit_transform(np.array(y).reshape(-1, 1))
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
        self.X, self.y = X, y.flatten()

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
                self.reference_lgbm = RidgeCV()
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
            Yp = multi_tree_evaluation(
                individual.gene,
                self.pset,
                X,
                self.original_features,
                sklearn_format=self.basic_primitives == "ML",
                register_array=individual.parameters["Register"]
                if self.mgp_mode == "Register"
                else None,
                configuration=self.evaluation_configuration,
                noise_configuration=self.pac_bayesian.noise_configuration,
            )
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
        self, X, individual, random_noise=0, random_seed=0, noise_configuration=None
    ):
        if individual.active_gene_num > 0:
            genes = individual.gene[: individual.active_gene_num]
        else:
            genes = individual.gene
        Yp = multi_tree_evaluation(
            genes,
            self.pset,
            X,
            self.original_features,
            configuration=self.evaluation_configuration,
            sklearn_format=self.basic_primitives == "ML",
            register_array=individual.parameters["Register"]
            if self.mgp_mode == "Register"
            else None,
            random_noise=random_noise,
            random_seed=random_seed,
            noise_configuration=noise_configuration,
        )
        if isinstance(Yp, torch.Tensor):
            Yp = Yp.detach().numpy()
        return Yp

    def predict(self, X, return_std=False):
        if self.normalize:
            # Scale X data if normalize flag is set
            X = self.x_scaler.transform(X)

        X = self.pretrain_predict(X)

        prediction_data_size = X.shape[0]
        if self.test_data is not None:
            # Concatenate new X data with existing X data in the transductive learning setting
            X = np.concatenate([self.X, X])

        # Prune genes in hall of fame using hoist mutation
        # if self.intron_gp:
        #     for h in self.hof:
        #         for gene in h.gene:
        #             # Find the best gene and hoist it to the top
        #             best_id = max(
        #                 [(k, getattr(g, "corr", 0)) for k, g in enumerate(gene)],
        #                 key=lambda x: (x[1], x[0]),
        #             )[0]
        #             hoistMutation(gene, best_id)
        #         # Reset the pipeline to the base model
        #         h.pipe = self.get_base_model()

        # Train the final model using lazy training
        self.final_model_lazy_training(self.hof, force_training=self.force_retrain)

        predictions = []
        weight_list = []

        # Generate features for each individual in the hall of fame
        for individual in self.hof:
            individual: MultipleGeneGP
            if len(individual.gene) == 0:
                continue
            if self.basic_primitives == "ML":
                # Use evolved pipeline to make predictions
                predicted = individual.pipe.predict(X)
            else:
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

            if self.bounded_prediction:
                predicted = np.clip(predicted, self.y.min(), self.y.max())

            if self.normalize:
                # Un-scale predicted values if normalize flag is set
                predicted = self.y_scaler.inverse_transform(
                    predicted.reshape(-1, 1)
                ).flatten()
            if (
                hasattr(self.hof, "ensemble_weight")
                and len(self.hof.ensemble_weight) > 0
            ):
                # Append ensemble weights if they exist
                weight_list.append(
                    self.hof.ensemble_weight[individual_to_tuple(individual)]
                )
            if len(weight_list) > 0 and not return_std:
                predictions.append(predicted)
            else:
                if len(predictions) == 0:
                    predictions = [predicted]
                else:
                    predictions[0] += predicted

        if len(predictions) != len(self.hof):
            predictions[0] = predictions[0] / len(self.hof)
            assert len(predictions) == 1

        if self.second_layer == "RF-Routing":
            self.ridge: RandomForestClassifier
            proba = self.ridge.predict_proba(X)
            final_prediction = (
                np.array(predictions)[self.ridge.candidates].T * proba
            ).sum(axis=1)
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
        self.validation_set_generation()
        gc.collect()
        if isinstance(self.mutation_scheme, MutationOperator):
            self.mutation_scheme.callback(self.pop)
        # callback function after each generation
        if self.verbose:
            pop = self.pop
            dt = defaultdict(int)
            # parameters = np.zeros(2)
            for p in pop:
                # dt[str(p.base_learner)] += 1
                # dt[str(p.dynamic_leaf_size)] += 1
                dt[str(p.dynamic_regularization)] += 1
            print(dt)
            if self.all_nodes_counter > 0:
                print(
                    "Ratio of Introns",
                    self.intron_nodes_counter / self.all_nodes_counter,
                )
            # print(parameters)

        if self.pac_bayesian.automatic_std and (
            self.current_gen % 20 == 0 and self.current_gen != 0
        ):
            automatic_perturbation_std(self, self.pop)

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
        if self.verbose:
            print("data shape", self.X.shape, self.y.shape)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        if self.reduction_ratio >= 1:
            self.population_reduction(population)

        # Evaluate the individuals with an invalid fitness
        if self.base_learner in ["RDT~LightGBM-Stump"]:
            invalid_ind = self.multiobjective_evaluation(toolbox, population)
        else:
            invalid_ind = self.population_evaluation(toolbox, population)

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
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
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
        # fitness decrease on validation set
        worse_iterations = 0
        # iterations of no fitness improvement on training set
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
        pop_size = self.n_pop
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

            if self.transformer_switch:
                self.transformer_tool.train()

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

            # offspring generation
            while len(new_offspring) < pop_size:
                if count > pop_size * 100:
                    raise Exception("Error!")
                count += 1
                offspring = self.select_pair_of_parents(
                    population, toolbox, elite_map, elites_archive, fitness_improvement
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
                        if (
                            random.random()
                            < self.crossover_configuration.semantic_crossover_probability
                            and self.crossover_configuration.semantic_crossover_mode
                            == CrossoverMode.Independent
                        ):
                            parent = [offspring[0], offspring[1]]
                            # copy individuals before any modifications
                            offspring = [
                                efficient_deepcopy(offspring[0]),
                                efficient_deepcopy(offspring[1]),
                            ]
                            if (
                                self.crossover_configuration.semantic_selection_mode
                                == SelectionMode.AngleDriven
                            ):
                                # either macro-crossover or macro-crossover
                                offspring = semanticFeatureCrossover(
                                    offspring[0], offspring[1], target=self.y
                                )
                            elif (
                                self.crossover_configuration.semantic_selection_mode
                                == SelectionMode.MAPElites
                            ):
                                offspring = mapElitesCrossover(
                                    offspring[0],
                                    offspring[1],
                                    target=self.y,
                                    map_elites_configuration=self.crossover_configuration.map_elites_configuration,
                                )
                            else:
                                raise Exception("Invalid Selection Mode!")
                            # mark as macro crossover
                            self.record_parent_fitness(
                                parent, offspring, crossover_type="Macro"
                            )
                        else:
                            if (
                                self.crossover_configuration.semantic_crossover_mode
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
                                crossover_configuration=self.crossover_configuration,
                                mutation_configuration=self.mutation_configuration,
                                algorithm=self,
                            )
                else:
                    offspring: MultipleGeneGP = varAnd(offspring, toolbox, cxpb, mutpb)

                self.torch_variable_clone(offspring)
                self.fix_very_trivial_trees(offspring)

                check_mutation = (
                    self.bloat_control is not None
                    and self.bloat_control.get("check_mutation", False)
                )
                check_initialization = (
                    self.bloat_control is not None
                    and self.bloat_control.get("check_initialization", False)
                )
                if check_mutation or check_initialization:
                    for o in offspring:
                        previous_set = set()
                        for id, gene in enumerate(o.gene):
                            # mutate constant genes and redundant genes
                            if str(gene) in previous_set or is_float(str(gene)):
                                if check_mutation:
                                    o.gene[id] = mutUniform(
                                        gene, self.toolbox.expr_mut, self.pset
                                    )[0]
                                if check_initialization:
                                    o.gene[id] = self.new_tree_generation()
                            previous_set.add(str(gene))

                offspring = self.prune_and_plant(offspring)

                # very strong mutation
                if self.intron_threshold > 0:
                    for o in offspring:
                        for id, gene in enumerate(o.gene):
                            if o.coef[id] < self.intron_threshold:
                                (o.gene[id],) = self.neutral_mutation(gene)

                self.semantic_repair_features(offspring, elites_archive)

                if self.base_learner == "Hybrid":
                    # Mutation for base learners
                    base_models = [o.base_model for o in offspring]
                    base_models = varAnd(
                        base_models, self.tpot_model._toolbox, 0.5, 0.1
                    )
                    for o, b in zip(offspring, base_models):
                        o.base_model = b

                self.self_adaptive_evolution(offspring)
                for o in offspring:
                    self.self_adaptive_mutation(o, population)

                    if len(new_offspring) < pop_size:
                        # checking redundant individuals
                        if (
                            self.allow_revisit
                            or (not individual_to_tuple(o) in self.evaluated_pop)
                            or (self.gene_num == 1 and count > pop_size * 10)
                        ):
                            # sometime, when gene num is very small, it is hard to generate a unique individual
                            self.evaluated_pop.add(individual_to_tuple(o))
                            new_offspring.append(o)

            assert len(new_offspring) == pop_size
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

            if self.pre_selection != None:
                (
                    offspring,
                    predicted_values,
                ) = self.surrogate_model.pre_selection_individuals(
                    population, new_offspring, self.n_pop
                )
                assert len(offspring) == self.n_pop, f"{len(offspring), self.n_pop}"
            else:
                offspring = new_offspring

            self.multi_fidelity_evaluation.update_evaluation_mode(self.current_gen)
            if self.reduction_ratio >= 1:
                self.population_reduction(population)

            if self.stage_flag:
                print("Start Evaluation")
            # Evaluate the individuals with an invalid fitness
            if self.base_learner in ["RDT~LightGBM-Stump"]:
                invalid_ind = self.multiobjective_evaluation(toolbox, offspring)
            else:
                invalid_ind = self.population_evaluation(toolbox, offspring)

            self.post_processing_after_evaluation(population, offspring)

            if self.verbose:
                p_value = self.get_p_value(offspring, population)
                print("P value of different population", p_value)

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

            if self.select in ["Auto", "Auto-MCTS"]:
                self.aos.update(population, offspring)

            # Update the hall of fame with the generated individuals
            if self.stage_flag:
                print("Start HOF updating")
            if self.dynamic_target:
                halloffame.clear()
            if self.lasso_hof is not None:
                self.lasso_hof.update(population)
            if halloffame is not None and worse_iterations == 0:
                halloffame.update(offspring)
                if self.verbose:

                    def replace(ss):
                        for s in range(
                            self.gene_num + self.X.shape[1], self.X.shape[1] - 1, -1
                        ):
                            ss = ss.replace(f"ARG{s}", f"INS{s - self.X.shape[1]}")
                        return ss

                    print("\n".join([replace(str(g)) for g in halloffame[0].gene]))
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
                if self.basic_primitives != "ML":
                    if self.active_gene_num == 0:
                        print(
                            "Unique Hash",
                            [
                                len(
                                    np.unique(
                                        [
                                            o.hash_result[i]
                                            for o in offspring
                                            if i < len(o.hash_result)
                                        ]
                                    )
                                )
                                for i in range(self.gene_num)
                            ],
                        )
                    else:
                        print(
                            "Unique Hash",
                            [
                                len(np.unique([o.hash_result[i] for o in offspring]))
                                for i in range(self.active_gene_num)
                            ],
                        )
                if self.base_learner == "AdaptiveLasso":
                    lasso_parameter = [o.parameters["Lasso"] for o in offspring]
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

            # multiobjective GP based on fitness-size
            if self.select == "Auto-MCTS":
                self.aos.store_in_archive(offspring)

            self.redundant_features_calculation(offspring)
            # Replace the current population by the offspring
            self.survival_selection(gen, population, offspring)
            # self.update_semantic_repair_input_matrix(population)
            elites_archive = self.update_external_archive(population, elites_archive)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
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
            assert number_of_evaluations == (self.current_gen + 1) * self.n_pop
        else:
            assert number_of_evaluations == total_evaluations

        if self.early_stop > 0:
            print("Final Generation", self.current_gen)
        if verbose:
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

    def fix_very_trivial_trees(self, offspring):
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
        if isinstance(self.score_func, R2PACBayesian):
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
            # clone all variables
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
            if (
                crossover_configuration.semantic_selection_mode
                == SelectionMode.MAPElites
            ):
                available_parent = mapElitesCrossover(
                    offspring[0],
                    offspring[1],
                    self.y,
                    crossover_configuration.map_elites_configuration,
                )
            elif (
                crossover_configuration.semantic_selection_mode
                == SelectionMode.AngleDriven
            ):
                available_parent = semanticFeatureCrossover(
                    offspring[0], offspring[1], self.y
                )
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

    def get_p_value(self, offspring, population):
        num_top_individuals = 30
        # num_top_individuals = self.n_pop
        p_value = ranksums(
            [
                o.fitness.wvalues[0]
                for o in sorted(
                    offspring, key=lambda x: x.fitness.wvalues[0], reverse=True
                )[:num_top_individuals]
            ],
            [
                o.fitness.wvalues[0]
                for o in sorted(
                    population, key=lambda x: x.fitness.wvalues[0], reverse=True
                )[:num_top_individuals]
            ],
        )
        return p_value[1]

    def get_validation_score(self, best_individual, force_training=False):
        self.final_model_lazy_training([best_individual], force_training=force_training)
        parent_input = self.feature_generation(self.valid_x, best_individual)
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

        if self.crossover_configuration.semantic_crossover_mode is not None:
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
            self.pop_avg_fitness_history.append(
                np.mean([ind.fitness.wvalues[0] for ind in population])
            )
            self.pop_diversity_history.append(self.diversity_calculation(population))
            if not isinstance(self, ClassifierMixin):
                self.pop_cos_distance_history.append(
                    self.cos_distance_calculation(population)
                )
                self.archive_cos_distance_history.append(
                    self.cos_distance_calculation(self.hof)
                )
            (
                genotype_sum_entropy,
                phenotype_sum_entropy,
            ) = self.gp_tree_entropy_calculation(population)
            self.tree_genotypic_diversity.append(genotype_sum_entropy)
            self.tree_phenotypic_diversity.append(phenotype_sum_entropy)
            self.avg_tree_size_history.append(
                np.mean([np.mean([len(g) for g in p.gene]) for p in population])
            )
            self.archive_fitness_history.append(
                np.mean([ind.fitness.wvalues[0] for ind in self.hof])
            )
            self.archive_diversity_history.append(self.diversity_calculation())

    def training_with_validation_set(self):
        # Train the final model with the validation set if data combination is enabled and validation set is provided
        if self.archive_configuration.data_combination and self.validation_size > 0:
            # Combine the training and validation sets
            X = np.concatenate((self.X, self.valid_x), axis=0)
            y = np.concatenate((self.y, self.valid_y), axis=0)
            # Train the final model using the combined set
            self.final_model_lazy_training(self.hof, X, y, force_training=True)

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
            o.parameters["Lasso"] = o.parameters["Lasso"] + factor * (
                a.parameters["Lasso"] - b.parameters["Lasso"]
            )
            o.parameters["Lasso"] = np.clip(o.parameters["Lasso"], -5, 0)
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
                # 0.25-> AA BB
                # 0.25-> AB BA
                parent_a = self.sample_model_name(parent_pool)
                offspring_a = toolbox.select(parent_a, 1)
                parent_b = self.sample_model_name(parent_pool)
                offspring_b = toolbox.select(parent_b, 1)
                offspring = [offspring_a[0], offspring_b[0]]
            else:
                parent_a = self.sample_model_name(parent_pool)
                offspring = toolbox.select(parent_a, 2)
        else:
            if self.number_of_parents > 0:
                # multi-parent generation, which is particularly useful for multi-tree GP
                offspring = []
                offspring.append(
                    self.offspring_generation(
                        toolbox, parent, self.number_of_parents, external_archive
                    )[0]
                )
                offspring.append(
                    self.offspring_generation(
                        toolbox, parent, self.number_of_parents, external_archive
                    )[0]
                )
            else:
                if (
                    isinstance(self.external_archive, int)
                    and self.external_archive >= 1
                ):
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
            if external_archive is None:
                external_archive = []
            for g in range(self.gene_num):
                if len(external_archive) == self.gene_num:
                    selected_features = {str(o[1]): o for o in external_archive[g]}
                else:
                    selected_features = {}
                for p in population:
                    score = p.fitness.wvalues[0] * p.coef[g]
                    gene_str = str(p.gene[g])
                    if gene_str in selected_features:
                        if score > selected_features[gene_str][0]:
                            selected_features[gene_str] = (score, p.gene[g])
                        else:
                            pass
                    else:
                        selected_features[gene_str] = (score, p.gene[g])
                final_archive = list(
                    sorted(
                        selected_features.values(), key=lambda x: x[0], reverse=True
                    )[: self.n_pop]
                )
                if len(external_archive) == self.gene_num:
                    external_archive[g] = final_archive
                else:
                    external_archive.append(final_archive)
        else:
            external_archive = None
        self.elites_archive = external_archive
        return external_archive

    def offspring_generation(self, toolbox, parent, count, external_archive=None):
        # Generate offspring based multiple parents
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
            elif self.map_elite_parameter["type"] == "Grid-Auto":
                elite_maps = []
                for id, parameters in enumerate(
                    [
                        {"fitness_ratio": x, "map_size": y}
                        for x in [0.2, 0.5, 0.8]
                        for y in [5, 10, 15]
                    ]
                ):
                    self.map_elite_parameter.update(parameters)
                    if len(elite_map) > 0:
                        elite_maps.append(
                            selMAPElites(
                                population, elite_map[id], self.map_elite_parameter
                            )
                        )
                    else:
                        elite_maps.append(
                            selMAPElites(population, {}, self.map_elite_parameter)
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
                    self.diversity_calculation(elite_map.values())
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
        model = random.choice(models)
        parent = list(filter(lambda x: x.base_model == model, parent))
        return parent

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
        elif (
            self.environmental_selection in ["NSGA2"]
            and self.base_learner == "RDT~LightGBM-Stump"
        ):
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
        elif self.environmental_selection in ["NSGA2-CV", "NSGA2-CV2"]:
            for ind in offspring + population:
                setattr(ind, "original_fitness", ind.fitness.values)
                fitness = ind.case_values
                if self.environmental_selection == "NSGA2-CV2":
                    fitness = fitness @ np.random.uniform(0, 1, size=(5, 2))
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
            genes = p.gene
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
            p.gene = genes

    def multiobjective_evaluation(self, toolbox, population):
        if self.base_learner == "RDT~LightGBM-Stump":
            second_pop = copy.deepcopy(population)
            for o in population:
                o.base_model = "Random-DT"
            for o in second_pop:
                o.base_model = "LightGBM-Stump"

        self.population_evaluation(toolbox, population)
        self.population_evaluation(toolbox, second_pop)
        population.extend(second_pop)

        for oa, ob in zip(population, second_pop):
            oa.fitness.weights = (-1, -1)
            oa.fitness.values = (oa.fitness.values[0], ob.fitness.values[0])
            ob.fitness.weights = (-1, -1)
            ob.fitness.values = (oa.fitness.values[0], ob.fitness.values[0])
        return population

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
        return invalid_ind

    def cos_distance_calculation(self, population=None):
        inds = self.get_diversity_matrix(population)
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

    def diversity_calculation(self, individuals=None):
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
            "RF-Routing",
        ]:
            all_ind = individuals
        elif self.second_layer in ["DiversityPrune", "TreeBaseline", "GA"]:
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
        if len(self.hof) == 1:
            assert len(self.hof) == 1
            best_ind = self.hof[0]
            return self.single_model(best_ind, mtl_id)
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
        return model_to_string(genes, learner, scaler)

    def pretrain_predict(self, X):
        if self.learner is not None:
            data = []
            for model in self.pretrain_models:
                data.append(model.predict(X).flatten())
            X = np.concatenate([X, np.array(data).T], axis=1)
        return X


def init_worker(function, data):
    function.data = data
