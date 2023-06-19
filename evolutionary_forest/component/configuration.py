import enum
from typing import Union

from deap.gp import PrimitiveSet

semantic_based_bloat_control = ['hoist_mutation']


def check_semantic_based_bc(bloat_control):
    if bloat_control is None:
        return False
    for bc in semantic_based_bloat_control:
        if bc in bloat_control and bloat_control[bc]:
            return True
    return False


class MABConfiguration():
    def __init__(self,
                 mode='Decay',
                 decay_ratio=0.5,
                 selection_operators='AutomaticLexicase,Tournament-7',
                 comparison_criterion='Case',
                 threshold=100,
                 **kwargs
                 ) -> None:
        self.mode = mode
        # Decay or Threshold Mode
        self.decay_ratio = decay_ratio
        # Selection
        self.selection_operators = selection_operators
        # Improvement on any case
        self.comparison_criterion = comparison_criterion
        self.threshold = threshold


class ArchiveConfiguration():
    def __init__(self, dynamic_validation=False, data_combination=True, **params):
        self.dynamic_validation = dynamic_validation
        # Training the final model with training data and validation data
        self.data_combination = data_combination


class CrossoverMode(enum.Enum):
    Independent = enum.auto()
    Sequential = enum.auto()

    @staticmethod
    def from_string(string):
        if isinstance(string, enum.Enum):
            return string
        for c_mode in CrossoverMode:
            if c_mode.name == string:
                return c_mode
        return None


class SelectionMode(enum.Enum):
    MAPElites = enum.auto()
    AngleDriven = enum.auto()

    @staticmethod
    def from_string(string):
        if isinstance(string, enum.Enum):
            return string
        for c_mode in SelectionMode:
            if c_mode.name == string:
                return c_mode
        return None


class CrossoverConfiguration():
    """
    Two possible ways of macro semantic crossover:
    1. Two individuals only apply macro crossover with a certain probability
    2. First apply two rounds of macro crossover to get two parents, then micro crossover
    """

    def __init__(self,
                 intron_parameters=None,
                 # single-tree crossover
                 root_crossover=False,
                 safe_crossover=False,
                 same_index=False,
                 leaf_biased=False,
                 # macro crossover
                 macro_crossover_rate=0,
                 independent_macro_crossover=True,
                 macro_semantic_crossover=False,
                 var_or=False,
                 # semantic macro crossover
                 semantic_crossover_probability=0,
                 semantic_crossover_mode=None,
                 semantic_selection_mode=None,
                 max_height=10,
                 number_of_invokes=0,
                 tree_selection='Random',
                 adaptive_power=1,
                 inverse_point=0.5,
                 sc_tournament_size=3,
                 sc_temperature=1 / 20, **params):
        # max height after crossover
        self.max_height = max_height
        self.intron_parameters: dict = intron_parameters
        self.root_crossover = root_crossover
        # ensure crossover on smaller trees
        self.safe_crossover = safe_crossover
        self.macro_semantic_crossover = macro_semantic_crossover
        # crossover on genes instead of trees
        self.macro_crossover_rate = macro_crossover_rate
        self.independent_macro_crossover = independent_macro_crossover
        # 90% weights to primitives, 10% to terminals
        self.leaf_biased = leaf_biased
        self.same_index = same_index
        self.var_or = var_or
        # semantic feature crossover
        # different from traditional semantic crossover, this semantic crossover is performed on the feature level
        self.semantic_crossover_probability = semantic_crossover_probability
        self.semantic_crossover_mode = CrossoverMode.from_string(semantic_crossover_mode)
        self.semantic_selection_mode = SelectionMode.from_string(semantic_selection_mode)
        self.map_elites_configuration = MAPElitesConfiguration(**params)
        self.number_of_invokes = number_of_invokes
        self.tree_selection = tree_selection

        # adaptive variation rate
        self.adaptive_power = adaptive_power
        self.inverse_point = inverse_point

        # self-competitive crossover
        self.sc_tournament_size = sc_tournament_size
        self.sc_temperature = sc_temperature


class MAPElitesConfiguration():
    def __init__(self,
                 map_elites_random_sample=False,
                 map_elites_bins=10,
                 **params):
        self.map_elites_random_sample = map_elites_random_sample
        self.map_elites_bins = map_elites_bins


class MutationConfiguration():
    def __init__(self,
                 intron_parameters=None,
                 safe_mutation=False,
                 mutation_expr_height=None,
                 max_height=10,
                 gene_add_del_rate=0,
                 **params):
        self.intron_parameters: dict = intron_parameters
        self.gene_addition_rate = gene_add_del_rate
        self.gene_deletion_rate = gene_add_del_rate
        self.safe_mutation = safe_mutation
        self.mutation_expr_height: str = mutation_expr_height
        self.max_height = max_height


class ImbalancedConfiguration():
    def __init__(self,
                 balanced_evaluation=False,
                 balanced_final_training=False,
                 balanced_fitness=False,
                 weight_on_x_space=True,
                 based_on_test=False,
                 **params):
        self.balanced_evaluation = balanced_evaluation
        self.balanced_final_training = balanced_final_training
        self.balanced_fitness = balanced_fitness
        self.weight_on_x_space = weight_on_x_space
        self.based_on_test = based_on_test


class EvaluationConfiguration():
    def __init__(self,
                 nn_prediction=None,
                 dynamic_target=False,
                 original_features=False,
                 test_data_size=0,
                 pset=None,
                 basic_primitives=True,
                 cross_validation=True,
                 feature_importance_method=False,
                 filter_elimination=False,
                 intron_gp=False,
                 bloat_control=None,
                 mini_batch=False,
                 semantic_crossover_probability=0,
                 **params):
        # prediction results of the neural network
        self.nn_prediction = nn_prediction
        # random generate cross-validation scheme
        self.dynamic_target = dynamic_target
        # using original features
        self.original_features = original_features
        # number of test data samples, which do not have the label
        self.test_data_size = test_data_size
        self.pset: Union[PrimitiveSet] = pset
        # check model is in sklearn format or not
        self.basic_primitives = basic_primitives
        self.sklearn_format = self.basic_primitives == 'ML'
        # using 5-fold CV
        self.cross_validation = cross_validation
        # feature importance method (Internal, SHAP, Permutation Importance)
        self.feature_importance_method = feature_importance_method
        # pre-elimination based on filter
        self.filter_elimination = filter_elimination
        self.intron_gp = intron_gp
        # determine mini-batch based on the current generation
        self.mini_batch = mini_batch
        self.batch_size = 32
        self.current_generation = 0
        # save semantic information of each tree
        self.semantic_crossover_probability = semantic_crossover_probability
        if self.semantic_crossover_probability > 0:
            self.save_semantics = True
        else:
            self.save_semantics = False

        # some parameters for bloat control
        self.bloat_control = bloat_control
        lsh_key = None
        self.intron_calculation = check_semantic_based_bc(self.bloat_control) or self.intron_gp
        if self.bloat_control is not None:
            if self.bloat_control.get("lsh_size", None) is None:
                lsh_key = self.bloat_control.get("key_item", 'String') == 'LSH'
            else:
                lsh_key = self.bloat_control["lsh_size"]
        # locality sensitive hashing, a very rare case
        self.lsh = lsh_key

        # weights
        self.sample_weight = None


class BloatControlConfiguration():
    def __init__(self,
                 hoist_before_selection=True,
                 **params):
        self.hoist_before_selection = hoist_before_selection


class BaseLearnerConfiguration():
    def __init__(self, ridge_alpha=1,
                 **params):
        self.ridge_alpha = ridge_alpha
