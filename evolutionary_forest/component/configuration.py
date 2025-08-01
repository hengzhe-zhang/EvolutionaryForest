import enum
from typing import Union

from deap.gp import PrimitiveSet

from evolutionary_forest.component.equation_learner.eql_util import (
    EQLSymbolicRegression,
)

semantic_based_bloat_control = ["hoist_mutation"]


def check_semantic_based_bc(bloat_control):
    if bloat_control is None:
        return False
    for bc in semantic_based_bloat_control:
        if bc in bloat_control and bloat_control[bc]:
            return True
    return False


class Configuration:
    pass


class NoiseConfiguration(Configuration):
    def __init__(
        self,
        noise_type="Normal",
        noise_to_terminal: Union[float, bool] = True,
        noise_normalization="Instance+",
        layer_adaptive=False,
        skip_root=True,
        only_terminal=False,
        sam_mix_bandwidth=1,
        stochastic_mode=False,
        strict_layer_mode=False,
        shuffle_scale=0.2,
        **params,
    ):
        """
        Args:
            noise_type (str, optional): The type of noise to use. Default is 'Normal'.
            noise_to_terminal (bool, optional): Whether to apply noise to the terminal layer. Default is True.
            noise_normalization (str, optional): The type of noise normalization to use. Default is 'Instance'.
            layer_adaptive (bool, optional): Whether to apply different noise to different layers. Default is False.
        """
        self.noise_type = noise_type
        self.noise_to_terminal = noise_to_terminal
        self.noise_normalization = noise_normalization
        self.layer_adaptive = layer_adaptive
        self.skip_root = skip_root
        self.only_terminal = only_terminal
        # Based on mixup to synthesize intermediate noise
        self.sam_mix_bandwidth = sam_mix_bandwidth
        self.stochastic_mode = stochastic_mode
        self.strict_layer_mode = strict_layer_mode

        self.shuffle_scale = shuffle_scale


class MABConfiguration(Configuration):
    def __init__(
        self,
        aos_mode="Decay",
        decay_ratio=0.5,
        selection_operators="AutomaticLexicase,Tournament-7",
        comparison_criterion="Case",
        aos_capping_threshold=100,
        bandit_clip_lower_bound=1e-2,
        **kwargs,
    ) -> None:
        self.aos_mode = aos_mode
        # Decay or Threshold Mode
        self.decay_ratio = decay_ratio
        # Selection
        self.selection_operators = selection_operators
        # Improvement on any case
        self.comparison_criterion = comparison_criterion
        self.aos_capped_threshold = aos_capping_threshold
        self.bandit_clip_lower_bound = bandit_clip_lower_bound


class ArchiveConfiguration(Configuration):
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
    ResXO = enum.auto()
    StageXO = enum.auto()

    @staticmethod
    def from_string(string):
        if isinstance(string, enum.Enum):
            return string
        for c_mode in SelectionMode:
            if c_mode.name == string:
                return c_mode
        return None


class CrossoverConfiguration(Configuration):
    """
    Two possible ways of macro semantic crossover:
    1. Two individuals only apply macro crossover with a certain probability
    2. First apply two rounds of macro crossover to get two parents, then micro crossover
    """

    def __init__(
        self,
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
        tree_selection="Random",
        adaptive_power=1,
        inverse_point=0.5,
        sc_tournament_size=3,
        sc_temperature=1 / 20,
        dimension_crossover_rate=0,
        merge_crossover=False,
        weighted_crossover=False,
        llm_crossover=False,
        llm_pattern_extraction=False,
        **params,
    ):
        # max height after crossover
        self.max_height = max_height
        self.root_crossover = root_crossover
        self.merge_crossover = merge_crossover
        # ensure crossover on smaller trees
        self.safe_crossover = safe_crossover
        self.macro_semantic_crossover = macro_semantic_crossover
        # crossover on genes instead of trees
        self.macro_crossover_rate = macro_crossover_rate
        self.independent_macro_crossover = independent_macro_crossover
        self.dimension_crossover_rate = dimension_crossover_rate
        # 90% weights to primitives, 10% to terminals
        self.leaf_biased = leaf_biased
        self.same_index = same_index
        self.var_or = var_or
        # semantic feature crossover
        # different from traditional semantic crossover, this semantic crossover is performed on the feature level
        self.semantic_crossover_probability = semantic_crossover_probability
        self.semantic_crossover_mode = CrossoverMode.from_string(
            semantic_crossover_mode
        )
        self.semantic_selection_mode = SelectionMode.from_string(
            semantic_selection_mode
        )
        self.map_elites_configuration = MAPElitesConfiguration(**params)
        self.number_of_invokes = number_of_invokes
        self.tree_selection = tree_selection

        # adaptive variation rate
        self.adaptive_power = adaptive_power
        self.inverse_point = inverse_point

        # self-competitive crossover
        self.sc_tournament_size = sc_tournament_size
        self.sc_temperature = sc_temperature

        # for racing selection
        self.weighted_crossover = weighted_crossover

        # LLM
        self.llm_crossover = llm_crossover
        self.llm_pattern_extraction = llm_pattern_extraction
        self.llm_pattern = ""


class MAPElitesConfiguration(Configuration):
    def __init__(self, map_elites_random_sample=False, map_elites_bins=10, **params):
        self.map_elites_random_sample = map_elites_random_sample
        self.map_elites_bins = map_elites_bins


class EQLHybridConfiguration(Configuration):
    def __init__(
        self,
        eql_hybrid=0,
        eql_hybrid_on_pareto_front=0,
        eql_size_limit=50,
        eql_only_initialization=True,
        **params,
    ):
        super().__init__()
        self.eql_hybrid = eql_hybrid
        self.eql_only_initialization = eql_only_initialization
        self.eql_hybrid_on_pareto_front = eql_hybrid_on_pareto_front
        self.eql_size_limit = eql_size_limit
        self.eql_learner = EQLSymbolicRegression(n_layers=1, verbose=True)


class MutationConfiguration(Configuration):
    def __init__(
        self,
        safe_mutation=False,
        mutation_expr_height=None,
        max_height=10,
        gene_addition_rate=0,
        gene_deletion_rate=0,
        weighted_deletion=False,
        addition_or_deletion=True,
        gene_addition_mode="Random",
        pool_based_addition=False,
        redundant_based_deletion=False,
        deletion_strategy="Importance",
        pool_addition_mode="Best",
        library_clustering_mode=False,
        pool_hard_instance_interval=0,
        change_semantic_after_deletion=True,
        include_subtree_to_lib=False,
        handle_objective_duplication=False,
        basic_primitives=None,
        semantic_local_search_pb: int = 1,
        pool_based_replacement_inner_probability=1,
        gene_replacement_rate=0,
        complementary_replacement=0,
        full_scaling_after_replacement=False,
        scaling_before_replacement=False,
        top_k_candidates=10,
        lib_feature_selection=None,
        trial_check=True,
        trial_check_ratio=1,
        lib_feature_selection_mode="Frequency",
        negative_data_augmentation=False,
        negative_local_search=True,
        independent_local_search=True,
        local_search_dropout=0,
        local_search_dropout_trigger=0,
        local_search_dropout_ensemble=False,
        local_search_dropout_bloat=False,
        neural_pool=0,
        neural_pool_start_generation=0,
        neural_pool_num_of_functions=5,
        mix_neural_pool_mode=True,
        neural_pool_continue=True,
        neural_pool_delete_semantics=True,
        neural_pool_greedy=True,
        weight_of_contrastive_learning=0.05,
        neural_pool_transformer_layer=1,
        neural_pool_dropout=0,
        neural_pool_hidden_size=64,
        neural_pool_mlp_layers=3,
        selective_retrain=True,
        retrieval_augmented_generation=True,
        **params,
    ):
        self.pool_hard_instance_interval = pool_hard_instance_interval
        self.library_clustering_mode = library_clustering_mode
        self.gene_addition_rate = gene_addition_rate
        self.gene_deletion_rate = gene_deletion_rate
        self.gene_addition_mode = gene_addition_mode
        self.weighted_deletion = weighted_deletion
        self.safe_mutation = safe_mutation
        self.mutation_expr_height: str = mutation_expr_height
        self.max_height = max_height
        self.addition_or_deletion = addition_or_deletion
        self.pool_based_addition = pool_based_addition
        self.pool_addition_mode = pool_addition_mode
        self.redundant_based_deletion = redundant_based_deletion
        self.deletion_strategy = deletion_strategy
        self.change_semantic_after_deletion = change_semantic_after_deletion

        # handle syntax duplication
        self.handle_objective_duplication = handle_objective_duplication
        self.basic_primitives = basic_primitives
        self.semantic_local_search_pb = semantic_local_search_pb
        self.neural_pool = neural_pool
        self.neural_pool_start_generation = neural_pool_start_generation
        self.pool_based_replacement_inner_probability = (
            pool_based_replacement_inner_probability
        )
        self.gene_replacement_rate = gene_replacement_rate

        """
        Useless parameters
        """
        self.complementary_replacement = complementary_replacement
        self.full_scaling_after_replacement = full_scaling_after_replacement
        self.scaling_before_replacement = scaling_before_replacement

        self.top_k_candidates = top_k_candidates
        self.lib_feature_selection = lib_feature_selection
        self.lib_feature_selection_mode = lib_feature_selection_mode
        self.local_search_dropout = local_search_dropout
        self.local_search_dropout_trigger = local_search_dropout_trigger
        self.local_search_dropout_ensemble = local_search_dropout_ensemble
        self.local_search_dropout_bloat = local_search_dropout_bloat

        # very important components
        self.include_subtree_to_lib = include_subtree_to_lib
        self.trial_check = trial_check
        self.trial_check_ratio = trial_check_ratio
        # negative data augmentation
        self.negative_data_augmentation = negative_data_augmentation
        self.negative_local_search = negative_local_search
        self.independent_local_search = independent_local_search

        """
        Some parameters related to neural pool
        """
        self.mix_neural_pool_mode = mix_neural_pool_mode
        self.neural_pool_continue = neural_pool_continue
        self.neural_pool_delete_semantics = neural_pool_delete_semantics
        self.neural_pool_num_of_functions = neural_pool_num_of_functions
        self.neural_pool_greedy = neural_pool_greedy
        self.weight_of_contrastive_learning = weight_of_contrastive_learning
        self.neural_pool_transformer_layer = neural_pool_transformer_layer
        self.neural_pool_hidden_size = neural_pool_hidden_size
        self.neural_pool_mlp_layers = neural_pool_mlp_layers
        self.neural_pool_dropout = neural_pool_dropout
        self.selective_retrain = selective_retrain
        self.retrieval_augmented_generation = retrieval_augmented_generation


class ImbalancedConfiguration(Configuration):
    def __init__(
        self,
        balanced_evaluation=False,
        balanced_final_training=False,
        balanced_fitness=False,
        weight_on_x_space=True,
        based_on_test=False,
        **params,
    ):
        self.balanced_evaluation = balanced_evaluation
        self.balanced_final_training = balanced_final_training
        self.balanced_fitness = balanced_fitness
        self.weight_on_x_space = weight_on_x_space
        self.based_on_test = based_on_test


class EvaluationConfiguration(Configuration):
    def __init__(
        self,
        dynamic_target=False,
        original_features=False,
        pset=None,
        basic_primitives=True,
        cross_validation=True,
        feature_importance_method=False,
        filter_elimination=False,
        intron_gp=False,
        bloat_control=None,
        mini_batch=False,
        semantic_crossover_probability=0,
        gradient_descent=False,
        transductive_learning=False,
        classification=False,
        max_height=None,
        ood_split=None,
        loss_discretization=None,
        semantic_library=None,
        constant_type=None,
        feature_smoothing=False,
        sample_weight=None,
        minor_sample_weight=0.1,
        two_stage_feature_selection=None,
        save_semantics=False,
        **params,
    ):
        # prediction results of the neural network
        self.feature_smoothing = feature_smoothing
        self.loss_discretization = loss_discretization
        self.ood_split = ood_split
        self.constant_type = constant_type
        # random generate cross-validation scheme
        self.dynamic_target = dynamic_target
        # using original features
        self.original_features = original_features
        self.pset: Union[PrimitiveSet] = pset
        # check model is in sklearn format or not
        self.basic_primitives = basic_primitives
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
        self.save_semantics = save_semantics
        if self.semantic_crossover_probability > 0:
            self.save_semantics = True
        self.gradient_descent = gradient_descent
        self.transductive_learning = transductive_learning

        # some parameters for bloat control
        self.bloat_control = bloat_control
        lsh_key = None
        self.intron_calculation = (
            check_semantic_based_bc(self.bloat_control) or self.intron_gp
        )
        if self.bloat_control is not None:
            if self.bloat_control.get("lsh_size", None) is None:
                lsh_key = self.bloat_control.get("key_item", "String") == "LSH"
            else:
                lsh_key = self.bloat_control["lsh_size"]
        # locality sensitive hashing, a very rare case
        self.lsh = lsh_key
        self.classification = classification

        # weights
        self.sample_weight = sample_weight
        self.minor_sample_weight = minor_sample_weight

        # max height, maybe useful for some evaluation stages
        self.max_height = max_height

        self.semantic_library = semantic_library
        self.enable_library = False

        self.contrastive_loss = None

        # Two Stage Feature Selection
        self.two_stage_feature_selection = two_stage_feature_selection


class BloatControlConfiguration(Configuration):
    def __init__(
        self,
        hoist_before_selection=True,
        lexicase_round=2,
        size_selection="Roulette",
        **params,
    ):
        self.hoist_before_selection = hoist_before_selection
        self.lexicase_round = lexicase_round
        self.size_selection = size_selection


class BaseLearnerConfiguration(Configuration):
    def __init__(self, ridge_alpha=1, **params):
        self.ridge_alpha = ridge_alpha


class ExperimentalConfiguration(Configuration):
    def __init__(self, pac_bayesian_comparison=False, **params) -> None:
        # Here are some configurations only used for experiments
        # This flag is only used for experimental comparison
        self.pac_bayesian_comparison = pac_bayesian_comparison


class DepthLimitConfiguration(Configuration):
    def __init__(
        self,
        max_height: Union[str, int] = 10,  # Maximum height of a GP tree
        min_height: int = 0,  # Minimum height of a GP tree
        **params,
    ):
        if isinstance(max_height, str):
            max_height = int(max_height.split("-")[1])
        self.max_height = max_height
        self.min_height = min_height
        assert self.max_height is not None
        assert self.min_height is not None
