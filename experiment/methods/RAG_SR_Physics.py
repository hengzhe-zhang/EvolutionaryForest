from sympy import parse_expr, preorder_traversal

from benchmark.utils.symbolic_check_utils import model_verification
from evolutionary_forest.forest import EvolutionaryForestRegressor

nn_parameters = {
    # "semantic_local_search_pb": 0.2,
    "neural_pool": 0.1,
    "neural_pool_num_of_functions": 5,
    "weight_of_contrastive_learning": 0.05,
    "neural_pool_dropout": 0.1,
    "neural_pool_transformer_layer": 1,
    "neural_pool_hidden_size": 64,
    "neural_pool_mlp_layers": 3,
    "selective_retrain": True,
    "negative_data_augmentation": True,
    "negative_local_search": False,
}

est = EvolutionaryForestRegressor(
    n_gen=100,
    n_pop=200,
    select="AutomaticLexicase",
    cross_pb=0.9,
    mutation_pb=0.1,
    max_height=10,
    ensemble_size=1,
    initial_tree_size="0-6",
    gene_num=10,
    basic_primitives="Add,Sub,Mul,Div,Sqrt,Square,Sin,Cos",
    base_learner="RidgeCV",
    ridge_alphas="Auto",
    verbose=False,
    boost_size=None,
    normalize=False,
    external_archive=1,
    max_trees=10000,
    library_clustering_mode="Worst",
    pool_addition_mode="Smallest~Auto",
    pool_hard_instance_interval=10,
    random_order_replacement=True,
    pool_based_addition=True,
    semantics_length=50,
    change_semantic_after_deletion=True,
    include_subtree_to_lib=True,
    library_updating_mode="Recent",
    root_crossover=True,
    scaling_before_replacement=False,
    score_func="R2",
    number_of_invokes=0,
    mutation_scheme="uniform-plus",
    environmental_selection=None,
    record_training_data=False,
    complementary_replacement=False,
    validation_size=0,
    constant_type="Float",
    full_scaling_after_replacement=False,
    **nn_parameters,
)


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())


if __name__ == "__main__":
    model_verification(est, complexity)
