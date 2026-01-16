from sympy import parse_expr, preorder_traversal

from evolutionary_forest.adaptive_selection_regressor import AdaptiveSelectionRegressor

est = AdaptiveSelectionRegressor(
    n_gen=100,
    n_pop=1000,
    select="CPSR-Correlation",
    cross_pb=0.9,
    mutation_pb=0.1,
    max_height=10,
    boost_size=1,
    initial_tree_size="0-6",
    gene_num=10,
    mutation_scheme="EDA-Terminal-PM",
    basic_primitives="optimal",
    base_learner="RidgeCV",
    verbose=False,
    normalize=True,
    external_archive=10,
    bloat_control={
        "hoist_mutation": True,
        "hoist_probability": 1,
        "iteratively_check": True,
        "key_item": "String",
    },
)


def complexity(est: AdaptiveSelectionRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())


if __name__ == "__main__":
    from benchmark.utils.symbolic_check_utils import model_verification

    model_verification(est, complexity)
