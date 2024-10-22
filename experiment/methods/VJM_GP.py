from sympy import preorder_traversal, simplify
from sympy import parse_expr

from benchmark.utils.symbolic_check_utils import model_verification
from evolutionary_forest.forest import EvolutionaryForestRegressor

hyper_params = [
    {},
]

est = EvolutionaryForestRegressor(
    n_gen=100,
    n_pop=200,
    select="AutomaticLexicase",
    cross_pb=0.9,
    mutation_pb=0.1,
    max_height=10,
    boost_size=1,
    initial_tree_size="0-3",
    gene_num=10,
    mutation_scheme="uniform-plus",
    basic_primitives=",".join(
        [
            "Add",
            "Sub",
            "Mul",
            "AQ",
            "ALog",
            "Square",
            "Abs",
            "Sqrt",
            "Neg",
            "Max",
            "Min",
            "RSin",
            "RCos",
        ]
    ),
    # base_learner="RidgeCV",
    base_learner="Bounded-RidgeCV",
    verbose=False,
    normalize=True,
    external_archive=None,
    # GP parameters
    root_crossover=True,
    number_of_invokes=1,
    constant_type="Float",
    bounded_prediction=True,
    # PAC-Bayesian parameters
    score_func="R2-VRM",
    sharpness_type="DataGPSource",
    objective_normalization=True,
    # alpha value in Beta-distribution, which is used to determine the mixing ratio
    perturbation_std=10,
    gene_deletion_rate=0.5,
    gene_addition_rate=0.5,
    environmental_selection="NSGA2",
    objective="R2,MaxSharpness-1~",
    knee_point="SUM",
    sharpness_distribution="I-MixUp",
    sharpness_iterations=10,
    mixup_bandwidth=0.5,
    intelligent_decision=False,
    adaptive_knee_point_metric="Adaptive",
    mixup_mode="Adaptive-ET",
)


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(simplify(parse_expr(est.model())))))


def model(est):
    return str(est.model())


if __name__ == "__main__":
    """
    OpenML Dataset 201 has constant features, these features are removed.
    """
    model_verification(est, complexity)
