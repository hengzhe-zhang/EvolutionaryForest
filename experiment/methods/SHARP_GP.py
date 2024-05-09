from sympy import parse_expr
from sympy import preorder_traversal

from evolutionary_forest.forest import EvolutionaryForestRegressor

hyper_params = [
    {},
]

all_primitives = ",".join(
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
        "Sigmoid",
    ]
)

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
    basic_primitives=f"Pipeline-{all_primitives}-Smooth",
    base_learner="RidgeCV",
    verbose=False,
    normalize=True,
    external_archive=None,
    # GP parameters
    root_crossover=True,
    number_of_invokes=1,
    constant_type=None,
    bounded_prediction=True,
    # PAC-Bayesian parameters
    score_func="R2-PAC-Bayesian",
    sharpness_type="Parameter",
    objective_normalization=True,
    perturbation_std=0.3,
    gene_deletion_rate=0.5,
    gene_addition_rate=0.5,
    environmental_selection="NSGA2",
    objective="R2,MaxSharpness-1-Base",
    knee_point="SAM",
    cached_sharpness=True,
    sharpness_iterations=10,
)


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())
