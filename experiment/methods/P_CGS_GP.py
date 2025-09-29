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
    select="CPSR-Correlation",
    cross_pb=0.9,
    mutation_pb=0.1,
    max_height=10,
    boost_size=1,
    initial_tree_size="2-6",
    gene_num=1,
    mutation_scheme="uniform-plus",
    basic_primitives="Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,SinPi,CosPi,Max,Min,Neg",
    base_learner="RidgeCV",
    verbose=False,
    normalize=True,
    external_archive=10,
    # GP parameters
    root_crossover=True,
    check_constants=True,
    number_of_invokes=1,
    constant_type="Float",
    # PAC-Bayesian parameters
    score_func="R2",
    environmental_selection=None,
    record_parent_expressions=True,
)


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(simplify(parse_expr(est.model())))))


def model(est):
    return str(est.model())


if __name__ == "__main__":
    model_verification(est, complexity)
