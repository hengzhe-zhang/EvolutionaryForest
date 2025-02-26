from sklearn import clone
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sympy import preorder_traversal
from sympy import symbols, lambdify, parse_expr

from evolutionary_forest.forest import EvolutionaryForestRegressor

hyper_params = [
    {},
]

est = EvolutionaryForestRegressor(
    n_gen=200,
    n_pop=200,
    select="AutomaticLexicase",
    cross_pb=0.9,
    mutation_pb=0.1,
    max_height=10,
    boost_size=1,
    initial_tree_size="0-3",
    gene_num=20,
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
    base_learner="RidgeCV",
    verbose=False,
    normalize=True,
    external_archive=1,
    # GP parameters
    root_crossover=True,
    number_of_invokes=1,
    constant_type="Float",
    bounded_prediction=True,
    gene_deletion_rate=0.5,
    gene_addition_rate=0.5,
    sharpness_type="Parameter",
    # Variance Reduction parameters
    score_func="R2-VarianceReduction",
    objective_normalization=True,
    perturbation_std=0.5,
    environmental_selection="NSGA2",
    objective="R2,MeanVariance",
    knee_point="SUM",
)


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())
