from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sympy import parse_expr, preorder_traversal

from evolutionary_forest.forest import EvolutionaryForestRegressor

est = EvolutionaryForestRegressor(
    n_gen=100,
    n_pop=1000,
    select="AutomaticLexicase",
    cross_pb=0.9,
    mutation_pb=0.1,
    max_height=10,
    boost_size=1,
    initial_tree_size="0-6",
    gene_num=10,
    mutation_scheme="EDA-Terminal-PM",
    basic_primitives="optimal",
    base_learner="RidgeCV",
    verbose=True,
    normalize=True,
    external_archive=10,
    bloat_control={
        "hoist_mutation": True,
        "hoist_probability": 1,
        "iteratively_check": True,
        "key_item": "String",
    },
)


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())


if __name__ == "__main__":
    # Test the complexity function
    X, y = load_diabetes(return_X_y=True)
    print(cross_val_score(est, X, y, n_jobs=-1))
    est.fit(X, y)
    print(complexity(est))
