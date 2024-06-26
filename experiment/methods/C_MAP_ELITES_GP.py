from sympy import parse_expr
from sympy import preorder_traversal

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
    boost_size=None,
    ensemble_size=100,
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
    base_learner="RidgeCV",
    verbose=False,
    normalize=True,
    external_archive=1,
    # GP parameters
    number_of_invokes=0,
    bounded_prediction=True,
    score_func="R2",
    categorical_encoder="TargetEncoder",
    clustering_method="KMeans-Cosine",
    log_item="AverageLoss,Ambiguity",
    environmental_selection=None,
    ensemble_selection="CVT-MAPElitesHOF",
    map_archive_candidate_size=100,
    gene_addition_rate=0.5,
    gene_deletion_rate=0.5,
)


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())


if __name__ == "__main__":
    model_verification(est, complexity)
