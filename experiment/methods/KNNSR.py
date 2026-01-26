# GSMX-GP

from benchmark.utils.symbolic_check_utils import model_verification
from evolutionary_forest.forest import EvolutionaryForestRegressor

est = EvolutionaryForestRegressor(
    n_gen=50,
    n_pop=200,
    select="DoubleLexicase",
    cross_pb=0.9,
    mutation_pb=0.1,
    max_height=10,
    ensemble_size=30,
    initial_tree_size="0-2",
    gene_num=10,
    basic_primitives="Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,SinPi,CosPi,Max,Min,Neg",
    base_learner="RidgeBoosted-LinearRankKNN",
    ensemble_selection="DPPHOF",
    verbose=False,
    normalize=True,
    external_archive=1,
    categorical_encoding="Target",
    root_crossover=True,
    check_constants=True,
    score_func="R2",
    number_of_invokes=0,
    mutation_scheme="uniform-plus",
    environmental_selection=None,
    record_training_data=False,
    validation_size=0,
    constant_type="Float",
    n_neighbors="Adaptive",
    cross_validation=False,
    allow_revisit=True,
    eql_hybrid=1,
    cv=None,
    contrastive_l2_regularization=1e-1,
)


def complexity(est: EvolutionaryForestRegressor):
    return est.complexity()


model = None

if __name__ == "__main__":
    model_verification(est, complexity)
