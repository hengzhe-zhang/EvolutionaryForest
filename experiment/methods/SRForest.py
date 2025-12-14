from sklearn.datasets import load_diabetes

from evolutionary_forest.forest import EvolutionaryForestRegressor

hyper_params = [
    {},
]

est = EvolutionaryForestRegressor(
    max_height=8,
    normalize=True,
    select="AutomaticLexicase",
    boost_size=100,
    basic_primitives="optimal",
    mutation_scheme="EDA-Terminal-PM",
    semantic_diversity="GreedySelection-Resampling",
    initial_tree_size="2-6",
    cross_pb=0.9,
    mutation_pb=0.1,
    ps_tree_ratio=0.2,
    gene_num=20,
    n_gen=100,
    n_pop=200,
    base_learner="Fast-RidgeDT-Plus",
)


def complexity(est: EvolutionaryForestRegressor):
    return est.complexity()


model = None

if __name__ == "__main__":
    # Test the complexity function
    X, y = load_diabetes(return_X_y=True)
    est.fit(X, y)
    print(complexity(est))
