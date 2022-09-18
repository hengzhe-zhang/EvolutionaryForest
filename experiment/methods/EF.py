from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

from evolutionary_forest.forest import EvolutionaryForestRegressor

hyper_params = [
    {
    },
]

"""
Removed components:
1. Random-DT as base learner
2. No ensemble selection
3. Traditional mutation operators
"""
est = EvolutionaryForestRegressor(max_height=8, normalize=True, select='AutomaticLexicase', boost_size=100,
                                  basic_primitives='sin-cos', mutation_scheme='uniform-plus',
                                  initial_tree_size='2-6', cross_pb=0.9, mutation_pb=0.1, gene_num=20, n_gen=100,
                                  n_pop=200, base_learner='Random-DT')


def complexity(est: EvolutionaryForestRegressor):
    return est.complexity()


model = None

if __name__ == '__main__':
    # Test the complexity function
    X, y = load_boston(return_X_y=True)
    print(cross_val_score(est, X, y, n_jobs=-1))
    est.fit(X, y)
    print(complexity(est))
