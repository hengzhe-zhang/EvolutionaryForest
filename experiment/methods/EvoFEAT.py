from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from evolutionary_forest.forest import EvolutionaryForestRegressor, EvolutionaryForestClassifier

hyper_params = [
    {
    },
]

est = EvolutionaryForestClassifier(max_height=8, normalize=True, select='AutomaticLexicase', boost_size=100,
                                   n_gen=30, n_pop=200, gene_num=30, basic_primitives='optimal',
                                   cross_pb=0.8, mutation_pb=0.05, mutation_scheme='EDA-Terminal-PM-SC',
                                   score_func='CrossEntropy', verbose=False, base_learner='Balanced-RDT-LR',
                                   semantic_diversity='GreedySelection-Resampling', class_weight='Balanced')


def complexity(est: EvolutionaryForestRegressor):
    return est.complexity()


model = None

if __name__ == '__main__':
    # Test the complexity of function
    X, y = load_iris(return_X_y=True)
    print(cross_val_score(est, X, y, n_jobs=-1))
    est.fit(X, y)
    print(complexity(est))
