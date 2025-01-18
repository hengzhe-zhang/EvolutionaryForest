from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.classifier import EvolutionaryForestClassifier

hyper_params = [
    {},
]

est = EvolutionaryForestClassifier(
    max_height=10,
    normalize=False,
    select="AutomaticLexicase",
    n_gen=50,
    n_pop=200,
    gene_num=10,
    cross_pb=0.9,
    mutation_pb=0.1,
    basic_primitives="optimal",
    mutation_scheme="uniform-plus",
    score_func="CrossEntropy",
    verbose=True,
    base_learner="LogisticRegression",
    class_weight="Balanced",
)


def complexity(est: EvolutionaryForestRegressor):
    return est.complexity()


model = None

if __name__ == "__main__":
    # Test the complexity of function
    X, y = load_iris(return_X_y=True)
    print(cross_val_score(LGBMClassifier(), X, y, n_jobs=-1))
    print(cross_val_score(est, X, y, n_jobs=-1))
