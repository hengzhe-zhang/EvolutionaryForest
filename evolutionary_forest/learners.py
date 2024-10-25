from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from evolutionary_forest.forest import EvolutionaryForestRegressor


class RidgeEvolutionaryFeatureLearner(EvolutionaryForestRegressor):
    def __init__(self, early_stop=-1, **params):
        parameter_dict = dict(
            n_gen=100,
            n_pop=200,
            select="AutomaticLexicase",
            cross_pb=0.9,
            mutation_pb=0.1,
            max_height=10,
            ensemble_size=1,
            initial_tree_size="0-6",
            gene_num=10,
            basic_primitives="Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,RSin,RCos,Max,Min,Neg",
            base_learner="RidgeCV",
            verbose=False,
            boost_size=None,
            normalize=True,
            external_archive=1,
            categorical_encoder="Target",
            root_crossover=True,
            score_func="R2",
            number_of_invokes=0,
            mutation_scheme="uniform-plus",
            environmental_selection=None,
            record_training_data=False,
            constant_type="Float",
            # early stopping
            early_stop=early_stop,
        )
        parameter_dict.update(params)
        super().__init__(**parameter_dict)


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    est = RidgeEvolutionaryFeatureLearner(early_stop=5, verbose=True)
    est.fit(train_X, train_y)
    print(est.score(test_X, test_y))
    est = RidgeEvolutionaryFeatureLearner(early_stop=10, verbose=True)
    est.fit(train_X, train_y)
    print(est.score(test_X, test_y))
