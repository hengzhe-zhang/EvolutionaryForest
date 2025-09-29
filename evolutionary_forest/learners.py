from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from evolutionary_forest.forest import EvolutionaryForestRegressor


class LASGPLearner(EvolutionaryForestRegressor):
    def __init__(
        self,
        gene_num=10,
        basic_primitives="Add,Sub,Mul,Div,Sqrt,AbsLog,Abs,Square,SinPi,CosPi,Max,Min,Neg",
        **params,
    ):
        parameter_dict = dict(
            n_gen=100,
            n_pop=200,
            select="AutomaticLexicase",
            cross_pb=0.9,
            mutation_pb=0.1,
            max_height=10,
            ensemble_size=1,
            initial_tree_size="0-6",
            gene_num=gene_num,
            basic_primitives=basic_primitives,
            base_learner="RidgeCV",
            verbose=False,
            boost_size=None,
            normalize="MinMax",
            external_archive=1,
            max_trees=10000,
            library_clustering_mode="Hardest-KMeans",
            pool_addition_mode="Smallest~Auto",
            pool_hard_instance_interval=10,
            random_order_replacement=True,
            pool_based_addition=True,
            semantics_length=50,
            change_semantic_after_deletion=True,
            include_subtree_to_lib=True,
            library_updating_mode="Recent",
            root_crossover=True,
            scaling_before_replacement=False,
            score_func="R2",
            number_of_invokes=0,
            mutation_scheme="EDA-SemanticLibrary",
            environmental_selection=None,
            record_training_data=False,
            complementary_replacement=False,
            validation_size=0,
            constant_type="Float",
            full_scaling_after_replacement=False,
            semantic_local_search_pb=0.2,
        )
        parameter_dict.update(params)
        super().__init__(**parameter_dict)


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
            basic_primitives="Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,SinPi,CosPi,Max,Min,Neg",
            base_learner="RidgeCV",
            verbose=False,
            boost_size=None,
            normalize=True,
            external_archive=1,
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
        self.parameter_dict = parameter_dict
        super().__init__(**parameter_dict)


class EnsembleLearner(RidgeEvolutionaryFeatureLearner):
    def __init__(
        self,
        ensemble_size=100,
        ps_tree_ratio=0.2,
        base_learner="Fast-RidgeBDT-Plus",
        **params,
    ):
        parameter_dict = dict(
            ensemble_size=ensemble_size,
            ps_tree_ratio=ps_tree_ratio,
            base_learner=base_learner,
        )
        parameter_dict.update(params)
        self.parameter_dict = parameter_dict
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
