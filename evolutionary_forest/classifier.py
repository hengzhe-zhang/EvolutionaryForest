import copy
import random

import numpy as np
from lightgbm import LGBMClassifier
from numpy.linalg import norm
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight
from tpot import TPOTClassifier

from evolutionary_forest.component.archive import EnsembleSelectionHallOfFame
from evolutionary_forest.component.evaluation import multi_tree_evaluation
from evolutionary_forest.component.fitness import Fitness
from evolutionary_forest.component.primitive_functions import individual_to_tuple
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.component.ensemble_learning.utils import GBDTLRClassifierX
from evolutionary_forest.model.PLTree import LRDTClassifier
from evolutionary_forest.model.SafetyLR import SafetyLogisticRegression
from evolutionary_forest.model.SafetyScaler import SafetyScaler
from evolutionary_forest.utility.classification_utils import calculate_cross_entropy
from evolutionary_forest.utils import save_array


class EvolutionaryForestClassifier(ClassifierMixin, EvolutionaryForestRegressor):
    def __init__(
        self,
        score_func="ZeroOne",
        # Balanced Classification
        class_weight="Balanced",
        **params
    ):
        super().__init__(score_func=score_func, **params)
        # Define a function that simply passes the input data through unchanged.
        identity_func = lambda x: x
        # Use FunctionTransformer to create a transformer object that applies the identity_func to input data.
        identity_transformer = FunctionTransformer(identity_func)
        self.y_scaler = identity_transformer
        self.class_weight = class_weight

        if self.base_learner == "Hybrid":
            config_dict = {
                "sklearn.linear_model.LogisticRegression": {
                    "penalty": ["l2"],
                    "C": [
                        1e-4,
                        1e-3,
                        1e-2,
                        1e-1,
                        0.5,
                        1.0,
                        5.0,
                        10.0,
                        15.0,
                        20.0,
                        25.0,
                    ],
                    "solver": ["liblinear"],
                },
                "sklearn.tree.DecisionTreeClassifier": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": range(1, 11),
                    "min_samples_split": range(2, 21),
                    "min_samples_leaf": range(1, 21),
                },
            }
            self.tpot_model = TPOTClassifier(
                config_dict=config_dict, template="Classifier"
            )
            self.tpot_model._fit_init()
        else:
            self.tpot_model = None

    def get_params(self, deep=True):
        params = super().get_params(deep)
        # Hack to make get_params return base class params...
        cp = copy.copy(self)
        cp.__class__ = EvolutionaryForestRegressor
        params.update(EvolutionaryForestRegressor.get_params(cp, deep))
        return params

    def oob_error(self, pop):
        # how to calculate the prediction result?
        count = np.zeros(len(self.y))
        # prediction = np.zeros((len(self.y), len(np.unique(self.y))))
        prediction = np.full((len(self.y), len(np.unique(self.y))), 0, dtype=np.float)
        for i, x in enumerate(pop):
            index = x.out_of_bag
            count[index] += 1
            prediction[index] += x.oob_prediction
        # label = stats.mode(np.array(prediction), axis=0, nan_policy='omit')[0].flatten()
        classes = pop[0].pipe["Ridge"].classes_
        label = classes.take(np.argmax(prediction, axis=1), axis=0)
        accuracy = accuracy_score(self.y, label)
        if self.verbose:
            print("oob score", accuracy)
        return accuracy

    def predict_proba(self, X):
        if self.normalize:
            X = self.x_scaler.transform(X)
        prediction_data_size = X.shape[0]
        if self.test_data is not None:
            X = np.concatenate([self.X, X])
        self.final_model_lazy_training(self.hof)
        selection_flag = np.ones(len(self.hof), dtype=bool)
        predictions = []
        weight_list = []
        for i, individual in enumerate(self.hof):
            Yp = multi_tree_evaluation(individual.gene, self.pset, X)
            if self.test_data is not None:
                Yp = Yp[-prediction_data_size:]
            predicted = individual.pipe.predict_proba(Yp)

            if (
                hasattr(self.hof, "loss_function")
                and self.hof.loss_function == "ZeroOne"
            ):
                # zero-one loss
                argmax = np.argmax(predicted, axis=1)
                predicted[:] = 0
                predicted[np.arange(0, len(predicted)), argmax] = 1

            if not np.all(np.isfinite(predicted)):
                # skip prediction results containing NaN
                selection_flag[i] = False
                continue
            predictions.append(predicted)
            if (
                hasattr(self.hof, "ensemble_weight")
                and len(self.hof.ensemble_weight) > 0
            ):
                weight_list.append(
                    self.hof.ensemble_weight[individual_to_tuple(individual)]
                )

        if self.second_layer != "None" and self.second_layer != None:
            assert len(self.hof) == len(self.tree_weight)
            predictions = np.array(predictions).T
            return (predictions @ self.tree_weight[selection_flag]).T
        elif len(weight_list) > 0:
            predictions = np.array(predictions).T
            weight_list = np.array(weight_list)
            return (predictions @ (weight_list / weight_list.sum())).T
        else:
            return np.mean(predictions, axis=0)

    def lazy_init(self, x):
        # sometimes, labels are not from 0-n-1, need to process
        self.order_encoder = LabelEncoder()
        self.y = self.order_encoder.fit_transform(self.y)
        # encoding target labels
        self.label_encoder = OneHotEncoder(sparse_output=False)
        self.label_encoder.fit(self.y.reshape(-1, 1))
        super().lazy_init(x)
        if self.class_weight == "Balanced":
            self.class_weight = compute_sample_weight(class_weight="balanced", y=self.y)
            if hasattr(self.hof, "class_weight"):
                self.hof.class_weight = np.reshape(self.class_weight, (-1, 1))
        if hasattr(self.hof, "task_type"):
            self.hof.task_type = "Classification"
        if isinstance(self.hof, EnsembleSelectionHallOfFame):
            self.hof.categories = len(np.unique(self.y))
            self.hof.label = self.label_encoder.transform(self.y.reshape(-1, 1))
        if isinstance(self.score_func, Fitness):
            self.score_func.classification = True
            self.pac_bayesian.classification = True
            self.score_func.instance_weights = self.class_weight

    def entropy_calculation(self):
        if self.score_func == "NoveltySearch":
            ensemble_value = np.mean([x.predicted_values for x in self.hof], axis=0)
            self.ensemble_value = ensemble_value
            return self.ensemble_value

    def calculate_case_values(self, individual, Y, y_pred):
        # Minimize case values
        if self.score_func == "NoveltySearch":
            Y = self.label_encoder.transform(Y.reshape(-1, 1))

        # smaller is better
        if self.score_func == "ZeroOne" or self.score_func == "ZeroOne-NodeCount":
            if len(y_pred.shape) == 2:
                y_pred = np.argmax(y_pred, axis=1)
            individual.case_values = -1 * (y_pred == Y)
        elif self.score_func == "CDFC":
            matrix = confusion_matrix(Y.flatten(), y_pred.flatten())
            score = matrix.diagonal() / matrix.sum(axis=1)
            individual.case_values = -1 * score
        elif (
            self.score_func == "CrossEntropy"
            or self.score_func == "NoveltySearch"
            or isinstance(self.score_func, Fitness)
        ):
            target_label = self.y
            cross_entropy = calculate_cross_entropy(target_label, y_pred)
            individual.case_values = cross_entropy
            assert not np.any(np.isnan(individual.case_values)), save_array(
                individual.case_values
            )
            assert np.size(individual.case_values) == np.size(self.y)
        elif "CV" in self.score_func:
            individual.case_values = -1 * y_pred
        else:
            raise Exception

        if self.score_func == "NoveltySearch":
            individual.original_case_values = individual.case_values
            # KL-Divergence for regularization
            if len(self.hof) != 0:
                """
                Cooperation vs Diversity:
                "Diversity with Cooperation: Ensemble Methods for Few-Shot Classification" ICCV 2019
                """
                ensemble_value = self.ensemble_value
                if self.diversity_metric == "CosineSimilarity":
                    # cosine similarity
                    # larger indicates similar
                    eps = 1e-15
                    ambiguity = np.sum((y_pred * ensemble_value), axis=1) / np.maximum(
                        norm(y_pred, axis=1) * norm(ensemble_value, axis=1), eps
                    )
                    assert not (
                        np.isnan(ambiguity).any() or np.isinf(ambiguity).any()
                    ), save_array((ambiguity, y_pred, ensemble_value))
                elif self.diversity_metric == "KL-Divergence":
                    # smaller indicates similar
                    kl_divergence = lambda a, b: np.sum(
                        np.nan_to_num(a * np.log(a / b), posinf=0, neginf=0), axis=1
                    )
                    ambiguity = (
                        kl_divergence(y_pred, ensemble_value)
                        + kl_divergence(ensemble_value, y_pred)
                    ) / 2
                    assert not (
                        np.isnan(ambiguity).any() or np.isinf(ambiguity).any()
                    ), save_array((ambiguity, y_pred, ensemble_value))
                    ambiguity *= -1
                else:
                    raise Exception
                ambiguity *= self.novelty_weight
                if self.ensemble_cooperation:
                    individual.case_values = individual.case_values - ambiguity
                else:
                    individual.case_values = individual.case_values + ambiguity
                assert not (
                    np.isnan(individual.case_values).any()
                    or np.isinf(individual.case_values).any()
                )

        if self.class_weight is not None:
            individual.case_values = individual.case_values * self.class_weight

    def get_diversity_matrix(self, all_ind):
        inds = []
        for p in all_ind:
            y_pred_one_hot = p.predicted_values
            inds.append(y_pred_one_hot.flatten())
        inds = np.array(inds)
        return inds

    def calculate_fitness_value(self, individual, estimators, Y, y_pred):
        # smaller is better, similar to negative R^2
        if self.score_func == "ZeroOne" or self.score_func == "CDFC":
            # larger is better
            if self.class_weight is not None:
                score = np.mean((y_pred.argmax(axis=1) == Y) * self.class_weight)
            else:
                score = np.mean(y_pred.argmax(axis=1) == Y)
            if self.weighted_coef:
                individual.coef = np.array(individual.coef) * (score)
            return (-1 * score,)
        elif self.score_func == "NoveltySearch":
            return (np.mean(individual.original_case_values),)
        elif self.score_func == "CrossEntropy":
            # weight is already included in case values
            return (np.mean(individual.case_values),)
        elif (
            self.score_func == "CV-NodeCount" or self.score_func == "ZeroOne-NodeCount"
        ):
            score = -1 * np.sum(y_pred.flatten() == Y.flatten())
            return (
                np.mean(
                    [
                        estimators[i]["Ridge"].tree_.node_count
                        for i in range(len(estimators))
                    ]
                ),
            )
        elif isinstance(self.score_func, Fitness):
            return self.score_func.fitness_value(individual, estimators, Y, y_pred)
        elif "CV" in self.score_func:
            return (-1 * np.mean(y_pred),)
        else:
            raise Exception

    def predict(self, X, return_std=False):
        predictions = self.predict_proba(X)
        assert np.all(np.sum(predictions, axis=1))
        argmax_predictions = np.argmax(predictions, axis=1)
        return self.order_encoder.inverse_transform(argmax_predictions)

    def get_base_model(self, regularization_ratio=1, base_model=None, **kwargs):
        base_model_str = base_model if isinstance(base_model, str) else ""
        if (
            self.base_learner == "DT"
            or self.base_learner == "PCA-DT"
            or self.base_learner == "Dynamic-DT"
            or base_model == "DT"
        ):
            ridge_model = DecisionTreeClassifier(
                max_depth=self.max_tree_depth, min_samples_leaf=self.min_samples_leaf
            )
        elif self.base_learner == "Balanced-DT" or base_model == "Balanced-DT":
            ridge_model = DecisionTreeClassifier(
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
            )
        elif self.base_learner == "DT-SQRT":
            ridge_model = DecisionTreeClassifier(
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
            )
        elif (
            self.base_learner == "LogisticRegression"
            or self.base_learner == "Fast-LRDT"
            or base_model_str == "LogisticRegression"
        ):
            ridge_model = SafetyLogisticRegression(
                max_iter=1000, solver="liblinear", random_state=0
            )
        elif (
            self.base_learner == "Balanced-LogisticRegression"
            or base_model_str == "Balanced-LogisticRegression"
        ):
            ridge_model = SafetyLogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=0,
            )
        elif self.base_learner == "Dynamic-LogisticRegression":
            ridge_model = LogisticRegression(
                C=regularization_ratio, max_iter=1000, solver="liblinear"
            )
        elif self.base_learner == "GBDT-PL":
            ridge_model = LGBMClassifier(
                n_estimators=10, learning_rate=1, max_depth=3, linear_tree=True
            )
        elif self.base_learner == "GBDT-LR":
            ridge_model = GBDTLRClassifierX(
                LGBMClassifier(n_estimators=10, learning_rate=1, max_depth=3),
                SafetyLogisticRegression(max_iter=1000, solver="liblinear"),
            )
        elif self.base_learner == "LightGBM":
            parameter_grid = {
                "learning_rate": [0.5],
                "n_estimators": [10],
            }
            parameter = random.choice(list(ParameterGrid(parameter_grid)))
            ridge_model = LGBMClassifier(
                **parameter, extra_trees=True, num_leaves=63, n_jobs=1
            )
        elif self.base_learner == "DT-Criterion":
            ridge_model = DecisionTreeClassifier(
                criterion=random.choice(["entropy", "gini"]),
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif (
            self.base_learner == "Random-DT"
            or self.base_learner == "Random-DT-Plus"
            or base_model == "Random-DT"
        ):
            ridge_model = DecisionTreeClassifier(
                splitter="random",
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif (
            self.base_learner == "Balanced-Random-DT"
            or base_model == "Balanced-Random-DT"
        ):
            ridge_model = DecisionTreeClassifier(
                splitter="random",
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
            )
        elif self.base_learner == "Random-DT-Criterion":
            ridge_model = DecisionTreeClassifier(
                criterion=random.choice(["entropy", "gini"]),
                splitter="random",
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif self.base_learner == "LogisticRegressionCV":
            ridge_model = LogisticRegressionCV(solver="liblinear")
        elif self.base_learner == "Random-DT-SQRT":
            ridge_model = DecisionTreeClassifier(
                splitter="random",
                max_depth=self.max_tree_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
            )
        elif self.base_learner == "LRDT":
            ridge_model = LRDTClassifier(
                decision_tree_count=self.decision_tree_count,
                max_leaf_nodes=self.max_leaf_nodes,
            )
        elif self.base_learner == "Hybrid":
            # extract base model
            ridge_model = self.tpot_model._compile_to_sklearn(base_model)[-1]
        elif isinstance(self.base_learner, ClassifierMixin):
            ridge_model = self.base_learner
        elif base_model_str != "" and base_model_str in self.base_model_dict:
            ridge_model = self.base_model_dict[base_model_str]
        elif self.base_learner != "" and self.base_learner in self.base_model_dict:
            ridge_model = self.base_model_dict[self.base_learner]
        else:
            raise Exception
        if self.base_learner == "PCA-DT":
            pipe = Pipeline(
                [
                    ("Scaler", StandardScaler()),
                    ("PCA", PCA()),
                    ("Ridge", ridge_model),
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ("Scaler", SafetyScaler()),
                    ("Ridge", ridge_model),
                ]
            )
        return pipe
