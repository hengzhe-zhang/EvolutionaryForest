import enum
import logging
import random
import time
from functools import lru_cache, wraps
from typing import List, Tuple

import dill
import joblib
import numpy as np
import shap
import torch
from cachetools import cached, LRUCache
from cachetools.keys import hashkey
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap.gp import PrimitiveTree, Primitive, Terminal
from numpy.testing import assert_almost_equal
from onedal.primitives import rbf_kernel
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, truncnorm
from sklearn import model_selection
from sklearn.base import RegressorMixin, ClassifierMixin, clone
from sklearn.datasets import make_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import (
    accuracy_score,
    r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import BaseDecisionTree

from evolutionary_forest.component.configuration import (
    EvaluationConfiguration,
    ImbalancedConfiguration,
    NoiseConfiguration,
)
from evolutionary_forest.component.generalization.cache.sharpness_memory import (
    TreeLRUCache,
)
from evolutionary_forest.component.generalization.local_sensitive_shuffle import (
    local_sensitive_shuffle_by_value,
)
from evolutionary_forest.component.generalization.pac_utils.random_node_selection import (
    select_one_node_per_path,
)
from evolutionary_forest.component.gradient_optimization.gradient_descent import (
    gradient_optimization,
)
from evolutionary_forest.component.post_processing.value_alignment import quick_fill
from evolutionary_forest.component.stgp.shared_type import (
    CategoricalFeature,
    FeatureLayer,
)
from evolutionary_forest.component.stgp.strongly_type_gp_utility import (
    multi_tree_evaluation_typed,
)
from evolutionary_forest.component.tree_utils import (
    node_depths_top_down,
)
from evolutionary_forest.model.MTL import MTLRidgeCV, MTLLassoCV
from evolutionary_forest.model.MixupPredictor import MixupRegressor
from evolutionary_forest.model.OptimalKNN import (
    WeightedKNNWithGP,
    WeightedKNNWithGPRidge,
)
from evolutionary_forest.model.RidgeGCV import RidgeGCV
from evolutionary_forest.multigene_gp import (
    result_post_process,
    MultiplePrimitiveSet,
    GPPipeline,
    IndividualConfiguration,
)
from evolutionary_forest.sklearn_utils import cross_val_predict
from evolutionary_forest.utility.ood_split import OutOfDistributionSplit
from evolutionary_forest.utility.sampling_utils import (
    sample_according_to_distance,
)
from evolutionary_forest.utils import (
    reset_random,
    cv_prediction_from_ridge,
    one_hot_encode,
)

np.seterr(invalid="ignore")
reset_random(0)
time_flag = False


def select_from_array(arr, s, x):
    s = s % len(arr)
    if s + x <= len(arr):
        return arr[s : s + x]
    else:
        return np.concatenate((arr[s : s + x], arr[: (s + x) % len(arr)]))


class EvaluationResults:
    def __init__(
        self,
        gp_evaluation_time=None,
        ml_evaluation_time=None,
        hash_result=None,
        correlation_results=None,
        introns_results=None,
        semantic_results=None,
    ):
        self.gp_evaluation_time: int = gp_evaluation_time
        self.ml_evaluation_time: int = ml_evaluation_time
        # hash value of all features
        self.hash_result: list = hash_result
        # correlation of all features
        self.correlation_results: list = correlation_results
        # correlation of each node
        self.introns_results: list = introns_results
        # semantic of each features
        self.semantic_results: list = semantic_results


def calculate_score(args):
    (X, Y, score_func, cv, configuration) = calculate_score.data
    configuration: EvaluationConfiguration
    configuration.enable_library = True
    dynamic_target = configuration.dynamic_target
    original_features = configuration.original_features
    pset = configuration.pset
    feature_importance_method = configuration.feature_importance_method
    filter_elimination = configuration.filter_elimination
    intron_calculation = configuration.intron_calculation
    sample_weight = configuration.sample_weight
    transductive_learning = configuration.transductive_learning

    if configuration.mini_batch:
        X = select_from_array(
            X,
            configuration.current_generation * configuration.batch_size // 4,
            configuration.batch_size,
        )
        Y = select_from_array(
            Y,
            configuration.current_generation * configuration.batch_size // 4,
            configuration.batch_size,
        )

    pipe: GPPipeline
    individual_configuration: IndividualConfiguration
    pipe, func, individual_configuration = args

    if not isinstance(func, list):
        func = dill.loads(func)

    if isinstance(sample_weight, str) and sample_weight.startswith("Adaptive"):
        sample_weight = individual_configuration.sample_weight

    # GP evaluation
    start_time = time.time()
    semantic_results = None

    # used for LGP-like mode
    results: EvaluationResults
    if hasattr(pipe, "register"):
        register_array = pipe.register
    else:
        register_array = None

    if configuration.basic_primitives.startswith("Pipeline"):
        Yp = multi_tree_evaluation_typed(
            func,
            pset,
            X,
            evaluation_configuration=configuration,
        )
        hash_result = None
        correlation_results = None
        introns_results = None
        if configuration.save_semantics:
            semantic_results = Yp
    else:
        Yp, results = multi_tree_evaluation(
            func,
            pset,
            X,
            original_features,
            need_hash=True,
            target=Y,
            configuration=configuration,
            register_array=register_array,
            similarity_score=intron_calculation,
            individual_configuration=individual_configuration,
        )
        hash_result, correlation_results, introns_results = (
            results.hash_result,
            results.correlation_results,
            results.introns_results,
        )
        if configuration.save_semantics:
            semantic_results = Yp
        if transductive_learning:
            Yp = Yp[: len(Y)]
        assert isinstance(Yp, (np.ndarray, torch.Tensor))
        if isinstance(Yp, np.ndarray):
            assert not np.any(np.isnan(Yp)), f"Type: {Yp.dtype}, Yp"
            assert not np.any(np.isinf(Yp))

        # only use for PS-Tree
        if hasattr(pipe, "partition_scheme"):
            partition_scheme = pipe.partition_scheme
            assert not np.any(np.isnan(partition_scheme))
            assert not np.any(np.isinf(partition_scheme))
            Yp = np.concatenate([Yp, np.reshape(partition_scheme, (-1, 1))], axis=1)
    if hasattr(pipe, "active_gene") and pipe.active_gene is not None:
        Yp = Yp[:, pipe.active_gene]

    gp_evaluation_time = time.time() - start_time

    if configuration.gradient_descent:
        gradient_optimization(Yp, Y, configuration, func)
        # evaluate again
        Yp = multi_tree_evaluation(
            func,
            pset,
            X,
            original_features,
            configuration=configuration,
            individual_configuration=individual_configuration,
        )

    if configuration.constant_type is None:
        gradient_operators = False
    else:
        gradient_operators = (
            configuration.gradient_descent
            or configuration.constant_type.startswith("GD")
        )
    if gradient_operators:
        Yp = Yp.detach().numpy()

    # ML evaluation
    start_time = time.time()
    base_model = pipe["Ridge"]
    regression_task = isinstance(base_model, RegressorMixin)

    if not configuration.cross_validation:
        pipe.fit(Yp, Y)
        if regression_task:
            y_pred = pipe.predict(Yp)
        else:
            y_pred = pipe.predict_proba(Yp)
        if isinstance(base_model, (MTLRidgeCV, MTLLassoCV)):
            y_pred = y_pred.flatten()
        estimators = [pipe]
    else:
        if configuration.ood_split:
            cv = OutOfDistributionSplit(n_splits=5, n_bins=5)
            y_pred, estimators = cross_val_predict(
                pipe,
                Yp,
                Y,
                cv=cv,
            )
        elif isinstance(base_model, (RidgeCV, RidgeGCV)) or (
            isinstance(base_model, MixupRegressor)
            and isinstance(base_model.regressor, RidgeGCV)
        ):
            if time_flag:
                cv_st = time.time()
            else:
                cv_st = 0

            # Ridge CV
            if sample_weight is not None:
                pipe.fit(Yp, Y, Ridge__sample_weight=sample_weight)
            else:
                # impute nan again
                Yp = np.nan_to_num(Yp, posinf=0, neginf=0)
                # avoid error in very special case
                Y = np.nan_to_num(Y, posinf=0, neginf=0)
                try:
                    pipe.fit(Yp, Y)
                except ValueError as e:
                    print("Value Error", Yp)
                    print("Value Error", Y)
                    raise e

            if time_flag:
                print("Cross Validation Time", time.time() - cv_st)

            """
            Potential Impacts:
            Given an incorrect best index
            """
            if isinstance(base_model, MTLRidgeCV):
                y_pred = base_model.cv_prediction(Y)
                estimators = [pipe]
            else:
                real_prediction = cv_prediction_from_ridge(Y, base_model)
                y_pred, estimators = real_prediction, [pipe]
        elif cv == 1:
            # single fold training (very fast)
            indices = np.arange(len(Y))
            x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
                Yp, Y, indices, test_size=0.2, random_state=0
            )
            clone_pipe: Pipeline = clone(pipe)
            estimators = [clone_pipe]
            clone_pipe.fit(x_train, y_train)
            consistency_check(pipe, clone_pipe)
            y_pred = clone_pipe.predict(x_test)
        else:
            # cross-validation
            if dynamic_target:
                cv = get_cv_splitter(base_model, cv, random.randint(0, int(1e9)))
            else:
                cv = get_cv_splitter(base_model, cv)

            if filter_elimination is not None and filter_elimination is not False:
                strategies = set(filter_elimination.split(","))
                mask = np.ones(X.shape[1], dtype=bool)
                if "Variance" in strategies:
                    # eliminate features based on variance
                    threshold = VarianceThreshold(threshold=0.01)
                    threshold.fit(Yp)
                    indices = threshold.get_support(indices=True)
                    mask[indices] = False
                if "Correlation" in strategies:
                    # eliminate features based on correlation
                    col_corr = []
                    corr_matrix = np.corrcoef(Yp)
                    for i in range(corr_matrix.shape[1]):
                        for j in range(i):
                            if abs(corr_matrix[i, j]) > threshold:
                                # highly correlated features
                                col_corr.append(i)
                                break
                    mask[col_corr] = False
                Yp = Yp[:, mask]

            if time_flag:
                cv_st = time.time()
            else:
                cv_st = 0

            if regression_task:
                y_pred, estimators = cross_val_predict(pipe, Yp, Y, cv=cv)
            else:
                y_pred, estimators = cross_val_predict(
                    pipe, Yp, Y, cv=cv, method="predict_proba"
                )

            if time_flag:
                print("Cross Validation Time", time.time() - cv_st)

            if feature_importance_method == "SHAP" and len(estimators) == cv.n_splits:
                for id, estimator in enumerate(estimators):
                    split_fold = list(cv.split(Yp, Y))
                    train_id, test_id = split_fold[id][0], split_fold[id][1]
                    if isinstance(
                        estimator["Ridge"], (LinearModel, LogisticRegression)
                    ):
                        explainer = shap.LinearExplainer(
                            estimator["Ridge"], Yp[train_id]
                        )
                    elif isinstance(estimator["Ridge"], BaseDecisionTree):
                        explainer = shap.TreeExplainer(estimator["Ridge"], Yp[train_id])
                    else:
                        raise Exception
                    if isinstance(estimator["Ridge"], BaseDecisionTree):
                        feature_importance = explainer.shap_values(Yp[test_id])[0]
                    else:
                        feature_importance = explainer.shap_values(Yp[test_id])
                    estimator["Ridge"].shap_values = np.abs(feature_importance).mean(
                        axis=0
                    )

            if (
                feature_importance_method == "PermutationImportance"
                and len(estimators) == cv.n_splits
            ):
                # Don't need to calculate the mean value here
                for id, estimator in enumerate(estimators):
                    split_fold = list(cv.split(Yp, Y))
                    train_id, test_id = split_fold[id][0], split_fold[id][1]
                    r = permutation_importance(
                        estimator["Ridge"],
                        Yp[test_id],
                        Y[test_id],
                        n_jobs=1,
                        n_repeats=1,
                    )
                    estimator["Ridge"].pi_values = np.abs(r.importances_mean)

            if np.any(np.isnan(y_pred)):
                np.save("error_data_x.npy", Yp)
                np.save("error_data_y.npy", Y)
                raise Exception
    ml_evaluation_time = time.time() - start_time
    configuration.enable_library = False
    return (
        y_pred,
        estimators,
        EvaluationResults(
            gp_evaluation_time=gp_evaluation_time,
            ml_evaluation_time=ml_evaluation_time,
            hash_result=hash_result,
            correlation_results=correlation_results,
            introns_results=introns_results,
            semantic_results=semantic_results,
        ),
    )


def consistency_check(pipe, clone_pipe):
    if isinstance(pipe["Ridge"], (WeightedKNNWithGP, WeightedKNNWithGPRidge)):
        assert pipe["Ridge"].distance == clone_pipe["Ridge"].distance


def calculate_permutation_importance(estimators, Yp, Y):
    if (
        not hasattr(estimators[0]["Ridge"], "coef_")
        and not hasattr(estimators[0]["Ridge"], "feature_importances_")
        and not hasattr(estimators[0]["Ridge"], "feature_importance")
    ):
        if time_flag:
            permutation_st = time.time()
        else:
            permutation_st = 0

        for id, estimator in enumerate(estimators):
            knn = estimator
            X_test, y_test = Yp, Y
            baseline_score = knn.score(X_test, y_test)
            importances = []
            for i in range(X_test.shape[1]):
                X_drop = X_test.copy()
                X_drop[:, i] = np.mean(X_drop[:, i])
                score = knn.score(X_drop, y_test)
                importances.append(baseline_score - score)
            feature_importances_ = np.array(importances)

            # importance_values = permutation_importance(estimator, Yp, Y, random_state=0)
            # feature_importances_ = importance_values.importances_mean
            if np.any(feature_importances_ < 0):
                feature_importances_ = feature_importances_ - feature_importances_.min()
            estimator["Ridge"].feature_importances_ = (
                feature_importances_ / feature_importances_.sum()
            )

        if time_flag:
            print("Permutation Cost", time.time() - permutation_st)


def statistical_best_alpha(base_model, Y, new_best_index):
    # new_best_index = statistical_best_alpha(base_model, Y, new_best_index)
    all_y_pred = base_model.cv_values_ + Y.mean()
    # get all error values
    errors = (Y.reshape(-1, 1) - all_y_pred) ** 2
    b_idx = None
    for idx in range(new_best_index + 1, errors.shape[1]):
        # signed rank test
        if wilcoxon(errors[:, new_best_index], errors[:, idx])[1] > 0.05:
            b_idx = idx
    if b_idx is not None:
        new_best_index = b_idx
    return new_best_index


def rbf_kernel_on_label(X, gamma):
    # Compute pairwise squared Euclidean distances
    distances_sq = cdist(X.reshape(-1, 1), X.reshape(-1, 1), "sqeuclidean")
    # Compute RBF kernel
    K = np.exp(-gamma * distances_sq)
    return K


def get_sample_weight(X, testX, Y, imbalanced_configuration: ImbalancedConfiguration):
    """
    When calculating the distance with test data, larger means more important.
    When calculating the distance with training data, larger is more important.
    :return:
    """
    # return None
    # Calculate the pairwise Euclidean distance
    X_space = imbalanced_configuration.weight_on_x_space
    based_on_test = imbalanced_configuration.based_on_test
    if X_space:
        if based_on_test:
            D = np.sqrt(
                np.sum((X[:, np.newaxis, :] - testX[np.newaxis, :, :]) ** 2, axis=-1)
            )
        else:
            X = np.concatenate([X, Y.reshape(-1, 1)], axis=1)
            D = np.sqrt(
                np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
            )
        # Set the gamma hyperparameter
        gamma = 1 / X.shape[1]
        # Calculate the RBF kernel matrix
        K = np.exp(-gamma * D**2)
    else:
        K = rbf_kernel_on_label(Y, 0.1)

    if not based_on_test or not X_space:
        K = np.sum(-np.sort(-K, axis=1)[:, :5], axis=1)
        K = 1 / K
    else:
        K = np.sum(-np.sort(-K, axis=1)[:, :5], axis=1)
    return K


def single_fold_validation(model, x_data, y_data, index):
    # only validate single fold, this method is faster than cross-validation
    cv = model_selection.KFold(n_splits=5, shuffle=False, random_state=0)
    current_index = 0
    scores = []
    for train_index, test_index in cv.split(x_data):
        if current_index == index:
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            model.fit(X_train, y_train)
            if isinstance(model["Ridge"], ClassifierMixin):
                scores.append(accuracy_score(y_test, model.predict(X_test)))
            elif isinstance(model["Ridge"], RegressorMixin):
                scores.append(r2_score(y_test, model.predict(X_test)))
            else:
                raise Exception
        else:
            scores.append(-1)
        current_index += 1
    assert np.any(scores != 0)
    return {
        "test_score": scores,
        "estimator": [model for _ in range(5)],
    }


def get_cv_splitter(base_model, cv, random_state=0):
    if isinstance(base_model, ClassifierMixin):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    return cv


def split_and_combine_data_decorator(
    data_arg_position=2,  # typically the third argument (indexing from 0)
    data_arg_name="data",
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Identifying the 'data' argument by its keyword or position
            data = kwargs.get(
                data_arg_name,
                args[data_arg_position] if len(args) > data_arg_position else None,
            )
            need_hash = kwargs.get("need_hash", False)

            # Check if data needs to be split
            step_size = 20000
            if data is not None and len(data) > step_size and not need_hash:
                # Split the data into slices of 50000 elements
                slices = [
                    data[i : i + step_size] for i in range(0, len(data), step_size)
                ]

                # Process each slice and collect results
                results = []
                for slice_data in slices:
                    if "data" in kwargs:
                        kwargs["data"] = slice_data
                    else:
                        args = list(args)
                        args[data_arg_position] = slice_data
                        args = tuple(args)

                    result = func(*args, **kwargs)
                    if isinstance(result, torch.Tensor):
                        result = result.detach().numpy()
                    results.append(result)
                    assert isinstance(
                        result, (np.ndarray, torch.Tensor)
                    ), f"{result}, {type(result)}"

                # Combine the results from all slices
                combined_results = np.concatenate(results)
                assert len(combined_results) == len(data)
                return combined_results
            else:
                # Call the original function if no slicing is required
                return func(*args, **kwargs)

        return wrapper

    return decorator


@split_and_combine_data_decorator()
def multi_tree_evaluation(
    gp_trees: List[PrimitiveTree],
    pset,
    data: np.ndarray,
    original_features=False,
    need_hash=False,
    register_array=None,
    target=None,
    configuration: EvaluationConfiguration = None,
    similarity_score=False,
    random_noise=0,
    random_seed=0,
    noise_configuration: NoiseConfiguration = None,
    reference_label: np.ndarray = None,
    individual_configuration: IndividualConfiguration = None,
    evaluation_cache: TreeLRUCache = None,
):
    if configuration is None:
        configuration = EvaluationConfiguration()

    if individual_configuration != None and isinstance(
        individual_configuration.dynamic_standardization, StandardScaler
    ):
        if not hasattr(individual_configuration.dynamic_standardization, "mean_"):
            data = individual_configuration.dynamic_standardization.fit_transform(data)
        else:
            data = individual_configuration.dynamic_standardization.transform(data)

    gradient_descent = configuration.gradient_descent
    if configuration.constant_type is None:
        gradient_optimization_flag = False
    else:
        gradient_optimization_flag = configuration.constant_type.startswith("GD")
    gradient_operators = gradient_descent or gradient_optimization_flag
    if gradient_operators and isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float().detach()

    # quick evaluate the result of features
    result = []
    hash_result = []
    correlation_results = []
    introns_results = []
    if hasattr(pset, "number_of_register"):
        # This part is useful for modular GP with registers
        # start with empty registers
        register = np.ones((data.shape[0], pset.number_of_register))
        for id, gene in enumerate(gp_trees):
            input_data = np.concatenate([data, register], axis=1)
            # evaluate GP trees
            quick_result = single_tree_evaluation(gene, pset, input_data)
            quick_result = quick_fill([quick_result], data)[0]
            # save hash of semantics
            if isinstance(quick_result, np.ndarray):
                hash_result.append(hash(quick_result.tostring()))
            else:
                hash_result.append(hash(str(quick_result)))
            # store to register
            register[:, register_array[id]] = quick_result
        result = register.T
    elif isinstance(pset, MultiplePrimitiveSet):
        # Modular GP
        for id, gene in enumerate(gp_trees):
            quick_result = single_tree_evaluation(gene, pset.pset_list[id], data)
            quick_result = quick_fill([quick_result], data)[0]
            add_hash_value(quick_result, hash_result)
            if target is not None:
                correlation_results.append(
                    np.abs(
                        cos_sim(
                            target - target.mean(), quick_result - quick_result.mean()
                        )
                    )
                )
            result.append(quick_result)
            data = np.concatenate([data, np.reshape(quick_result, (-1, 1))], axis=1)
    else:
        # ordinary GP evaluation
        for gene in gp_trees:
            # support a cache system
            if (
                evaluation_cache is not None
                and evaluation_cache.query(gene, random_noise, random_seed) is not None
            ):
                # cache to skip evaluation
                feature = evaluation_cache.query(gene, random_noise, random_seed)
                result.append(feature)
                continue

            feature, semantic_similarity = single_tree_evaluation(
                gene,
                pset,
                data,
                target=target if similarity_score else None,
                return_subtree_information=True,
                random_noise=random_noise,
                random_seed=random_seed,
                evaluation_configuration=configuration,
                noise_configuration=noise_configuration,
                reference_label=reference_label,
            )
            if isinstance(feature, np.ndarray) and len(feature.shape) == 2:
                # multi-dimensional output
                result.extend(list(feature.T))
            else:
                introns_results.append(semantic_similarity)
                simple_feature = quick_fill([feature], data)[0]
                add_hash_value(simple_feature, hash_result)
                if target is not None and configuration.intron_calculation:
                    abs_pearson_correlation = np.abs(
                        cos_sim(
                            target - target.mean(),
                            simple_feature - simple_feature.mean(),
                        )
                    )
                    correlation_results.append(abs_pearson_correlation)
                result.append(feature)

            if evaluation_cache is not None:
                # support a cache system
                evaluation_cache.store(gene, feature, random_noise, random_seed)

    if not gradient_operators:
        result = result_post_process(result, data, original_features)
        # result = np.reshape(result[:, -1], (-1, 1))
    if gradient_operators:
        result = quick_fill(result, data)
        result = torch.stack(tuple(result)).T

    if not need_hash:
        return result
    else:
        return result, EvaluationResults(
            hash_result=hash_result,
            correlation_results=correlation_results,
            introns_results=introns_results,
        )


def add_hash_value(quick_result, hash_result):
    if quick_result.var() != 0:
        hash_value = quick_result - quick_result.mean() / quick_result.var()
    else:
        hash_value = quick_result - quick_result.mean()
    if isinstance(quick_result, np.ndarray):
        hash_result.append(joblib.hash(hash_value))
    else:
        hash_result.append(hash(str(hash_value)))


cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# @custom_lru_cache(maxsize=10000, key_func=gp_key_func)
def single_tree_evaluation(
    expr: PrimitiveTree,
    pset,
    data,
    prefix="ARG",
    target=None,
    return_subtree_information=False,
    random_noise=0,
    random_seed=0,
    evaluation_configuration: EvaluationConfiguration = None,
    noise_configuration: NoiseConfiguration = None,
    reference_label=None,
) -> Tuple[np.ndarray, dict]:
    if evaluation_configuration is None:
        evaluation_configuration = EvaluationConfiguration()
    # random noise is vital for sharpness aware minimization
    # quickly evaluate a primitive tree
    lsh = evaluation_configuration.lsh
    classification = evaluation_configuration.classification
    if lsh:
        random_matrix = lsh_matrix_initialization(lsh, data)
    result = None
    stack = []
    subtree_information = {}
    depth_information, _ = node_depths_top_down(expr, current_depth=0)
    best_score = None
    # save the semantics of the best subtree
    best_subtree_semantics = None
    if noise_configuration is not None and noise_configuration.strict_layer_mode:
        if noise_configuration.skip_root:
            random_nodes = select_one_node_per_path(expr, forbidden_nodes={0})
        else:
            random_nodes = select_one_node_per_path(expr)
    else:
        # default value
        random_nodes = []
    for id, node in enumerate(expr):
        stack.append((node, [], id))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args, id = stack.pop()
            equivalent_subtree = -1
            if isinstance(prim, Primitive):
                try:
                    # for idx in range(len(args)):
                    #     if isinstance(args[idx], (float, int)):
                    #         args[idx] = np.full(len(data), args[idx])
                    result = pset.context[prim.name](*args)
                    if (
                        evaluation_configuration.enable_library
                        and evaluation_configuration.semantic_library is not None
                    ):
                        if isinstance(result, np.ndarray) and result.size > 1:
                            # Add subtree to semantic lib
                            evaluation_configuration.semantic_library.append_subtree(
                                result, PrimitiveTree(expr[expr.searchSubtree(id)])
                            )
                    if (
                        random_noise > 0
                        and (isinstance(result, (np.ndarray)) and result.size > 1)
                        and (isinstance(result, (torch.Tensor)) and result.shape[0] > 1)
                    ):
                        if (
                            (
                                # not add noise to the root node
                                len(stack) == 0
                                and noise_configuration.skip_root
                            )
                            or noise_configuration.only_terminal
                            or prim.ret == FeatureLayer
                        ):
                            pass
                        else:
                            layer_random_noise = get_adaptive_noise(
                                noise_configuration.layer_adaptive,
                                depth_information[id],
                                random_noise,
                            )
                            if (
                                noise_configuration.strict_layer_mode
                                and id not in random_nodes
                            ):
                                layer_random_noise = 0
                            result = inject_noise_to_data(
                                result,
                                layer_random_noise,
                                noise_configuration,
                                random_seed=random_seed,
                                reference_label=reference_label,
                                sam_mix_bandwidth=noise_configuration.sam_mix_bandwidth,
                            )
                except OverflowError as e:
                    result = args[0]
                    logging.error(
                        "Overflow error occurred: %s, args: %s", str(e), str(args)
                    )
                if target is not None:
                    for arg_id, a in enumerate(args):
                        if np.array_equal(result, a):
                            equivalent_subtree = arg_id
            elif isinstance(prim, Terminal):
                if prefix in prim.name:
                    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
                        result = data[:, int(prim.name.replace(prefix, ""))]
                    elif isinstance(data, (dict, enum.EnumMeta)):
                        result = data[prim.name]
                    else:
                        raise ValueError("Unsupported data type!")
                    if (
                        random_noise > 0
                        and isinstance(result, (np.ndarray, torch.Tensor))
                        # not a trivial expression
                        and len(result) > 1
                        and noise_configuration.noise_to_terminal is not False
                        # not add noise to only a terminal node, len(stack) == 0 represents only terminal node
                        # but this restriction only available in skip_root mode
                        and not (len(stack) == 0 and noise_configuration.skip_root)
                        # not to add noise to categorical features
                        and not (prim.ret == CategoricalFeature)
                    ):
                        layer_random_noise = get_adaptive_noise(
                            noise_configuration.layer_adaptive,
                            depth_information[id],
                            random_noise,
                        )
                        if (
                            noise_configuration.strict_layer_mode
                            and id not in random_nodes
                        ):
                            layer_random_noise = 0
                        result = inject_noise_to_data(
                            result,
                            layer_random_noise,
                            noise_configuration,
                            random_seed=random_seed,
                            reference_label=reference_label,
                            sam_mix_bandwidth=noise_configuration.sam_mix_bandwidth,
                        )
                else:
                    if isinstance(prim.value, str):
                        result = float(prim.value)
                    else:
                        result = prim.value
            else:
                raise Exception
            if target is not None:
                if len(np.unique(result)) == 1:
                    # If this primitive outputs constant, this can be viewed as an intron
                    subtree_information[id] = -1
                else:
                    if classification:
                        one_hot = one_hot_encode(target)
                        similarity = max(
                            map(
                                lambda t: np.abs(cos_sim(result - result.mean(), t)),
                                one_hot.T,
                            )
                        )
                        subtree_information[id] = similarity
                    else:
                        subtree_information[id] = np.abs(
                            cos_sim(result - result.mean(), target - target.mean())
                        )
                    if lsh:
                        hash_id = local_sensitive_hash(random_matrix, result)
                        subtree_information[id] = (subtree_information[id], hash_id)
                    if (
                        best_subtree_semantics is None
                        or best_score < subtree_information[id]
                    ):
                        best_score = subtree_information[id]
                        best_subtree_semantics = result
                    if equivalent_subtree >= 0:
                        # check whether a subtree is equal to its parent
                        subtree_information[id] = (
                            subtree_information[id],
                            equivalent_subtree,
                            "",
                        )
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(result)
    if len(subtree_information) > 0:
        assert len(subtree_information) == len(expr)
    if return_subtree_information:
        return result, subtree_information
    else:
        return result


def get_adaptive_noise(layer_adaptive, node_depth, random_noise):
    if layer_adaptive == "InvLinear":
        layer_random_noise = 1 / node_depth * random_noise
    elif layer_adaptive == "Linear":
        # Add noise to deep layers
        layer_random_noise = node_depth * random_noise
    elif layer_adaptive == "Log":
        # Less aggressive
        layer_random_noise = 1 / np.log(1 + node_depth) * random_noise
    elif layer_adaptive == "Sqrt":
        layer_random_noise = 1 / np.sqrt(node_depth) * random_noise
    elif layer_adaptive == "Cbrt":
        layer_random_noise = 1 / np.cbrt(node_depth) * random_noise
    elif layer_adaptive == "Log+":
        layer_random_noise = np.log(1 + node_depth) * random_noise
    elif layer_adaptive == "Sqrt+":
        layer_random_noise = np.sqrt(node_depth) * random_noise
    elif layer_adaptive == "Cbrt+":
        layer_random_noise = np.cbrt(node_depth) * random_noise
    else:
        layer_random_noise = random_noise
    return layer_random_noise


def lsh_matrix_initialization(lsh, data):
    rng = np.random.RandomState(0)
    if lsh == "Log":
        random_matrix = rng.randn(len(data), int(np.ceil(np.log2(len(data)))))
    elif not isinstance(lsh, bool) and isinstance(lsh, int):
        random_matrix = rng.randn(len(data), int(lsh))
    else:
        random_matrix = rng.randn(len(data), 10)
    return random_matrix


def local_sensitive_hash(random_matrix: np.ndarray, result):
    """
    LSH works by projecting data points into a lower-dimensional space using random projection matrices,
    and then quantizing these projections into hash codes.
    Similar data points are likely to be mapped to the same or nearby hash codes.
    """
    # locality-sensitive hash
    hash_id = 0
    final_sign = (result - result.mean()) @ random_matrix
    for s_id, s in enumerate(final_sign):
        if s > 0:
            hash_id += 1 << s_id
    return hash_id


@lru_cache()
def random_sample(size_of_noise, random_seed):
    return np.random.randint(0, size_of_noise, size_of_noise)


@cached(
    cache=LRUCache(maxsize=128),
    key=lambda random_seed, reference_label, gamma: hashkey(random_seed, gamma),
)
def weighted_sampling_cached(random_seed, reference_label, gamma=1):
    return weighted_sampling(random_seed, reference_label, gamma)


def weighted_sampling(random_seed, reference_label, gamma=1):
    distance_matrix = rbf_kernel(reference_label.reshape(-1, 1), gamma=gamma)
    return sample_according_to_distance(
        distance_matrix, np.arange(0, len(reference_label))
    )


def inject_noise_to_data(
    result,
    random_noise_magnitude: float,
    noise_configuration: NoiseConfiguration,
    noise_vector: np.ndarray = None,
    random_seed=0,
    reference_label=None,
    sam_mix_bandwidth=None,
):
    if random_noise_magnitude == 0:
        return result
    noise_type = noise_configuration.noise_type
    if noise_type == "Shuffle":
        return local_sensitive_shuffle_by_value(
            result, variability_scale=noise_configuration.shuffle_scale
        )
    size_of_noise = len(result)
    if noise_vector is not None:
        noise = noise_vector
    else:
        if noise_configuration.stochastic_mode:
            noise = noise_generation.__wrapped__(
                noise_type, size_of_noise, random_noise_magnitude, random_seed=-1
            )
        else:
            noise = noise_generation(
                noise_type, size_of_noise, random_noise_magnitude, random_seed
            )

    if isinstance(result, torch.Tensor):
        # from numpy
        noise = torch.from_numpy(noise).float()

    if noise_configuration.noise_normalization == "Mix":
        sampled_instances = random_sample(len(result), random_seed)
        result = (1 - noise) * result + noise * result[sampled_instances]
    elif noise_configuration.noise_normalization in ["MixT", "MixT+"]:
        # For this distance matrix, the larger, the near
        if noise_configuration.stochastic_mode:
            sampled_instances = weighted_sampling(
                random_seed, reference_label, gamma=sam_mix_bandwidth
            )
        else:
            sampled_instances = weighted_sampling_cached(
                random_seed, reference_label, gamma=sam_mix_bandwidth
            )
        result = (1 - noise) * result + noise * result[sampled_instances]
    elif noise_configuration.noise_normalization == "Instance+":
        result = result + noise * random_noise_magnitude * result
    elif noise_configuration.noise_normalization == "Instance":
        result = result + noise * random_noise_magnitude * np.abs(result)
    elif noise_configuration.noise_normalization == "STD":
        result = result + noise * random_noise_magnitude * np.std(result)
    elif noise_configuration.noise_normalization in ["None", None]:
        result = result + noise * random_noise_magnitude
    else:
        raise Exception("Invalid noise normalization type")

    return result


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


@lru_cache
def noise_generation(noise_type, size_of_noise, random_noise_magnitude, random_seed):
    """
    Obviously, it's possible to use cache technique to avoid the expensive sampling process
    """
    if random_seed == -1:
        rng = np.random
    else:
        rng = np.random.RandomState(random_seed)
    if noise_type == "Normal":
        noise = rng.normal(0, 1, size_of_noise)
    elif noise_type == "TruncatedNormal":
        noise = get_truncated_normal(0, 1, -1, 1).rvs(size_of_noise)
    elif noise_type == "Uniform":
        noise = rng.uniform(-1, 1, size_of_noise)
    elif noise_type == "Laplace":
        noise = rng.laplace(0, 1, size_of_noise)
    elif noise_type == "Beta":
        noise = rng.beta(1, random_noise_magnitude, size_of_noise)
        # ensure large part of original samples, and less part is noise
        noise = np.minimum(noise, 1 - noise)
    elif noise_type == "Ensemble":
        choices = [
            rng.normal(0, 1, size_of_noise),
            rng.uniform(-1, 1, size_of_noise),
            rng.laplace(0, 1, size_of_noise),
        ]
        noise = choices[random_seed % len(choices)]
    elif noise_type == "Binomial":
        noise = rng.choice([-1, 1], size_of_noise)
    else:
        raise Exception("Invalid noise type")
    return noise.astype(np.float32)


def minimal_task():
    # A minimal task for SR
    x, y = make_regression(n_samples=1000)
    pset = gp.PrimitiveSet("MAIN", x.shape[1])
    pset.addPrimitive(np.add, 2, name="vadd")
    pset.addPrimitive(np.subtract, 2, name="vsub")
    pset.addPrimitive(np.multiply, 2, name="vmul")
    pset.addPrimitive(np.negative, 1, name="vneg")
    pset.addPrimitive(np.cos, 1, name="vcos")
    pset.addPrimitive(np.sin, 1, name="vsin")
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=8)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    pop = toolbox.population(n=10000)
    st = time.time()
    avg_a = np.zeros(x.shape[0])
    for ind in pop:
        avg_a += single_tree_evaluation(ind, pset, x)
    print("time", time.time() - st)
    st = time.time()
    avg_b = np.zeros(x.shape[0])
    for ind in pop:
        func = gp.compile(ind, pset)
        avg_b += func(*x.T)
    print("time", time.time() - st)
    assert_almost_equal(avg_a, avg_b)


def minimal_feature_importance():
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge

    diabetes = load_diabetes()
    X_train, X_val, y_train, y_val = train_test_split(
        diabetes.data, diabetes.target, random_state=0
    )
    model = Ridge(alpha=1e-2).fit(X_train, y_train)
    model.score(X_val, y_val)
    r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=0)
    print(r.importances_mean)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(
                f"{diabetes.feature_names[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}"
            )


if __name__ == "__main__":
    # minimal_task()
    minimal_feature_importance()
