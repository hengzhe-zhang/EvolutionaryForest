import enum
import logging
import random
import time
from typing import List, Tuple

import dill
import numpy as np
import shap
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap.gp import PrimitiveTree, Primitive, Terminal
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn import model_selection
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.datasets import make_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, \
    f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import BaseDecisionTree

from evolutionary_forest.component.configuration import EvaluationConfiguration, ImbalancedConfiguration
from evolutionary_forest.model.MTL import MTLRidgeCV
from evolutionary_forest.multigene_gp import result_post_process, MultiplePrimitiveSet, quick_fill, GPPipeline
from evolutionary_forest.sklearn_utils import cross_val_predict
from evolutionary_forest.utils import reset_random, cv_prediction_from_ridge

np.seterr(invalid='ignore')
reset_random(0)
time_flag = False


def select_from_array(arr, s, x):
    s = s % len(arr)
    if s + x <= len(arr):
        return arr[s:s + x]
    else:
        return np.concatenate((arr[s:s + x], arr[:(s + x) % len(arr)]))


class EvaluationResults():
    def __init__(self,
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
    nn_prediction = configuration.nn_prediction
    dynamic_target = configuration.dynamic_target
    original_features = configuration.original_features
    test_data_size = configuration.test_data_size
    pset = configuration.pset
    sklearn_format = configuration.sklearn_format
    cross_validation = configuration.cross_validation
    feature_importance_method = configuration.feature_importance_method
    filter_elimination = configuration.filter_elimination
    intron_calculation = configuration.intron_calculation
    sample_weight = configuration.sample_weight

    if configuration.mini_batch:
        X = select_from_array(X, configuration.current_generation * configuration.batch_size // 4,
                              configuration.batch_size)
        Y = select_from_array(Y, configuration.current_generation * configuration.batch_size // 4,
                              configuration.batch_size)

    pipe: GPPipeline
    pipe, func = args
    if not isinstance(func, list):
        func = dill.loads(func)

    # GP evaluation
    start_time = time.time()
    semantic_results = None
    if sklearn_format:
        Yp = quick_result_calculation(func, pset, X, original_features, sklearn_format)
        pipe = pipe_combine(Yp, pipe)
        Yp = X
        hash_result = None
        correlation_results = None
        introns_results = None
    else:
        results: EvaluationResults
        if hasattr(pipe, 'register'):
            Yp, results = quick_result_calculation(func, pset, X, original_features,
                                                   need_hash=True, target=Y,
                                                   register_array=pipe.register,
                                                   configuration=configuration)
        else:
            Yp, results = quick_result_calculation(func, pset, X, original_features,
                                                   need_hash=True, target=Y,
                                                   configuration=configuration,
                                                   similarity_score=intron_calculation)
        hash_result, correlation_results, introns_results = results.hash_result, \
            results.correlation_results, results.introns_results
        if configuration.save_semantics:
            semantic_results = Yp
        if nn_prediction is not None:
            Yp = np.concatenate([Yp, nn_prediction], axis=1)
        if test_data_size > 0:
            Yp = Yp[:-test_data_size]
        assert isinstance(Yp, np.ndarray)
        assert not np.any(np.isnan(Yp))
        assert not np.any(np.isinf(Yp))

        # only use for PS-Tree
        if hasattr(pipe, 'partition_scheme'):
            partition_scheme = pipe.partition_scheme
            assert not np.any(np.isnan(partition_scheme))
            assert not np.any(np.isinf(partition_scheme))
            Yp = np.concatenate([Yp, np.reshape(partition_scheme, (-1, 1))], axis=1)
    if hasattr(pipe, 'active_gene') and pipe.active_gene is not None:
        Yp = Yp[:, pipe.active_gene]

    gp_evaluation_time = time.time() - start_time

    # ML evaluation
    start_time = time.time()
    if isinstance(score_func, str) and 'CV' in score_func:
        # custom cross validation scheme
        if '-Random' in score_func:
            cv = StratifiedKFold(shuffle=True)
        else:
            cv = None

        if score_func == 'CV-Accuracy-Recall':
            custom_scorer = {'accuracy': make_scorer(accuracy_score),
                             'balanced_accuracy': make_scorer(balanced_accuracy_score),
                             'precision': make_scorer(precision_score, average='macro'),
                             'recall': make_scorer(recall_score, average='macro'),
                             'f1': make_scorer(f1_score, average='macro')}
            result = cross_validate(pipe, Yp, Y, cv=cv, return_estimator=True, scoring=custom_scorer)
        elif score_func == 'CV-F1-Score':
            result = cross_validate(pipe, Yp, Y, cv=cv, return_estimator=True,
                                    scoring=make_scorer(f1_score, average='macro'))
        elif score_func == 'CV-SingleFold':
            result = single_fold_validation(pipe, Yp, Y, index=random.randint(0, 4))
        else:
            result = cross_validate(pipe, Yp, Y, cv=cv, return_estimator=True)

        if score_func == 'CV-Accuracy-Recall':
            y_pred = np.array([np.mean(result['test_accuracy']),
                               np.mean(result['test_balanced_accuracy']),
                               np.mean(result['test_precision']),
                               np.mean(result['test_recall']),
                               np.mean(result['test_f1'])])
        else:
            y_pred = result['test_score']
        estimators = result['estimator']
    elif not cross_validation:
        pipe.fit(Yp, Y)
        y_pred = pipe.predict(Yp)
        estimators = [pipe]
    else:
        if sklearn_format:
            base_model = pipe['model']['Ridge']
        else:
            base_model = pipe['Ridge']
        regression_task = isinstance(base_model, RegressorMixin)
        if isinstance(base_model, RidgeCV):
            if time_flag:
                cv_st = time.time()
            else:
                cv_st = 0

            # Ridge CV
            if sample_weight is not None:
                pipe.fit(Yp, Y, Ridge__sample_weight=sample_weight)
            else:
                pipe.fit(Yp, Y)

            if time_flag:
                print('Cross Validation Time', time.time() - cv_st)

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
            # single fold training (not recommend)
            indices = np.arange(len(Y))
            x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
                Yp, Y, indices, test_size=0.2)
            estimators = [pipe]
            pipe.fit(x_train, y_train)
            y_pred = np.ones_like(Y)
            y_pred[idx_test] = pipe.predict(x_test)
        else:
            # cross-validation
            if dynamic_target:
                cv = get_cv_splitter(base_model, cv, random.randint(0, int(1e9)))
            else:
                cv = get_cv_splitter(base_model, cv)

            if filter_elimination is not None and filter_elimination is not False:
                strategies = set(filter_elimination.split(','))
                mask = np.ones(X.shape[1], dtype=bool)
                if 'Variance' in strategies:
                    # eliminate features based on variance
                    threshold = VarianceThreshold(threshold=0.01)
                    threshold.fit(Yp)
                    indices = threshold.get_support(indices=True)
                    mask[indices] = False
                if 'Correlation' in strategies:
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
                y_pred, estimators = cross_val_predict(pipe, Yp, Y, cv=cv, method='predict_proba')

            if time_flag:
                print('Cross Validation Time', time.time() - cv_st)

            if feature_importance_method == 'SHAP' and len(estimators) == cv.n_splits:
                for id, estimator in enumerate(estimators):
                    split_fold = list(cv.split(Yp, Y))
                    train_id, test_id = split_fold[id][0], split_fold[id][1]
                    if isinstance(estimator['Ridge'], (LinearModel, LogisticRegression)):
                        explainer = shap.LinearExplainer(estimator['Ridge'], Yp[train_id])
                    elif isinstance(estimator['Ridge'], BaseDecisionTree):
                        explainer = shap.TreeExplainer(estimator['Ridge'], Yp[train_id])
                    else:
                        raise Exception
                    if isinstance(estimator['Ridge'], BaseDecisionTree):
                        feature_importance = explainer.shap_values(Yp[test_id])[0]
                    else:
                        feature_importance = explainer.shap_values(Yp[test_id])
                    estimator['Ridge'].shap_values = np.abs(feature_importance).mean(axis=0)

            if feature_importance_method == 'PermutationImportance' and len(estimators) == cv.n_splits:
                # Don't need to calculate the mean value here
                for id, estimator in enumerate(estimators):
                    split_fold = list(cv.split(Yp, Y))
                    train_id, test_id = split_fold[id][0], split_fold[id][1]
                    r = permutation_importance(estimator['Ridge'], Yp[test_id], Y[test_id], n_jobs=1, n_repeats=1)
                    estimator['Ridge'].pi_values = np.abs(r.importances_mean)

            # calculate_permutation_importance(estimators, Yp, Y)

            if np.any(np.isnan(y_pred)):
                np.save('error_data_x.npy', Yp)
                np.save('error_data_y.npy', Y)
                raise Exception
    ml_evaluation_time = time.time() - start_time
    return y_pred, estimators, EvaluationResults(gp_evaluation_time=gp_evaluation_time,
                                                 ml_evaluation_time=ml_evaluation_time,
                                                 hash_result=hash_result,
                                                 correlation_results=correlation_results,
                                                 introns_results=introns_results,
                                                 semantic_results=semantic_results)


def calculate_permutation_importance(estimators, Yp, Y):
    if not hasattr(estimators[0]['Ridge'], 'coef_') \
        and not hasattr(estimators[0]['Ridge'], 'feature_importances_') \
        and not hasattr(estimators[0]['Ridge'], 'feature_importance'):
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
            estimator['Ridge'].feature_importances_ = feature_importances_ / feature_importances_.sum()

        if time_flag:
            print('Permutation Cost', time.time() - permutation_st)


def statistical_best_alpha(base_model, Y, new_best_index):
    # new_best_index = statistical_best_alpha(base_model, Y, new_best_index)
    all_y_pred = (base_model.cv_values_ + Y.mean())
    # get all error values
    errors = (Y.reshape(-1, 1) - all_y_pred) ** 2
    b_idx = None
    for idx in range(new_best_index + 1, errors.shape[1]):
        # singed rank test
        if wilcoxon(errors[:, new_best_index], errors[:, idx])[1] > 0.05:
            b_idx = idx
    if b_idx is not None:
        new_best_index = b_idx
    return new_best_index


def rbf_kernel_on_label(X, gamma):
    # Compute pairwise squared Euclidean distances
    distances_sq = cdist(X.reshape(-1, 1), X.reshape(-1, 1), 'sqeuclidean')
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
            D = np.sqrt(np.sum((X[:, np.newaxis, :] - testX[np.newaxis, :, :]) ** 2, axis=-1))
        else:
            X = np.concatenate([X, Y.reshape(-1, 1)], axis=1)
            D = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))
        # Set the gamma hyperparameter
        gamma = 1 / X.shape[1]
        # Calculate the RBF kernel matrix
        K = np.exp(-gamma * D ** 2)
    else:
        K = rbf_kernel_on_label(Y, 0.1)

    if not based_on_test or not X_space:
        K = np.sum(-np.sort(-K, axis=1)[:, :5], axis=1)
        K = 1 / K
    else:
        K = np.sum(-np.sort(-K, axis=1)[:, :5], axis=1)
    return K


def pipe_combine(Yp, pipe):
    Yp = list(filter(lambda x: not isinstance(x, (int, float, np.ndarray, enum.Enum)), Yp))
    if len(Yp) == 0:
        pipe = Pipeline([
            ("feature", "passthrough"),
            ('model', pipe)
        ])
    else:
        pipe = Pipeline([
            ("feature", FeatureUnion(
                [(f'preprocessing_{id}', p) for id, p in enumerate(Yp)]
            )),
            ('model', pipe)
        ])
    return pipe


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
            if isinstance(model['Ridge'], ClassifierMixin):
                scores.append(accuracy_score(y_test, model.predict(X_test)))
            elif isinstance(model['Ridge'], RegressorMixin):
                scores.append(r2_score(y_test, model.predict(X_test)))
            else:
                raise Exception
        else:
            scores.append(-1)
        current_index += 1
    assert np.any(scores != 0)
    return {
        'test_score': scores,
        'estimator': [model for _ in range(5)],
    }


def get_cv_splitter(base_model, cv, random_state=0):
    if isinstance(base_model, ClassifierMixin):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    return cv


def quick_result_calculation(func: List[PrimitiveTree], pset, data, original_features=False,
                             sklearn_format=False, need_hash=False, register_array=None,
                             target=None, configuration: EvaluationConfiguration = None,
                             similarity_score=False,
                             random_noise=0):
    if configuration is None:
        configuration = EvaluationConfiguration()
    intron_gp = configuration.intron_gp
    lsh = configuration.lsh
    # evaluate an individual rather than a gene
    if sklearn_format:
        data = enum.Enum('Enum', {f"ARG{i}": i for i in range(data.shape[1])})
    # quick evaluate the result of features
    result = []
    hash_result = []
    correlation_results = []
    introns_results = []
    if hasattr(pset, 'number_of_register'):
        # This part is useful for modular GP with registers
        register = np.ones((data.shape[0], pset.number_of_register))
        for id, gene in enumerate(func):
            input_data = np.concatenate([data, register], axis=1)
            quick_result = quick_evaluate(gene, pset, input_data)
            quick_result = quick_fill([quick_result], data)[0]
            if isinstance(quick_result, np.ndarray):
                hash_result.append(hash(quick_result.tostring()))
            else:
                hash_result.append(hash(str(quick_result)))
            register[:, register_array[id]] = quick_result
        result = register.T
    elif isinstance(pset, MultiplePrimitiveSet):
        # Modular GP
        for id, gene in enumerate(func):
            quick_result = quick_evaluate(gene, pset.pset_list[id], data)
            quick_result = quick_fill([quick_result], data)[0]
            add_hash_value(quick_result, hash_result)
            if target is not None:
                correlation_results.append(np.abs(cos_sim(target - target.mean(), quick_result - quick_result.mean())))
            result.append(quick_result)
            data = np.concatenate([data, np.reshape(quick_result, (-1, 1))], axis=1)
    else:
        # ordinary GP evaluation
        for gene in func:
            feature, intron_ids = quick_evaluate(gene, pset, data, target=target if similarity_score else None,
                                                 intron_gp=intron_gp, lsh=lsh,
                                                 return_subtree_information=True,
                                                 random_noise=random_noise)
            introns_results.append(intron_ids)
            simple_feature = quick_fill([feature], data)[0]
            add_hash_value(simple_feature, hash_result)
            if target is not None and configuration.intron_calculation:
                correlation_results.append(np.abs(cos_sim(target - target.mean(),
                                                          simple_feature - simple_feature.mean())))
            result.append(feature)
    if not sklearn_format:
        result = result_post_process(result, data, original_features)
        # result = np.reshape(result[:, -1], (-1, 1))
    if not need_hash:
        return result
    else:
        return result, EvaluationResults(hash_result=hash_result,
                                         correlation_results=correlation_results,
                                         introns_results=introns_results)


def add_hash_value(quick_result, hash_result):
    if quick_result.var() != 0:
        hash_value = quick_result - quick_result.mean() / quick_result.var()
    else:
        hash_value = quick_result - quick_result.mean()
    if isinstance(quick_result, np.ndarray):
        hash_result.append(hash(hash_value.tostring()))
    else:
        hash_result.append(hash(str(hash_value)))


cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def quick_evaluate(expr: PrimitiveTree, pset, data, prefix='ARG', target=None,
                   intron_gp=False, lsh=False, return_subtree_information=False,
                   random_noise=0) -> Tuple[np.ndarray, dict]:
    # random noise is very important for sharpness aware minimization
    # quickly evaluate a primitive tree
    if lsh:
        rng = np.random.RandomState(0)
        if lsh == 'Log':
            random_matrix = rng.randn(len(data), int(np.ceil(np.log2(len(data)))))
        elif not isinstance(lsh, bool) and isinstance(lsh, int):
            random_matrix = rng.randn(len(data), int(lsh))
        else:
            random_matrix = rng.randn(len(data), 10)
    result = None
    stack = []
    subtree_information = {}
    best_score = None
    # save the semantics of the best subtree
    best_subtree_semantics = None
    for id, node in enumerate(expr):
        stack.append((node, [], id))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args, id = stack.pop()
            equivalent_subtree = -1
            if isinstance(prim, Primitive):
                try:
                    result = pset.context[prim.name](*args)
                    if random_noise > 0 and isinstance(result, np.ndarray) and len(result) > 1:
                        result += np.random.normal(0, random_noise*np.std(result), len(result))
                except OverflowError as e:
                    result = args[0]
                    logging.error("Overflow error occurred: %s, args: %s", str(e), str(args))
                if target is not None:
                    for arg_id, a in enumerate(args):
                        if np.array_equal(result, a):
                            equivalent_subtree = arg_id
            elif isinstance(prim, Terminal):
                if prefix in prim.name:
                    if isinstance(data, np.ndarray):
                        result = data[:, int(prim.name.replace(prefix, ''))]
                    elif isinstance(data, (dict, enum.EnumMeta)):
                        result = data[prim.name]
                    else:
                        raise ValueError("Unsupported data type!")
                else:
                    result = float(prim.name)
            else:
                raise Exception
            if target is not None:
                if len(np.unique(result)) == 1:
                    # If this primitive outputs constant, this can be viewed as an intron
                    subtree_information[id] = -1
                else:
                    subtree_information[id] = np.abs(cos_sim(result - result.mean(), target - target.mean()))
                    if lsh:
                        # locality sensitive hash
                        hash_id = 0
                        final_sign = (result - result.mean()) @ random_matrix
                        for s_id, s in enumerate(final_sign):
                            if s > 0:
                                hash_id += 1 << s_id
                        subtree_information[id] = (subtree_information[id], hash_id)
                    if best_subtree_semantics is None or best_score < subtree_information[id]:
                        best_score = subtree_information[id]
                        best_subtree_semantics = result
                    if equivalent_subtree >= 0:
                        # check whether a subtree is equal to its parent
                        subtree_information[id] = (subtree_information[id], equivalent_subtree, '')
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(result)
    if len(subtree_information) > 0:
        assert len(subtree_information) == len(expr)
    if intron_gp:
        return best_subtree_semantics, subtree_information
    else:
        if return_subtree_information:
            return result, subtree_information
        else:
            return result


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
        avg_a += quick_evaluate(ind, pset, x)
    print('time', time.time() - st)
    st = time.time()
    avg_b = np.zeros(x.shape[0])
    for ind in pop:
        func = gp.compile(ind, pset)
        avg_b += func(*x.T)
    print('time', time.time() - st)
    assert_almost_equal(avg_a, avg_b)


def minimal_feature_importance():
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge

    diabetes = load_diabetes()
    X_train, X_val, y_train, y_val = train_test_split(
        diabetes.data, diabetes.target, random_state=0)
    model = Ridge(alpha=1e-2).fit(X_train, y_train)
    model.score(X_val, y_val)
    r = permutation_importance(model, X_val, y_val,
                               n_repeats=30,
                               random_state=0)
    print(r.importances_mean)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{diabetes.feature_names[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")


if __name__ == '__main__':
    # minimal_task()
    minimal_feature_importance()
