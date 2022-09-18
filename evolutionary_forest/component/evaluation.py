import random
import time

import dill
import numpy as np
from sklearn import model_selection
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, \
    f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, KFold
from sklearn.pipeline import Pipeline

from evolutionary_forest.model.PLTree import SoftPLTreeRegressor, SoftPLTreeRegressorEM
from evolutionary_forest.multigene_gp import result_calculation
from evolutionary_forest.sklearn_utils import cross_val_predict


def calculate_score(args):
    (X, Y, score_func, cv, other_parameters) = calculate_score.data
    # nn_prediction: prediction results of the neural network
    # dynamic_target: random generate cross-validation scheme
    nn_prediction, dynamic_target = other_parameters['nn_prediction'], other_parameters['dynamic_target']
    original_features = other_parameters['original_features']
    test_data_size = other_parameters['test_data_size']

    pipe: Pipeline
    pipe, func, basic_gene = args
    if not isinstance(func, list):
        func = dill.loads(func)
    if not isinstance(basic_gene, list):
        basic_gene = dill.loads(basic_gene)

    # only used for PS-Tree
    best_label = None
    altered_labels = None
    # GP evaluation
    start_time = time.time()
    if len(basic_gene) > 0:
        # synthesize some basic features
        x_basic = result_calculation(basic_gene, X, original_features)
        X = np.concatenate([X, x_basic], axis=1)
        # X = x_basic
    Yp = result_calculation(func, X, original_features)
    if nn_prediction is not None:
        Yp = np.concatenate([Yp, nn_prediction], axis=1)
    if test_data_size > 0:
        Yp = Yp[:-test_data_size]
    gp_evaluation_time = time.time() - start_time
    assert isinstance(Yp, np.ndarray)
    assert not np.any(np.isnan(Yp))
    assert not np.any(np.isinf(Yp))

    if hasattr(pipe, 'partition_scheme'):
        partition_scheme = pipe.partition_scheme
        assert not np.any(np.isnan(partition_scheme))
        assert not np.any(np.isinf(partition_scheme))
        Yp = np.concatenate([Yp, np.reshape(partition_scheme, (-1, 1))], axis=1)

    # ML evaluation
    start_time = time.time()
    if 'CV' in score_func:
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
            result = data_test(pipe, Yp, Y, index=random.randint(0, 4))
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
    else:
        base_model = pipe['Ridge']
        regression_task = isinstance(base_model, RegressorMixin)
        if isinstance(base_model, RidgeCV):
            pipe.fit(Yp, Y)
            y_pred, estimators = base_model.cv_values_[:, np.argmax(base_model.cv_values_.sum(axis=0))], [pipe]
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

            if regression_task:
                y_pred, estimators = cross_val_predict(pipe, Yp, Y, cv=cv)
            else:
                y_pred, estimators = cross_val_predict(pipe, Yp, Y, cv=cv, method='predict_proba')

            if np.any(np.isnan(y_pred)):
                np.save('error_data_x.npy', Yp)
                np.save('error_data_y.npy', Y)
                raise Exception

            if isinstance(base_model, SoftPLTreeRegressor) and not isinstance(base_model, SoftPLTreeRegressorEM):
                # determine the best partition scheme
                cv_label = other_parameters['cv_label']
                if cv_label:
                    best_label = np.zeros(len(Y))
                else:
                    partition_num = len(np.unique(pipe.partition_scheme))
                    best_label = np.zeros((len(Y), partition_num))
                for index, estimator in zip(cv.split(Y), estimators):
                    estimator: SoftPLTreeRegressor
                    if cv_label:
                        # determine the label of PS-Tree through cross-validation
                        train_index, test_index = index
                        X_test, y_test = Yp[test_index], Y[test_index]
                        best_label[test_index] = estimator.score(X_test, np.reshape(y_test, (-1, 1)))
                    else:
                        train_index, test_index = index
                        score = estimator.score(Yp[train_index], np.reshape(Y[train_index], (-1, 1)))
                        best_label[train_index, score] += 1
                if not cv_label:
                    best_label = np.argmax(best_label, axis=1)
                altered_labels = np.sum(pipe.partition_scheme != best_label)
    ml_evaluation_time = time.time() - start_time
    if altered_labels is not None:
        print('altered_labels', altered_labels)
    return y_pred, estimators, {
        'gp_evaluation_time': gp_evaluation_time,
        'ml_evaluation_time': ml_evaluation_time,
        'best_label': best_label,
        'altered_labels': altered_labels,
    }


def data_test(model, x_data, y_data, index):
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
