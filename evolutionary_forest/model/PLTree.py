from abc import abstractmethod
from typing import Dict, List, Union, Callable

import numpy as np
from mlxtend.classifier import LogisticRegression
from scipy.special import softmax
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_wine, load_diabetes
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RidgeCV, LinearRegression, Ridge, LogisticRegression
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import BaseMixture
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, BaseDecisionTree


class PLTree(BaseDecisionTree):
    @abstractmethod
    def model_controller(self):
        pass

    def post_fit(self, X, y, partition_scheme=None):
        # construct linear models in each partition
        self.model_map: Dict = {}
        model_coefs = []
        if partition_scheme is None:
            labels = super().apply(X)
            for l in np.unique(labels):
                idx = labels == l
                model = self.model_controller()
                if len(np.unique(y[idx])) == 1:
                    model = DummyClassifier()
                    model.coef_ = np.zeros(X.shape[1])
                    model = model.fit(X[idx], y[idx])
                else:
                    model = model.fit(X[idx], y[idx])
                self.model_map[l] = model
                # model.predict_proba(X)
                if len(model.coef_.shape) == 2:
                    model_coefs.append(np.max(model.coef_, axis=0))
                else:
                    model_coefs.append(model.coef_)
        else:
            for l, p in enumerate(partition_scheme.T):
                model = self.model_controller()
                index = p > 0
                model = model.fit(X[index], y[index], sample_weight=p[index])
                self.model_map[l] = model
                model_coefs.append(model.coef_)

        lr_coef = np.abs(np.array(model_coefs)).max(axis=0)
        lr_coef /= np.sum(lr_coef)
        self.feature_importance = np.max(
            [lr_coef, self.feature_importances_[: len(lr_coef)]], axis=0
        )
        self.feature_importance /= np.sum(self.feature_importance)
        assert len(self.feature_importance) == X.shape[1]

    def post_predict(self, X, labels, method=None, classes=None):
        classes_map = {}
        if method == "predict_proba":
            # for classification task, we need to predict probabilities
            predictions = np.zeros((X.shape[0], self.n_classes_))
            classes_map = {v: k for k, v in enumerate(self.classes_)}
        elif method == "soft_prediction":
            if isinstance(self.partition_tree, DecisionTreeClassifier):
                predictions = np.zeros((X.shape[0], self.partition_tree.n_classes_))
            elif isinstance(self.partition_tree, LogisticRegression):
                predictions = np.zeros((X.shape[0], len(self.partition_tree.classes_)))
            elif isinstance(self.partition_tree, GaussianMixture):
                n_classes = len(np.unique(self.partition_tree.classes_))
                predictions = np.zeros((X.shape[0], n_classes))
            else:
                raise Exception
        else:
            predictions = np.zeros(X.shape[0])

        if method == "soft_prediction":
            # get all prediction results
            for l in classes:
                model = self.model_map[l]
                predictions[:, l] = model.predict(X)
        else:
            for l in np.sort(np.unique(labels)):
                idx = labels == l
                model = self.model_map[l]
                if method == "predict_proba":
                    # predict classification probabilities
                    temp_proba = model.predict_proba(X[idx])
                    for c, p in zip(model.classes_, temp_proba.T):
                        predictions[idx, classes_map[c]] = p
                else:
                    predictions[idx] = model.predict(X[idx])

        return predictions

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.post_fit(X, y)

    def predict(self, X, check_input=True):
        labels = super().apply(X)
        return self.post_predict(X, labels)


class RandomWeightRidge(Ridge):
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-3,
        solver="auto",
        positive=False,
        random_state=None,
        initial_weight=None,
    ):
        super().__init__(
            alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )
        self.initial_weight = initial_weight

    def fit(self, X, y, sample_weight=None):
        return super().fit(
            X, y, np.abs(X[:, -1]) if np.sum(np.abs(X[:, -1])) != 0 else None
        )


class PLTreeRegressor(DecisionTreeRegressor, PLTree):
    """
    A simple implementation of piecewise linear regression tree
    """

    def __init__(self, base_model="Ridge", **kwargs):
        """
        base_model: The local model
        """
        self.base_model = base_model
        super().__init__(**kwargs)

    def model_controller(self):
        if self.base_model == "RidgeCV":
            cv = RidgeCV()
        elif self.base_model == "Ridge":
            cv = Ridge()
        elif self.base_model == "LR":
            cv = LinearRegression()
        else:
            raise Exception
        return cv


class RPLBaseModel(BaseEstimator):
    base_model: Callable
    ridge: Union[LinearModel, LogisticRegression]

    def __init__(self, decision_tree_count, max_leaf_nodes):
        self.decision_tree_count = decision_tree_count
        self.max_leaf_nodes = max_leaf_nodes
        self.regression = isinstance(self, RegressorMixin)
        self.dt: List[Union[DecisionTreeRegressor, DecisionTreeClassifier]] = [
            self.base_model(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=10)
            if self.regression
            # For classification cases
            else MultiOutputRegressor(
                self.base_model(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=10)
            )
            for _ in range(self.decision_tree_count)
        ]

    def fit(self, X, y):
        self.ridge.fit(X, y)
        if self.regression:
            prediction = self.ridge.predict(X)
        else:
            y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
            prediction = self.ridge.predict_proba(X)
        residual = y - prediction
        for dt in self.dt:
            dt.fit(X, residual)
            if self.regression:
                prediction += dt.predict(X)
                residual = y - prediction
            else:
                prediction += dt.predict(X)
                p = softmax(prediction)
                residual = y - p

        # Calculate feature importance values
        # Feature importance in the global model
        self.feature_importance = np.abs(self.ridge.coef_)
        if len(self.feature_importance.shape) == 2:
            self.feature_importance = np.max(self.feature_importance, axis=0)
        self.feature_importance = (
            self.feature_importance / self.feature_importance.sum()
        )
        for dt in self.dt:
            # It is possible that decision tree not uses any features.
            # In this case, feature importance values will be zero.
            # assert np.isclose(np.sum(dt.feature_importances_), 1)
            if hasattr(dt, "feature_importance"):
                self.feature_importance = np.max(
                    [dt.feature_importance, self.feature_importance], axis=0
                )
            elif hasattr(dt, "feature_importances_"):
                self.feature_importance = np.max(
                    [dt.feature_importances_, self.feature_importance], axis=0
                )
            elif isinstance(dt, MultiOutputRegressor):
                feature_importance = np.max(
                    [d.feature_importances_ for d in dt.estimators_], axis=0
                )
                self.feature_importance = np.max(
                    [feature_importance, self.feature_importance], axis=0
                )
            else:
                raise Exception
        self.feature_importance /= self.feature_importance.sum()
        self.feature_importance = np.nan_to_num(self.feature_importance)
        assert np.all(self.feature_importance >= 0)
        return self

    def predict_proba(self, X):
        prediction = self.ridge.predict_proba(X)
        if np.isnan(prediction).any():
            raise Exception
        # assert not np.isnan(prediction).any(), f"Coefficient of LR: {self.ridge.coef_},{self.ridge.intercept_},{prediction},{X}"
        for dt in self.dt:
            prediction += dt.predict(X)
        prediction = softmax(prediction, axis=1)
        return prediction

    def predict(self, X):
        if not self.regression:
            return np.argmax(self.predict_proba(X))
        prediction = self.ridge.predict(X)
        for dt in self.dt:
            prediction += dt.predict(X)
        return prediction


class RidgeDT(RPLBaseModel, RegressorMixin):
    def __init__(self, decision_tree_count=0, max_leaf_nodes=4):
        if max_leaf_nodes is None:
            max_leaf_nodes = 4
        self.ridge = Ridge()
        self.base_model = PLTreeRegressor
        super().__init__(decision_tree_count, max_leaf_nodes)


class LRDTClassifier(RPLBaseModel, ClassifierMixin):
    def __init__(self, decision_tree_count=1, max_leaf_nodes=4):
        if max_leaf_nodes is None:
            max_leaf_nodes = 4
        self.ridge = LogisticRegression(max_iter=1000, solver="liblinear")
        self.base_model = PLTreeRegressor
        super().__init__(decision_tree_count, max_leaf_nodes)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return super().fit(X, y)


class RidgeDTPlus(RidgeDT):
    """
    Using a decision tree regressor to further improve the performance.
    """

    def __init__(
        self,
        decision_tree_count=0,
        max_leaf_nodes=4,
        min_samples_leaf=1,  # min_samples_leaf of the last decision tree
        final_model_splitter="random",
    ):
        super().__init__(decision_tree_count, max_leaf_nodes)
        self.final_model_splitter = final_model_splitter
        self.min_samples_leaf = min_samples_leaf
        self.dt.append(
            DecisionTreeRegressor(
                splitter=final_model_splitter, min_samples_leaf=min_samples_leaf
            )
        )


class PLTreeClassifier(DecisionTreeClassifier, PLTree):
    """
    A simple implementation of piecewise linear regression tree
    """

    def __init__(self, base_model="LR", **kwargs):
        self.base_model = base_model
        super().__init__(**kwargs)

    def model_controller(self):
        if self.base_model == "LR":
            cv = LogisticRegression(solver="liblinear")
        elif isinstance(self.base_model, ClassifierMixin):
            cv = self.base_model
        else:
            raise Exception
        return cv

    def predict_proba(self, X, check_input=True):
        labels = super().apply(X)
        return self.post_predict(X, labels, method="predict_proba")


class SoftPLTreeRegressor(RegressorMixin, PLTree):
    """
    A simple implementation of piecewise linear regression tree
    """

    def __init__(
        self,
        *,
        base_model="RidgeCV",
        feature_num=None,
        gene_num=None,
        only_constructed_features=True,
        only_original_features=True,
        partition_model="DecisionTree",
        partition_number=1,
        **kwargs,
    ):
        """
        :param feature_num: Number of all features
        :param gene_num: Number of constructed features
        """
        self.base_model = base_model
        self.feature_num = feature_num
        self.gene_num = gene_num
        self.importance_value = None
        self.only_constructed_features = only_constructed_features
        self.only_original_features = only_original_features
        self.partition_model = partition_model
        self.partition_number = partition_number
        if partition_model == "DecisionTree":
            self.partition_tree = DecisionTreeClassifier(**kwargs)
        elif partition_model == "DecisionTree-Regression":
            self.partition_tree = DecisionTreeRegressor(max_leaf_nodes=partition_number)
        elif partition_model == "LogisticRegression":
            self.partition_tree = LogisticRegression(solver="liblinear")
        elif partition_model == "GMM":
            self.partition_tree = GaussianMixture(n_components=partition_number)
        elif partition_model == "K-Means":
            self.partition_tree = KMeans(n_clusters=partition_number)
        else:
            raise Exception
        super().__init__(**kwargs)

    def model_controller(self):
        if self.base_model == "RidgeCV":
            cv = RidgeCV()
        elif self.base_model == "RidgeCV-Log":
            cv = RidgeCV(alphas=(0.1, 1, 10, 100, 1000))
        elif self.base_model == "Ridge":
            cv = Ridge()
        elif self.base_model == "LR":
            cv = LinearRegression()
        elif isinstance(self.base_model, ClassifierMixin):
            cv = self.base_model
        else:
            raise Exception
        return cv

    def complexity(self):
        # count decision tree nodes
        count = 0
        count += self.partition_tree.tree_.node_count
        # count local model coefficients
        for k, v in self.model_map.items():
            v: LinearRegression
            count += len(v.coef_)
        return count

    def fit(self, X, y, sample_weight=None, check_input=True):
        X, label = X[:, :-1], X[:, -1]
        label = LabelEncoder().fit_transform(label)
        assert X.shape[1] == self.feature_num
        if isinstance(self.partition_tree, DecisionTreeClassifier):
            if X.shape[1] > self.gene_num and self.only_original_features:
                # construct the space partition tree with original features
                self.partition_tree.fit(
                    X[:, self.gene_num :],
                    label,
                    sample_weight=sample_weight,
                    check_input=check_input,
                )
                self.importance_value = np.zeros(self.gene_num)
            else:
                # construct the space partition tree with constructed features
                self.partition_tree.fit(
                    X, label, sample_weight=sample_weight, check_input=check_input
                )
                self.importance_value = self.partition_tree.feature_importances_
        elif isinstance(self.partition_tree, DecisionTreeRegressor):
            self.partition_tree.fit(X, label)
            self.importance_value = self.partition_tree.feature_importances_
        elif isinstance(self.partition_tree, GaussianMixture):
            self.partition_tree.fit(X)
            self.importance_value = np.zeros(X.shape[1])
            self.partition_tree.classes_ = self.partition_tree.predict(X)
        elif isinstance(self.partition_tree, KMeans):
            label = self.partition_tree.fit_predict(X)
            self.partition_tree = DecisionTreeClassifier()
            self.partition_tree.fit(
                X, label, sample_weight=sample_weight, check_input=check_input
            )
            self.importance_value = self.partition_tree.feature_importances_
            self.partition_tree.classes_ = self.partition_tree.predict(X)
        else:
            self.partition_tree.fit(X, label)
            self.importance_value = np.max(self.partition_tree.coef_, axis=0)
        if isinstance(self.partition_tree, (ClassifierMixin, BaseMixture)):
            if X.shape[1] > self.gene_num and self.only_original_features:
                partition_scheme = self.partition_tree.predict_proba(
                    X[:, self.gene_num :]
                )
            else:
                partition_scheme = self.partition_tree.predict_proba(X)
        elif isinstance(self.partition_tree, DecisionTreeRegressor):
            partition_scheme = self.partition_tree.apply(X)
        else:
            raise Exception
        # training final results
        if self.only_constructed_features:
            X = X[:, : self.gene_num]
        self.post_fit(X, y, partition_scheme=partition_scheme)

    def predict(self, X, check_input=True):
        labels_prob, predict_values = self.basic_prediction(X)
        predict_values = np.sum(predict_values * labels_prob, axis=1)
        return predict_values

    def basic_prediction(self, X):
        X = X[:, : self.feature_num]
        if X.shape[1] > self.gene_num and self.only_original_features:
            # get the space partition results with original features
            labels = self.partition_tree.predict(X[:, self.gene_num :])
            # probability matrix (size: samples * clusters)
            labels_prob = self.partition_tree.predict_proba(X[:, self.gene_num :])
        else:
            # get the space partition results with constructed features
            labels = self.partition_tree.predict(X)
            # probability matrix (size: samples * clusters)
            labels_prob = self.partition_tree.predict_proba(X)
        if self.only_constructed_features:
            X = X[:, : self.gene_num]
        # prediction result matrix (size: samples * clusters)
        predict_values = self.post_predict(
            X, labels, method="soft_prediction", classes=self.partition_tree.classes_
        )
        return labels_prob, predict_values

    def score(self, X, y, sample_weight=None):
        # best partition scheme
        _, predict_values = self.basic_prediction(X)
        # regret = (((np.sum((predict_values * self.partition_tree.predict_proba(X[:, :-1])), axis=1)
        #             - y.flatten()) ** 2 - np.min((predict_values - y) ** 2, axis=1)))
        # print('Max Regret', np.max(regret))
        return np.argmin((predict_values - y) ** 2, axis=1).astype(int)

    def em_algorithm(self, X, y, random_initialization=False):
        if random_initialization:
            partition_scheme = np.random.randint(0, self.partition_number, len(X))
        else:
            partition_scheme = Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    # ('K-Means', KMeans(self.partition_number)),
                    # ('K-Means', SpectralClustering(self.partition_number)),
                    ("K-Means", DBSCAN()),
                ]
            ).fit_predict(X)
            print(np.unique(partition_scheme))

        inconsistent_partition = np.inf
        iteration = 0
        max_iteration = 20
        cv_em = False
        while inconsistent_partition >= 10:
            if iteration >= max_iteration:
                break
            cv = KFold(n_splits=5, shuffle=True, random_state=0)
            if cv_em:
                # determine the best partition scheme through cross-validation
                new_partition_scheme = np.zeros(len(y))
                for index in cv.split(X):
                    train_index, test_index = index
                    SoftPLTreeRegressor.fit(
                        self,
                        np.concatenate(
                            [
                                X[train_index],
                                np.reshape(partition_scheme[train_index], (-1, 1)),
                            ],
                            axis=1,
                        ),
                        y[train_index],
                    )
                    new_partition_scheme[test_index] = SoftPLTreeRegressor.score(
                        self, X[test_index], y[test_index].reshape(-1, 1)
                    )
            else:
                SoftPLTreeRegressor.fit(
                    self,
                    np.concatenate([X, np.reshape(partition_scheme, (-1, 1))], axis=1),
                    y,
                )
                new_partition_scheme = SoftPLTreeRegressor.score(
                    self, X, y.reshape(-1, 1)
                )
            inconsistent_partition = np.sum(new_partition_scheme != partition_scheme)
            print(
                "iteration", iteration, "inconsistent partition", inconsistent_partition
            )
            partition_scheme = new_partition_scheme
            iteration += 1
        return iteration

    @property
    def feature_importances_(self):
        return self.importance_value


class SoftPLTreeRegressorEM(SoftPLTreeRegressor):
    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"
    ):
        super().em_algorithm(X, y)


def regression_task_demo():
    X, y = load_diabetes(return_X_y=True)
    X, y = np.array(X), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    for i in range(10):
        lr = BaggingRegressor(RidgeDTPlus(decision_tree_count=i), n_estimators=100)
        lr.fit(x_train, y_train)
        print("Training Score", r2_score(lr.predict(x_train), y_train))
        print("Testing Score", r2_score(lr.predict(x_test), y_test))


def classification_task_demo():
    X, y = load_wine(return_X_y=True)
    X, y = np.array(X), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    for dt_count in [0, 1, 2]:
        lr = LRDTClassifier(decision_tree_count=dt_count)
        lr.fit(x_train, y_train)
        print(
            "Training Score",
            roc_auc_score(
                y_train, lr.predict_proba(x_train), average="macro", multi_class="ovo"
            ),
        )
        print(
            "Testing Score",
            roc_auc_score(
                y_test, lr.predict_proba(x_test), average="macro", multi_class="ovo"
            ),
        )


if __name__ == "__main__":
    regression_task_demo()
    # classification_task_demo()
