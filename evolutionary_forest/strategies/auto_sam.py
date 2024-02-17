import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


def auto_tune_sam(X, y, std):
    _, lb, ub, criterion = std.split("-")
    ub = float(ub)
    lb = float(lb)

    model = DecisionTreeRegressor()
    test_scores = []
    generalization_gap = []

    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        test_score = r2_score(y_test, model.predict(X_test))
        train_score = r2_score(y_train, model.predict(X_train))
        generalization_gap.append(train_score - test_score)
        test_scores.append(test_score)
    if criterion == "GAP":
        # more gap, more penalty
        return lb + (ub - lb) * max(np.mean(generalization_gap), 0)
    elif criterion == "Test":
        # high test score, less penalty
        return ub - (ub - lb) * max(np.mean(test_scores), 0)
    else:
        raise Exception
