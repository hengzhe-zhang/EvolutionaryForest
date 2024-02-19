import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


def calculate_adjusted_upper_bound(lb, ub, generalization_gap, mode="Increase"):
    gap_mean = np.mean(generalization_gap)
    gap_mean_clipped = np.clip(gap_mean, 0, 1)  # Clip the mean gap between 0 and 1

    # Convert bounds to linear scale
    lb_linear = np.log(lb)
    ub_linear = np.log(ub)

    # Calculate adjusted upper bound
    if mode == "Increase":
        adjusted_ub_linear = lb_linear + (ub_linear - lb_linear) * gap_mean_clipped
    else:
        adjusted_ub_linear = ub_linear - (ub_linear - lb_linear) * gap_mean_clipped

    # Convert adjusted upper bound back to log scale
    adjusted_ub = np.exp(adjusted_ub_linear)

    return adjusted_ub


def auto_tune_sam(X, y, std):
    _, lb, ub, criterion = std.split("-")
    ub = float(ub)
    lb = float(lb)

    generalization_gap, test_scores = get_error_based_on_decision_tree(X, y)
    if criterion == "GAP":
        # more gap, more penalty
        return lb + (ub - lb) * np.clip(np.mean(generalization_gap), 0, 1)
    elif criterion == "Test":
        # high test score, less penalty
        return ub - (ub - lb) * np.clip(np.mean(test_scores), 0, 1)
    elif criterion == "LogGAP":
        return calculate_adjusted_upper_bound(
            lb, ub, np.mean(generalization_gap), mode="Increase"
        )
    elif criterion == "LogTest":
        return calculate_adjusted_upper_bound(
            lb, ub, np.mean(test_scores), mode="Decrease"
        )
    else:
        raise Exception


def get_error_based_on_decision_tree(X, y):
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
    return generalization_gap, test_scores


def auto_sam_scaling(X, y, strategy):
    # base_sam_std = 0.25
    _, base_sam_std, criterion = strategy.split("-")
    base_sam_std = float(base_sam_std)
    base_size = 100
    if criterion == "Size" or len(X) < base_size:
        return base_size / len(X) * base_sam_std
    generalization_gap, test_scores = get_error_based_on_decision_tree(X, y)
    index = np.arange(0, len(X))
    np.random.shuffle(index)
    index = index[:100]
    generalization_gap_mini, test_scores_mini = get_error_based_on_decision_tree(
        X[index], y[index]
    )
    generalization_gap = np.mean(generalization_gap)
    generalization_gap_mini = np.mean(generalization_gap_mini)
    if criterion == "GAP":
        return generalization_gap / generalization_gap_mini * base_sam_std
    test_scores = np.mean(test_scores)
    test_scores_mini = np.mean(test_scores_mini)
    if criterion == "Test":
        return test_scores_mini / test_scores * base_sam_std
