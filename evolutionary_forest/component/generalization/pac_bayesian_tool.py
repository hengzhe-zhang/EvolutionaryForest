import copy
from typing import TYPE_CHECKING, Callable
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from evolutionary_forest.component.generalization.pac_bayesian import (
    pac_bayesian_estimation,
    SharpnessType,
    PACBayesianConfiguration,
)

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def automatic_perturbation_std_calculation_classification(X, y):
    model = ExtraTreesClassifier(random_state=0)

    # Perform cross-validation to calculate the score
    scores = cross_val_score(
        model, X, y, cv=5, scoring="balanced_accuracy"
    )  # Using R^2 score
    mean_score = np.mean(scores)

    return criterion_normal(mean_score)


def automatic_perturbation_std_calculation(X, y, criterion="Normal"):
    # Initialize the ExtraTreesRegressor
    model = ExtraTreesRegressor(random_state=0)

    # Perform cross-validation to calculate the score
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")  # Using R^2 score
    mean_score = np.mean(scores)

    if criterion == "Normal":
        return criterion_normal(mean_score)
    elif criterion == "High":
        return criterion_high(mean_score)
    elif criterion == "Low":
        return criterion_low(mean_score)


def criterion_normal(mean_score):
    # Return standard deviation based on the score
    if mean_score < 0.5:
        return 0.5
    elif mean_score < 0.8:
        return 0.3
    else:
        return 0.1


def criterion_high(mean_score):
    # Return standard deviation based on the score
    if mean_score < 0.5:
        return 0.5
    else:
        return 0.3


def criterion_low(mean_score):
    # Return standard deviation based on the score
    if mean_score < 0.8:
        return 0.3
    else:
        return 0.1


def automatic_perturbation_std(self: "EvolutionaryForestRegressor", population):
    pac_bayesian: PACBayesianConfiguration = self.pac_bayesian
    automatic_std_model = pac_bayesian.automatic_std_model
    for individual in population:
        individual.backup_predictions = individual.predicted_values
    best_score = np.inf
    best_alpha = 0.1
    possible_alphas = [1, 3e-1, 1e-1, 3e-2, 1e-2, 1e-3, 0]
    for alpha in possible_alphas:
        pac_bayesian.perturbation_std = alpha
        scores = []
        val_scores = []

        # automatically determine parameter
        for individual in population:
            feature_generator = (
                lambda data,
                random_noise=0,
                noise_configuration=None: self.feature_generation(
                    data,
                    individual,
                    random_noise=random_noise,
                    noise_configuration=noise_configuration,
                )
            )
            kf_scores = []
            kf_val_scores = []

            kf = KFold(n_splits=5)
            for train_index, val_index in kf.split(self.X):
                X_train, X_val = self.X[train_index], self.X[val_index]
                y_train, y_val = self.y[train_index], self.y[val_index]
                features = feature_generator(X_train)

                # please do not touch the original pipeline
                if automatic_std_model == "KNN+":
                    pipe = get_cross_validation_model("KNN", features, None, y_train)
                elif automatic_std_model == "DT+":
                    pipe = get_cross_validation_model("DT", features, None, y_train)
                else:
                    pipe = copy.deepcopy(individual.pipe)
                pipe.fit(features, y_train)

                individual.predicted_values = pipe.predict(features)

                # PAC-Bayesian
                estimation = pac_bayesian_estimation(
                    features,
                    X_train,
                    y_train,
                    pipe,
                    individual,
                    pac_bayesian,
                    SharpnessType.Parameter,
                    feature_generator=feature_generator,
                )
                sharpness_value = estimation[1][0]
                mse_training = mean_squared_error(y_train, individual.predicted_values)
                final_score = sharpness_value + mse_training
                kf_scores.append(final_score)

                pipe = get_cross_validation_model(
                    automatic_std_model, features, pipe, y_train
                )

                mse_validation = predict_on_validation_set(
                    X_val, y_val, feature_generator, pipe
                )
                kf_val_scores.append(mse_validation)
            scores.append(np.mean(kf_scores))
            val_scores.append(np.mean(kf_val_scores))
        best_val_score = val_scores[np.argmin(scores)]
        # print('alpha', alpha,
        #       'correlation', spearman(scores, val_scores),
        #       'best score', best_val_score,
        #       'best sharpness', np.min(scores))
        if best_score > best_val_score:
            best_score = best_val_score
            best_alpha = alpha
    pac_bayesian.perturbation_std = best_alpha
    print("Current best alpha", best_alpha)
    for individual in population:
        individual.predicted_values = individual.backup_predictions
        del individual.backup_predictions

    # need to refresh all individuals
    # otherwise, an unfair comparison will happen,
    # for example, if an individual is evaluated on low perturbation,
    # it definitely has low sharpness
    self.post_processing_after_evaluation([], self.pop)


def predict_on_validation_set(X_val, y_val, feature_generator, pipe):
    # prediction on validation
    val_features = feature_generator(X_val)
    val_prediction = pipe.predict(val_features)
    mse_validation = mean_squared_error(y_val, val_prediction)
    return mse_validation


def get_cross_validation_model(automatic_std_model, features, pipe, y_train):
    # real cross-validation
    if automatic_std_model == "KNN":
        pipe = Pipeline(
            [
                ("Scaler", StandardScaler()),
                ("Ridge", KNeighborsRegressor()),
            ]
        )
        pipe.fit(features, y_train)
    elif automatic_std_model == "DT":
        pipe = Pipeline(
            [
                ("Scaler", StandardScaler()),
                ("Ridge", DecisionTreeRegressor()),
            ]
        )
        pipe.fit(features, y_train)
    else:
        pass
    return pipe


def tune_perturbation_std(model_function: Callable, X, y):
    # Define the range of perturbation_std values to try
    perturbation_std_values = [0.05, 0.1, 0.25, 0.5]

    # Create an empty list to store R2 scores for each perturbation_std value
    r2_scores = []

    # Perform 3-fold cross-validation manually
    kf = KFold(n_splits=3, shuffle=True, random_state=0)

    for perturbation_std in perturbation_std_values:
        r2_fold_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create an EvolutionaryForestRegressor model and set the perturbation_std parameter
            model = model_function()
            model.pac_bayesian.perturbation_std = perturbation_std
            model.early_stop = 10

            # Train the model on the training data
            model.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = model.predict(X_test)

            # Calculate the R2 score for this fold
            r2_fold = r2_score(y_test, y_pred)
            r2_fold_scores.append(r2_fold)

        # Calculate the average R2 score across all folds for this perturbation_std value
        r2_mean = np.mean(r2_fold_scores)
        r2_scores.append(r2_mean)

    # Find the best perturbation_std value based on R2 performance
    best_perturbation_std = perturbation_std_values[np.argmax(r2_scores)]
    best_score = r2_scores[np.argmax(r2_scores)]
    return best_perturbation_std


def get_individual_depth(individual):
    # Compute average depth across all trees in the individual
    return np.mean([tree.height for tree in individual.gene])


def sharpness_based_dynamic_depth_limit(individuals, initial_max_depth, alpha=0.05):
    """
    Adjust the depth limit based on the correlation between sharpness and average tree depth.

    individuals: List of individuals, where each individual contains multiple GP trees.
    initial_max_depth: Initial maximum depth set for GP trees.
    alpha: Significance level for correlation. Default is 0.05.

    Returns: adjusted maximum depth.
    """
    sharpnesses = [-1 * ind.fitness.wvalues[1] for ind in individuals]
    avg_depths = [get_individual_depth(ind) for ind in individuals]

    correlation, p_value = pearsonr(sharpnesses, avg_depths)

    # print('correlation', correlation, 'p_value', p_value)

    # Check if the correlation is statistically significant
    if p_value < alpha:
        adjustment_factor = 1 - abs(correlation)
        adjusted_depth = int(initial_max_depth * adjustment_factor)

        # Apply safety limits
        adjusted_depth = max(int(0.5 * initial_max_depth), adjusted_depth)
        adjusted_depth = min(initial_max_depth, adjusted_depth)

        return adjusted_depth
    else:
        return initial_max_depth
