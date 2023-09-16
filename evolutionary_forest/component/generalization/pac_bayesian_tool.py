import copy
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from evolutionary_forest.component.generalization.pac_bayesian import pac_bayesian_estimation, SharpnessType, \
    PACBayesianConfiguration

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


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
            feature_generator = lambda data, random_noise=0, noise_configuration=None: \
                self.feature_generation(data, individual, random_noise=random_noise,
                                        noise_configuration=noise_configuration)
            kf_scores = []
            kf_val_scores = []

            kf = KFold(n_splits=5)
            for train_index, val_index in kf.split(self.X):
                X_train, X_val = self.X[train_index], self.X[val_index]
                y_train, y_val = self.y[train_index], self.y[val_index]
                features = feature_generator(X_train)

                # please do not touch the original pipeline
                if automatic_std_model == 'KNN+':
                    pipe = get_cross_validation_model('KNN', features, None, y_train)
                elif automatic_std_model == 'DT+':
                    pipe = get_cross_validation_model('DT', features, None, y_train)
                else:
                    pipe = copy.deepcopy(individual.pipe)
                pipe.fit(features, y_train)

                individual.predicted_values = pipe.predict(features)

                # PAC-Bayesian
                estimation = pac_bayesian_estimation(features, X_train, y_train, pipe, individual,
                                                     self.evaluation_configuration.cross_validation,
                                                     pac_bayesian, SharpnessType.Parameter,
                                                     feature_generator=feature_generator)
                sharpness_value = estimation[1][0]
                mse_training = mean_squared_error(y_train, individual.predicted_values)
                final_score = sharpness_value + mse_training
                kf_scores.append(final_score)

                pipe = get_cross_validation_model(automatic_std_model, features, pipe, y_train)

                mse_validation = predict_on_validation_set(X_val, y_val, feature_generator, pipe)
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
    print('Current best alpha', best_alpha)
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
    if automatic_std_model == 'KNN':
        pipe = Pipeline(
            [
                ('Scaler', StandardScaler()),
                ('Ridge', KNeighborsRegressor()),
            ]
        )
        pipe.fit(features, y_train)
    elif automatic_std_model == 'DT':
        pipe = Pipeline(
            [
                ('Scaler', StandardScaler()),
                ('Ridge', DecisionTreeRegressor()),
            ]
        )
        pipe.fit(features, y_train)
    else:
        pass
    return pipe


def tune_perturbation_std(regr: "EvolutionaryForestRegressor", X, y):
    # Define the range of perturbation_std values to try
    perturbation_std_values = [0.05, 0.1, 0.25, 0.5]

    # Create an empty list to store cross-validation scores for each perturbation_std value
    cv_scores = []

    # Evaluate the performance of each perturbation_std value using cross-validation with R2 scoring
    for perturbation_std in perturbation_std_values:
        regr = copy.deepcopy(regr)
        regr.pac_bayesian.perturbation_std = perturbation_std
        regr.early_stop = 10

        # Use 5-fold cross-validation to assess model performance with R2 scoring
        scores = cross_val_score(regr, X, y, cv=3, scoring='r2')
        # Calculate the average R2 score and add it to the cv_scores list
        cv_scores.append(np.mean(scores))

    # Find the best perturbation_std value based on R2 performance
    best_perturbation_std = perturbation_std_values[np.argmax(cv_scores)]
    best_score = cv_scores[np.argmax(cv_scores)]
    return best_perturbation_std
