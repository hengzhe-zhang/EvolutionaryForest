import copy
import itertools
import random

import numpy as np
from deap import creator, base, tools
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class PACBayesianConfiguration():
    def __init__(self, kl_term_weight: float = 1,
                 perturbation_std: float = 0.05,
                 objective='Perturbed-MSE,KL-Divergence',
                 l2_penalty=0,
                 complexity_estimation_ratio=1,
                 bound_reduction=False,
                 optimal_design=False,
                 **params):
        self.optimal_design = optimal_design
        self.bound_reduction = bound_reduction
        self.complexity_estimation_ratio = complexity_estimation_ratio
        self.kl_term_weight = kl_term_weight
        self.perturbation_std = perturbation_std
        self.objective = objective
        self.l2_penalty = l2_penalty


def kl_term_function(m, w, sigma, delta=0.1):
    """
        Parameters:
        m (int): The number of training samples.
        w (numpy.array): The weight vector of the model.
        sigma (float): The standard deviation of the Gaussian prior on the weights.
        delta (float): The confidence parameter (0 < delta < 1), used to control the
                       trade-off between the bound's tightness and the probability
                       that the bound holds.

        Returns:
        float: The calculated PAC-Bayesian bound term.
    """

    w_norm = np.linalg.norm(w)
    kl_term = (w_norm ** 2) / (2 * sigma ** 2)
    log_term = np.log(2 * m / delta)
    result = (1 / m) * (kl_term + log_term)
    return 4 * np.sqrt(result)


def pac_bayesian_estimation(X, y, estimator, configuration: PACBayesianConfiguration):
    original_mse = mean_squared_error(y, get_cv_predictions(estimator, y))

    # Define the number of iterations
    num_iterations = 10
    X = StandardScaler().fit_transform(X)

    # Create an array to store the R2 scores
    mse_scores = np.zeros(num_iterations)
    std = configuration.perturbation_std
    # Iterate over the number of iterations
    for i in range(num_iterations):
        # Add random Gaussian noise to the coefficients and intercept
        X_noise = X + np.random.normal(scale=std, size=X.shape)
        estimator_noise = copy.deepcopy(estimator)
        # Use the modified Ridge model to predict the outcome variable
        estimator_noise.fit(X_noise, y)

        y_pred = get_cv_predictions(estimator_noise, y)

        # Calculate the R2 score between the predicted outcomes and the true outcomes
        mse_scores[i] = mean_squared_error(y, y_pred)

    # Compute the mean and standard deviation of the R2 scores
    perturbation_mse = np.mean(mse_scores)

    if np.sum(std) == 0:
        kl_divergence = np.inf
    else:
        kl_divergence = kl_term_function(len(X.flatten()), X.flatten(), std)

    objectives = []
    for s in configuration.objective.split(','):
        if '*' in s:
            weight, s = s.split('*')
            weight = float(weight)
        else:
            weight = 1

        if s == 'MSE':
            objectives.append((original_mse, -1 * weight))
        elif s == 'Perturbed-MSE':
            objectives.append((perturbation_mse, -1 * weight))
        elif s == 'KL-Divergence':
            objectives.append((kl_divergence, -1 * weight))
        else:
            raise ValueError("Unknown objective function!")
    return tuple(objectives)
    # return len(X) * perturbation_mse + kl_divergence


def get_cv_predictions(estimator_noise, y):
    base_model = estimator_noise['Ridge']
    all_y_pred = (base_model.cv_values_ + y.mean())
    error_list = ((y.reshape(-1, 1) - all_y_pred) ** 2).sum(axis=0)
    new_best_index = np.argmin(error_list)
    y_pred = base_model.cv_values_[:, new_best_index]
    return y_pred


def get_adaptive_std(estimator):
    ridge_model: RidgeCV = estimator['Ridge']
    coef_intercept = np.concatenate((ridge_model.coef_, np.array([ridge_model.intercept_])))
    std = np.mean(np.abs(coef_intercept))
    return std


def assign_rank(population, hof, external_archive):
    # Combine population and hall of fame
    if external_archive != None:
        all_individuals = population + list(hof) + list(external_archive)
    else:
        all_individuals = population + list(hof)

    # Compute ranks for each fitness dimension
    fitness_dims = len(all_individuals[0].fitness_list)
    for ind in all_individuals:
        ind.rank_list = np.zeros((fitness_dims,), dtype=np.int)

    # Rank 1-> The best individual
    for d in range(fitness_dims):
        # fitness_list are minimization objectives
        sorted_inds = sorted(all_individuals, key=lambda x: x.fitness_list[d])
        rank = 0
        for _, group in itertools.groupby(sorted_inds, key=lambda x: x.fitness_list[d]):
            group_list = list(group)
            for ind in group_list:
                ind.rank_list[d] = rank
            rank += len(group_list)
        # after this stage, for R2 scores, smaller values will get a smaller rank
        # we need to reverse this

    # For rank, smaller is better.
    for i, ind in enumerate(all_individuals):
        if isinstance(ind.fitness_list[0], tuple):
            # weight rank by weights in the fitness-weight vector
            ind.fitness.values = (
                np.mean(list(itertools.starmap(lambda rank, fitness_weight: rank * (-fitness_weight[1]),
                                               zip(ind.rank_list, ind.fitness_list)))),
            )
            # after this stage, R2 scores are weighted by a negative weight
            # better values will get a smaller rank, which is correct
        else:
            ind.fitness.values = (np.mean(ind.rank_list),)
    return all_individuals


def pac_bayesian_example():
    X, y = load_diabetes(return_X_y=True)
    # X, y = make_friedman1(n_samples=100, n_features=10)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Standardize X
    scaler_X = StandardScaler()
    X_train_standardized = scaler_X.fit_transform(X_train)
    X_test_standardized = scaler_X.transform(X_test)
    # Standardize y
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_standardized = (y_train - y_mean) / y_std
    y_test_standardized = (y_test - y_mean) / y_std

    config = PACBayesianConfiguration(kl_term_weight=1, perturbation_std=0.01)
    estimator = Pipeline(steps=[('Ridge', Ridge(alpha=0.01))])
    estimator.fit(X_train, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(X_test)))
    print('A', pac_bayesian_estimation(X_train, y_train, estimator, config))

    pf = PolynomialFeatures(degree=2)
    X_features = pf.fit_transform(X_train)
    estimator = Pipeline(steps=[('Ridge', Ridge(alpha=0.01))])
    estimator.fit(X_features, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(pf.fit_transform(X_test))))
    print('B', pac_bayesian_estimation(X_features, y_train, estimator, config))

    pf = PolynomialFeatures(degree=3)
    X_features = pf.fit_transform(X_train)
    estimator = Pipeline(steps=[('Ridge', Ridge(alpha=0.01))])
    estimator.fit(X_features, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(pf.fit_transform(X_test))))
    print('C', pac_bayesian_estimation(X_features, y_train, estimator, config))

    X_features = pf.fit_transform(X_train)
    estimator = Pipeline(steps=[('Ridge', Ridge(alpha=0.01))])
    # assume y_train is a NumPy array
    shuffled_X_train = np.copy(X_features)
    np.random.shuffle(shuffled_X_train)
    estimator.fit(shuffled_X_train, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(pf.fit_transform(X_test))))
    print('D', pac_bayesian_estimation(shuffled_X_train, y_train, estimator, config))


def rank_fitness_example():
    # Define the fitness function
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    # Generate a population of 50 individuals with random fitness values
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=50)
    for ind in population:
        ind.fitness_list = tuple(random.uniform(0.0, 1.0) for _ in range(3))
    # Call the assign_rank function on the population
    hof = []
    all_individuals = assign_rank(population, hof)
    # Check that the ranks are correctly assigned
    for d in range(3):
        sorted_inds = sorted(all_individuals, key=lambda x: x.fitness_list[d])
        for i, ind in enumerate(sorted_inds):
            assert ind.rank_list[d] == i, (ind.rank_list[d], i)
    for ind in all_individuals:
        assert ind.fitness.values == (-1 * sum(ind.rank_list),)


if __name__ == '__main__':
    pac_bayesian_example()
    # rank_fitness_example()
