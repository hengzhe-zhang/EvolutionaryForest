import itertools
import random
import re
from enum import Enum, auto

import numpy as np
from deap import creator, base, tools
from lightgbm import LGBMRegressor
from numba import njit
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from analysis.pac_bayesian.evaluation.dropout import dropout_features
from evolutionary_forest.component.configuration import (
    NoiseConfiguration,
    Configuration,
)
from evolutionary_forest.component.evaluation import (
    noise_generation,
    weighted_sampling_cached,
)
from evolutionary_forest.component.generalization.cache.sharpness_memory import (
    TreeLRUCache,
)
from evolutionary_forest.model.WKNN import GaussianKNNRegressor
from evolutionary_forest.utility.classification_utils import (
    calculate_cross_entropy,
    calculate_zero_one_loss,
)
from evolutionary_forest.utility.sampling_utils import sample_indices_gaussian_kernel
from evolutionary_forest.utils import cv_prediction_from_ridge


@njit(cache=True)
def m_sharpness(baseline, k=4):
    final_baseline = np.zeros(baseline.shape[0])
    random_indices = np.random.choice(len(baseline), baseline.shape[0], replace=False)

    for i in range(0, len(random_indices), k):
        # batch
        g = random_indices[i : i + k]
        tmp = np.zeros(baseline.shape[1])
        for idx in g:
            tmp += baseline[idx]
        # large sharpness
        base_id = np.argmax(tmp)
        # final sharpness over batch
        for idx in g:
            final_baseline[idx] = baseline[idx][base_id]
    return final_baseline


class RademacherConfiguration(Configuration):
    def __init__(self, bound_reduction=False, direct_reduction=False, **params):
        # Using definition of Rademacher complexity to reduce estimated time
        self.bound_reduction = bound_reduction
        self.direct_reduction = direct_reduction


class PACBayesianConfiguration(Configuration):
    def __init__(
        self,
        kl_term_weight: float = 1,
        perturbation_std: float = 1,
        objective="R2,Perturbed-MSE,KL-Divergence",
        l2_penalty=0,
        complexity_estimation_ratio=1,
        optimal_design=False,
        reference_model="KR",
        sharpness_iterations=5,
        automatic_std=False,
        automatic_std_model="KNN",
        only_hard_instance=0,
        sharpness_decay=0,
        structural_sharpness=0,
        adaptive_depth=False,
        classification=False,
        sam_knn_neighbors=1,
        weighted_sam=False,
        prediction_based_sharpness=0,
        intelligent_decision=False,
        dropout_rate=0.2,
        cached_sharpness=False,
        allow_extrapolate_mixup=False,
        kl_mechanism=False,
        sam_standardization=False,
        linear_regularization=False,
        mixup_mode="",
        adaptive_knee_point_metric=None,
        pac_bayesian_subsample=0,
        mixup_skip_self=False,
        classification_loss="CrossEntropy",
        **params,
    ):
        # For dropout
        self.sam_standardization = sam_standardization
        self.kl_mechanism = kl_mechanism
        self.dropout_rate = dropout_rate

        # For VCD
        self.adaptive_depth = adaptive_depth
        self.optimal_design = optimal_design

        # This is a general parameter: not to evaluate individuals with very bad performance
        self.complexity_estimation_ratio = complexity_estimation_ratio
        self.kl_term_weight = kl_term_weight
        self.perturbation_std = perturbation_std
        self.automatic_std = automatic_std
        self.objective = objective
        self.l2_penalty = l2_penalty
        self.sharpness_iterations = sharpness_iterations
        self.automatic_std_model = automatic_std_model
        self.structural_sharpness = structural_sharpness
        self.only_hard_instance = only_hard_instance
        self.sharpness_decay = sharpness_decay
        self.noise_configuration = NoiseConfiguration(**params)
        self.reference_model = reference_model
        self.classification = classification
        self.sam_knn_neighbors = sam_knn_neighbors
        self.weighted_sam = weighted_sam

        # allow missing value imputation
        self.prediction_based_sharpness = prediction_based_sharpness

        # intelligent decision
        self.intelligent_decision = intelligent_decision
        # efficient evaluation
        self.cached_sharpness = cached_sharpness
        self.tree_sharpness_cache = TreeLRUCache()
        # self.cache_radius = cache_radius
        # self.tree_sharpness_cache = LearningTreeCache(self.cache_radius)

        # mixup setting
        self.allow_extrapolate_mixup = allow_extrapolate_mixup
        self.mixup_skip_self = mixup_skip_self
        self.linear_regularization = linear_regularization
        self.mixup_mode = mixup_mode

        # Adaptive Knee
        self.adaptive_knee_point_metric = adaptive_knee_point_metric
        self.pac_bayesian_subsample = pac_bayesian_subsample

        self.classification_loss = classification_loss


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
    kl_term = (w_norm**2) / (2 * sigma**2)
    log_term = np.log(2 * m / delta)
    result = (1 / m) * (kl_term + log_term)
    return 4 * np.sqrt(result)


class SharpnessType(Enum):
    Data = auto()
    Semantics = auto()
    DataGP = auto()
    DataLGBM = auto()
    Parameter = auto()
    DataRealVariance = auto()
    DataGPSource = auto()
    MaxMargin = auto()
    DataGPHybrid = auto()
    ParameterPlus = auto()
    GKNN = auto()
    Dropout = auto()
    ReconstructionLoss = auto()


def pac_bayesian_estimation(
    # X: constructed features
    X,
    # original_X: input features
    original_X,
    y,
    estimator,
    individual,
    configuration: PACBayesianConfiguration,
    sharpness_type: SharpnessType,
    feature_generator=None,
    data_generator=None,
    reference_model: LGBMRegressor = None,
    sharpness_vector=None,
    instance_weights=None,
    historical_best_score=None,
):
    """
    Please pay attention, when calculating the sharpness,
    do not use cross-validation prediction as the predictions.
    Here, we should strictly follow the process of sharpness estimation.
    """
    if configuration.classification:
        original_predictions = individual.pipe.predict_proba(X)
        if configuration.classification_loss == "ZeroOne":
            # This is error rate
            entropy = calculate_zero_one_loss(y, original_predictions)
        else:
            # Two point:
            # 1. KNN
            # 2. ZeroOne Loss
            assert configuration.classification_loss == "CrossEntropy"
            entropy = calculate_cross_entropy(y, original_predictions)
        if instance_weights is not None:
            baseline = entropy * instance_weights
        else:
            baseline = entropy
    else:
        original_predictions = individual.pipe.predict(X)
        baseline = (y - original_predictions) ** 2
        individual.naive_mse = np.mean(baseline)
    if configuration.structural_sharpness > 0:
        if hasattr(individual, "baseline_mse"):
            individual.structural_sharpness = np.mean(
                np.maximum(baseline - individual.baseline_mse, 0)
            )
        else:
            individual.structural_sharpness = 0
        individual.baseline_mse = baseline
    R2 = individual.fitness.wvalues[0]
    # Define the number of iterations
    num_iterations = configuration.sharpness_iterations
    sc = estimator["Scaler"]

    # Create an array to store the R2 scores
    index = None
    if configuration.only_hard_instance != 0:
        mse_scores = np.zeros(
            (num_iterations, int(configuration.only_hard_instance * len(y)))
        )
        index = sample_indices_gaussian_kernel(
            y.flatten(), int(configuration.only_hard_instance * len(y)), replace=False
        )
        baseline = baseline[index]
    else:
        mse_scores = np.zeros((num_iterations, len(X)))
        kl_scores = np.zeros((num_iterations, len(X)))
    mse_scores = mse_scores.astype(np.float32)
    kl_scores = kl_scores.astype(np.float32)

    if sharpness_type == SharpnessType.GKNN:
        base_knn = GaussianKNNRegressor(configuration.sam_knn_neighbors)
        base_knn.fit(X, y)

    derivatives = []
    std = configuration.perturbation_std
    # Iterate over the number of iterations
    for i in range(num_iterations):
        if configuration.intelligent_decision and historical_best_score is not None:
            regularization_objective = configuration.objective.split(",")[1]
            if regularization_objective == "MaxSharpness-1~":
                scores = np.max(mse_scores, axis=0)
                max_sharpness = np.mean(scores)
            elif regularization_objective == "MaxSharpness-1-Base":
                scores = np.vstack((baseline, mse_scores))
                scores = np.max(scores, axis=0)
                scores -= baseline
                max_sharpness = np.mean(scores)
            else:
                raise Exception("Not support!")
            base_score = np.mean(individual.case_values)
            ratio = 1
            if configuration.adaptive_knee_point_metric is not None:
                ratio = configuration.adaptive_knee_point_metric
            if i >= 1 and historical_best_score < ratio * max_sharpness + base_score:
                # At least one iteration is needed
                # don't need evaluation
                continue

        X_noise_plus = None
        # default using original data
        data = X
        target_y = y
        if sharpness_type == SharpnessType.Dropout:
            dropout_rate = configuration.dropout_rate
            X_noise = dropout_features(X, dropout_rate=dropout_rate)
        elif (
            sharpness_type == SharpnessType.Data
            or sharpness_type == SharpnessType.DataGP
            or sharpness_type == SharpnessType.DataLGBM
            or sharpness_type == SharpnessType.DataGPSource
            or sharpness_type == SharpnessType.GKNN
            or sharpness_type == SharpnessType.ReconstructionLoss
        ):
            # Generate some random noise data
            data = data_generator(random_seed=i)
            if isinstance(data, tuple):
                # in some cases, function may return both data and label
                if len(data) == 3:
                    source_indices: list[int]
                    # target y may represent mix-up generated label
                    data, target_y, source_indices = data
                if len(data) == 2:
                    data, target_y = data
            X_noise = sc.transform(feature_generator(data, random_seed=i))
        elif sharpness_type in [SharpnessType.Parameter, SharpnessType.ParameterPlus]:
            # Sharpness-aware minimization
            if configuration.only_hard_instance != 0:
                # worst x%
                input_x = original_X[index]
                target_y = y[index]
            else:
                input_x = original_X
            X_noise = sc.transform(
                feature_generator(
                    input_x,
                    random_noise=configuration.perturbation_std,
                    random_seed=i,
                    noise_configuration=configuration.noise_configuration,
                )
            )
        elif sharpness_type == SharpnessType.DataGPHybrid:
            data = data_generator(random_seed=i)
            if len(data) == 3:
                source_indices: list[int]
                data, target_y, source_indices = data
            X_noise = X
        else:
            raise Exception("Unknown sharpness type!")

        if configuration.sam_standardization:
            # standardization usually doesn't need to be fine-tuned
            # because sharpness is usually estimated based on a fixed model
            X_noise = StandardScaler().fit_transform(X_noise)
        # in most cases, don't need to refit the model
        y_pred_on_noise = get_cv_predictions(
            estimator, X_noise, y, direct_prediction=True
        )

        if (
            isinstance(configuration.objective, str)
            and "Derivative" in configuration.objective
        ):
            # numerical differentiation
            y_pred_plus = get_cv_predictions(
                estimator, X_noise_plus, y, direct_prediction=True
            )
            derivatives.append(np.mean(np.abs((y_pred_plus - y_pred_on_noise))))

        # Calculate the R2 score between the predicted outcomes and the true outcomes
        if sharpness_type == SharpnessType.DataLGBM:
            mse_scores[i] = (
                reference_model.predict(data).flatten() - y_pred_on_noise
            ) ** 2
        elif sharpness_type == SharpnessType.ReconstructionLoss:
            ridge = Ridge()
            ridge.fit(X_noise, original_X)
            mse_scores[i] = np.mean((ridge.predict(X_noise) - original_X) ** 2, axis=1)
        elif sharpness_type == SharpnessType.GKNN:
            mse_scores[i] = (base_knn.predict(X_noise).flatten() - y_pred_on_noise) ** 2
        elif (
            sharpness_type == SharpnessType.DataGP
            or sharpness_type == SharpnessType.ParameterPlus
        ):
            # The squared error between GP predictions and the predictions on noisy data
            # This is the traditional VRM framework
            gp_original_predictions = get_cv_predictions(
                estimator, X, y, direct_prediction=True
            )
            mse_scores[i] = (gp_original_predictions.flatten() - y_pred_on_noise) ** 2
        elif sharpness_type == SharpnessType.DataGPSource:
            # The squared error between GP predictions and interpolated targets
            # This is the mode in VRM decomposition
            # gp_original_predictions: prediction on clean data
            gp_original_predictions = get_cv_predictions(
                estimator, X, y, direct_prediction=True
            )
            mix_target = np.zeros_like(y_pred_on_noise)
            for index, ratio in source_indices:
                mix_target += gp_original_predictions[index] * ratio
            mse_scores[i] = (mix_target - y_pred_on_noise) ** 2
        else:
            if configuration.classification:
                if configuration.classification_loss == "ZeroOne":
                    # This is error rate
                    cross_entropy = calculate_zero_one_loss(
                        target_y,
                        y_pred_on_noise,
                    )
                else:
                    cross_entropy = calculate_cross_entropy(
                        target_y,
                        y_pred_on_noise,
                    )
                if instance_weights is not None:
                    mse_scores[i] = cross_entropy * instance_weights
                else:
                    mse_scores[i] = cross_entropy
            else:
                if configuration.noise_configuration.noise_normalization == "MixT+":
                    random_seed = i
                    size_of_noise = len(y)
                    noise_type = configuration.noise_configuration.noise_type
                    random_noise_magnitude = configuration.perturbation_std
                    reference_label = y
                    sam_mix_bandwidth = (
                        configuration.noise_configuration.sam_mix_bandwidth
                    )

                    noise = noise_generation(
                        noise_type, size_of_noise, random_noise_magnitude, random_seed
                    )
                    sampled_instances = weighted_sampling_cached(
                        random_seed, reference_label, gamma=sam_mix_bandwidth
                    )

                    target_y = (1 - noise) * y + noise * y[sampled_instances]

                mse_scores[i] = (target_y - y_pred_on_noise) ** 2
                kl_scores[i] = y_pred_on_noise**2

    if sharpness_type == SharpnessType.DataRealVariance:
        # This is the real variance
        mean_score = np.mean(mse_scores, axis=0)
        for i in range(len(mse_scores)):
            mse_scores[i] = (mse_scores[i] - mean_score) ** 2

    objectives = []
    for s in configuration.objective.split(","):
        if "*" in s:
            weight, s = s.split("*")
            weight = float(weight)
        else:
            weight = 1

        if s == "PVRM":
            s = "MaxSharpness-1~"

        """
        Actually, the weight in fitness list doesn't matter, because we didn't consider it.
        The rule is: Maximize first, minimize others.
        """
        if s == "R2":
            objectives.append((R2, 1 * weight))
        elif s == "Perturbed-MSE" or s == "MeanSharpness" or s == "MeanVariance":
            # Compute the mean
            perturbed_mse = np.mean(mse_scores)
            # mean-sharpness, which follows PAC-Bayesian
            objectives.append((perturbed_mse, -1 * weight))
        elif s == "MeanSharpness-Base":
            # mean-sharpness, which strictly follows PAC-Bayesian theory
            # subtract baseline MSE
            baseline_mse = mean_squared_error(y, original_predictions)
            # average over samples
            sharp_mse = np.mean(mse_scores, axis=1)
            # average over perturbations
            """
            In very rare cases, the average sharpness could be less than 0.
            This is undesired because it indicates the model located at the a very bad loss maximum point,
            and adding arbitrary noise could improve the performance.
            """
            protected_sharpness = max(np.mean(sharp_mse - baseline_mse), 0)
            objectives.append((protected_sharpness, -1 * weight))
        elif s == "MaxSharpness":
            # n-SAM, reduce the maximum sharpness over all samples
            # average over samples
            sharp_mse = np.mean(mse_scores, axis=1)
            # max over perturbations
            objectives.append((np.max(sharp_mse), -1 * weight))
        elif s == "MaxSharpness-Base" or s == "MaxSharpness-Base+":
            # n-SAM, reduce the maximum sharpness over all samples
            # subtract baseline MSE
            baseline_mse = mean_squared_error(y, original_predictions)
            mse_scores = np.vstack((mse_scores, baseline))
            if s == "MaxSharpness+":
                best_index = np.argmax(np.mean(mse_scores, axis=1))
                max_sharp = mse_scores[best_index]
                sharpness_vector[:] = max_sharp
            # average over samples
            sharp_mse = np.mean(mse_scores, axis=1)
            # max over perturbations
            max_sharpness = np.max(sharp_mse - baseline_mse)
            objectives.append((max_sharpness, -1 * weight))
        elif s == "MaxSharpness-1" or s == "MaxSharpness-1~":
            # 1-SAM, reduce the maximum sharpness over each sample
            """
            Warning: Please be caution to include baseline, because sometimes the sharpness is not a full loss
            Sharpness: Loss after perturbation
            """
            assert not (
                sharpness_type == SharpnessType.DataGP and s == "MaxSharpness-1"
            )
            if s == "MaxSharpness-1":
                # include baseline
                mse_scores = np.vstack((mse_scores, baseline))
            # max for each sample
            max_sharp = np.max(mse_scores, axis=0)
            max_sharpness = np.mean(max_sharp)
            objectives.append((max_sharpness, -1 * weight))
        elif s == "MaxSharpness-1-Base" or s == "MaxSharpness-1-Base+":
            # 1-SAM, reduce the maximum sharpness over each sample
            # subtract baseline MSE
            mse_scores = np.vstack((mse_scores, baseline))
            # max for each sample
            max_sharp = np.max(mse_scores, axis=0)
            max_sharp -= baseline
            if configuration.weighted_sam != False:
                individual.sharpness_vector = max_sharp

            if s == "MaxSharpness-1-Base+":
                sharpness_vector[:] = max_sharp
            max_sharpness = np.mean(max_sharp)
            if configuration.kl_mechanism == "Sharp":
                kl_scores = np.vstack((kl_scores, original_predictions**2))
                kl_scores = np.maximum(kl_scores, 0)
                kl_scores -= original_predictions**2
                max_kl_scores = np.max(kl_scores, axis=0)
                # print("np.mean(kl_scores)", np.mean(kl_scores))
                max_sharpness = max_sharpness + np.mean(max_kl_scores)
            elif configuration.kl_mechanism == "KL":
                kl_score = kl_term_function(
                    len(X),
                    original_predictions**2,
                    std,
                )
                # print("kl_score", kl_score)
                max_sharpness = max_sharpness + kl_score
            elif configuration.kl_mechanism == "Weighted-KL":
                kl_weight = configuration.kl_term_weight
                max_sharpness = max_sharpness + kl_weight * np.mean(
                    (original_predictions**2)
                )
                # print("Mean prediction", np.mean((original_predictions**2)))
            elif configuration.kl_mechanism == "Weighted-KL+":
                kl_weight = configuration.kl_term_weight
                kl_scores = np.vstack((kl_scores, original_predictions**2))
                max_kl_scores = np.max(kl_scores, axis=0)
                max_sharpness = max_sharpness + kl_weight * np.mean(max_kl_scores)
                # print("best KL", np.mean(max_kl_scores))

            objectives.append((max_sharpness, -1 * weight))
        elif check_format(s):
            # K-SAM, reduce the maximum sharpness over K samples
            # subtract baseline MSE
            mse_scores = np.vstack((mse_scores, baseline))
            _, k, _ = s.split("-")
            max_sharp = m_sharpness(mse_scores.T, int(k))
            max_sharp -= baseline
            if s.endswith("+"):
                sharpness_vector[:] = max_sharp
            max_sharpness = np.mean(max_sharp)
            objectives.append((max_sharpness, -1 * weight))
        elif s == "Derivative":
            derivative = np.max(derivatives)
            objectives.append((derivative, -1 * weight))
        elif s == "KL-Divergence":
            if np.sum(std) == 0:
                kl_divergence = np.inf
            else:
                kl_divergence = kl_term_function(len(X.flatten()), X.flatten(), std)
            objectives.append((kl_divergence, -1 * weight))
        elif s == "Size":
            objectives.append((np.sum([len(g) for g in individual.gene]), -1 * weight))
        else:
            raise ValueError("Unknown objective function!")
    return tuple(objectives)


def check_format(input_string):
    pattern = r"^MaxSharpness-(\d+)-Base\+?$"
    match = re.match(pattern, input_string)
    return bool(match)


def get_cv_predictions(estimator, X, y, direct_prediction=True):
    base_model = estimator["Ridge"]
    if isinstance(base_model, RidgeCV) and not direct_prediction:
        y_pred = cv_prediction_from_ridge(y, base_model)
    else:
        if isinstance(base_model, ClassifierMixin):
            y_pred = base_model.predict_proba(X)
        else:
            y_pred = base_model.predict(X)
    return y_pred


def get_adaptive_std(estimator):
    ridge_model: RidgeCV = estimator["Ridge"]
    coef_intercept = np.concatenate(
        (ridge_model.coef_, np.array([ridge_model.intercept_]))
    )
    std = np.mean(np.abs(coef_intercept))
    return std


def assign_rank(population, hof, external_archive):
    all_individuals = combine_individuals(population, hof, external_archive)

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
                np.mean(
                    list(
                        itertools.starmap(
                            lambda rank, fitness_weight: rank * (-fitness_weight[1]),
                            zip(ind.rank_list, ind.fitness_list),
                        )
                    )
                ),
            )
            # after this stage, R2 scores are weighted by a negative weight
            # better values will get a smaller rank, which is correct
        else:
            ind.fitness.values = (np.mean(ind.rank_list),)
    return all_individuals


def combine_individuals(population, hof, external_archive):
    # Combine population and hall of fame
    if external_archive != None:
        all_individuals = population + list(hof) + list(external_archive)
    else:
        all_individuals = population + list(hof)
    return all_individuals


def pac_bayesian_example():
    X, y = load_diabetes(return_X_y=True)
    # X, y = make_friedman1(n_samples=100, n_features=10)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
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
    estimator = Pipeline(steps=[("Ridge", Ridge(alpha=0.01))])
    estimator.fit(X_train, y_train)
    # Calculate the R2 score on the test set
    print("Test R2", r2_score(y_test, estimator.predict(X_test)))
    print("A", pac_bayesian_estimation(X_train, y_train, estimator, config))

    pf = PolynomialFeatures(degree=2)
    X_features = pf.fit_transform(X_train)
    estimator = Pipeline(steps=[("Ridge", Ridge(alpha=0.01))])
    estimator.fit(X_features, y_train)
    # Calculate the R2 score on the test set
    print("Test R2", r2_score(y_test, estimator.predict(pf.fit_transform(X_test))))
    print("B", pac_bayesian_estimation(X_features, y_train, estimator, config))

    pf = PolynomialFeatures(degree=3)
    X_features = pf.fit_transform(X_train)
    estimator = Pipeline(steps=[("Ridge", Ridge(alpha=0.01))])
    estimator.fit(X_features, y_train)
    # Calculate the R2 score on the test set
    print("Test R2", r2_score(y_test, estimator.predict(pf.fit_transform(X_test))))
    print("C", pac_bayesian_estimation(X_features, y_train, estimator, config))

    X_features = pf.fit_transform(X_train)
    estimator = Pipeline(steps=[("Ridge", Ridge(alpha=0.01))])
    # assume y_train is a NumPy array
    shuffled_X_train = np.copy(X_features)
    np.random.shuffle(shuffled_X_train)
    estimator.fit(shuffled_X_train, y_train)
    # Calculate the R2 score on the test set
    print("Test R2", r2_score(y_test, estimator.predict(pf.fit_transform(X_test))))
    print("D", pac_bayesian_estimation(shuffled_X_train, y_train, estimator, config))


def rank_fitness_example():
    # Define the fitness function
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    # Generate a population of 50 individuals with random fitness values
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3
    )
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


if __name__ == "__main__":
    # pac_bayesian_example()
    # rank_fitness_example()
    # Example usage:
    input_string1 = "MaxSharpness-4-Base"
    input_string2 = "MaxSharpness-4-Base+"
    input_string3 = "MaxSharpness-Base"
    input_string4 = "MaxSharpness-123-Base+"

    print(check_format(input_string1))  # Output: True
    print(check_format(input_string2))  # Output: True
    print(check_format(input_string3))  # Output: False
    print(check_format(input_string4))  # Output: True
