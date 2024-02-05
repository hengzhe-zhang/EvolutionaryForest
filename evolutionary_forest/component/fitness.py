import random
import time
from abc import abstractmethod
from functools import partial, lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch
from deap.gp import PrimitiveTree, Primitive, Terminal
from deap.tools import sortNondominated
from sklearn.base import ClassifierMixin
from sklearn.metrics import r2_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from torch import optim

from evolutionary_forest.component.evaluation import multi_tree_evaluation
from evolutionary_forest.component.generalization.iodc import (
    create_z,
    create_w,
    calculate_iodc,
)
from evolutionary_forest.component.generalization.pac_bayesian import (
    assign_rank,
    pac_bayesian_estimation,
    SharpnessType,
    combine_individuals,
)
from evolutionary_forest.component.generalization.rademacher_complexity import (
    generate_rademacher_vector,
    rademacher_complexity_estimation,
)
from evolutionary_forest.component.generalization.vc_dimension import (
    vc_dimension_estimation,
)
from evolutionary_forest.component.generalization.wcrv import (
    calculate_WCRV,
    calculate_mic,
)
from evolutionary_forest.model.ASGAN import ASGAN
from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utility.classification_utils import calculate_cross_entropy
from evolutionary_forest.utils import tuple_to_list, list_to_tuple

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class Fitness:
    def __init__(self):
        self.classification = False
        self.instance_weights = None

    def fitness_value(self, individual, estimators, Y, y_pred):
        # basic fitness evaluation
        # warning: Minimization
        if self.classification:
            if self.instance_weights is not None:
                return (
                    np.mean(calculate_cross_entropy(Y, y_pred) * self.instance_weights),
                )
            else:
                return (np.mean(calculate_cross_entropy(Y, y_pred)),)
        else:
            return (-1 * r2_score(Y, y_pred),)

    @abstractmethod
    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        pass


class MTLR2(Fitness):
    def __init__(self, number_of_tasks):
        self.number_of_tasks = number_of_tasks

    def fitness_value(self, individual, estimators, Y, y_pred):
        number_of_tasks = self.number_of_tasks
        Y_partitions = np.array_split(Y, number_of_tasks)
        y_pred_partitions = np.array_split(y_pred, number_of_tasks)

        r_squared_values = []

        for i in range(number_of_tasks):
            mean_y = np.mean(Y_partitions[i])
            tss = np.sum((Y_partitions[i] - mean_y) ** 2)
            rss = np.sum((Y_partitions[i] - y_pred_partitions[i]) ** 2)
            r_squared = 1 - (rss / tss)
            r_squared_values.append(r_squared)
        return (-1 * np.mean(r_squared_values),)


class RademacherComplexityR2(Fitness):
    def __init__(
        self,
        algorithm: "EvolutionaryForestRegressor",
        rademacher_mode="Analytical",
        **params
    ):
        super().__init__()
        self.algorithm = algorithm
        self.size_objective = False
        self.feature_count_objective = False
        self.all_objectives = False
        self.historical_best_bounded_complexity = None
        self.historical_best_bounded_complexity_list = None
        self.rademacher_mode = rademacher_mode

    def fitness_value(self, individual, estimators, Y, y_pred):
        if self.all_objectives:
            individual.l2_norm = np.linalg.norm(y_pred) ** 2
        # very simple fitness evaluation
        return (-1 * r2_score(Y, y_pred),)

    def assign_complexity(self, individual, estimator):
        algorithm = self.algorithm
        # Extract features from input data based on individual
        X_features = algorithm.feature_generation(algorithm.X, individual)
        y = algorithm.y
        # Calculate R2 score, Rademacher complexity and Rademacher complexity list
        (
            estimation,
            bounded_rademacher,
            bounded_rademacher_list,
        ) = rademacher_complexity_estimation(
            X_features,
            y,
            estimator,
            generate_rademacher_vector(algorithm.X),
            self.historical_best_bounded_complexity_list,
            algorithm.pac_bayesian,
            self.rademacher_mode,
        )
        # Store results in individual's fitness list
        weighted_rademacher = estimation[1]
        individual.fitness_list = self.get_fitness_list(
            individual, weighted_rademacher[0]
        )
        # Normalize mean squared error
        normalize_factor = np.mean((np.mean(y) - y) ** 2)
        if algorithm.pac_bayesian.bound_reduction:
            # strictly follow the definition of Rademacher complexity
            bounded_mse = np.mean(
                np.clip(individual.case_values / normalize_factor, 0, 1)
            )
        else:
            bounded_mse = np.mean(individual.case_values / normalize_factor)
        if (
            algorithm.pac_bayesian.bound_reduction
            or algorithm.pac_bayesian.direct_reduction
        ):
            # Reduce training time based on the Rademacher bound
            current_bound = bounded_mse + 2 * bounded_rademacher
            current_bound_list = bounded_mse + 2 * np.array(bounded_rademacher_list)
            if self.historical_best_bounded_complexity is None:
                # Store historical best bound
                self.historical_best_bounded_complexity = current_bound
                self.historical_best_bounded_complexity_list = current_bound_list
            elif self.historical_best_bounded_complexity > current_bound:
                self.historical_best_bounded_complexity = current_bound
                self.historical_best_bounded_complexity_list = current_bound_list
        # assert abs(individual.fitness.wvalues[0] - individual.fitness_list[0][0]) <= 1e-6

    def assign_complexity_pop(self, pop):
        algorithm = self.algorithm
        y = algorithm.y
        normalize_factor = np.mean((np.mean(y) - y) ** 2)
        if algorithm.pac_bayesian.bound_reduction:
            reduced_evaluation = 0
            for p in pop:
                # Calculate bounded MSE
                bounded_mse = np.mean(np.clip(p.case_values / normalize_factor, 0, 1))
                # Update best bounded complexity if needed
                if (
                    self.historical_best_bounded_complexity is None
                    or bounded_mse < self.historical_best_bounded_complexity
                ):
                    self.assign_complexity(p, p.pipe)
                else:
                    p.fitness_list = self.get_fitness_list(p, np.inf)
                    reduced_evaluation += 1
            if algorithm.verbose:
                print("reduced_evaluation: ", reduced_evaluation)
        # If a complexity estimation ratio < 1 is used
        elif algorithm.pac_bayesian.complexity_estimation_ratio < 1:
            self.skip_based_on_threshold(algorithm, pop)
        else:
            for p in pop:
                self.assign_complexity(p, p.pipe)

    def get_fitness_list(self, individual: MultipleGeneGP, rademacher_complexity):
        objective_weight = 1
        if self.size_objective:
            # Calculate tree size
            tree_size = sum([len(tree) for tree in individual.gene])
            return [
                (individual.fitness.wvalues[0], 1),
                (rademacher_complexity, objective_weight),
                (tree_size, objective_weight),
            ]
        elif self.feature_count_objective:
            feature_count = len(individual.gene)
            return [
                (individual.fitness.wvalues[0], 1),
                (rademacher_complexity, objective_weight),
                (feature_count, objective_weight),
            ]
        elif self.all_objectives:
            tree_size = sum([len(tree) for tree in individual.gene])
            feature_count = len(individual.gene)
            return [
                (individual.fitness.wvalues[0], 1),
                (rademacher_complexity, objective_weight),
                (tree_size, objective_weight),
                (feature_count, objective_weight),
                (individual.l2_norm, objective_weight),
            ]
        else:
            return [
                (individual.fitness.wvalues[0], 1),
                (rademacher_complexity, objective_weight),
            ]

    def skip_based_on_threshold(self, algorithm, pop):
        # Get R2 score threshold
        q = np.quantile(
            [p.fitness.wvalues[0] for p in pop],
            q=1 - algorithm.pac_bayesian.complexity_estimation_ratio,
        )
        reduced_evaluation = 0
        for p in pop:
            # If fitness is better than q
            if p.fitness.wvalues[0] > q:
                self.assign_complexity(p, p.pipe)
            else:
                p.fitness_list = self.get_fitness_list(p, np.inf)
                reduced_evaluation += 1

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        reassign_objective_values(parent, population)


def reassign_objective_values(parent, pop):
    if parent != None:
        pop = parent + pop
    valid_components = [
        p.fitness_list[1][0]
        for p in pop
        if p.fitness_list[1][0] < np.inf and not np.isnan(p.fitness_list[1][0])
    ]
    if len(valid_components) == 0:
        max_rademacher = np.inf
    else:
        max_rademacher = max(valid_components)
    for individual in pop:
        individual.fitness.weights = tuple(
            -1 for _ in range(len(individual.fitness_list))
        )
        # R2 should be maximized, other should be minimized
        r2 = [-1 * individual.fitness_list[0][0]]
        rademacher = list(map(lambda x: x[0], individual.fitness_list[1:]))
        if rademacher[0] >= np.inf or np.isnan(rademacher[0]):
            # do not use inf because this will make NSGA-2 problematic
            rademacher[0] = max_rademacher + 1
        individual.fitness.values = tuple(r2 + rademacher)


class LocalRademacherComplexityR2(RademacherComplexityR2):
    def get_fitness_list(self, individual, rademacher_complexity):
        algorithm = self.algorithm
        weights = individual.pipe["Ridge"].coef_
        objective_weight = 1
        return [
            (individual.fitness.wvalues[0], 1),
            (rademacher_complexity, objective_weight),
            (np.linalg.norm(weights, ord=2), objective_weight),
        ]


class RademacherComplexitySizeR2(RademacherComplexityR2):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        super().__init__(algorithm, **params)
        self.size_objective = True


class RademacherComplexityFeatureCountR2(RademacherComplexityR2):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        super().__init__(algorithm, **params)
        self.feature_count_objective = True


class RademacherComplexityAllR2(RademacherComplexityR2):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        super().__init__(algorithm, **params)
        self.all_objectives = True


class RademacherComplexityR2Scaler(RademacherComplexityR2):
    """
    Aggregating fitness values into a scalar value
    """

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        assign_rank(population, hall_of_fame, elite_archive)


class RademacherComplexitySizeR2Scaler(RademacherComplexitySizeR2):
    """
    Aggregating fitness values into a scalar value
    """

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        assign_rank(population, hall_of_fame, elite_archive)


class LocalRademacherComplexityR2Scaler(LocalRademacherComplexityR2):
    """
    Aggregating fitness values into a scalar value
    """

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        assign_rank(population, hall_of_fame, elite_archive)


class TikhonovR2(Fitness):
    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        coef_norm = np.linalg.norm(y_pred, ord=2) ** 2
        return (-1 * score, coef_norm)


class R2GrandComplexity(Fitness):
    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        coef_norm = np.linalg.norm(y_pred, ord=2) ** 2
        tree_size = sum([len(tree) for tree in individual.gene])
        individual.grand_complexity = (coef_norm, tree_size)
        return (-1 * score, 0)

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        # This post-processing is before environmental selection
        if parent is not None:
            population = parent + population
        all_individuals = combine_individuals(population, hall_of_fame, elite_archive)
        for ind in all_individuals:
            ind.original_fitness = ind.fitness.values
            ind.fitness.weights = len(ind.grand_complexity) * (-1,)
            ind.fitness.values = ind.grand_complexity
        layers = sortNondominated(all_individuals, len(all_individuals))
        for gid, layer in enumerate(layers):
            for ind in layer:
                ind.fitness.values = (ind.original_fitness[0], gid)


class R2WCRV(Fitness):
    def __init__(self, algorithm: "EvolutionaryForestRegressor"):
        super().__init__()
        self.algorithm = algorithm

    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        prefix = "ARG"
        # get used variables
        variables = []
        for gene in individual.gene:
            variables.extend(
                [
                    int(node.name.replace(prefix, ""))
                    for node in gene
                    if prefix in node.name
                ]
            )
        variables = set(variables)
        # calculate the WCRV
        inputs = np.array([self.algorithm.X[:, x] for x in variables]).T
        if len(inputs) == 0:
            return (-1 * score, 0)
        residuals = Y - y_pred
        weights = [
            calculate_mic(inputs[:, i], residuals) for i in range(inputs.shape[1])
        ]  # Calculate the weight for each input variable
        median_weight = np.median(weights)  # Calculate the median weight
        wcrv = calculate_WCRV(inputs, residuals, weights, median_weight)
        return (-1 * score, wcrv)


class R2IODC(Fitness):
    def __init__(self, algorithm: "EvolutionaryForestRegressor"):
        super().__init__()
        self.algorithm = algorithm

    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        z = create_z(self.algorithm.X)
        w = create_w(y_pred)
        # maximize IODC
        iodc = calculate_iodc(z, w)
        return (-1 * score, -1 * iodc)


class VCDimensionR2(Fitness):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        super().__init__()
        self.algorithm = algorithm
        self.size_objective = False

    def assign_complexity(self, individual, estimator):
        # reducing the time of estimating VC-Dimension
        algorithm = self.algorithm
        X_features = algorithm.feature_generation(algorithm.X, individual)
        feature_generator = partial(algorithm.feature_generation, individual=individual)
        y = algorithm.y

        # VCD estimation
        gene_length = sum([len(g) for g in individual.gene])
        estimation = vc_dimension_estimation(
            X_features,
            y,
            estimator,
            input_dimension=algorithm.X.shape[1],
            estimated_vcd=gene_length,
            feature_generator=feature_generator,
            optimal_design=algorithm.pac_bayesian.optimal_design,
        )
        individual.fitness_list = estimation
        return (-1 * individual.fitness_list[0][0],)

    def assign_complexity_pop(self, pop):
        # get minimum r2
        ratio = self.algorithm.pac_bayesian.complexity_estimation_ratio
        q = np.quantile([p.fitness.wvalues[0] for p in pop], q=1 - ratio)

        reduced_evaluation = 0
        for p in pop:
            if p.fitness.wvalues[0] > q:
                # better fitness value
                self.assign_complexity(p, p.pipe)
            else:
                p.fitness_list = [(p.fitness.wvalues[0], 1), (np.inf, -1)]
                reduced_evaluation += 1


class R2Size(Fitness):
    def fitness_value(self, individual, estimators, Y, y_pred):
        if self.classification:
            score = np.mean(calculate_cross_entropy(Y, y_pred))
        else:
            score = r2_score(Y, y_pred)
        tree_size = sum([len(tree) for tree in individual.gene])
        return (-1 * score, tree_size)


class R2BootstrapError(Fitness):
    def fitness_value(self, individual, estimators, Y, y_pred):
        if self.classification:
            score = np.mean(calculate_cross_entropy(Y, y_pred))
        else:
            score = r2_score(Y, y_pred)
        bse = []
        semantic_loss = individual.case_values
        for _ in range(50):
            sample_indices = np.arange(len(semantic_loss))
            bootstrap_indices = np.random.choice(
                sample_indices, size=sample_indices.shape[0], replace=True
            )
            bse.append(semantic_loss[bootstrap_indices].mean())
        # minimize standard deviation
        return (-1 * score, np.std(bse))


class R2FeatureCount(Fitness):
    def fitness_value(self, individual: MultipleGeneGP, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        return (-1 * score, individual.gene_num)


class R2SizeScaler(Fitness):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        super().__init__()
        self.algorithm = algorithm

    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        tree_size = sum([len(tree) for tree in individual.gene])
        individual.fitness_list = ((score, 1), (tree_size, -1))
        return (-1 * score,)

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        assign_rank(population, hall_of_fame, elite_archive)


class R2PACBayesian(Fitness):
    def __init__(
        self,
        algorithm: "EvolutionaryForestRegressor",
        sharpness_type="Semantics",
        sharpness_distribution="Normal",
        sharpness_loss_weight=0.5,
        mixup_bandwidth=1,
        **params
    ):
        super().__init__()
        self.sharpness_distribution = sharpness_distribution
        self.algorithm = algorithm
        if sharpness_type == "Data":
            sharpness_type = SharpnessType.Data
        if sharpness_type == "Semantics":
            sharpness_type = SharpnessType.Semantics
        if sharpness_type == "DataGP":
            sharpness_type = SharpnessType.DataGP
        if sharpness_type == "DataLGBM":
            sharpness_type = SharpnessType.DataLGBM
        if sharpness_type == "DataGPSource":
            sharpness_type = SharpnessType.DataGPSource
        if sharpness_type == "DataGPHybrid":
            sharpness_type = SharpnessType.DataGPHybrid
        if sharpness_type == "MaxMargin":
            sharpness_type = SharpnessType.MaxMargin
        if sharpness_type == "Parameter":
            sharpness_type = SharpnessType.Parameter
        if sharpness_type == "DataRealVariance":
            sharpness_type = SharpnessType.DataRealVariance
        self.sharpness_type = sharpness_type
        self.sharpness_loss_weight = sharpness_loss_weight
        self.mixup_bandwith = mixup_bandwidth

    def lazy_init(self):
        if self.sharpness_distribution.startswith(
            "GAN"
        ) or self.sharpness_distribution.startswith("ASGAN"):
            from ctgan import CTGAN

            start = time.time()
            if self.sharpness_distribution.startswith("GAN"):
                self.gan = CTGAN()
            elif self.sharpness_distribution.startswith("ASGAN"):
                self.gan = ASGAN(batch_size=len(self.algorithm.X))
            self.gan.fit(
                np.concatenate(
                    [self.algorithm.X, self.algorithm.y.reshape(-1, 1)], axis=1
                )
            )
            end = time.time()
            gan_verbose = True
            if gan_verbose:
                print("GAN Training Time ", end - start)

    def sample_according_to_probability(
        self, distance_matrix, indices_a, inverse_prob=False
    ):
        """
        Sample indices according to the probability distribution given by the distance matrix.
        """
        prob_distribution = distance_matrix[
            indices_a
        ]  # Extract probabilities for the given indices
        if inverse_prob:
            # inverse the probability vector
            prob_distribution[prob_distribution != 0] = (
                1 / prob_distribution[prob_distribution != 0]
            )
        # Normalize to form a valid probability distribution
        prob_distribution = prob_distribution / np.sum(
            prob_distribution, axis=1, keepdims=True
        )
        # Sample indices according to the probability distribution
        indices_b = [
            np.random.choice(len(distance_matrix), p=prob_distribution[i])
            for i in range(len(indices_a))
        ]
        return indices_b

    @lru_cache(maxsize=128)
    def mixup_gaussian(self, random_seed=0):
        if random.random() < 0.5:
            return self.mixup(random_seed=random_seed, mixup_strategy="I-MixUp")
        else:
            return self.gaussian_noise(
                random_seed=random_seed,
                std=self.algorithm.pac_bayesian.perturbation_std,
            )

    @lru_cache(maxsize=128)
    def mixup(
        self,
        random_seed=0,
        mixup_strategy="C-MixUp",
        alpha_beta=None,
    ):
        # MixUp for data augmentation
        algorithm = self.algorithm
        # Temporarily using perturbation_std as the MixUp parameter
        alpha_beta = (
            self.algorithm.pac_bayesian.perturbation_std
            if alpha_beta is None
            else alpha_beta
        )
        # For this distance matreix, the larger, the near
        distance_matrix = rbf_kernel(
            algorithm.y.reshape(-1, 1), gamma=self.mixup_bandwith
        )
        if alpha_beta == "Adaptive":
            ratio = None
        elif isinstance(alpha_beta, str) and alpha_beta.startswith("Fix"):
            ratio = float(alpha_beta.split("-")[1])
        else:
            ratio = np.random.beta(alpha_beta, alpha_beta, len(algorithm.X))
        indices_a = np.random.randint(0, len(algorithm.X), len(algorithm.X))
        if mixup_strategy in ["I-MixUp", "IN-MixUp"]:
            indices_a = np.arange(0, len(algorithm.X))
            indices_b = self.sample_according_to_probability(distance_matrix, indices_a)
            if alpha_beta == "Adaptive":
                ratio = (
                    1
                    - 0.5
                    * distance_matrix[indices_a][range(0, len(indices_a)), indices_b]
                )
            if mixup_strategy == "I-MixUp":
                ratio = np.where(ratio < 1 - ratio, 1 - ratio, ratio)
        elif mixup_strategy == "D-MixUp":
            """
            1. First, determine the high density and low density data
            2. Then, determine a random index for cross-over
            """
            indices_a = np.arange(0, len(algorithm.X))
            distance_matrix = rbf_kernel(algorithm.y.reshape(-1, 1))
            probability = np.sum(distance_matrix, axis=1)
            probability = probability / np.sum(probability)
            # emphasize on high density region
            indices_b = np.random.choice(
                len(distance_matrix),
                p=probability,
                size=len(distance_matrix),
            )
        elif mixup_strategy == "C-MixUp":
            # sample indices
            distance_matrix = rbf_kernel(
                algorithm.y.reshape(-1, 1), gamma=self.mixup_bandwith
            )
            # for the distance, the large the near, because it's Gaussian
            indices_b = self.sample_according_to_probability(distance_matrix, indices_a)
        else:
            indices_b = np.random.randint(0, len(algorithm.X), len(algorithm.X))
        data = algorithm.X[indices_a] * ratio.reshape(-1, 1) + algorithm.X[
            indices_b
        ] * (1 - ratio.reshape(-1, 1))
        label = algorithm.y[indices_a] * ratio + algorithm.y[indices_b] * (1 - ratio)
        return data, label, ((indices_a, ratio), (indices_b, 1 - ratio))

    @lru_cache(maxsize=128)
    def GAN(self, random_seed=0):
        # GAN for data augmentation
        sampled_data = self.gan.sample(len(self.algorithm.X))
        X, y = sampled_data[:, :-1], sampled_data[:, -1]
        if self.sharpness_distribution.endswith("-N"):
            dis = pairwise_distances(self.algorithm.X, X)
            all_index = dis.argmax(axis=1)
            return X[all_index], y[all_index]
        if self.sharpness_distribution.endswith("-T"):
            lgbm_predict = self.algorithm.reference_lgbm.predict(X)
            dis = pairwise_distances(
                self.algorithm.y.reshape(-1, 1), lgbm_predict.reshape(-1, 1)
            )
            all_index = dis.argmax(axis=1)
            return X[all_index], y[all_index]
        return X, y

    @lru_cache(maxsize=128)
    def gaussian_noise(self, random_seed=0, std=None):
        algorithm = self.algorithm
        return algorithm.X + np.random.normal(
            scale=(self.algorithm.pac_bayesian.perturbation_std if std is None else std)
            * algorithm.X.std(axis=0),
            size=algorithm.X.shape,
        )

    @lru_cache(maxsize=128)
    def uniform_noise(self, random_seed=0, std=None):
        algorithm = self.algorithm
        return algorithm.X + np.random.uniform(
            high=(self.algorithm.pac_bayesian.perturbation_std if std is None else std)
            * algorithm.X.std(axis=0),
            size=algorithm.X.shape,
        )

    @lru_cache(maxsize=128)
    def laplace_noise(self, random_seed=0, std=None):
        algorithm = self.algorithm
        return algorithm.X + np.random.laplace(
            scale=(self.algorithm.pac_bayesian.perturbation_std if std is None else std)
            * algorithm.X.std(axis=0),
            size=algorithm.X.shape,
        )

    def assign_complexity(self, individual, estimator):
        # reducing the time of estimating VC-Dimension
        algorithm = self.algorithm
        X_features = algorithm.feature_generation(algorithm.X, individual)
        # random generate training data
        if self.sharpness_distribution == "Normal":
            data_generator = self.gaussian_noise
        elif self.sharpness_distribution == "Uniform":
            data_generator = self.uniform_noise
        elif self.sharpness_distribution == "Laplace":
            data_generator = self.laplace_noise
        elif self.sharpness_distribution == "NormalMixUp":
            # An ensemble of Gaussian noise and MixUp
            data_generator = self.mixup_gaussian
        elif (
            self.sharpness_distribution == "MixUp"
            or self.sharpness_distribution.endswith("MixUp")
        ):
            if self.sharpness_distribution == "MixUp":
                data_generator = self.mixup
            else:
                data_generator = partial(
                    self.mixup, mixup_strategy=self.sharpness_distribution
                )
        elif self.sharpness_distribution.startswith(
            "GAN"
        ) or self.sharpness_distribution.startswith("ASGAN"):
            data_generator = self.GAN
        else:
            raise Exception
        feature_generator = lambda data, random_noise=0, random_seed=0, noise_configuration=None: algorithm.feature_generation(
            data,
            individual,
            random_noise=random_noise,
            random_seed=random_seed,
            noise_configuration=noise_configuration,
        )
        y = algorithm.y

        sharpness_vector = []
        # PAC-Bayesian estimation
        # return a tuple
        if self.algorithm.constant_type == "GD":
            torch.set_grad_enabled(True)
            torch_variables, trees = self.transform_gene(individual)

            if all(
                (
                    all(isinstance(x, Terminal) for x in gene)
                    or
                    # no primitives
                    all(
                        isinstance(x.value, torch.Tensor)
                        for x in filter(lambda x: isinstance(x, Terminal), gene)
                    )
                )
                # or only constant primitives
                for gene in individual.gene
            ):
                sharpness = np.inf
            else:
                features = multi_tree_evaluation(
                    trees,
                    self.algorithm.pset,
                    self.algorithm.X,
                    self.algorithm.original_features,
                    configuration=self.algorithm.evaluation_configuration,
                )
                if torch.any(torch.isnan(features)):
                    sharpness = np.inf
                else:
                    ridge = estimator["Ridge"]
                    # extract coefficients from linear model
                    weights = ridge.coef_
                    bias = ridge.intercept_
                    weights_torch = torch.tensor(
                        weights, dtype=torch.float32, requires_grad=False
                    )
                    bias_torch = torch.tensor(
                        bias, dtype=torch.float32, requires_grad=False
                    )

                    mean = features.mean(dim=0)
                    std = features.std(dim=0)
                    epsilon = 1e-5
                    features = (features - mean) / (std + epsilon)
                    Y_pred = torch.mm(features, weights_torch.view(-1, 1)) + bias_torch

                    criterion = torch.nn.MSELoss()
                    mse_old = (Y_pred.detach().numpy().flatten() - y) ** 2
                    loss = criterion(Y_pred, torch.from_numpy(y).detach().float())
                    loss.backward()

                    self.sharpness_gradient_ascent(torch_variables)

                    features = multi_tree_evaluation(
                        trees,
                        self.algorithm.pset,
                        self.algorithm.X,
                        self.algorithm.original_features,
                        configuration=self.algorithm.evaluation_configuration,
                    )
                    mean = features.mean(dim=0)
                    std = features.std(dim=0)
                    features = (features - mean) / (std + epsilon)
                    Y_pred = torch.mm(features, weights_torch.view(-1, 1)) + bias_torch
                    # mse_new = criterion(Y_pred, torch.from_numpy(y).detach().float()).item()
                    mse_new = (Y_pred.detach().numpy().flatten() - y) ** 2
                    sharpness = np.maximum(mse_new - mse_old, 0).mean()
                    # print('R2 After', r2_score(y, Y_pred.detach().numpy()))
            sharpness = np.nan_to_num(sharpness, nan=np.inf)
            assert sharpness >= 0
            # print('Sharpness', sharpness, individual.case_values.mean())
            estimation = [(individual.fitness.wvalues[0], 1), (sharpness, -1)]
        else:
            estimation = pac_bayesian_estimation(
                X_features,
                algorithm.X,
                y,
                estimator,
                individual,
                self.algorithm.evaluation_configuration.cross_validation,
                self.algorithm.pac_bayesian,
                self.sharpness_type,
                feature_generator=feature_generator,
                data_generator=data_generator,
                reference_model=self.algorithm.reference_lgbm,
                sharpness_vector=sharpness_vector,
                instance_weights=self.instance_weights,
            )
        if (
            hasattr(individual, "fitness_list")
            and self.algorithm.pac_bayesian.sharpness_decay > 0
        ):
            old_sharpness = individual.fitness_list[1][0]
            estimation = tuple_to_list(estimation)
            estimation[1][0] = (
                self.algorithm.pac_bayesian.sharpness_decay * old_sharpness
                + (1 - self.algorithm.pac_bayesian.sharpness_decay) * estimation[1][0]
            )
            estimation = list_to_tuple(estimation)
        if (
            hasattr(individual, "structural_sharpness")
            and self.algorithm.pac_bayesian.structural_sharpness > 0
        ):
            estimation = tuple_to_list(estimation)
            estimation[1][0] = (
                self.algorithm.pac_bayesian.structural_sharpness
                * individual.structural_sharpness
                + (1 - self.algorithm.pac_bayesian.structural_sharpness)
                * estimation[1][0]
            )
            estimation = list_to_tuple(estimation)

        individual.fitness_list = estimation
        assert len(individual.case_values) > 0
        # [(training R2, 1), (sharpness, -1)]
        sharpness_value = estimation[1][0]
        """
        Here, using cross-validation loss is reasonable
        """
        # using SAM loss as the final selection criterion
        naive_mse = np.mean(individual.case_values)
        # sharpness value is a numerical value
        individual.sam_loss = naive_mse + sharpness_value
        # print('SAM loss: ', individual.sam_loss, naive_mse, sharpness_value)
        if len(sharpness_vector) > 0:
            # if the sharpness vector is available,
            # smaller is better
            individual.case_values = individual.case_values + sharpness_vector
        return (-1 * individual.fitness_list[0][0],)

    def sharpness_gradient_ascent(self, torch_variables):
        # Reverse the direction of the gradients for gradient ascent
        torch_variables = list(filter(lambda v: v.grad is not None, torch_variables))
        for v in torch_variables:
            v.grad = -v.grad
        if self.algorithm.pac_bayesian.noise_configuration.noise_normalization not in [
            None,
            False,
        ]:
            # Compute the norm of the gradient
            gradient_norm = torch.norm(
                torch.stack([v.grad.norm() for v in torch_variables])
            )
            # Scale the gradients to have a norm of 1
            for v in torch_variables:
                v.grad /= gradient_norm
        if self.algorithm.pac_bayesian.perturbation_std == "Adaptive":
            lr = len(torch_variables)
        else:
            lr = self.algorithm.pac_bayesian.perturbation_std
        optimizer = optim.SGD(torch_variables, lr=lr)
        # Update model parameters using an optimizer
        optimizer.step()
        optimizer.zero_grad()

    def transform_gene(self, individual):
        """
        Convert genes into a format with gradient-enabled tensors.
        """
        # Gradient Ascent is supported
        trees = []
        torch_variables = []
        for gene in individual.gene:
            new_gene: PrimitiveTree = PrimitiveTree([])
            for i in range(len(gene)):
                if isinstance(gene[i], Primitive):
                    new_gene.append(self.algorithm.pset.mapping["Add"])
                    v = torch.zeros(1, requires_grad=True, dtype=torch.float32)
                    gp_v = Terminal(v, False, object)
                    new_gene.append(gp_v)
                    torch_variables.append(v)
                    new_gene.append(gene[i])
                elif isinstance(gene[i], Terminal) and isinstance(
                    gene[i].value, torch.Tensor
                ):
                    # avoid gradient interference
                    node = torch.tensor(
                        [gene[i].value.item()], dtype=torch.float32
                    ).requires_grad_(False)
                    new_gene.append(Terminal(node, False, object))
                else:
                    new_gene.append(gene[i])
            trees.append(new_gene)
        return torch_variables, trees

    def assign_complexity_pop(self, pop):
        for p in pop:
            self.assign_complexity(p, p.pipe)

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        # get minimum r2
        ratio = self.algorithm.pac_bayesian.complexity_estimation_ratio
        q = np.quantile([p.fitness.wvalues[0] for p in population], q=1 - ratio)

        reduced_evaluation = 0
        for p in population:
            if p.fitness.wvalues[0] >= q:
                # better fitness value
                self.assign_complexity(p, p.pipe)
            else:
                p.fitness_list = [(p.fitness.wvalues[0], 1), (np.inf, -1)]
                reduced_evaluation += 1

        reassign_objective_values(parent, population)


class PACBayesianR2Scaler(R2PACBayesian):
    """
    Aggregating fitness values into a scalar value
    """

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        for individual in population:
            individual.fitness.values = (individual.sam_loss,)
        # assign_rank(population, hall_of_fame, elite_archive)
