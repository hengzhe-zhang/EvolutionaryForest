import time
from abc import abstractmethod
from functools import partial, lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch
from deap.gp import PrimitiveTree, Primitive, Terminal
from deap.tools import sortNondominated
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, pairwise_distances, mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from evolutionary_forest.component.evaluation import (
    multi_tree_evaluation,
)
from evolutionary_forest.component.generalization.cache.radius_neighbor_cache import (
    LearningTreeCache,
)
from evolutionary_forest.component.generalization.iodc import (
    create_z,
    create_w,
    calculate_iodc,
)
from evolutionary_forest.component.generalization.mixup_utils.mixup_mode_check import (
    mixup_mode_check,
)
from evolutionary_forest.component.generalization.mixup_utils.safety_mixup import (
    safe_mixup,
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
from evolutionary_forest.utility.gradient_optimization.scaling import (
    gradient_agnostic_standardization,
)
from evolutionary_forest.utility.sampling_utils import sample_according_to_distance
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
        """
        Actually, the weight in fitness list doesn't matter, because we didn't consider it.
        The rule is: Maximize first, minimize others.
        """
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
    """
    PAC-Bayesian and Rademacher are very special, they need to be re-assigned.
    Actually, the weight in fitness list doesn't matter, because we didn't consider it.
    """
    if parent != None:
        pop = parent + pop
    for p in pop:
        p.fitness_list = [[element[0]] for element in p.fitness_list]
        assert all([len(element) == 1 for element in p.fitness_list])
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
        coef_norm = np.linalg.norm(y_pred, ord=2)
        return (-1 * score, coef_norm)


class R2GrandComplexity(Fitness):
    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        coef_norm = np.linalg.norm(y_pred, ord=2)
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
        individual.sam_loss = mean_squared_error(Y, y_pred) * np.std(bse)
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


class R2CVGap(Fitness):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        super().__init__()

    def fitness_value(self, individual: MultipleGeneGP, estimators, Y, y_pred):
        cv_mse = np.mean((y_pred - Y) ** 2)
        r2 = r2_score(Y, y_pred)
        trees = self.algorithm.feature_generation(self.algorithm.X, individual)
        prediction = individual.pipe.predict(trees)
        # this should be smaller
        training_mse = np.mean((prediction - Y) ** 2)
        gap = max(cv_mse - training_mse, 0)
        """
        Actually, the weight in fitness list doesn't matter, because we didn't consider it.
        The rule is: Maximize first, minimize others.
        """
        individual.fitness_list = ((r2, 1), (gap, 1))
        individual.sam_loss = cv_mse + gap
        # minimize generalization gap
        return (cv_mse, gap)


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
        if sharpness_type == "Dropout":
            sharpness_type = SharpnessType.Dropout
        if sharpness_type == "ParameterPlus":
            sharpness_type = SharpnessType.ParameterPlus
        if sharpness_type == "GKNN":
            sharpness_type = SharpnessType.GKNN
        if sharpness_type == "DataRealVariance":
            sharpness_type = SharpnessType.DataRealVariance
        self.sharpness_type = sharpness_type
        self.sharpness_loss_weight = sharpness_loss_weight
        self.mixup_bandwidth = mixup_bandwidth

    def lazy_init(self):
        algorithm = self.algorithm
        algorithm.pac_bayesian.mixup_mode = mixup_mode_check(
            algorithm.X, algorithm.y, algorithm.pac_bayesian.mixup_mode
        )

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

    @lru_cache(maxsize=128)
    def mixup_gaussian(self, random_seed=0):
        data, label, ratio = self.mixup(
            random_seed=random_seed, mixup_strategy="I-MixUp", alpha_beta=10
        )
        data = data + np.random.normal(
            scale=self.algorithm.pac_bayesian.perturbation_std * data.std(axis=0),
            size=data.shape,
        )
        return data, label, ratio

    @lru_cache(maxsize=128)
    def mixup(
        self,
        random_seed=0,
        mixup_strategy="C-MixUp",
        alpha_beta=None,
    ):
        allow_extrapolate_mixup = self.algorithm.pac_bayesian.allow_extrapolate_mixup
        # MixUp for data augmentation
        algorithm = self.algorithm
        # Temporarily using perturbation_std as the MixUp parameter
        alpha_beta = (
            self.algorithm.pac_bayesian.perturbation_std
            if alpha_beta is None
            else alpha_beta
        )
        # For this distance matrix, the larger, the near
        mixup_mode = self.algorithm.pac_bayesian.mixup_mode
        if isinstance(self.mixup_bandwidth, str):
            if self.mixup_bandwidth == "AdaptiveMax":
                max_value = np.max(self.algorithm.y)
                range_value = np.max(self.algorithm.y) - np.min(self.algorithm.y)
                if max_value < 2 and range_value < 4:
                    gamma_value = 100
                else:
                    gamma_value = 0.01
            else:
                raise ValueError("Unknown bandwidth")
        else:
            gamma_value = self.mixup_bandwidth

        distance_matrix = rbf_kernel(algorithm.y.reshape(-1, 1), gamma=gamma_value)
        if alpha_beta == "Adaptive":
            ratio = None
        elif isinstance(alpha_beta, str) and alpha_beta.startswith("Fix"):
            ratio = float(alpha_beta.split("-")[1])
        else:
            ratio = np.random.beta(alpha_beta, alpha_beta, len(algorithm.X))
        indices_a = np.random.randint(0, len(algorithm.X), len(algorithm.X))
        if mixup_strategy in ["I-MixUp"]:
            if mixup_mode != "":
                return safe_mixup(
                    algorithm.X,
                    algorithm.y,
                    gamma_value,
                    alpha_beta,
                    mode=mixup_mode,
                )
            indices_a = np.arange(0, len(algorithm.X))
            indices_b = sample_according_to_distance(distance_matrix, indices_a)
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
            distance_matrix = rbf_kernel(algorithm.y.reshape(-1, 1), gamma=gamma_value)
            # for the distance, the large the near, because it's Gaussian
            indices_b = sample_according_to_distance(distance_matrix, indices_a)
        else:
            indices_b = np.random.randint(0, len(algorithm.X), len(algorithm.X))

        data = algorithm.X[indices_a] * ratio.reshape(-1, 1) + algorithm.X[
            indices_b
        ] * (1 - ratio.reshape(-1, 1))
        label = algorithm.y[indices_a] * ratio + algorithm.y[indices_b] * (1 - ratio)
        if allow_extrapolate_mixup:
            # For some point, the ratio could be larger than 1 to simulate extrapolation.
            # However, 1.5 is a very dangerous value.
            temp_ratio = (1 + ratio).reshape(-1, 1)
            data_extrapolation = temp_ratio * algorithm.X[indices_a] + (
                (1 - temp_ratio) * algorithm.X[indices_b]
            )
            label_extrapolation = temp_ratio.flatten() * algorithm.y[indices_a] + (
                (1 - temp_ratio).flatten() * algorithm.y[indices_b]
            )
            # only consider out of distribution samples
            replace_index = (label_extrapolation > algorithm.y.max()) | (
                label_extrapolation < algorithm.y.min()
            )
            temp_ratio = temp_ratio.flatten()
            ratio[replace_index] = temp_ratio[replace_index]
            # print("Extrapolation instances", np.sum(replace_index))
            data[replace_index] = data_extrapolation[replace_index]
            label[replace_index] = label_extrapolation[replace_index]
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

    @lru_cache(maxsize=1)
    def linear_prediction(self):
        prediction = (
            RidgeCV().fit(self.algorithm.X, self.algorithm.y).predict(self.algorithm.X)
        )
        return prediction

    def assign_complexity(self, individual: MultipleGeneGP, estimator):
        # reducing the time of estimating VC-Dimension
        algorithm = self.algorithm
        X_features = algorithm.feature_generation(algorithm.X, individual)
        # random generate training data
        if self.sharpness_distribution == "Normal":
            if algorithm.stochastic_mode:
                data_generator = self.gaussian_noise.__wrapped__
            else:
                data_generator = self.gaussian_noise
        elif self.sharpness_distribution == "Uniform":
            data_generator = self.uniform_noise
        elif self.sharpness_distribution == "Laplace":
            data_generator = self.laplace_noise
        elif self.sharpness_distribution == "GaussianMixUp":
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
        if self.algorithm.constant_type in ["GD+", "GD-", "GD--"]:
            torch.set_grad_enabled(True)
            size_of_data = len(self.algorithm.X)
            torch_variables, trees = self.transform_gp_tree_with_tensors(
                individual, size_of_data
            )

            if all(
                (
                    # has no function
                    all(isinstance(x, Terminal) for x in gene)
                    or
                    # has function, but only constant terminal
                    all(
                        isinstance(x.value, torch.Tensor)
                        for x in filter(lambda x: isinstance(x, Terminal), gene)
                    )
                )
                # or only constant primitives
                for gene in individual.gene
            ):
                sharpness = 0
            else:
                features = self.get_constructed_features(individual, trees)
                if torch.any(torch.isnan(features)):
                    sharpness = np.inf
                else:
                    scaler: StandardScaler = estimator["Scaler"]
                    gradient_agnostic_standardization(features, scaler)

                    ridge = estimator["Ridge"]
                    (
                        bias_torch,
                        weights_torch,
                    ) = self.extract_weights_and_bias_from_linear_model(ridge)
                    Y_pred = self.get_predictions_on_linear_model(
                        features, weights_torch, bias_torch
                    )
                    mse_old = (Y_pred.detach().numpy().flatten() - y) ** 2
                    self.calculate_gradient(Y_pred, y)

                    """
                    Two modes:
                    1. Gradient-based sharpness perturbation
                    2. Gradient Regularization
"""
                    traditional_sam = False
                    if traditional_sam:
                        self.sharpness_gradient_ascent(torch_variables)
                        features = self.get_constructed_features(individual, trees)
                        gradient_agnostic_standardization(features, scaler)
                        Y_pred = self.get_predictions_on_linear_model(
                            features, weights_torch, bias_torch
                        )
                        mse_new = (Y_pred.detach().numpy().flatten() - y) ** 2
                        sharpness = np.maximum(mse_new - mse_old, 0).mean()
                        # print("Sharpness", sharpness, "MSE", mse_old.mean())
                    else:
                        one_step_gradient_ascent = False
                        if one_step_gradient_ascent:
                            self.sharpness_gradient_ascent(torch_variables)
                            features = self.get_constructed_features(individual, trees)
                            gradient_agnostic_standardization(features, scaler)
                            Y_pred = self.get_predictions_on_linear_model(
                                features, weights_torch, bias_torch
                            )
                            self.calculate_gradient(Y_pred, y)

                        # Collect gradients into a list
                        gradients = [
                            v.grad.numpy()
                            for v in torch_variables
                            if v.grad is not None
                        ]

                        # Compute norms of gradients
                        norms = [np.linalg.norm(grad) for grad in gradients]

                        # Compute norm of the vector containing norms
                        norm_of_norms = np.linalg.norm(norms)

                        sharpness = norm_of_norms
                    # print('R2 After', r2_score(y, Y_pred.detach().numpy()))
            gradient_sharpness = np.nan_to_num(sharpness, nan=np.inf)
            assert gradient_sharpness >= 0

        if self.algorithm.constant_type in ["GD+", "GD-"]:
            sharpness_value = gradient_sharpness
            estimation = [(individual.fitness.wvalues[0], 1), (sharpness_value, -1)]
        else:
            if len(self.algorithm.hof) > 0:
                historical_best_score = self.algorithm.hof[0].sam_loss
            else:
                historical_best_score = None

            # no matter what, always need Gaussian estimation
            estimation = pac_bayesian_estimation(
                X_features,
                algorithm.X,
                y,
                estimator,
                individual,
                self.algorithm.pac_bayesian,
                self.sharpness_type,
                feature_generator=feature_generator,
                data_generator=data_generator,
                reference_model=self.algorithm.reference_lgbm,
                sharpness_vector=sharpness_vector,
                instance_weights=self.instance_weights,
                historical_best_score=historical_best_score,
            )
            if (
                hasattr(individual, "fitness_list")
                and self.algorithm.pac_bayesian.sharpness_decay > 0
            ):
                old_sharpness = individual.fitness_list[1][0]
                estimation = tuple_to_list(estimation)
                estimation[1][0] = (
                    self.algorithm.pac_bayesian.sharpness_decay * old_sharpness
                    + (1 - self.algorithm.pac_bayesian.sharpness_decay)
                    * estimation[1][0]
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
        # Encoded information: [(training R2, 1), (sharpness, -1)]
        sharpness_value = estimation[1][0]

        if self.algorithm.constant_type in ["GD--"]:
            """
            In the mode is GD--, not only use gradient norm but also use weight perturbation.
            """
            if algorithm.verbose and sharpness_value > gradient_sharpness:
                pass
            sharpness_value = np.maximum(gradient_sharpness, sharpness_value)
            # sharpness value needs to be updated, so that objective values are consistent
            individual.fitness_list = [
                (individual.fitness.wvalues[0], 1),
                (sharpness_value, -1),
            ]
        """
        Here, using cross-validation loss is reasonable
        """
        # using SAM loss as the final selection criterion
        naive_mse = np.mean(individual.case_values)

        linear_regularization_flag = self.algorithm.pac_bayesian.linear_regularization
        if linear_regularization_flag:
            prediction = individual.pipe.predict(X_features)
            linear_regularization = np.mean(
                (self.linear_prediction() - prediction) ** 2
            )
            if self.algorithm.verbose:
                print("linear score", linear_regularization)
            if linear_regularization_flag == "Max":
                # similar to SAM
                linear_regularization = np.mean(
                    np.maximum(
                        (self.linear_prediction() - prediction) ** 2
                        - (y - prediction) ** 2,
                        0,
                    )
                )
            sharpness_value += linear_regularization

        # sharpness value is a numerical value
        individual.sam_loss = naive_mse + sharpness_value
        # print("SAM loss: ", individual.sam_loss, naive_mse, sharpness_value)
        if len(sharpness_vector) > 0:
            # if the sharpness vector is available,
            # smaller is better
            individual.case_values = individual.case_values + sharpness_vector
        return (-1 * individual.fitness_list[0][0],)

    def extract_weights_and_bias_from_linear_model(self, ridge):
        # extract coefficients from linear model
        weights = ridge.coef_
        bias = ridge.intercept_
        weights_torch = torch.tensor(weights, dtype=torch.float32, requires_grad=False)
        bias_torch = torch.tensor(bias, dtype=torch.float32, requires_grad=False)
        return bias_torch, weights_torch

    def get_predictions_on_linear_model(self, features, weights_torch, bias_torch):
        Y_pred = torch.mm(features, weights_torch.view(-1, 1)) + bias_torch
        return Y_pred

    def calculate_gradient(self, Y_pred, Y_label):
        criterion = torch.nn.MSELoss()
        loss = criterion(Y_pred, torch.from_numpy(Y_label).detach().float())
        loss.backward()

    def get_constructed_features(self, individual, trees):
        features = multi_tree_evaluation(
            trees,
            self.algorithm.pset,
            self.algorithm.X,
            self.algorithm.original_features,
            configuration=self.algorithm.evaluation_configuration,
            individual_configuration=individual.individual_configuration,
        )
        return features

    def sharpness_gradient_ascent(self, torch_variables):
        # Reverse the direction of the gradients for gradient ascent
        torch_variables = list(filter(lambda v: v.grad is not None, torch_variables))
        for v in torch_variables:
            v.grad = -v.grad
        if self.algorithm.pac_bayesian.noise_configuration.noise_normalization not in [
            None,
            False,
        ]:
            instance_wise = True
            if instance_wise:
                # Compute the norm of the gradient
                # gradient_norm = torch.norm(
                #     torch.stack([v.grad for v in torch_variables])
                # )
                gradient_norm = torch.norm(
                    torch.stack([v.grad for v in torch_variables]), dim=0
                )
                # Scale the gradients to have a norm of 1
                for v in torch_variables:
                    v.grad /= gradient_norm * len(torch_variables)
        if self.algorithm.pac_bayesian.perturbation_std == "Adaptive":
            lr = len(torch_variables)
        else:
            lr = self.algorithm.pac_bayesian.perturbation_std
        # optimizer = optim.SGD(torch_variables, lr=lr)
        # # Update model parameters using an optimizer
        # optimizer.step()
        # optimizer.zero_grad()

        # Manual update
        with torch.no_grad():  # Disable gradient calculation for the update step
            for param in torch_variables:
                param -= lr * param.grad  # Update the parameters using gradient descent

    def transform_gp_tree_with_tensors(self, individual, length):
        """
        Convert genes into a format with gradient-enabled tensors.
        """
        # Gradient Ascent is supported
        trees = []
        torch_variables = []
        for gene in individual.gene:
            new_gene: PrimitiveTree = PrimitiveTree([])
            for i in range(len(gene)):
                if isinstance(gene[i], Terminal) and isinstance(
                    gene[i].value, torch.Tensor
                ):
                    # avoid gradient interference
                    node = torch.tensor(
                        [gene[i].value.item()], dtype=torch.float32
                    ).requires_grad_(False)
                    # save it a constant
                    new_gene.append(Terminal(node, False, object))
                # elif i != 0 and (
                #     isinstance(gene[i], Primitive) or isinstance(gene[i], Terminal)
                # ):
                elif isinstance(gene[i], Primitive) or isinstance(gene[i], Terminal):
                    # unless the root node, add a noise term
                    # including both terminal and non-terminal nodes
                    new_gene.append(self.algorithm.pset.mapping["Add"])
                    v = torch.zeros(length, requires_grad=True, dtype=torch.float32)
                    gp_v = Terminal(v, False, object)
                    new_gene.append(gp_v)
                    torch_variables.append(v)
                    new_gene.append(gene[i])
                else:
                    new_gene.append(gene[i])
            trees.append(new_gene)
        return torch_variables, trees

    def assign_complexity_pop(self, pop):
        for p in pop:
            self.assign_complexity(p, p.pipe)

    # @time_it
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

        if isinstance(
            self.algorithm.pac_bayesian.tree_sharpness_cache, LearningTreeCache
        ):
            self.algorithm.pac_bayesian.tree_sharpness_cache.retrain()


class PACBayesianR2Scaler(R2PACBayesian):
    """
    Aggregating fitness values into a scalar value
    """

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        for individual in population:
            individual.fitness.values = (individual.sam_loss,)
        # assign_rank(population, hall_of_fame, elite_archive)
