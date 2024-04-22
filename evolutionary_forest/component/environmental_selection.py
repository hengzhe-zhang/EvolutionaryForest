from abc import abstractmethod
from functools import partial
from operator import attrgetter
from typing import TYPE_CHECKING, Union

import numpy as np
from deap.tools import (
    selNSGA2,
    sortNondominated,
    selSPEA2,
    selBest,
    selNSGA3,
    uniform_reference_points,
)
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from scipy.stats import wilcoxon
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.knee_point_eurogp.utility_function import knee_point_by_utility
from evolutionary_forest.application.baes_class import spearman
from evolutionary_forest.component.bloat_control.simple_simplification import (
    simple_simplification,
)
from evolutionary_forest.component.decision_making.bend_angle_knee import (
    find_knee_based_on_bend_angle,
)
from evolutionary_forest.component.decision_making.euclidian_knee_selection import (
    point_to_line_distance,
    euclidian_knee,
)
from evolutionary_forest.component.decision_making.harmonic_rank import (
    best_harmonic_rank,
)
from evolutionary_forest.component.fitness import R2PACBayesian
from evolutionary_forest.component.primitive_functions import individual_to_tuple
from evolutionary_forest.multigene_gp import (
    multiple_gene_compile,
    result_calculation,
    MultipleGeneGP,
)
from evolutionary_forest.strategies.instance_weighting import calculate_instance_weights
from evolutionary_forest.utility.multiobjective.fitness_normalization import (
    fitness_normalization,
    fitness_restore_back,
)
from evolutionary_forest.utility.statistics.variance_tool import mean_without_outliers

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def knee_point_detection(front, knee_point_strategy: Union[bool, str] = "Knee"):
    front = np.array(front)
    denominator = np.max(front, axis=0) - np.min(front, axis=0)
    denominator = np.where(denominator > 0, denominator, 1)
    front = (front - np.min(front, axis=0)) / denominator
    if knee_point_strategy == "BendAngleKnee":
        # turn to a minimization problem
        _, index = find_knee_based_on_bend_angle(-1 * front)
        return index
    elif knee_point_strategy == "AngleKnee":
        # turn to a minimization problem
        _, index = find_knee_based_on_bend_angle(-1 * front, local=True)
        return index
    elif knee_point_strategy == "UtilityFunction":
        # turn to a minimization problem
        index, _, _ = knee_point_by_utility(-1 * front)
        return index
    elif knee_point_strategy == "AngleKneeF":
        # turn to a minimization problem
        _, index = find_knee_based_on_bend_angle(
            -1 * front, local=True, four_neighbour=True
        )
        return index
    elif isinstance(knee_point_strategy, str) and knee_point_strategy.startswith(
        "AngleKS"
    ):
        weight = float(knee_point_strategy.split("~")[1])
        # turn to a minimization problem
        _, index = find_knee_based_on_bend_angle(
            -1 * front, local=True, number_of_cluster=weight
        )
        return index
    elif isinstance(knee_point_strategy, str) and knee_point_strategy.startswith(
        "AngleMKS"
    ):
        weight = float(knee_point_strategy.split("~")[1])
        # turn to a minimization problem
        _, index = find_knee_based_on_bend_angle(
            -1 * front, local=True, number_of_cluster=weight, minimal_complexity=True
        )
        return index
    elif isinstance(knee_point_strategy, str) and knee_point_strategy.startswith(
        "AngleFKS"
    ):
        weight = float(knee_point_strategy.split("~")[1])
        # turn to a minimization problem
        _, index = find_knee_based_on_bend_angle(
            -1 * front, local=True, number_of_cluster=weight, four_neighbour=True
        )
        return index
    elif knee_point_strategy == "Knee" or knee_point_strategy == True:
        # turn to a minimization problem
        return euclidian_knee(-1 * front)
    elif knee_point_strategy == "BestAdditionalObjetive":
        return np.argmax(front[:, 1])
    elif knee_point_strategy == "BestMainObjetive":
        return np.argmax(front[:, 0])
    elif knee_point_strategy == "BestHarmonicRank":
        return best_harmonic_rank(front)
    elif knee_point_strategy == "HighTradeoff":
        ht = HighTradeoffPoints()
        try:
            # convert to minimization
            ans = ht.do(-1 * front)
            if ans is None:
                print("Empty Answer", front)
                # if no trade-off point, then choosing the point with the highest R2
                return max(range(len(front)), key=lambda x: front[x][0])
            else:
                return max(ans, key=lambda x: front[x][0])
        except (ValueError, IndexError):
            print("Value Error", front)
            # Unknown Exception
            return max(range(len(front)), key=lambda x: front[x][0])
    elif knee_point_strategy == "BestSum":
        pf = front
        pf = (pf - np.min(pf, axis=0)) / (np.max(pf, axis=0) - np.min(pf, axis=0))
        id = pf.sum(axis=1).argmax()
        return id
    elif knee_point_strategy == "BestSumCbrt":
        pf = front
        pf = (pf - np.min(pf, axis=0)) / (np.max(pf, axis=0) - np.min(pf, axis=0))
        pf[:, 1] = np.cbrt(pf[:, 1])
        id = pf.sum(axis=1).argmax()
        return id
    else:
        raise Exception("Unknown Knee Point Strategy")


class EnvironmentalSelection:
    @abstractmethod
    def select(self, population, offspring):
        pass


class Objective:
    @abstractmethod
    def set(self, individuals):
        pass


def get_unique_individuals(individuals):
    """
    Old version: Check by string, without considering permutation
    """
    generated = set()
    result = []

    for ind in individuals:
        s = ind
        encoded_ind = individual_to_tuple(s)
        if encoded_ind not in generated:
            generated.add(encoded_ind)
            result.append(s)
    return result


class NSGA2(EnvironmentalSelection):
    def __init__(
        self,
        algorithm: "EvolutionaryForestRegressor",
        objective_function: Objective = None,
        objective_normalization=False,
        knee_point=False,
        bootstrapping_selection=False,
        first_objective_weight=1,
        max_cluster_point=True,
        handle_objective_duplication=False,
        n_pop=0,
        **kwargs
    ):
        self.handle_objective_duplication = handle_objective_duplication
        self.first_objective_weight = first_objective_weight
        self.bootstrapping_selection = bootstrapping_selection
        self.algorithm: "EvolutionaryForestRegressor" = algorithm
        self.objective_function = objective_function
        self.objective_normalization = objective_normalization
        self.knee_point: Union[str, bool] = knee_point
        # if using clustering for ensemble, max point in each cluster
        self.max_cluster_point = max_cluster_point
        self.selection_operator = selNSGA2
        self.validation_x = None
        self.validation_y = None
        # golden standard
        self.n_pop = n_pop

    def select(self, population, offspring):
        """
        Old version: The size of the population is limited by the number of unique individuals in the first generation
        """
        individuals = population + offspring
        if self.algorithm.pac_bayesian.weighted_sam != False:
            weights = calculate_instance_weights(
                self.algorithm.y, individuals, self.algorithm.pac_bayesian.weighted_sam
            )
            for ind in individuals:
                sam_loss = np.sum(ind.sharpness_vector * weights)
                ind.fitness.values = (ind.fitness.values[0], sam_loss)
                mse = np.mean((ind.predicted_values - self.algorithm.y) ** 2)
                # np.mean(ind.sharpness_vector) + mse
                ind.sam_loss = sam_loss + mse
        # remove exactly the same individuals
        individuals = get_unique_individuals(individuals)
        # remove individuals with the same objective values
        if self.handle_objective_duplication:
            individuals = self.objective_space_duplication_removal(individuals)

        if self.objective_normalization:
            classification_task = isinstance(self.algorithm, ClassifierMixin)
            fitness_normalization(individuals, classification_task)

        population[:] = self.selection_operator(individuals, self.n_pop)
        if self.algorithm.validation_size > 0:
            """
            When have validation data, the knee point is selected using the validation data.
            This has the first priority.
            """
            first_pareto_front = sortNondominated(population, self.n_pop)[0]
            self.algorithm.hof = first_pareto_front
        elif self.knee_point != False:
            if self.knee_point == "Ensemble":
                first_pareto_front = sortNondominated(population, self.n_pop)[0]
                self.algorithm.hof = first_pareto_front
            elif self.knee_point == "Top-10":
                first_pareto_front = sortNondominated(population, self.n_pop)[0]
                best_individuals = sorted(
                    first_pareto_front, key=lambda x: x.fitness.wvalues
                )[-10:]
                self.algorithm.hof = best_individuals
            elif self.knee_point in [
                "Cluster+Ensemble",
                "Cluster+Ensemble+Euclidian",
                "Cluster+Ensemble+Fitness",
            ] or self.knee_point.startswith("Cluster+Ensemble"):
                first_pareto_front = sortNondominated(population, self.n_pop)[0]
                if "-" in self.knee_point:
                    n_clusters = int(self.knee_point.split("-")[1])
                    knee_point_mode = self.knee_point.split("-")[0]
                else:
                    n_clusters = 10
                    knee_point_mode = self.knee_point

                if len(first_pareto_front) <= n_clusters:
                    self.algorithm.hof = first_pareto_front
                else:
                    fitness_values = np.array(
                        [x.fitness.wvalues for x in first_pareto_front]
                    )
                    fitness_values = StandardScaler().fit_transform(fitness_values)
                    if knee_point_mode == "Cluster+Ensemble+Fitness":
                        labels = KMeans(n_clusters=n_clusters).fit_predict(
                            fitness_values
                        )
                    elif knee_point_mode == "Cluster+Ensemble+Fitness+Spectral":
                        labels = SpectralClustering(n_clusters=n_clusters).fit_predict(
                            fitness_values
                        )
                    elif knee_point_mode == "Cluster+Ensemble+Fitness+AC":
                        labels = AgglomerativeClustering(
                            n_clusters=n_clusters
                        ).fit_predict(fitness_values)
                    else:
                        # normalization + reference point synthesis
                        # (1,1,1)-(0,0,0)=(1,1,1,)
                        semantics = np.array(
                            [
                                p.predicted_values - self.algorithm.y
                                for p in first_pareto_front
                            ]
                        )
                        # (0,0,0)-((1,1,1)-(0,0,0))=(-1,-1,-1)
                        inverse_semantics = np.array(
                            [
                                self.algorithm.y
                                - (p.predicted_values - self.algorithm.y)
                                for p in first_pareto_front
                            ]
                        )
                        symmetric_semantics = np.concatenate(
                            [semantics, inverse_semantics]
                        )
                        if knee_point_mode == "Cluster+Ensemble+Euclidian":
                            model = StandardScaler(with_mean=False)
                        elif knee_point_mode == "Cluster+Ensemble+Cosine":
                            model = KernelPCA(kernel="cosine")
                        else:
                            raise Exception("Unknown Knee Point Strategy")
                        model.fit(symmetric_semantics)
                        semantics = model.transform(semantics)
                        labels = KMeans(n_clusters=n_clusters).fit_predict(semantics)

                    # get the best individual from each cluster
                    best_individuals = []
                    for cluster_label in range(n_clusters):
                        cluster_indices = np.where(labels == cluster_label)[0]
                        cluster_front = [first_pareto_front[i] for i in cluster_indices]
                        if len(cluster_front) == 0:
                            continue
                        if self.max_cluster_point:
                            best_individual = max(
                                cluster_front, key=lambda x: x.fitness.wvalues
                            )
                        else:
                            best_individual = min(
                                cluster_front, key=lambda x: x.fitness.wvalues
                            )
                        best_individuals.append(best_individual)
                    self.algorithm.hof = best_individuals
            elif self.knee_point == "Validation":
                first_pareto_front = sortNondominated(population, self.n_pop)[0]
                scores = []
                for ind in first_pareto_front:
                    features = self.algorithm.feature_generation(self.validation_x, ind)
                    scores.append(
                        r2_score(self.validation_y, ind.pipe.predict(features))
                    )
                self.algorithm.hof = [first_pareto_front[np.argmax(scores)]]
                # refit
                ind = self.algorithm.hof[0]
                concatenate_X = np.concatenate(
                    [self.algorithm.X, self.validation_x], axis=0
                )
                concatenate_y = np.concatenate(
                    [self.algorithm.y, self.validation_y.flatten()]
                )
                concatenate_X = self.algorithm.feature_generation(concatenate_X, ind)
                ind.pipe.fit(concatenate_X, concatenate_y)
            elif (
                self.knee_point == "SelfDistillation"
                or self.knee_point == "SelfDistillation-SAM"
            ):
                if self.knee_point == "SelfDistillation-SAM":
                    inds = sorted(individuals, key=lambda x: x.sam_loss)[:30]
                else:
                    inds = selBest(population, 30)
                target = np.mean([ind.predicted_values for ind in inds], axis=0)
                knee = np.argmin(
                    [
                        np.mean((ind.predicted_values - target) ** 2)
                        for ind in population
                    ]
                )
                self.algorithm.hof = [population[knee]]
            else:
                first_pareto_front = sortNondominated(population, self.n_pop)[0]
                if self.knee_point == "Overshot-SAM":
                    first_pareto_front = []
                    for ind in population:
                        prediction_error = ind.naive_mse
                        cv_error = np.mean(ind.case_values)
                        if prediction_error <= cv_error:
                            first_pareto_front.append(ind)

                if self.knee_point == "Adaptive-SAM":
                    all_mse = []
                    all_sharpness = []
                    for ind in population + list(self.algorithm.hof):
                        if hasattr(ind, "fitness_list"):
                            # minimize
                            sharpness = ind.fitness_list[1][0]
                        else:
                            sharpness = -1 * ind.fitness.wvalues[1]
                        naive_mse = np.mean(ind.case_values)
                        all_mse.append(naive_mse)
                        all_sharpness.append(sharpness)
                    metric_std = "Std"
                    ratio = mean_without_outliers(
                        np.array(all_mse), metric=metric_std
                    ) / mean_without_outliers(
                        np.array(all_sharpness), metric=metric_std
                    )
                    if self.algorithm.verbose:
                        print("STD Ratio", ratio)
                    for ind in population + list(self.algorithm.hof):
                        if hasattr(ind, "fitness_list"):
                            # minimize
                            sharpness = ind.fitness_list[1][0]
                        else:
                            sharpness = -1 * ind.fitness.wvalues[1]
                        naive_mse = np.mean(ind.case_values)
                        ind.sam_loss = naive_mse + ratio * sharpness

                if (
                    self.knee_point == "SAM"
                    or self.knee_point.startswith("SAM-")
                    or self.knee_point == "SUM"
                    or self.knee_point
                    in ["Duel-SAM", "Duel-SAM+", "Duel-SAM++", "Adaptive-SAM"]
                    or self.knee_point
                    in ["KNN-SAM", "LR-SAM", "WKNN-SAM", "Overshot-SAM"]
                    or self.knee_point in ["M-SAM", "M-SAM+"]
                ):
                    if not isinstance(self.algorithm.score_func, R2PACBayesian):
                        pac = R2PACBayesian(self.algorithm, **self.algorithm.param)
                        for ind in first_pareto_front:
                            if not hasattr(ind, "sam_loss"):
                                pac.assign_complexity(ind, ind.pipe)
                    knee = np.argmin([p.sam_loss for p in first_pareto_front])
                    if self.knee_point in ["KNN-SAM", "LR-SAM", "WKNN-SAM"]:
                        knee = self.external_regressor_based_duel_selection(
                            first_pareto_front, knee
                        )
                    if self.knee_point in ["Duel-SAM+", "Duel-SAM++"]:
                        best_ind_id = np.argmax(
                            [p.fitness.wvalues[0] for p in first_pareto_front]
                        )
                        best_ind = first_pareto_front[best_ind_id]
                        useful_models = []
                        p_value_threshold = 0.05
                        for model in first_pareto_front:
                            if np.any(best_ind.case_values != model.case_values):
                                # within safety individual
                                signed_test_score = wilcoxon(
                                    best_ind.case_values / model.case_values,
                                    np.ones_like(best_ind.case_values),
                                    alternative="less",
                                )[1]
                                if signed_test_score >= p_value_threshold:
                                    useful_models.append(model)
                        if self.knee_point == "Duel-SAM+":
                            knee = np.argmax([p.sam_loss for p in useful_models])
                        elif self.knee_point == "Duel-SAM++":
                            # worst accuracy
                            knee = np.argmin(
                                [p.fitness.wvalues[0] for p in useful_models]
                            )
                    if self.knee_point == "Duel-SAM":
                        knee = self.duel_model_selection(first_pareto_front, knee)

                elif "+" in self.knee_point:
                    knee = []
                    for strategy in self.knee_point.split("+"):
                        knee.append(
                            knee_point_detection(
                                [p.fitness.wvalues for p in first_pareto_front],
                                knee_point_strategy=strategy,
                            )
                        )
                else:
                    knee = knee_point_detection(
                        [p.fitness.wvalues for p in first_pareto_front],
                        knee_point_strategy=self.knee_point,
                    )
                if isinstance(knee, list):
                    self.algorithm.hof = [first_pareto_front[k] for k in knee]
                else:
                    if self.knee_point == "SAM":
                        if self.algorithm.verbose:
                            print("Number of models on PF", len(first_pareto_front))
                        current_best = self.algorithm.hof[0]
                        """
                        The newly added one should be better than the historical one.
                        """
                        if current_best.sam_loss > first_pareto_front[knee].sam_loss:
                            self.algorithm.hof = [first_pareto_front[knee]]
                        else:
                            if (
                                self.algorithm.verbose
                                and current_best.sam_loss
                                < first_pareto_front[knee].sam_loss
                            ):
                                """
                                In very rare cases, the best individual could be eliminated.
                                """
                                print(
                                    "Bad case!",
                                    current_best.sam_loss,
                                    first_pareto_front[knee].sam_loss,
                                )
                    else:
                        # Select the knee point as the final model
                        self.algorithm.hof = [first_pareto_front[knee]]

        if self.bootstrapping_selection:
            first_pareto_front: list = sortNondominated(population, self.n_pop)[0]

            def quick_evaluation(ind):
                r2_scores = []
                func = multiple_gene_compile(ind, self.algorithm.pset)
                for train_index, test_index in KFold(
                    shuffle=True, random_state=0
                ).split(self.algorithm.X, self.algorithm.y):
                    Yp = result_calculation(func, self.algorithm.X, False)
                    ind.pipe.fit(Yp[train_index], self.algorithm.y[train_index])
                    r2_scores.append(
                        r2_score(
                            self.algorithm.y[test_index],
                            ind.pipe.predict(Yp[test_index]),
                        )
                    )
                return np.mean(r2_scores)

            # Select the minimal cross-validation error as the final model
            self.algorithm.hof = [max(first_pareto_front, key=quick_evaluation)]

        if self.objective_normalization:
            # must-change back to original fitness to avoid any potential error
            fitness_restore_back(individuals)
        return population

    def external_regressor_based_duel_selection(self, first_pareto_front, knee):
        best_ind_id = np.argmax([p.fitness.wvalues[0] for p in first_pareto_front])
        best_ind: MultipleGeneGP = first_pareto_front[best_ind_id]
        best_features = self.algorithm.feature_generation(self.algorithm.X, best_ind)
        hof_features = self.algorithm.feature_generation(
            self.algorithm.X, first_pareto_front[knee]
        )
        if self.knee_point == "KNN-SAM":
            regressor = KNeighborsRegressor()
        elif self.knee_point == "LR-SAM":
            regressor = LinearRegression()
        elif self.knee_point == "WKNN-SAM":
            regressor = KNeighborsRegressor(weights="distance")
        else:
            raise Exception("Unknown Knee Point Strategy")
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", regressor),
            ]
        )
        hof_score = np.mean(cross_val_score(pipe, hof_features, self.algorithm.y, cv=3))
        best_score = np.mean(
            cross_val_score(pipe, best_features, self.algorithm.y, cv=3)
        )
        if hof_score < best_score:
            knee = best_ind_id
        return knee

    def duel_model_selection(self, first_pareto_front, knee):
        best_ind_id = np.argmax([p.fitness.wvalues[0] for p in first_pareto_front])
        best_ind = first_pareto_front[best_ind_id]
        knee_ind = first_pareto_front[knee]
        p_value_threshold = 0.05
        if np.any(best_ind.case_values != knee_ind.case_values):
            signed_test_score = wilcoxon(
                best_ind.case_values / knee_ind.case_values,
                np.ones_like(best_ind.case_values),
                alternative="less",
            )[1]
            if signed_test_score < p_value_threshold:
                knee = best_ind_id
        return knee

    def objective_space_duplication_removal(self, combined_population):
        # Remove duplicates based on semantics/fitness values
        hash_library = {}
        for individual in combined_population:
            for tree, tree_hash in zip(individual.gene, individual.hash_result):
                if tree_hash not in hash_library:
                    hash_library[tree_hash] = individual
                else:
                    # maintain a minimal sized tree library
                    if len(tree) < len(hash_library[tree_hash]):
                        hash_library[tree_hash] = individual

        for individual in combined_population:
            for idx, tree in enumerate(individual.gene):
                # replace with a smaller tree
                if len(tree) > len(hash_library[individual.hash_result[idx]]):
                    hash_library[individual.hash_result[idx]] = tree

        for individual in combined_population:
            # perform simplification
            simple_simplification(individual)

        # Create a dictionary to hold unique individuals
        unique_individuals = {}

        for individual in combined_population:
            """
            The fitness duplication purely based on the first objective, i.e., training loss.
            """
            # Convert fitness values to a hashable type (tuple)
            fitness = individual.fitness.values[0]
            tree_size = np.sum([len(tree) for tree in individual.gene])

            # If this fitness value hasn't been added yet, add the individual to the dictionary
            if fitness not in unique_individuals:
                unique_individuals[fitness] = individual
            else:
                historical_tree_size = np.sum(
                    [len(tree) for tree in unique_individuals[fitness].gene]
                )
                if tree_size < historical_tree_size:
                    unique_individuals[fitness] = individual

        if self.algorithm.verbose:
            print("Number of unique individuals", len(unique_individuals))
        # Extract the individuals from the dictionary to form a new population without duplicates
        return list(unique_individuals.values())


class SPEA2(NSGA2):
    def __init__(
        self,
        algorithm: "EvolutionaryForestRegressor",
        objective_function: Objective = None,
        objective_normalization=False,
        knee_point=False,
        bootstrapping_selection=False,
        **kwargs
    ):
        super().__init__(
            algorithm,
            objective_function,
            objective_normalization,
            knee_point,
            bootstrapping_selection,
            **kwargs
        )
        self.selection_operator = selSPEA2


class Best(EnvironmentalSelection):
    def __init__(self, fit_attr="fitness"):
        super().__init__()
        self.fit_attr = fit_attr

    def select(self, population, offspring):
        if self.fit_attr == "sam_loss":
            individuals = population + offspring
            k = len(population)
            return sorted(individuals, key=attrgetter(self.fit_attr))[:k]
        else:
            return selBest(population + offspring, len(population), self.fit_attr)


class NSGA3(NSGA2):
    def __init__(
        self,
        algorithm: "EvolutionaryForestRegressor",
        objective_function: Objective = None,
        objective_normalization=False,
        knee_point=False,
        bootstrapping_selection=False,
        first_objective_weight=1,
        **kwargs
    ):
        super().__init__(
            algorithm,
            objective_function,
            objective_normalization,
            knee_point,
            bootstrapping_selection,
            first_objective_weight,
            **kwargs
        )
        self.selection_operator = partial(
            selNSGA3, ref_points=uniform_reference_points(3)
        )


if __name__ == "__main__":
    p1 = np.array([0, -4 / 3])
    p2 = np.array([2, 0])
    p3 = np.array([5, 6])
    distance = point_to_line_distance(p1, p2, p3)
    assert abs(distance - 3.32) < 0.01
    print(distance)
