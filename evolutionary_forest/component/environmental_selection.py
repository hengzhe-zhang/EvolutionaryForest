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
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from analysis.knee_point_eurogp.utility_function import knee_point_by_utility
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
from evolutionary_forest.multigene_gp import multiple_gene_compile, result_calculation

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


def unique(individuals):
    generated = set()
    result = []

    for ind in individuals:
        s = ind
        if str(s) not in generated:
            generated.add(str(s))
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
        **kwargs
    ):
        self.handle_objective_duplication = handle_objective_duplication
        self.first_objective_weight = first_objective_weight
        self.bootstrapping_selection = bootstrapping_selection
        self.algorithm = algorithm
        self.objective_function = objective_function
        self.objective_normalization = objective_normalization
        self.knee_point: Union[str, bool] = knee_point
        # if using clustering for ensemble, max point in each cluster
        self.max_cluster_point = max_cluster_point
        self.selection_operator = selNSGA2
        self.validation_x = None
        self.validation_y = None

    def select(self, population, offspring):
        individuals = population + offspring
        # remove exactly the same individuals
        individuals = unique(individuals)
        # remove individuals with the same objective values
        if self.handle_objective_duplication:
            individuals = self.objective_space_duplication_removal(individuals)

        if self.objective_normalization:
            for ind in individuals:
                ind.unnormalized_fitness = ind.fitness.values
            dims = len(individuals[0].fitness.values)
            min_max = []
            for d in range(dims):
                values = [ind.fitness.values[d] for ind in individuals]
                min_val = min(values)
                max_val = max(values)
                min_max.append((min_val, max_val))
            for ind in individuals:
                values = []
                for d in range(dims):
                    min_val, max_val = min_max[d]
                    normalized_fitness = ind.fitness.values[d] - min_val
                    if (max_val - min_val) > 0:
                        normalized_fitness = normalized_fitness / (max_val - min_val)
                    values.append(normalized_fitness)
                ind.fitness.values = values

        population[:] = self.selection_operator(individuals, len(population))

        if self.knee_point != False:
            if self.knee_point == "Ensemble":
                first_pareto_front = sortNondominated(population, len(population))[0]
                self.algorithm.hof = first_pareto_front
            elif self.knee_point == "Top-10":
                first_pareto_front = sortNondominated(population, len(population))[0]
                best_individuals = sorted(
                    first_pareto_front, key=lambda x: x.fitness.wvalues
                )[-10:]
                self.algorithm.hof = best_individuals
            elif self.knee_point in [
                "Cluster+Ensemble",
                "Cluster+Ensemble+Euclidian",
                "Cluster+Ensemble+Fitness",
            ] or self.knee_point.startswith("Cluster+Ensemble"):
                first_pareto_front = sortNondominated(population, len(population))[0]
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
                first_pareto_front = sortNondominated(population, len(population))[0]
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
            else:
                first_pareto_front = sortNondominated(population, len(population))[0]
                if self.knee_point == "SAM" or self.knee_point == "SUM":
                    if not isinstance(self.algorithm.score_func, R2PACBayesian):
                        pac = R2PACBayesian(self.algorithm, **self.algorithm.param)
                        for ind in first_pareto_front:
                            if not hasattr(ind, "sam_loss"):
                                pac.assign_complexity(ind, ind.pipe)
                    knee = np.argmin([[p.sam_loss for p in first_pareto_front]])
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
                    # Select the knee point as the final model
                    self.algorithm.hof = [first_pareto_front[knee]]

        if self.bootstrapping_selection:
            first_pareto_front: list = sortNondominated(population, len(population))[0]

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
            # must change back to original fitness to avoid any potential error
            for ind in individuals:
                ind.fitness.values = ind.unnormalized_fitness
        return population

    def objective_space_duplication_removal(self, combined_population):
        # Remove duplicates based on fitness values
        # Create a dictionary to hold unique individuals
        unique_individuals = {}

        for individual in combined_population:
            # Convert fitness values to a hashable type (tuple)
            fitness = tuple(individual.fitness.values)

            # If this fitness value hasn't been added yet, add the individual to the dictionary
            if fitness not in unique_individuals:
                unique_individuals[fitness] = individual

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
