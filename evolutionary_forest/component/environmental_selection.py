from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Union

import numpy as np
from deap.tools import selNSGA2, sortNondominated, selSPEA2, selBest, selNSGA3, uniform_reference_points
from numpy.linalg import norm
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.mcdm.pseudo_weights import PseudoWeights
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from evolutionary_forest.multigene_gp import multiple_gene_compile, result_calculation

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def knee_point_detection(front, knee_point_strategy: Union[bool, str] = 'Knee'):
    front = np.array(front)
    if knee_point_strategy == 'Knee' or knee_point_strategy == True:
        pf = front
        pf = (pf - np.min(pf, axis=0)) / (np.max(pf, axis=0) - np.min(pf, axis=0))

        p1 = pf[pf[:, 0].argmax()]
        p2 = pf[pf[:, 1].argmax()]
        # 自动选择拐点
        ans = max([i for i in range(len(pf))],
                  key=lambda i: norm(np.cross(p2 - p1, p1 - pf[i])) / norm(p2 - p1))
        return ans
    elif knee_point_strategy == 'BestAdditionalObjetive':
        return np.argmax(front[:, 1])
    elif knee_point_strategy == 'BestMainObjetive':
        return np.argmax(front[:, 0])
    elif knee_point_strategy == 'SecondAdditionalObjetive':
        if len(front) > 1:
            return np.argsort(front[:, 1])[-2]
        else:
            return 0
    elif knee_point_strategy == 'SecondMainObjetive':
        if len(front) > 1:
            return np.argsort(front[:, 0])[-2]
        else:
            return 0
    elif knee_point_strategy == 'HighTradeoff':
        front = (front - np.min(front, axis=0)) / (np.max(front, axis=0) - np.min(front, axis=0))
        ht = HighTradeoffPoints()
        try:
            # convert to minimization
            ans = ht.do(-1 * front)
            if ans is None:
                print('Empty Answer', front)
                # if no trade-off point, then choosing the point with the highest R2
                return max(range(len(front)), key=lambda x: front[x][0])
            else:
                return max(ans, key=lambda x: front[x][0])
        except (ValueError, IndexError):
            print('Value Error', front)
            # Unknown Exception
            return max(range(len(front)), key=lambda x: front[x][0])
    elif knee_point_strategy.startswith('CP'):
        cp_ratio = float(knee_point_strategy.split('-')[1])
        decomp = ASF()
        # convert to a minimization problem
        I = decomp(-1 * front, np.array([cp_ratio, 1 - cp_ratio])).argmin()
        return I
    elif knee_point_strategy.startswith('PW'):
        pw_ratio = float(knee_point_strategy.split('-')[1])
        # convert to a minimization problem
        I = PseudoWeights(np.array([pw_ratio, 1 - pw_ratio])).do(-1 * front)
        return I
    else:
        raise Exception('Unknown Knee Point Strategy')


class EnvironmentalSelection():
    @abstractmethod
    def select(self, population, offspring):
        pass


class Objective():
    @abstractmethod
    def set(self, individuals):
        pass

    def restore(self, individuals):
        for ind in individuals:
            ind.fitness.weights = (-1,)
            ind.fitness.values = getattr(ind, 'original_fitness')


class TreeSizeObjective(Objective):
    def __init__(self):
        pass

    def set(self, individuals):
        for ind in individuals:
            setattr(ind, 'original_fitness', ind.fitness.values)
            ind.fitness.weights = (-1, -1)
            ind.fitness.values = (ind.fitness.values[0], np.sum([len(y) for y in ind.gene]))


class NSGA2(EnvironmentalSelection):

    def __init__(self,
                 algorithm: "EvolutionaryForestRegressor",
                 objective_function: Objective = None,
                 objective_normalization=False,
                 knee_point=False,
                 bootstrapping_selection=False,
                 first_objective_weight=1, **kwargs):
        self.first_objective_weight = first_objective_weight
        self.bootstrapping_selection = bootstrapping_selection
        self.algorithm = algorithm
        self.objective_function = objective_function
        self.objective_normalization = objective_normalization
        self.knee_point: str = knee_point
        self.selection_operator = selNSGA2
        self.validation_x = None
        self.validation_y = None

    def select(self, population, offspring):
        individuals = population + offspring
        if self.objective_function != None:
            self.objective_function.set(individuals)

        if self.objective_normalization:
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
                    values.append((ind.fitness.values[d] - min_val) / (max_val - min_val))
                ind.fitness.values = values

        population[:] = self.selection_operator(individuals, len(population))

        if self.knee_point != False:
            if self.knee_point == 'Ensemble':
                first_pareto_front = sortNondominated(population, len(population))[0]
                self.algorithm.hof = first_pareto_front
            elif self.knee_point == 'Top-10':
                first_pareto_front = sortNondominated(population, len(population))[0]
                best_individuals = sorted(first_pareto_front, key=lambda x: x.fitness.wvalues)[-10:]
                self.algorithm.hof = best_individuals
            elif self.knee_point in ['Cluster+Ensemble',
                                     'Cluster+Ensemble+Euclidian',
                                     'Cluster+Ensemble+Fitness'] \
                or self.knee_point.startswith('Cluster+Ensemble'):
                first_pareto_front = sortNondominated(population, len(population))[0]
                if '-' in self.knee_point:
                    n_clusters = int(self.knee_point.split('-')[1])
                    knee_point_mode = self.knee_point.split('-')[0]
                else:
                    n_clusters = 10
                    knee_point_mode = self.knee_point

                semantics = np.array([p.predicted_values - self.algorithm.y for p in first_pareto_front])
                if knee_point_mode == 'Cluster+Ensemble+Euclidian':
                    semantics = StandardScaler(with_mean=False).fit_transform(semantics)
                else:
                    semantics = KernelPCA(kernel='cosine').fit_transform(semantics)
                if len(first_pareto_front) <= n_clusters:
                    self.algorithm.hof = first_pareto_front
                else:
                    fitness_values = np.array([x.fitness.wvalues for x in first_pareto_front])
                    fitness_values = StandardScaler().fit_transform(fitness_values)
                    if knee_point_mode == 'Cluster+Ensemble+Fitness':
                        labels = KMeans(n_clusters=n_clusters).fit_predict(fitness_values)
                    elif knee_point_mode == 'Cluster+Ensemble+Fitness+Spectral':
                        labels = SpectralClustering(n_clusters=n_clusters).fit_predict(fitness_values)
                    elif knee_point_mode == 'Cluster+Ensemble+Fitness+AC':
                        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(fitness_values)
                    else:
                        labels = KMeans(n_clusters=n_clusters).fit_predict(semantics)

                    # get the best individual from each cluster
                    best_individuals = []
                    for cluster_label in range(n_clusters):
                        cluster_indices = np.where(labels == cluster_label)[0]
                        cluster_front = [first_pareto_front[i] for i in cluster_indices]
                        if len(cluster_front) == 0:
                            continue
                        best_individual = max(cluster_front, key=lambda x: x.fitness.wvalues)
                        best_individuals.append(best_individual)
                    self.algorithm.hof = best_individuals
            elif self.knee_point == 'Validation':
                first_pareto_front = sortNondominated(population, len(population))[0]
                scores = []
                for ind in first_pareto_front:
                    features = self.algorithm.feature_generation(self.validation_x, ind)
                    scores.append(r2_score(self.validation_y, ind.pipe.predict(features)))
                self.algorithm.hof = [first_pareto_front[np.argmax(scores)]]
                # refit
                ind = self.algorithm.hof[0]
                concatenate_X = np.concatenate([self.algorithm.X, self.validation_x], axis=0)
                concatenate_y = np.concatenate([self.algorithm.y, self.validation_y.flatten()])
                concatenate_X = self.algorithm.feature_generation(concatenate_X, ind)
                ind.pipe.fit(concatenate_X, concatenate_y)
            else:
                first_pareto_front = sortNondominated(population, len(population))[0]
                knee = knee_point_detection([p.fitness.wvalues for p in first_pareto_front],
                                            knee_point_strategy=self.knee_point)
                # Select the knee point as the final model
                self.algorithm.hof = [first_pareto_front[knee]]

        if self.bootstrapping_selection:
            first_pareto_front: list = sortNondominated(population, len(population))[0]

            def quick_evaluation(ind):
                r2_scores = []
                func = multiple_gene_compile(ind, self.algorithm.pset)
                for train_index, test_index in KFold(shuffle=True, random_state=0).split(self.algorithm.X,
                                                                                         self.algorithm.y):
                    Yp = result_calculation(func, self.algorithm.X, False)
                    ind.pipe.fit(Yp[train_index], self.algorithm.y[train_index])
                    r2_scores.append(r2_score(self.algorithm.y[test_index], ind.pipe.predict(Yp[test_index])))
                return np.mean(r2_scores)

            # Select the minimal cross-validation error as the final model
            self.algorithm.hof = [max(first_pareto_front, key=quick_evaluation)]

        if self.objective_function != None:
            self.objective_function.restore(individuals)
        return population


class SPEA2(NSGA2):

    def __init__(self, algorithm: "EvolutionaryForestRegressor", objective_function: Objective = None,
                 objective_normalization=False, knee_point=False, bootstrapping_selection=False, **kwargs):
        super().__init__(algorithm, objective_function, objective_normalization, knee_point, bootstrapping_selection, **kwargs)
        self.selection_operator = selSPEA2


class Best(EnvironmentalSelection):

    def select(self, population, offspring):
        return selBest(population + offspring, len(population))


class NSGA3(NSGA2):

    def __init__(self, algorithm: "EvolutionaryForestRegressor", objective_function: Objective = None,
                 objective_normalization=False, knee_point=False, bootstrapping_selection=False, first_objective_weight=1,
                 **kwargs):
        super().__init__(algorithm, objective_function, objective_normalization, knee_point, bootstrapping_selection,
                         first_objective_weight, **kwargs)
        self.selection_operator = partial(selNSGA3, ref_points=uniform_reference_points(3))


if __name__ == "__main__":
    pass
