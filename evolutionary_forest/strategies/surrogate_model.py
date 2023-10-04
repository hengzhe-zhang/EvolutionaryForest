from itertools import chain
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from deap.gp import Terminal
from scipy.spatial import KDTree
from scipy.spatial.distance import correlation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from evolutionary_forest.component.evaluation import quick_evaluate
from evolutionary_forest.multigene_gp import result_calculation
from evolutionary_forest.sklearn_utils import cross_val_predict

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class SurrogateModel():
    """
    Some useful functions for building a surrogate model
    Pre-selection strategy:
    1. simple-task: A simple surrogate task
    """

    def __init__(self, algorithm):
        self.algorithm: "EvolutionaryForestRegressor" = algorithm

    def feature_surrogate_model(self):
        algorithm = self.algorithm
        # construct a surrogate model to predict whether a feature is useful or not
        # negative samples
        data = []
        label = []
        for ind in sorted(algorithm.pop, key=lambda x: x.fitness.wvalues[0])[:algorithm.n_pop // 2]:
            funcs = algorithm.toolbox.compile(ind)
            for func, coef in zip(funcs, ind.coef):
                Yp = result_calculation([func], algorithm.X[:5], algorithm.original_features)
                if coef < 0.001:
                    # label.append(1)
                    pass
                else:
                    data.append(Yp.flatten())
                    label.append(0)
                # label.append(ind.fitness.wvalues[0])
        for ind in sorted(algorithm.pop, key=lambda x: x.fitness.wvalues[0])[algorithm.n_pop // 2:]:
            funcs = algorithm.toolbox.compile(ind)
            for func, coef in zip(funcs, ind.coef):
                Yp = result_calculation([func], algorithm.X[:5], algorithm.original_features)
                if coef < 0.001:
                    # label.append(1)
                    pass
                else:
                    data.append(Yp.flatten())
                    label.append(2)
        data = np.array(data)
        label = np.array(label)
        cross_val_predict(KNeighborsClassifier(n_neighbors=5, metric=correlation), data, label, cv=5)

    def get_features(self, sample, individuals, flatten=False):
        algorithm = self.algorithm
        all_Yp = []
        for individual in individuals:
            func = algorithm.toolbox.compile(individual)
            Yp = result_calculation(func, algorithm.X[sample], algorithm.original_features)
            if flatten:
                all_Yp.append(Yp.flatten())
                # all_Yp.append((cross_val_predict(LinearRegression(), Yp, algorithm.y[sample]) - algorithm.y[sample]) ** 2)
            else:
                all_Yp.append(Yp)
        return np.array(all_Yp)

    def pre_selection_individuals(self, parents, offspring, pop_size):
        algorithm = self.algorithm
        predicted_values = algorithm.pre_selection_score(offspring, parents)
        final_offspring = []
        if algorithm.pre_selection == 'model-size-medium':
            start_index = pop_size // 2
            index = np.argsort(-1 * predicted_values)[start_index:start_index + pop_size]
        else:
            index = np.argsort(-1 * predicted_values)[:pop_size]
        for i in index:
            final_offspring.append(offspring[i])
        return final_offspring, predicted_values[index]

    def pre_selection_score(self, offspring, parents):
        algorithm = self.algorithm
        # larger is better
        if algorithm.pre_selection == 'surrogate-model':
            predicted_values = algorithm.calculate_score_by_surrogate_model(offspring, parents)
        elif algorithm.pre_selection == 'simple-task':
            predicted_values = algorithm.calculate_score_by_statistical_methods(offspring, parents)
        elif algorithm.pre_selection in ['model-size', 'model-size-medium']:
            predicted_values = algorithm.calculate_score_by_model_size(offspring, parents)
        elif algorithm.pre_selection == 'diversity':
            plot = False
            # Get the most hard 20 data points
            all_values = np.array([p.case_values for p in parents])
            sample = np.sum(all_values, axis=0).argsort()[-20:]

            pset = algorithm.toolbox.expr.keywords['pset']

            def quick_get_features(individuals):
                all_Yp = []
                for individual in individuals:
                    Yp = []
                    for gene in individual.gene:
                        input_data = algorithm.X[sample]
                        temp_result = quick_evaluate(gene, pset, input_data)
                        if np.size(temp_result) < len(input_data):
                            temp_result = np.full(len(input_data), temp_result)
                        Yp.append(temp_result)
                    Yp = np.array(Yp).T
                    all_Yp.append(Yp.flatten())
                return np.array(all_Yp)

            offspring_features = quick_get_features(offspring)

            offspring_features = StandardScaler().fit_transform(offspring_features)
            pca = PCA(n_components=10)
            offspring_features = pca.fit_transform(offspring_features)
            if plot:
                plt.scatter(offspring_features[:, 0], offspring_features[:, 1])
            kmeans = KMeans(n_clusters=algorithm.n_pop)
            kmeans.fit(offspring_features)

            predicted_values = np.zeros(len(offspring))
            kd = KDTree(offspring_features)
            nearest_inds = []
            # Get the nearest point for each cluster center
            for c in kmeans.cluster_centers_:
                dist, ind = kd.query(c)
                nearest_inds.append(ind)
            predicted_values[np.array(nearest_inds)] = 1
            if plot:
                plt.scatter(offspring_features[np.array(nearest_inds), 0],
                            offspring_features[np.array(nearest_inds), 1], marker="s", color='red')
                plt.scatter(offspring_features[np.arange(0, 100), 0],
                            offspring_features[np.arange(0, 100), 1], marker=".", color='green')
                plt.show()
        else:
            raise Exception
        return predicted_values

    def calculate_score_by_model_size(self, offspring, parents):
        return np.array([-1 * np.mean([len(y) for y in x.gene]) for x in offspring])

    def calculate_score_by_statistical_methods(self, offspring, parents):
        algorithm = self.algorithm
        all_values = np.array([p.case_values for p in parents])
        sample = np.sum(all_values, axis=0).argsort()[-50:]
        offspring_features = algorithm.get_features(sample, offspring, False)
        all_score = []
        for f in offspring_features:
            pipe = algorithm.get_base_model()
            y_pred = cross_val_predict(pipe, f, algorithm.y[sample], cv=3).flatten()
            score = r2_score(algorithm.y[sample], y_pred)
            all_score.append(score)
        return np.array(all_score)

    def calculate_score_by_surrogate_model(self, offspring, parents):
        algorithm = self.algorithm

        sample = np.random.randint(0, len(algorithm.y), size=10)
        # select individuals based on a surrogate model
        parent_features = algorithm.get_features(sample, chain(parents, algorithm.hof), True)
        target = np.array([p.fitness.wvalues[0] for p in chain(parents, algorithm.hof)])

        surrogate_model = XGBRegressor(n_jobs=1, objective='rank:pairwise')
        surrogate_model.fit(parent_features, target)
        offspring_features = algorithm.get_features(sample, offspring, True)
        predicted_values = surrogate_model.predict(offspring_features)

        # scoring = make_scorer(spearman)
        # print('CV', np.mean(cross_val_score(surrogate_model, parent_features, target, cv=3, scoring=scoring)))
        # print('CV', np.mean(cross_val_score(surrogate_model, parent_features, target, cv=3)))
        return predicted_values

    def get_genotype_features(self, pop, get_label=False):
        algorithm = self.algorithm
        data = []
        for p in pop:
            height = np.mean([g.height for g in p.gene])
            size = np.mean([len(g) for g in p.gene])
            primitives_dict = {t.name: 0 for t in algorithm.pset.primitives[object]}
            terminals_dict = {t.name: 0 for t in filter(lambda x: isinstance(x, Terminal),
                                                        algorithm.pset.terminals[object])}
            for g in p.gene:
                for x in g:
                    if x.name in primitives_dict:
                        primitives_dict[x.name] += 1
                    if x.name in terminals_dict:
                        terminals_dict[x.name] += 1
            used_variables = len(list(filter(lambda x: x > 0, terminals_dict.values())))
            frequency = list(map(lambda x: x / size, list(primitives_dict.values()) + list(terminals_dict.values())))
            if hasattr(p, 'parent_fitness'):
                parent_fitness = [max(p.parent_fitness), min(p.parent_fitness), np.mean(p.parent_fitness)]
            else:
                parent_fitness = [0, 0, 0]
            data.append([height, size, used_variables] + frequency + parent_fitness)
        if get_label:
            label = []
            for p in pop:
                label.append(p.fitness.wvalues[0])
            return np.array(data), np.array(label)
        else:
            return np.array(data)

    def surrogate_model_construction(self, pop):
        algorithm = self.algorithm
        # construct a surrogate model
        data, label = algorithm.get_genotype_features(pop, get_label=True)
        surrogate_model = ExtraTreesRegressor(n_estimators=100, n_jobs=1)
        surrogate_model.fit(data, label)
        return surrogate_model
