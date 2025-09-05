from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score

from evolutionary_forest.component.fitness import Fitness


class TransportR2(Fitness):
    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        distance = wasserstein_distance(Y, y_pred)
        return (-1 * score, distance)
