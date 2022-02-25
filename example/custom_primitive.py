import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score

from evolutionary_forest import forest


class EvolutionaryForestRegressor(forest.EvolutionaryForestRegressor):

    def lazy_init(self, x):
        super().lazy_init(x)
        # add "sin" and "cos" to the primitive set
        self.pset.addPrimitive(np.sin, 1)
        self.pset.addPrimitive(np.cos, 1)


X, y = load_diabetes(return_X_y=True)

score = cross_val_score(ExtraTreesRegressor(), X, y, n_jobs=5)
print("CV Score", score, np.mean(score))
reg = EvolutionaryForestRegressor(max_height=3, normalize=True, select='AutomaticLexicase',
                                  gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                  base_learner='Random-DT', verbose=True)
score = cross_val_score(reg, X, y, n_jobs=5)
print("CV Score", score, np.mean(score))
