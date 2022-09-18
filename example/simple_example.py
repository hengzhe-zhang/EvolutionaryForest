import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score

from evolutionary_forest.forest import EvolutionaryForestRegressor

X, y = load_diabetes(return_X_y=True)
score = cross_val_score(ExtraTreesRegressor(), X, y, n_jobs=5)
print("CV Score", score, np.mean(score))
reg = EvolutionaryForestRegressor(max_height=3, normalize=True, select='AutomaticLexicase',
                                  gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                  base_learner='Random-DT', verbose=True)
score = cross_val_score(reg, X, y, n_jobs=5)
print("CV Score", score, np.mean(score))
