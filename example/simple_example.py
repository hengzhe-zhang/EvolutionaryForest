import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score

from evolutionary_forest.forest import EvolutionaryForestRegressor

X, y = load_diabetes(return_X_y=True)
score = cross_val_score(ExtraTreesRegressor(), X, y, n_jobs=5)
print("CV Score", score, np.mean(score))
reg = EvolutionaryForestRegressor(max_height=8, normalize=True, select='AutomaticLexicase',
                                  gene_num=10, boost_size=100, n_gen=100, base_learner='Random-DT')
score = cross_val_score(reg, X, y, n_jobs=5)
print("CV Score", score, np.mean(score))
