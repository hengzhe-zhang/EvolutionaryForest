from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from evolutionary_forest.component.test_function import TestFunction
from evolutionary_forest.forest import EvolutionaryForestRegressor

X, y = load_diabetes(return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=0)
reg = EvolutionaryForestRegressor(max_height=3, normalize=True, select='AutomaticLexicase',
                                  gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                  base_learner='Random-DT', verbose=True)
reg.test_fun = [TestFunction(train_X, train_y, reg), TestFunction(val_X, val_y, reg)]
reg.fit(train_X, train_y)
