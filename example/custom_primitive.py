"""
This demo is used to illustrate two cases:
1. How to add custom defined functions to Evolutionary Forest?
1. How to add an online-learning support to Evolutionary Forest?
"""
from itertools import chain

import numpy as np
from sklearn import clone
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from evolutionary_forest import forest
from evolutionary_forest.component.primitives import individual_to_tuple


class EvolutionaryForestRegressor(forest.EvolutionaryForestRegressor):

    def lazy_init(self, x):
        super().lazy_init(x)
        # add "sin" and "cos" to the primitive set
        self.pset.addPrimitive(np.sin, 1)
        self.pset.addPrimitive(np.cos, 1)

    def partial_fit(self, X, y, generation=5, distribution_drift=False):
        assert len(self.pop) > 0, "The partial fit function must be called after invoking the fit function!"
        if distribution_drift:
            # remove syntax equivalent individual
            unique_pop = {individual_to_tuple(p): p for p in (self.pop + list(self.hof))}.values()
            self.pop = list(sorted(unique_pop, key=lambda x: x.fitness.wvalues, reverse=True)[:self.n_pop])
            # history fitness is invalid
            self.hof.clear()
            for p in self.pop:
                p.fitness.delValues()
        else:
            for h in chain.from_iterable([self.hof, self.pop]):
                h.pipe = clone(h.pipe)

        # backup
        lazy_init_backup = self.lazy_init
        generation_backup = self.n_gen
        # temporarily mocking the initialization function
        self.lazy_init = lambda _: None
        self.n_gen = generation
        super().fit(X, y)
        # restoration
        self.lazy_init = lazy_init_backup
        self.n_gen = generation_backup
        return None


X, y = make_friedman1()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
r = EvolutionaryForestRegressor(max_height=3, normalize=True, select='AutomaticLexicase',
                                gene_num=10, boost_size=100, n_gen=3, n_pop=200, cross_pb=1,
                                base_learner='Random-DT', verbose=True)
r.fit(x_train, y_train)
print(r2_score(y_test, r.predict(x_test)))
for generation in range(10):
    r.partial_fit(x_train, y_train, generation=3)
    print(generation, r2_score(y_test, r.predict(x_test)))
