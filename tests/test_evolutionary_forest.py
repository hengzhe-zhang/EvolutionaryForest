#!/usr/bin/env python

"""Tests for `evolutionary_forest` package."""

from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from evolutionary_forest.forest import EvolutionaryForestRegressor


def test_simple_data():
    X, y = make_regression(n_samples=100, n_features=5, n_informative=5)
    gp = EvolutionaryForestRegressor(
        max_height=8,
        normalize=True,
        select="AutomaticLexicase",
        boost_size=10,
        n_gen=2,
        gene_num=5,
        base_learner="Random-DT",
    )
    gp.fit(X, y)
    assert_almost_equal(mean_squared_error(y, gp.predict(X)), 0)
    assert len(gp.hof) == 10
    assert len(gp.hof[0].gene) == 5
