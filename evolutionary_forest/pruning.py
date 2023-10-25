import random

import numpy
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap.tools import cxTwoPoint
from sklearn.metrics import r2_score


def oob_pruning(oob, y_train):
    ensemble_size = len(oob)

    def oob_error(x):
        x = np.array(x).astype(bool)
        if (np.sum((oob[x] != 0), axis=0) == 0).any():
            return (-10 * np.sum(np.sum((oob[x] != 0), axis=0) == 0),)
        prediction = np.sum(oob[x], axis=0) / np.sum((oob[x] != 0), axis=0)
        if np.isnan(prediction).any():
            return (-10 * np.sum(np.isnan(prediction)),)
        oob_score = r2_score(y_train, prediction)
        return (oob_score,)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=ensemble_size,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", oob_error)
    toolbox.register("mate", cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=10)

    pop = toolbox.population(n=100)

    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=numpy.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=50,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    weight = np.array((hof[0]))
    return weight
