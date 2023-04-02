"""
An example of alpha dominance-based multi-objective symbolic regression.
"""
import math
import operator
import random

import numpy
import numpy as np
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap.benchmarks.tools import hypervolume
from deap.tools import sortNondominated


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    return math.fsum(sqerrors) / len(points), len(individual)


toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main(seed=None, alpha_dominance=True):
    alpha = 0
    step_size = 90
    random.seed(seed)

    NGEN = 50
    MU = 100
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    historical_smallest = min([len(p) for p in pop])
    historical_largest = max([len(p) for p in pop])

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0], fit[1]

        max_size = max([len(x) for x in pop + offspring])
        for ind in pop + offspring:
            fit = ind.fitness.values
            ind.fitness.values = fit[0], fit[1] / max_size + alpha * fit[0]

        historical_smallest = min(historical_smallest, min([len(p) for p in pop]))
        historical_largest = max(historical_largest, max([len(p) for p in pop]))
        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        if alpha_dominance:
            # Update alpha
            first_pareto_front = sortNondominated(offspring + pop, MU)[0]
            theta = np.rad2deg(np.arctan(alpha))
            avg_size = np.mean([len(p) for p in first_pareto_front])
            print('Average Size', avg_size, 'Degree', theta)
            # update angle
            theta = theta + (historical_largest + historical_smallest - 2 * avg_size) / \
                    (historical_largest - historical_smallest) * step_size
            theta = np.clip(theta, 0, 90)
            alpha = np.tan(np.deg2rad(theta))
            print(alpha)
        else:
            assert alpha == 0
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    for ind in pop:
        fit = ind.fitness.values
        ind.fitness.values = fit[0], len(ind)
        print(ind.fitness.wvalues)
    print('Hyper Volume', hypervolume(pop, [10, 500]))
    arg = np.argmax([p.fitness.wvalues[0] for p in pop])
    print("Best Individual", pop[arg].fitness.values[0])
    print("Best Individual Size", len(pop[arg]))
    return pop, logbook


if __name__ == "__main__":
    main(alpha_dominance=True)
    main(alpha_dominance=False)
