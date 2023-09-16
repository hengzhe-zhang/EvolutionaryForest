import array
import random

import numpy
import numpy as np
import pyade
from deap import base
from deap import creator
from deap import tools


class EnsembleSelection():
    def __init__(self, NDIM=10, generation=10, population_size=50, evaluation_function=None, verbose=False):
        # Problem dimension
        self.generation = generation
        self.population_size = population_size
        self.NDIM = NDIM

        if not hasattr(creator, "EnsembleFitnessMin"):
            creator.create("EnsembleFitnessMin", base.Fitness, weights=(1.0,))
            creator.create("EnsembleIndividual", array.array, typecode='d', fitness=creator.EnsembleFitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat, creator.EnsembleIndividual, toolbox.attr_float, NDIM)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k=3)
        toolbox.register("evaluate", evaluation_function)
        self.toolbox = toolbox
        self.verbose = verbose

    def run(self, initial_vector=None):
        toolbox = self.toolbox
        # Differential evolution parameters
        CR = 0.25
        F = 1
        MU = self.population_size
        NGEN = self.generation

        # ensure no worse than last iteration
        # first NDIM individuals are inherited from the previous generation
        pop = toolbox.population(n=MU - 2)
        pop += [creator.EnsembleIndividual(np.linspace(-1, 1, self.NDIM))]
        pop += [creator.EnsembleIndividual(initial_vector)]
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        if self.verbose:
            print(logbook.stream)

        for g in range(1, NGEN):
            for k, agent in enumerate(pop):
                a, b, c = toolbox.select(pop)
                y = toolbox.clone(agent)
                index = random.randrange(self.NDIM)
                for i, value in enumerate(agent):
                    if i == index or random.random() < CR:
                        y[i] = a[i] + F * (b[i] - c[i])
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(pop), **record)
            if self.verbose:
                print(logbook.stream)

        # print("Best individual is ", hof[0], hof[0].fitness.values[0])
        return hof[0]


class EnsembleSelectionADE():
    def __init__(self, NDIM=10, generation=10, population_size=50, evaluation_function=None, verbose=False,
                 de_algorithm='IL-SHADE'):
        # You may want to use a variable so its easier to change it if we want
        algorithm = {
            'IL-SHADE': pyade.ilshade,
            'DE': pyade.de,
            'JADE': pyade.jade,
            'MPEDE': pyade.mpede,
            'JSO': pyade.jso,
        }[de_algorithm]

        # We get default parameters for a problem with two variables
        params = algorithm.get_default_params(dim=NDIM)

        # We define the boundaries of the variables
        params['bounds'] = np.array([[-1, 1]] * NDIM)

        # We indicate the function we want to minimize
        params['func'] = evaluation_function
        params['population_size'] = population_size
        params['max_evals'] = generation * population_size
        self.algorithm = algorithm
        self.params = params

    def run(self, initial_vector=None):
        # We run the algorithm and obtain the results
        solution, fitness = self.algorithm.apply(**self.params)
        print('OOB', fitness)
        return solution


if __name__ == "__main__":
    e = EnsembleSelection()
    e.run()
