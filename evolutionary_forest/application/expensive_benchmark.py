import operator

from deap import gp, tools, creator
from deap.algorithms import eaSimple
from deap.tools import HallOfFame
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from evolutionary_forest.component.evaluation import single_tree_evaluation
from evolutionary_forest.component.primitive_functions import analytical_quotient
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.forest import spearman
from evolutionary_forest.multigene_gp import *
from evolutionary_forest.utils import gene_to_string


class ExpensiveBenchmarkGenerator(BaseEstimator):
    def __init__(self, n_gen=30, n_pop=100, cross_pb=0.9, mutation_pb=0.1):
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

        self.number_of_variables = 5
        self.number_of_objectives = 2
        self.gene_num = self.number_of_variables + self.number_of_objectives
        self.X = np.random.randn(5, 50, self.number_of_variables)
        self.pset = MultiplePrimitiveSet("MAIN", 1, "ARG")
        self.pset.pset_list = [
            self.get_pset(1) for _ in range(self.number_of_variables)
        ] + [
            self.get_pset(self.number_of_variables)
            for _ in range(self.number_of_objectives)
        ]

        self.toolbox = TypedToolbox()
        toolbox = self.toolbox
        toolbox.expr = partial(gp.genHalfAndHalf, pset=self.pset, min_=0, max_=2)
        toolbox.individual = partial(
            multiple_gene_initialization,
            MultipleGeneGP,
            toolbox.expr,
            gene_num=self.gene_num,
        )
        toolbox.population = partial(tools.initRepeat, list, toolbox.individual)
        toolbox.evaluate = self.fitness_evaluation
        crossover_configuration = CrossoverConfiguration()
        crossover_configuration.same_index = True
        toolbox.register(
            "mate",
            cxOnePoint_multiple_gene,
            crossover_configuration=crossover_configuration,
        )
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register(
            "mutate", mutUniform_multiple_gene, expr=toolbox.expr_mut, pset=self.pset
        )
        toolbox.select = partial(selTournament, tournsize=7)

        max_value = 3
        toolbox.decorate(
            "mate",
            staticLimit_multiple_gene(
                key=operator.attrgetter("height"), max_value=max_value
            ),
        )
        toolbox.decorate(
            "mutate",
            staticLimit_multiple_gene(
                key=operator.attrgetter("height"), max_value=max_value
            ),
        )

        self.n_gen = n_gen
        self.n_pop = n_pop
        self.cross_pb = cross_pb
        self.mutation_pb = mutation_pb
        self.hof = HallOfFame(10)

    def get_pset(self, input_vars):
        pset = gp.PrimitiveSet("MAIN", input_vars, "ARG")
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(analytical_quotient, 2)
        pset.addPrimitive(np.negative, 1)
        pset.addPrimitive(np.absolute, 1)
        pset.addPrimitive(np.sin, 1)
        pset.addPrimitive(np.cos, 1)
        pset.addPrimitive(np.maximum, 2)
        pset.addPrimitive(np.minimum, 2)
        return pset

    def print_code(self, ind):
        for id, g in enumerate(ind.gene):
            if id < self.number_of_variables:
                print(f"VAR{id}=lambda:", gene_to_string(g).replace("ARG", "X"))
            else:
                print(
                    f"O{id - self.number_of_variables}=lambda:",
                    gene_to_string(g).replace("ARG", "VAR"),
                )

    def fitness_evaluation(self, ind: MultipleGeneGP):
        all_scores = []
        for group_X in self.X:
            final_array = []
            c_id = 0
            for g in ind.gene[: self.number_of_variables]:
                y_array = []
                for x in group_X.T:
                    y = single_tree_evaluation(
                        g, self.pset.pset_list[c_id], np.reshape(x, (-1, 1))
                    )
                    y_array.append(y)
                final_array.append(np.mean(y_array, axis=0))
                c_id += 1
            final_array = np.array(final_array)

            objectives = []
            for g in ind.gene[self.number_of_variables :]:
                y = single_tree_evaluation(g, self.pset.pset_list[c_id], final_array.T)
                objectives.append(y)
                c_id += 1
            pipe = Pipeline(
                [
                    ("Scaler", StandardScaler()),
                    ("GP", GaussianProcessRegressor(normalize_y=True)),
                ]
            )
            scoring = make_scorer(spearman)
            average_score = max(
                np.mean(cross_val_score(pipe, group_X, objectives[0], scoring=scoring)),
                np.mean(cross_val_score(pipe, group_X, objectives[1], scoring=scoring)),
            )
            all_scores.append(average_score)
        all_scores = np.mean(all_scores)
        return (all_scores,)

    def fit(self):
        pop = self.toolbox.population(self.n_pop)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.wvalues)
        stats_size = tools.Statistics(lambda x: len(x) // len(x.gene))
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std, axis=0)
        mstats.register("25%", partial(np.quantile, q=0.25), axis=0)
        mstats.register("75%", partial(np.quantile, q=0.75), axis=0)
        mstats.register("median", np.median, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)

        eaSimple(
            pop,
            self.toolbox,
            self.cross_pb,
            self.mutation_pb,
            self.n_gen,
            stats=mstats,
            halloffame=self.hof,
            verbose=True,
        )


if __name__ == "__main__":
    exp = ExpensiveBenchmarkGenerator()
    exp.fit()
    for g in exp.hof:
        print(g.fitness.wvalues)
        exp.print_code(g)
