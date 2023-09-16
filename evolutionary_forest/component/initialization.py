from evolutionary_forest.component.adaptive_crossover import cxOnePointAdaptive
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.multigene_gp import *


def initialize_crossover_operator(self: "EvolutionaryForestRegressor", toolbox: TypedToolbox):
    # Define the crossiover operator based on the EDA operator
    if 'Biased' in self.mutation_scheme:
        toolbox.mate = cxOnePoint_multiple_gene_biased
    elif 'SameWeight' in self.mutation_scheme:
        toolbox.mate = cxOnePoint_multiple_gene_same_weight
    elif self.mutation_scheme.endswith('-SC'):
        toolbox.mate = partial(cxOnePoint_multiple_gene_SC,
                               crossover_configuration=self.crossover_configuration)
    elif self.mutation_scheme.endswith('-BSC'):
        # Good Biased
        toolbox.mate = partial(cxOnePoint_multiple_gene_BSC,
                               crossover_configuration=self.crossover_configuration)
    elif self.mutation_scheme.endswith('-TSC'):
        # Tournament SC
        toolbox.mate = partial(cxOnePoint_multiple_gene_TSC,
                               crossover_configuration=self.crossover_configuration)
    elif 'SameIndex' in self.mutation_scheme:
        toolbox.mate = cxOnePoint_multiple_gene_same_index
    elif 'AllGene' in self.mutation_scheme:
        toolbox.mate = cxOnePoint_all_gene
    elif 'AdaptiveCrossover' in self.mutation_scheme:
        toolbox.mate = cxOnePointAdaptive
    else:
        toolbox.mate = partial(cxOnePoint_multiple_gene,
                               pset=self.pset,
                               crossover_configuration=self.get_crossover_configuration())
