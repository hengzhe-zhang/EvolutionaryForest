# GSMX-GP
from sympy import parse_expr, preorder_traversal

from evolutionary_forest.forest import EvolutionaryForestRegressor

est = EvolutionaryForestRegressor(n_gen=100, n_pop='30N-300', select='AutomaticLexicase',
                                  cross_pb=0.9, mutation_pb=0.1, max_height=8,
                                  boost_size=1, initial_tree_size='0-2', gene_num=20,
                                  mutation_scheme='EDA-Terminal-PM', basic_primitives='optimal',
                                  base_learner='RidgeCV', verbose=False, normalize=True, external_archive=1,
                                  **{
                                      'semantic_crossover_probability': 0.2,
                                      'semantic_selection_mode': 'MAPElites',
                                      'semantic_crossover_mode': 'Independent'
                                  }
                                  )


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())
