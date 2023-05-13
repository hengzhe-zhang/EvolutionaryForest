from sympy import parse_expr, preorder_traversal

from evolutionary_forest.forest import EvolutionaryForestRegressor


est = EvolutionaryForestRegressor(n_gen=100, n_pop='30N-300', select='AutomaticLexicase',
                                  cross_pb='DynamicAdaptive-0.9', mutation_pb='DynamicAdaptive-0.1', max_height=2,
                                  boost_size=1, initial_tree_size='1-2', gene_num=20,
                                  mutation_scheme='EDA-Terminal-PM', basic_primitives='optimal',
                                  base_learner='RidgeCV', verbose=False, normalize=True,
                                  ridge_alphas='Auto', external_archive=1,
                                  **{
                                      'strict_layer_mgp': True,
                                      'mgp_scope': 10,
                                      'mgp_mode': True,
                                      'number_of_parents': 2,
                                      'macro_crossover_rate': 0.25,
                                      'root_crossover': True,
                                      'delete_irrelevant': True,
                                      'delete_redundant': True,
                                      'min_height': 0,
                                      'shared_eda': False,
                                      # Adaptive GP parameter
                                      'adaptive_power': 2,
                                      'inverse_point': 0.75
                                  }
                                  )


def complexity(est: EvolutionaryForestRegressor):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())
