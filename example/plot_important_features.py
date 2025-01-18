from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deap.gp import graph
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance


def plot_trees(reg: EvolutionaryForestRegressor, number_of_features=10):
    plt.figure(figsize=(10, 3))

    all_genes_map = defaultdict(int)
    all_genes_map_instance = {}
    for x in reg.hof:
        for o_g, c in zip(x.gene, np.abs(x.coef)):
            # Taking the fitness of each model into consideration
            importance_value = c
            all_genes_map[str(o_g)] += importance_value
            all_genes_map_instance[str(o_g)] = o_g
    all_genes_map = sorted(
        all_genes_map.items(), key=lambda x: (x[1], x[0]), reverse=True
    )[:number_of_features]
    genes = [all_genes_map_instance[k] for k, v in all_genes_map]

    function_name = {
        "multiply": "MUL",
        "analytical_quotient": "AQ",
        "subtract": "SUB",
        "add": "ADD",
    }
    for i in range(len(genes)):
        gene = genes[i]
        for g in gene:
            if (
                hasattr(g, "value")
                and isinstance(g.value, str)
                and g.value.startswith("ARG")
            ):
                g.value = g.value.replace("ARG", "X")
            if g.name in function_name:
                g.name = function_name[g.name]

        ax = plt.subplot(2, len(genes) // 2, i + 1)

        nodes, edges, labels = graph(gene)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#b3cde0", node_size=500)
        nx.draw_networkx_edges(g, pos, ax=ax)
        nx.draw_networkx_labels(g, pos, labels, ax=ax)

        plt.title(f"Feature #{i + 1}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    reg = EvolutionaryForestRegressor(
        max_height=10,
        normalize=True,
        select="AutomaticLexicase",
        gene_num=10,
        boost_size=10,
        n_gen=5,
        n_pop=100,
        base_learner="Random-DT",
        verbose=True,
    )
    reg.fit(X, y)
    print(r2_score(y_train, reg.predict(x_train)))
    print(r2_score(y_test, reg.predict(x_test)))

    # Plot symbolic trees of top-10 features
    plot_trees(reg, number_of_features=10)

    # Calculate feature importance values
    feature_importance_dict = get_feature_importance(reg)
    plot_feature_importance(feature_importance_dict)
