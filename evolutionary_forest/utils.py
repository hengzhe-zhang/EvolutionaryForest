from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from deap.gp import Primitive
from matplotlib import pyplot as plt
from sympy import simplify, latex
from sympy.printing.latex import latex_escape


def get_feature_importance(regr, simple_version=True):
    all_genes_map = defaultdict(int)
    for x in regr.hof:
        if simple_version:
            latex_string = lambda g: latex(simplify(gene_to_string(g).replace("ARG", "X").replace("_", "-")))
            genes = [f'${latex_string(g)}$' for g in x.gene]
        else:
            def code_generation(gene):
                code = str(gene)
                args = ",".join(arg for arg in regr.pset.arguments)
                code = "lambda {args}: {code}".format(args=args, code=code)
                return code

            genes = [f'{code_generation(g)}' for g in x.gene]
        for g, c in zip(genes, x.coef):
            all_genes_map[g] += 1 * c
    feature_importance_dict = {k: v for k, v in sorted(all_genes_map.items(), key=lambda item: -item[1])}
    return feature_importance_dict


def select_top_features(code_importance_dict, ratio=None):
    if ratio == None:
        mean_importance = np.mean(list(code_importance_dict.values()))
    else:
        mean_importance = np.quantile(list(code_importance_dict.values()), ratio)
    index = list(np.array(list(code_importance_dict.values())) <= mean_importance).index(True)
    features = list(code_importance_dict.keys())[:index]
    return features


def feature_append(regr, X, features, only_new_features=False):
    if only_new_features:
        data = []
    else:
        data = [X]
    for f in features:
        func = eval(f, regr.pset.context)
        data.append(func(*X.T).reshape(-1, 1))
    return np.concatenate(data, axis=1)


def plot_feature_importance(feature_importance_dict, save_fig=False):
    names, importance = list(feature_importance_dict.keys()), list(feature_importance_dict.values())
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = names

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    sns.set(style="white", font_scale=1.5)
    # Define size of bar plot
    plt.figure(figsize=(12, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'][:15], y=fi_df['feature_names'][:15])

    # Add chart labels
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    if save_fig:
        plt.savefig('feature_importance.png', format='png')
        plt.savefig('feature_importance.eps', format='eps')
    plt.show()


infix_map = {
    'add': '+',
    'subtract': '-',
    'multiply': '*',
}


def gene_to_string(gene):
    string = ""
    stack = []
    for node in gene:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            # string = prim.format(*args)
            if type(prim) is Primitive:
                string = '('
                if prim.name == 'analytical_quotient':
                    string += f'{args[0]}/sqrt(1+{args[1]}*{args[1]})'
                elif prim.name not in infix_map:
                    string += prim.format(*args)
                else:
                    string += args[0]
                    for a in args[1:]:
                        string += f'{infix_map[prim.name]}{a}'
                string += ')'
            else:
                string = prim.name
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string
