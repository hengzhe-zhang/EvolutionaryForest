import copy
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from deap.gp import Primitive
from matplotlib import pyplot as plt
from numba import njit
from sympy import latex, parse_expr

from evolutionary_forest.component.primitives import individual_to_tuple


def efficient_deepcopy(self, memo=None):
    if memo is None:
        memo = {}
    cls = self.__class__  # Extract the class of the object
    result = cls.__new__(cls)  # Create a new instance of the object based on extracted class
    memo[id(self)] = result
    for k, v in self.__dict__.items():
        if k in ('predicted_values', 'case_values', 'pipe'):
            continue
        # Copy over attributes by copying directly or in case of complex objects like lists
        # for example calling the `__deepcopy()__` method defined by them.
        # Thus recursively copying the whole tree of objects.
        setattr(result, k, copy.deepcopy(v, memo))
    return result


def get_feature_importance(regr, simple_version=True, fitness_weighted=False, mean_fitness=False,
                           ensemble_weighted=True):
    """
    :param regr: evolutionary forest
    :param simple_version: return simplified symbol, which is used for printing
    :param fitness_weighted: assign different weights to features based on fitness values
    :param mean_fitness: return mean feature importance instead of summative feature importance
    :return:
    """
    if mean_fitness:
        all_genes_map = defaultdict(list)
    else:
        all_genes_map = defaultdict(int)
    for x in regr.hof:
        if simple_version:
            latex_string = lambda g: latex(parse_expr(gene_to_string(g).replace("ARG", "X").replace("_", "-")),
                                           mul_symbol='dot')
            genes = [f'${latex_string(g)}$' for g in x.gene]
        else:
            # Genearting a Python code fragment
            def code_generation(gene):
                code = str(gene)
                args = ",".join(arg for arg in regr.pset.arguments)
                code = "lambda {args}: {code}".format(args=args, code=code)
                return code

            genes = [f'{code_generation(g)}' for g in x.gene]
        for g, c in zip(genes, np.abs(x.coef)):
            # Taking the fitness of each model into consideration
            importance_value = c
            if fitness_weighted:
                importance_value = importance_value * x.fitness.wvalues[0]
            if ensemble_weighted and hasattr(regr.hof, 'ensemble_weight'):
                importance_value = importance_value * regr.hof.ensemble_weight[individual_to_tuple(x)]
            # Deciding using summation or mean technique to calculate the importance value
            if mean_fitness:
                all_genes_map[g].append(importance_value)
            else:
                all_genes_map[g] += importance_value
    if mean_fitness:
        for k, v in all_genes_map.items():
            all_genes_map[k] = np.mean(v)
    feature_importance_dict = {k: v for k, v in sorted(all_genes_map.items(), key=lambda item: -item[1])}
    sum_value = np.sum([v for k, v in feature_importance_dict.items()])
    feature_importance_dict = {k: v / sum_value for k, v in feature_importance_dict.items()}
    return feature_importance_dict


def select_top_features(code_importance_dict, ratio=None):
    """
    :param code_importance_dict: feature importance dict
    :param ratio: only select top features
    :return:
    """
    if ratio == None:
        mean_importance = np.mean(list(code_importance_dict.values()))
    else:
        mean_importance = np.quantile(list(code_importance_dict.values()), ratio)
    features = list(map(lambda kv: kv[0], filter(lambda kv: kv[1] >= mean_importance, code_importance_dict.items())))
    return features


def feature_append(regr, X, features, only_new_features=False):
    if only_new_features:
        data = []
    else:
        data = [X]
    for f in features:
        func = eval(f, regr.pset.context)
        features = func(*X.T)
        if isinstance(features, np.ndarray) and len(features) == len(X):
            data.append(features.reshape(-1, 1))
    transformed_features = np.concatenate(data, axis=1)
    # Fix outliers (In extreme cases, some functions will produce unexpected results)
    transformed_features = np.nan_to_num(transformed_features, posinf=0, neginf=0)
    return transformed_features


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

    # Define size of bar plot
    plt.figure(figsize=(12, 8))
    sns.set(style="white", font_scale=1.5)
    # Plot Seaborn bar chart
    sns.barplot(x=fi_df['feature_importance'][:15], y=fi_df['feature_names'][:15], palette='bone')

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
    'protect_division': '/',
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
                elif prim.name == 'analytical_loge':
                    string += f'log(1+Abs({args[0]}))'
                elif prim.name == 'protect_sqrt':
                    string += f'sqrt(Abs({args[0]}))'
                elif prim.name == 'maximum':
                    string += f'Max({args[0]}, {args[1]})'
                elif prim.name == 'minimum':
                    string += f'Min({args[0]}, {args[1]})'
                elif prim.name == 'negative':
                    string += f'-{args[0]}'
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


@njit(cache=True)
def pairwise_data(x, y):
    data = []
    label = []
    for a, va in zip(x, y):
        for b, vb in zip(x, y):
            data.append(a - b)
            # data.append(np.concatenate((a, b)))
            # label.append(va > vb)
            label.append(va > vb)
    return data, label


def pairwise_data_np(x, y):
    data, label = pairwise_data(x, y)
    return np.array(data), np.array(label)


@njit(cache=True)
def calculate_value(x):
    data = []
    count = np.round(np.sqrt(len(x)))
    for i in range(0, len(x), count):
        s = 0
        for j in range(i, i + count):
            s += x[j]
        s = s / count
        data.append(s)
    return data


def get_activations(clf, X):
    # get hidden layer outputs
    hidden_layer_sizes = clf.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [X.shape[1]] + hidden_layer_sizes + \
                  [clf.n_outputs_]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(np.empty((X.shape[0],
                                     layer_units[i + 1])))
    clf._forward_pass(activations)
    return activations


if __name__ == '__main__':
    print(parse_expr('log(1)'))
    print(latex(parse_expr('X1*X2'), mul_symbol='dot'))


@njit(cache=True)
def numba_seed(a):
    random.seed(a)
    np.random.seed(a)


def reset_random(s):
    random.seed(s)
    np.random.seed(s)
    numba_seed(s)


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=0)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights, axis=0)
    return (average, np.sqrt(variance))


def save_array(array):
    save_object(array, f'np_{time.time()}.pkl')


def save_object(obj, path):
    import pickle
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_object(path):
    import pickle
    with open(path, 'rb') as file:
        obj = pickle.load(file)
        return obj
