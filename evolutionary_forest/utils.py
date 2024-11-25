import copy
import math
import os
import pickle as cPickle
import random
import time
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from deap.gp import Primitive
from matplotlib import pyplot as plt
from numba import njit
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sympy import latex, parse_expr

from evolutionary_forest.component.primitive_functions import individual_to_tuple


from evolutionary_forest.model.RidgeGCV import RidgeGCV

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class MeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.mean(X, axis=1)


class MedianRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.median(X, axis=1)


def extract_numbers(dimension, s):
    """
    Extracts two integers from a string of the form 'aN-b'.

    Args:
    s (str): The input string to extract integers from.

    Returns:
    A tuple containing two integers. The first integer is the number
    before the letter 'N' in the input string, and the second integer
    is the number after the dash.
    """

    # Split the string into two parts around the dash
    parts = s.split("-")

    # Extract the first number before the letter 'N'
    num1 = int(parts[0].rstrip("N"))

    # Extract the second number after the dash
    num2 = int(parts[1])

    return min(num1 * dimension, num2)


def cross_scale(array, cross_pb, inverse, power=1):
    # scaling based on cross probabilities
    array = np.array(array)
    array = np.power(array, power)
    if inverse:
        max_value = np.max(array)
        return cross_pb * (array / max_value)
    else:
        min_value = np.min(array)
        return cross_pb * (min_value / array)


def get_important_features(parents):
    # Get important features
    genes = list(chain.from_iterable([x.gene for x in parents]))
    coefficients = list(chain.from_iterable([x.coef for x in parents]))
    hash_codes = list(chain.from_iterable([x.hash_result for x in parents]))
    results = dict()
    for g, c, h in zip(genes, coefficients, hash_codes):
        if h not in results:
            results[h] = [float(c), g]
        else:
            results[h][0] += float(c)
    features = list(sorted(results.items(), key=lambda item: -1 * item[1][0]))
    coefs = list(map(lambda x: x[1][0], features))
    features = list(map(lambda x: x[1][1], features))
    return features, coefs


def efficient_deepcopy(self, memo=None, custom_filter=None):
    if memo is None:
        memo = {}
    cls = self.__class__  # Extract the class of the object
    result = cls.__new__(
        cls
    )  # Create a new instance of the object based on extracted class
    memo[id(self)] = result
    for k, v in self.__dict__.items():
        if custom_filter is None:
            custom_filter = ("predicted_values", "case_values", "estimators")
        if k in custom_filter:
            continue
        if k in ["content"]:
            setattr(result, k, v)
            continue
        # Copy over attributes by copying directly or in case of complex objects like lists
        # for example calling the `__deepcopy()__` method defined by them.
        # Thus, recursively copying the whole tree of objects.
        setattr(result, k, copy.deepcopy(v, memo))
    return result


# Generating a Python code fragment
def code_generation(regressor, tree):
    code = str(tree)
    args = ",".join(arg for arg in regressor.pset.arguments)
    code = "lambda {args}: {code}".format(args=args, code=code)
    return code


def get_feature_importance(
    regressor_model: "EvolutionaryForestRegressor",
    latex_version=True,
    fitness_weighted=False,
    mean_fitness=False,
    ensemble_weighted=True,
    simple_version=None,
    **params,
):
    """
    :param regressor_model: evolutionary forest
    :param latex_version: return simplified symbol, which is used for printing
    :param fitness_weighted: assign different weights to features based on fitness values
    :param mean_fitness: return mean feature importance instead of summative feature importance
    :param simple_version: alias for latex_version
    :return:
    """

    """
    Alias setting: just for compatibility
    """
    if simple_version is not None:
        latex_version = simple_version

    if mean_fitness:
        all_genes_map = defaultdict(list)
    else:
        all_genes_map = defaultdict(int)
    hash_dict = {}

    # Processing function
    if latex_version:
        # latex_string = lambda g: latex(
        #     parse_expr(gene_to_string(g).replace("ARG", "X").replace("_", "-")),
        #     mul_symbol="dot",
        # )
        # processing_code = lambda g: f"${latex_string(g)}$"
        processing_code = lambda g: gene_to_string(g)
    else:
        processing_code = lambda g: f"{code_generation(regressor_model, g)}"

    for x in regressor_model.hof:
        for o_g, h, c in zip(x.gene, x.hash_result, np.abs(x.coef)):
            # Taking the fitness of each model into consideration
            importance_value = c
            if fitness_weighted:
                importance_value = importance_value * x.fitness.wvalues[0]

            if ensemble_weighted and hasattr(regressor_model.hof, "ensemble_weight"):
                importance_value = (
                    importance_value
                    * regressor_model.hof.ensemble_weight[individual_to_tuple(x)]
                )

            # Merge features with equivalent hash values
            if h not in hash_dict or len(o_g) < len(hash_dict[h]):
                hash_dict[h] = o_g

            # Deciding to use summation or mean technique to calculate the importance value
            if mean_fitness:
                all_genes_map[h].append(importance_value)
            else:
                all_genes_map[h] += importance_value

    # Forming a dict
    all_genes_map = {
        processing_code(hash_dict[k]): all_genes_map[k] for k in all_genes_map.keys()
    }
    if mean_fitness:
        for k, v in all_genes_map.items():
            all_genes_map[k] = np.mean(v)

    feature_importance_dict = {
        k: v for k, v in sorted(all_genes_map.items(), key=lambda item: -item[1])
    }
    sum_value = np.sum([v for k, v in feature_importance_dict.items()])
    feature_importance_dict = {
        k: v / sum_value for k, v in feature_importance_dict.items()
    }
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
    features = list(
        map(
            lambda kv: kv[0],
            filter(lambda kv: kv[1] >= mean_importance, code_importance_dict.items()),
        )
    )
    return features


def plot_feature_importance(feature_importance_dict, save_fig=False):
    import textwrap

    names, importance = list(feature_importance_dict.keys()), list(
        feature_importance_dict.values()
    )
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = names

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Wrap long feature names
    fi_df["feature_names"] = fi_df["feature_names"].apply(
        lambda x: "\n".join(textwrap.wrap(x, width=45))
    )

    # Define size of bar plot
    plt.figure(figsize=(16, 16))
    sns.set(style="white", font_scale=1.5)
    # Plot Seaborn bar chart
    sns.barplot(
        x=fi_df["feature_importance"][:15],
        y=fi_df["feature_names"][:15],
        palette="bone",
    )

    # Add chart labels
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    if save_fig:
        plt.savefig("result/feature_importance.png", format="png")
        plt.savefig("result/feature_importance.eps", format="eps")
    plt.show()


infix_map = {
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
}


def gene_to_string(gene):
    string = ""
    stack = []
    for node in gene:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            # string = prim.format(*args)
            if isinstance(prim, Primitive):
                string = "("
                if prim.name == "AQ":
                    string += f"{args[0]}/sqrt(1+{args[1]}*{args[1]})"
                elif prim.name == "AQ-Signed":
                    string += (
                        f"(sign({args[1]})*({args[0]}/sqrt(1+{args[1]}*{args[1]})))"
                    )
                elif prim.name == "Log":
                    string += f"log(sqrt(1+{args[0]}*{args[0]}))"
                elif prim.name == "ALog":
                    string += f"log(sqrt(1+{args[0]}*{args[0]}))"
                elif prim.name == "AbsLog":
                    string += f"log(1+abs({args[0]}))"
                elif prim.name == "ALog-Signed":
                    string += f"(sign({args[0]})*log(sqrt(1+{args[0]}*{args[0]})))"
                elif prim.name == "Log10":
                    string += f"log(sqrt(1+{args[0]}*{args[0]}),10)"
                elif prim.name == "Square":
                    string += f"{args[0]}*{args[0]}"
                elif prim.name == "Cube":
                    string += f"{args[0]}*{args[0]}*{args[0]}"
                elif prim.name == "Sqrt":
                    string += f"sqrt(Abs({args[0]}))"
                elif prim.name == "Sqrt-Signed":
                    string += f"(sign({args[0]})*sqrt(Abs({args[0]})))"
                elif prim.name == "Add":
                    string += f"({args[0]}+{args[1]})"
                elif prim.name == "Sub":
                    string += f"({args[0]}-{args[1]})"
                elif prim.name == "Mul":
                    string += f"({args[0]}*{args[1]})"
                elif prim.name == "Max":
                    string += f"Max({args[0]}, {args[1]})"
                elif prim.name == "Min":
                    string += f"Min({args[0]}, {args[1]})"
                elif prim.name == "Neg":
                    string += f"-{args[0]}"
                elif prim.name == "Tanh":
                    string += f"tanh({args[0]})"
                elif prim.name == "Sigmoid":
                    string += f"expit({args[0]})"
                elif prim.name == "Arctan":
                    string += f"atan({args[0]})"
                elif prim.name == "Square":
                    string += f"Pow({args[0]},2)"
                elif prim.name == "Cube":
                    string += f"Pow({args[0]},3)"
                elif prim.name == "Cbrt":
                    string += f"Pow({args[0]},1/3)"
                elif prim.name == "Sin":
                    string += f"sin({args[0]})"
                elif prim.name == "Cos":
                    string += f"cos({args[0]})"
                elif prim.name in ["SinPi", "RSin"]:
                    string += f"sin({math.pi}*{args[0]})"
                elif prim.name in ["CosPi", "RCos"]:
                    string += f"cos({math.pi}*{args[0]})"
                elif prim.name == "Abs":
                    string += f"Abs({args[0]})"
                elif prim.name not in infix_map:
                    string += prim.format(*args)
                else:
                    string += args[0]
                    for a in args[1:]:
                        string += f"{infix_map[prim.name]}{a}"
                string += ")"
            else:
                string = str(prim.value)
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
    layer_units = [X.shape[1]] + hidden_layer_sizes + [clf.n_outputs_]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(np.empty((X.shape[0], layer_units[i + 1])))
    clf._forward_pass(activations)
    return activations


if __name__ == "__main__":
    print(parse_expr("log(1)"))
    print(latex(parse_expr("X1*X2"), mul_symbol="dot"))


@njit(cache=True)
def numba_seed(a):
    random.seed(a)
    np.random.seed(a)


def reset_random(s):
    random.seed(s)
    np.random.seed(s)
    numba_seed(s)
    torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


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
    save_object(array, f"np_{time.time()}.pkl")


def save_object(obj, path):
    import pickle

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_object(path):
    import pickle

    with open(path, "rb") as file:
        obj = pickle.load(file)
        return obj


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def pickle_deepcopy(a):
    return cPickle.loads(cPickle.dumps(a, -1))


def cv_prediction_from_ridge(Y, base_model: (RidgeCV, RidgeGCV)):
    """
    Scikit-learn _preprocess_data function will center Y, so we need to add the mean back
    """
    # all_y_pred = base_model.cv_results_ + Y.mean()
    # errors = (Y.reshape(-1, 1) - all_y_pred) ** 2
    # assert errors.size == Y.size * len(base_model.alphas)
    # error_list = errors.sum(axis=0)
    # new_best_index = np.argmin(error_list)

    # get optimal alpha
    new_best_index = tuple(base_model.alphas).index(base_model.alpha_)
    # get cv_values
    real_prediction = base_model.cv_results_[:, new_best_index]
    # restore to correct scale
    real_prediction = real_prediction + Y.mean()
    return real_prediction


def pareto_front_2d(points, mode="min") -> [np.ndarray, float]:
    """
    Calculate the Pareto front for a 2D problem using pymoo.

    Parameters:
    - points: List of objectives, where each element is a tuple (objective1, objective2).
    - mode: A string indicating if the problem is 'max' or 'min'. Default is 'min'.

    Returns:
    - pareto_front: List of Pareto-optimal points.
    - pareto_indices: Indices of the Pareto-optimal points.
    """

    # Convert the mode into the correct objective directions for non-dominated sorting
    if mode == "min":
        mask = np.array([1, 1])
    elif mode == "max":
        mask = np.array([-1, -1])
    else:
        raise ValueError("Invalid mode. Choose 'min' or 'max'.")

    # Adjust points according to the mode (minimization or maximization)
    adjusted_points = np.array(points) * mask

    # Perform non-dominated sorting
    nds = NonDominatedSorting().do(adjusted_points)

    # Extract the Pareto front
    pareto_front = np.array(points)[nds[0]]
    pareto_indices = nds[0]

    return pareto_front, pareto_indices.tolist()


def one_hot_encode(categories):
    """
    Perform One-Hot Encoding on a given NumPy array of integer categories.

    Parameters:
        categories (np.array): The NumPy array containing integer categories.

    Returns:
        np.array: A NumPy array with One-Hot Encoded columns.
    """

    # Find all unique categories and their corresponding indices in the original array
    unique_categories, inverse = np.unique(categories, return_inverse=True)

    # Initialize a zero array with shape (number of samples, number of unique categories)
    one_hot = np.zeros((categories.shape[0], unique_categories.shape[0]))

    # Set appropriate positions to 1 based on original category positions
    one_hot[np.arange(categories.shape[0]), inverse] = 1

    return one_hot


def tuple_to_list(nested_tuple):
    if isinstance(nested_tuple, tuple):
        return [tuple_to_list(item) for item in nested_tuple]
    else:
        return nested_tuple


def list_to_tuple(nested_list):
    if isinstance(nested_list, list):
        return tuple(list_to_tuple(item) for item in nested_list)
    else:
        return nested_list


def model_to_string(genes, learner, scaler):
    coefs = learner.coef_
    model_str = ""
    for id, g, c in zip(range(0, len(genes)), genes, coefs):
        gene_string = gene_to_string(g)
        if scaler != None:
            mean, std = scaler.mean_[id], scaler.scale_[id]
            gene_string = f"(({gene_string}-{mean})/{std})"
        if id == 0:
            model_str += str(c) + "*" + gene_string
        else:
            model_str += "+" + str(c) + "*" + gene_string
    model_str += "+" + str(learner.intercept_)
    return model_str.replace("ARG", "x")
