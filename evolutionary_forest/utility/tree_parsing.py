import operator
import random
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from deap import base, creator, gp, tools
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def visualize_predictions_and_uncertainties(predictions, uncertainties):
    # Create a DataFrame for easy manipulation and plotting
    data = pd.DataFrame(
        {
            "Prediction": predictions,
            "Uncertainty": uncertainties,
            "Index": range(len(predictions)),
        }
    )

    # Sort the DataFrame by Uncertainty
    data = data.sort_values(by="Uncertainty", ascending=False).reset_index(drop=True)

    # Use Seaborn's color palette
    palette = sns.color_palette("deep", len(data))

    # Initialize a matplotlib plot
    plt.figure(figsize=(12, 6))

    # Plot predictions as bars with the seaborn palette
    plt.bar(
        data["Index"], data["Prediction"], color=palette, alpha=0.6, label="Predictions"
    )

    # Add error bars representing uncertainties
    plt.errorbar(
        data["Index"],
        data["Prediction"],
        yerr=data["Uncertainty"],
        fmt="o",
        color="black",
        label="Uncertainties",
        capsize=5,
    )

    # Customize the plot
    plt.xlabel("Samples (sorted by uncertainty)")
    plt.ylabel("Prediction Value")
    plt.title("Predictions and Uncertainties (Sorted by Uncertainty)")
    plt.legend()

    # Display the plot
    plt.show()


def visualize_clusters(one_hot_encoded, clusters, use_tsne=False):
    # Dimensionality Reduction for Visualization
    if use_tsne:
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    reduced_data = reducer.fit_transform(one_hot_encoded)

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap="viridis", s=50
    )
    plt.title("Cluster Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    plt.show()


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    If the tree is directed and the direction is from root to leaves,
    then pos will be calculated from the root downwards.
    """
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(
    G,
    root,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    xcenter=0.5,
    pos=None,
    parent=None,
    parsed=[],
):
    """
    Helper function for hierarchy_pos.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)  # this should be for directed graphs
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(
                G,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=nextx,
                pos=pos,
                parent=root,
                parsed=parsed,
            )
    return pos


def plot_tree(individual):
    nodes, edges, labels = gp.graph(individual)

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    pos = hierarchy_pos(g, root=0)  # Adjust the root to start at the top
    nx.draw(g, pos, with_labels=False, arrows=False)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()


def mark_node_levels_recursive(individual, index=0, level=0, original_primitive=False):
    """
    Recursively parses a GP tree and marks the level of each node.

    Args:
    - individual: The GP tree to be parsed.
    - index: The current index in the GP tree list (default is 0, root).
    - level: The current level of the node (default is 0, root level).

    Returns:
    - A list of tuples, where each tuple contains (node, level).
    - The next index to be processed after this subtree.
    """
    if index >= len(individual):
        return [], index

    node = individual[index]
    if original_primitive:
        node_levels = [(node, level)]
    else:
        node_levels = [
            (node.name if isinstance(node, gp.Primitive) else str(node.value), level)
        ]

    if isinstance(node, gp.Primitive):  # If it's a function node
        child_index = index + 1
        for _ in range(node.arity):
            # Recursively process each child node
            child_levels, child_index = mark_node_levels_recursive(
                individual, child_index, level + 1, original_primitive
            )
            node_levels.extend(child_levels)
    else:
        child_index = index + 1  # Move to the next node

    return node_levels, child_index


def collect_nodes_by_level(node_levels):
    """
    Collects nodes by their levels.

    Args:
    - node_levels: A list of tuples, where each tuple contains (node, level).

    Returns:
    - A dictionary where keys are levels and values are lists of nodes at that level.
    """
    level_dict = defaultdict(list)

    for node, level in node_levels:
        level_dict[level].append(node)

    return level_dict


def process_tree_and_concat_levels(random_tree, cutoff=5):
    # Step 1: Mark the levels of each node
    node_levels, _ = mark_node_levels_recursive(random_tree)

    # Print the levels of each node
    # for node, level in node_levels:
    #     print(f"Node: {node}, Level: {level}")

    # Step 2: Collect nodes by their levels
    level_dict = collect_nodes_by_level(node_levels)

    # Output the nodes level by level
    # for level in sorted(level_dict.keys()):
    #     print(f"Level {level}: {level_dict[level]}")

    # Concatenate nodes in different levels, cut off by 5 nodes, and return the result
    concatenated_nodes = []
    for level in sorted(level_dict.keys()):
        concatenated_nodes.extend(level_dict[level])
        if len(concatenated_nodes) >= cutoff:
            break
    return concatenated_nodes[:cutoff]


def sort_tree(nodes_list: List[List[str]]) -> List[str]:
    # Sort each list of nodes and then concatenate them into one list
    sorted_nodes = []
    for nodes in sorted(nodes_list):
        sorted_nodes.extend(nodes)
    return sorted_nodes


def gp_tree_clustering(inds, n_clusters=3):
    one_hot_encoded = extract_gp_tree_features(inds)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(one_hot_encoded)

    # Select one index from each cluster
    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        selected_indices.append(
            cluster_indices[0]
        )  # Select the first individual in the cluster
    # Visualize the clusters
    # visualize_clusters(one_hot_encoded, clusters, True)
    inds = [inds[i] for i in selected_indices]
    return inds


def extract_gp_tree_features(inds, encoder=None):
    all_features = feature_extraction(inds)
    encoder, one_hot_encoded = feature_encoding(all_features, encoder)
    return one_hot_encoded


def feature_encoding(all_features, encoder=None):
    # One-hot encoding
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        one_hot_encoded = encoder.fit_transform(all_features)
    else:
        one_hot_encoded = encoder.transform(all_features)
    return encoder, one_hot_encoded


def feature_extraction(inds):
    all_features = []
    # Extract and process features from each individual's gene
    for ind in inds:
        features = []
        for tree in ind.gene:
            cutoff = 3
            feature = process_tree_and_concat_levels(tree, cutoff=cutoff)
            # Ensure each feature list has exactly 5 elements by appending empty strings if necessary
            if len(feature) < cutoff:
                feature.extend([""] * (cutoff - len(feature)))
            features.append(feature)
        features = sort_tree(features)
        all_features.append(features)
    return all_features


class HistoricalData:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.data = []
        self.labels = []

    def extend(self, data, labels):
        # Extend the data and labels lists
        self.data.extend(data)
        self.labels.extend(labels)

        # Ensure that the size of data and labels does not exceed max_size
        if len(self.data) > self.max_size:
            # Calculate the number of elements to remove
            excess = len(self.data) - self.max_size

            # Remove the oldest elements to maintain the size
            self.data = self.data[excess:]
            self.labels = self.labels[excess:]

    def get_data_and_labels(self):
        return self.data, self.labels


def gp_tree_prediction(
    parents,
    inds,
    historical_best: HistoricalData,
    surrogate_model,
    top_inds=10,
    min_samples_leaf=10,
    threshold=0.2,
    surrogate_model_type="RandomForest",
):
    all_features, labels = historical_best.get_data_and_labels()
    current_samples = len(all_features)

    current_all_features = feature_extraction(parents)
    current_labels = [ind.fitness.wvalues[0] for ind in parents]
    historical_best.extend(current_all_features, current_labels)
    current_labels = np.array(current_labels)
    if current_samples == 0:
        return inds[:top_inds], None

    if surrogate_model is None:
        # Create a pipeline with OneHotEncoder and RandomForestRegressor
        if surrogate_model_type == "RandomForest":
            model = RandomForestRegressor(min_samples_leaf=min_samples_leaf)
        elif surrogate_model_type == "ExtraTrees":
            model = ExtraTreesRegressor(min_samples_leaf=min_samples_leaf)
        else:
            raise ValueError("Invalid surrogate model type")
        model_pipeline = Pipeline(
            [
                ("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore")),
                ("rf", model),
            ]
        )
        # Train the pipeline on the training data
        model_pipeline.fit(all_features[:current_samples], labels[:current_samples])
    else:
        model_pipeline = surrogate_model

    prediction = model_pipeline.predict(current_all_features)
    # Calculate the Spearman correlation coefficient
    spearman_score = spearmanr(current_labels, prediction)[0]
    if spearman_score < threshold:
        return inds[:top_inds], None

    # Extract features for test data (automatically encoded by the pipeline)
    all_features_test = feature_extraction(inds)
    predictions = model_pipeline.predict(all_features_test)

    # Calculate prediction uncertainties
    rf = model_pipeline.named_steps["rf"]
    uncertainties = np.std(
        [
            tree.predict(
                model_pipeline.named_steps["onehot"].transform(all_features_test)
            )
            for tree in rf.estimators_
        ],
        axis=0,
    )

    # Combine predictions and uncertainties
    combined_scores = predictions + uncertainties
    # visualize_predictions_and_uncertainties(predictions, uncertainties)
    # Select top individuals based on combined scores
    top_indices = np.argsort(combined_scores)[-top_inds:]
    selected_inds = [inds[i] for i in top_indices]

    return selected_inds, model_pipeline


if __name__ == "__main__":
    # Step 1: Define the primitive set
    pset = gp.PrimitiveSet("MAIN", 1)  # 1 input variable
    pset.addPrimitive(operator.add, 2)  # Add function with arity 2
    pset.addPrimitive(operator.sub, 2)  # Subtract function with arity 2
    pset.addPrimitive(operator.mul, 2)  # Multiply function with arity 2
    pset.addPrimitive(operator.neg, 1)  # Negation function with arity 1

    pset.addEphemeralConstant(
        "rand101", lambda: random.randint(-10, 10)
    )  # Ephemeral constant

    pset.renameArguments(ARG0="x")  # Rename the single input variable

    # Step 2: Define fitness and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Step 3: Create the toolbox and register the necessary functions
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Step 4: Generate a random GP tree
    random_tree = toolbox.individual()
    print("Randomly generated GP tree:")
    print(random_tree)

    plot_tree(random_tree)

    print(process_tree_and_concat_levels(random_tree, cutoff=5))
