from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from deap.gp import PrimitiveTree, Primitive

from evolutionary_forest.multigene_gp import MultipleGeneGP


def detect_communities_louvain(graph):
    import community as community_louvain

    """
    Perform community detection using the Louvain method.

    Parameters:
    - graph: A NetworkX graph object.

    Returns:
    - partition: A dictionary mapping node identifiers to their respective community.
    - communities: A list of sets, where each set contains the nodes belonging to a community.
    """

    # Use the Louvain method to find the best partition
    partition = community_louvain.best_partition(graph)

    # Calculate the modularity of the partition
    modularity_score = community_louvain.modularity(partition, graph)
    print(f"Modularity: {modularity_score}")

    # Invert the partition dictionary to group nodes by their community
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = set()
        communities[community_id].add(node)

    # Convert the communities dict to a list for easier handling
    communities_list = list(communities.values())

    return partition, communities_list


def plot_graph_with_communities(G, communities):
    """
    Plots the graph with nodes colored by their community, labels displayed, edge weights shown,
    and node size based on betweenness centrality to indicate importance.

    Parameters:
    - G: A NetworkX graph object.
    - communities: A list of sets, where each set contains the nodes belonging to a community.
    """

    # Calculate betweenness centrality for each node
    centrality = nx.betweenness_centrality(G)

    # Normalize and scale centrality values for use as node sizes
    # Multiply by a constant to scale sizes to a suitable range for your plot
    centrality_values = np.array(list(centrality.values()))
    sizes = (
        centrality_values / centrality_values.max()
    ) * 1000  # Adjust multiplier as needed

    # Set the seaborn color palette
    colors = sns.color_palette("hsv", len(communities))

    # Create a figure and axis for the plot
    plt.figure(figsize=(12, 12))

    # Generate a layout that can potentially spread out the communities better
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # Draw nodes with colors based on their community and sizes based on centrality
    for idx, community in enumerate(communities):
        community_sizes = [sizes[list(G.nodes).index(node)] for node in community]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=community,
            node_color=[colors[idx]] * len(community),
            node_size=community_sizes,
            label=f"Community {idx + 1}",
            alpha=0.8,
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Add legend for communities
    plt.legend(scatterpoints=1)

    plt.title("Graph with Communities, Labels, Edge Weights, and Node Importance")
    plt.axis("off")  # Turn off the axis
    plt.show()


def is_constant_terminal(node, labels):
    label = labels[node]
    try:
        float(label)  # Try converting the label to a float
        return True  # Conversion succeeded, so it's a numeric constant
    except ValueError:
        return False  # Conversion failed, so it's not a numeric constant


def graph(expr):
    nodes = list(range(len(expr)))
    edges = list()
    labels = dict()
    types = dict()  # Dictionary to hold the type of each node

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = node.name if isinstance(node, Primitive) else node.value
        types[i] = (
            "primitive" if isinstance(node, Primitive) else "terminal"
        )  # Determine the type of the node
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels, types


def merge_trees_to_graph(inds: List[MultipleGeneGP]):
    # Initialize an empty weighted graph using NetworkX
    G = nx.Graph()

    # Iterate over each individual in the population
    for ind in inds:
        # Iterate over each gene (tree) in the individual
        for tree in ind.gene:
            # Ensure the tree is treated as a PrimitiveTree
            assert isinstance(tree, PrimitiveTree), (
                "Tree is not an instance of PrimitiveTree"
            )

            # Extract nodes, edges, labels, and types from the tree
            nodes, edges, labels, types = graph(tree)

            # Add nodes to the graph using labels as identifiers to merge duplicate nodes
            for node in nodes:
                if not is_constant_terminal(
                    node, labels
                ):  # Check if the node is not a constant terminal
                    label = labels[node]
                    if label not in G:
                        # Add node with its type ('primitive' or 'terminal')
                        G.add_node(label, type=types[node])

            # Add or update edges with weights
            for edge in edges:
                if not is_constant_terminal(
                    edge[0], labels
                ) and not is_constant_terminal(edge[1], labels):
                    node1_label = labels[edge[0]]
                    node2_label = labels[edge[1]]

                    if G.has_edge(node1_label, node2_label):
                        # If the edge already exists, increase its weight
                        G[node1_label][node2_label]["weight"] += 1
                    else:
                        # Add new edge with a weight of 1
                        G.add_edge(node1_label, node2_label, weight=1)

    # Return the combined graph
    return G


def merge_trees_list_to_graph(inds: List[MultipleGeneGP]):
    # Initialize an empty weighted graph using NetworkX
    G = nx.Graph()

    # Iterate over each individual in the population
    for tree in inds:
        # Ensure the tree is treated as a PrimitiveTree
        assert isinstance(tree, PrimitiveTree), (
            "Tree is not an instance of PrimitiveTree"
        )

        # Extract nodes, edges, labels, and types from the tree
        nodes, edges, labels, types = graph(tree)

        # Add nodes to the graph using labels as identifiers to merge duplicate nodes
        for node in nodes:
            if not is_constant_terminal(
                node, labels
            ):  # Check if the node is not a constant terminal
                label = labels[node]
                if label not in G:
                    # Add node with its type ('primitive' or 'terminal')
                    G.add_node(label, type=types[node])

        # Add or update edges with weights
        for edge in edges:
            if not is_constant_terminal(edge[0], labels) and not is_constant_terminal(
                edge[1], labels
            ):
                node1_label = labels[edge[0]]
                node2_label = labels[edge[1]]

                if G.has_edge(node1_label, node2_label):
                    # If the edge already exists, increase its weight
                    G[node1_label][node2_label]["weight"] += 1
                else:
                    # Add new edge with a weight of 1
                    G.add_edge(node1_label, node2_label, weight=1)

    # Return the combined graph
    return G


def plot_graph_with_centrality(G, current_gen, centrality_type="eigenvector"):
    """
    Plots the graph with nodes sized by their betweenness centrality.

    Parameters:
    - G: A NetworkX graph object.
    """
    # Calculate betweenness centrality for each node
    if centrality_type == "degree":
        centrality = nx.degree_centrality(G, weight="weight")
    elif centrality_type == "betweenness":
        centrality = nx.betweenness_centrality(G, weight="weight")
    elif centrality_type == "closeness":
        centrality = nx.closeness_centrality(G, weight="weight")
    elif centrality_type == "eigenvector":
        centrality = nx.eigenvector_centrality(
            G, max_iter=1000, weight="weight"
        )  # max_iter may need adjustment
    else:
        raise Exception(f"Unsupported centrality type: {centrality_type}")

    # Normalize the centrality values to use for node size
    # Multiply by a constant to scale the sizes to your preference
    centrality_values = np.array(list(centrality.values()))
    sizes = (
        centrality_values / centrality_values.max()
    ) * 1000  # Adjust the multiplier as needed

    # Explicitly create a figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(
        G, k=0.1, iterations=20
    )  # Adjust layout parameters as needed

    # Draw the nodes, using centrality values for sizing
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=sizes, node_color=sizes, cmap=plt.cm.viridis, alpha=0.8, ax=ax
    )

    # Draw the edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)

    plt.title(f"Graph with Nodes Sized by Betweenness Centrality, Gen {current_gen}")
    plt.axis("off")  # Turn off the axis

    # Fix: Provide the `ax` argument to `plt.colorbar()`
    plt.colorbar(nodes, ax=ax, label="Betweenness Centrality")

    plt.savefig("result/graph_with_centrality.eps", format="eps")
    plt.show()


def get_important_nodes_labels(graph, centrality_type="betweenness", threshold=0.1):
    """
    Identify and get the labels of important nodes in a graph based on a specified centrality measure.

    Parameters:
    - graph: A NetworkX graph object.
    - centrality_type: Type of centrality measure to use ('degree', 'betweenness', 'closeness', 'eigenvector').
    - threshold: Centrality threshold to consider a node as important (range: 0 to 1).

    Returns:
    - important_nodes: A list of labels of the important nodes.
    """

    # Calculate centrality for all nodes based on the specified centrality type
    if centrality_type == "degree":
        centrality = nx.degree_centrality(graph, weight="weight")
    elif centrality_type == "betweenness":
        centrality = nx.betweenness_centrality(graph, weight="weight")
    elif centrality_type == "closeness":
        centrality = nx.closeness_centrality(graph, weight="weight")
    elif centrality_type == "eigenvector":
        centrality = nx.eigenvector_centrality(
            graph, max_iter=1000, weight="weight"
        )  # max_iter may need adjustment
    else:
        raise ValueError(f"Unsupported centrality type: {centrality_type}")

    # Identify nodes with centrality above the threshold
    important_nodes = [
        node
        for node, centrality_value in centrality.items()
        if centrality_value >= threshold
    ]

    # Get the labels of important nodes (assuming the node itself is used as label, adjust as needed)
    important_nodes_labels = [
        graph.nodes[node].get("label", node) for node in important_nodes
    ]

    return important_nodes_labels


def get_all_node_labels(graph):
    """
    Retrieve all node labels from a NetworkX graph.

    Parameters:
    - graph: A NetworkX graph object.

    Returns:
    - node_labels: A dictionary where keys are node identifiers and values are their labels.
    """

    # Initialize an empty dictionary to hold node labels
    node_labels = {}

    # Iterate through all nodes in the graph
    for node in graph.nodes(data=True):
        node_id, attributes = node
        # Retrieve the 'label' attribute for each node, default to node_id if 'label' is not found
        node_labels[node_id] = attributes.get("label", node_id)

    return node_labels


def get_centrality_score(good_graph, centrality_type="betweenness"):
    """
    Calculate the ratio of centrality values for nodes with the same labels in good_graph and bad_graph.

    Parameters:
    - good_graph: A NetworkX graph object considered as the "good" graph.
    - bad_graph: A NetworkX graph object considered as the "bad" graph.
    - centrality_type: Type of centrality measure to use ('degree', 'betweenness', 'closeness', 'eigenvector').

    Returns:
    - centrality_scores: A dictionary mapping node labels to their centrality ratio (good_graph / bad_graph).
    """

    # Function to calculate centrality based on type
    def calculate_centrality(graph, type):
        if type == "degree":
            return nx.degree_centrality(graph)
        elif type == "betweenness":
            return nx.betweenness_centrality(graph, weight="weight")
        elif type == "closeness":
            return nx.closeness_centrality(graph)
        elif type == "eigenvector":
            return nx.eigenvector_centrality(graph, max_iter=1000)
        else:
            raise ValueError(f"Unsupported centrality type: {type}")

    # Calculate centrality for all nodes in both graphs
    good_centrality = calculate_centrality(good_graph, centrality_type)

    # Compute the ratio of centrality values for nodes with the same label
    centrality_scores = {}
    for node in good_graph.nodes():
        label = good_graph.nodes[node].get("label", node)
        centrality_scores[label] = good_centrality[label]

    return centrality_scores


def select_important_nodes_by_ratio(centrality_ratios, threshold=2.0):
    """
    Select important nodes based on the ratio of their centrality values in good_graph to bad_graph.

    Parameters:
    - centrality_ratios: A dictionary mapping node labels to their centrality ratios.
    - threshold: The minimum ratio to consider a node as important.

    Returns:
    - important_nodes: A list of labels of the important nodes based on the centrality ratio.
    """
    important_nodes = [
        label for label, ratio in centrality_ratios.items() if ratio >= threshold
    ]
    return important_nodes


def get_top_nodes_by_centrality_ratios(graph, centrality_ratios, node_type, top_k=10):
    """
    Get the top log K nodes of a specific type (primitive or terminal) based on centrality ratios.

    Parameters:
    - graph: A NetworkX graph object.
    - centrality_ratios: A dictionary mapping node labels to their centrality ratios.
    - node_type: The type of node to consider ('primitive' or 'terminal').

    Returns:
    - top_nodes: A list of labels of the top log K nodes of the specified type.
    """
    # Filter nodes by the specified type and their presence in the centrality_ratios
    filtered_nodes = {
        label: ratio
        for label, ratio in centrality_ratios.items()
        if graph.nodes[label].get("type") == node_type and label in graph
    }

    # Calculate log K, where K is the number of nodes of the specified type
    K = len(filtered_nodes)
    if K == 0:
        return []  # Return an empty list if there are no nodes of the specified type
    # top_k = math.ceil(math.log(K, 2))
    top_k = max(top_k, 1)  # Ensure at least one node is selected

    # Sort the filtered nodes by their centrality ratio in descending order
    sorted_nodes = sorted(filtered_nodes.items(), key=lambda x: x[1], reverse=True)

    # Select the top log K nodes
    top_nodes = [node[0] for node in sorted_nodes[:top_k]]

    return top_nodes


def get_top_primitives_and_terminals(good_graph, centrality_type, threshold: int):
    centrality_score = get_centrality_score(good_graph, centrality_type=centrality_type)
    top_primitives = get_top_nodes_by_centrality_ratios(
        good_graph,
        centrality_score,
        node_type="primitive",
        top_k=threshold,
    )
    top_terminals = get_top_nodes_by_centrality_ratios(
        good_graph,
        centrality_score,
        node_type="terminal",
        top_k=threshold,
    )
    nodes = top_primitives + top_terminals
    return nodes


def get_bad_primitives_and_terminals(
    good_graph, bad_graph, centrality_type, threshold: int
):
    centrality_score = get_centrality_score(good_graph, centrality_type=centrality_type)
    top_primitives = get_top_nodes_by_centrality_ratios(
        good_graph,
        centrality_score,
        node_type="primitive",
        top_k=threshold,
    )
    top_terminals = get_top_nodes_by_centrality_ratios(
        good_graph,
        centrality_score,
        node_type="terminal",
        top_k=threshold,
    )
    centrality_score = get_centrality_score(bad_graph, centrality_type=centrality_type)
    bad_primitives = get_top_nodes_by_centrality_ratios(
        bad_graph,
        centrality_score,
        node_type="primitive",
        top_k=threshold,
    )
    bad_terminals = get_top_nodes_by_centrality_ratios(
        bad_graph,
        centrality_score,
        node_type="terminal",
        top_k=threshold,
    )
    primitive_nodes = list(set(list(bad_terminals)) - set(list(top_terminals)))
    terminal_nodes = list(set(list(bad_primitives)) - set(list(top_primitives)))
    nodes = primitive_nodes + terminal_nodes
    return nodes


def get_primitives_and_terminals_by_ratio(
    good_graph, centrality_type, threshold: float
):
    good_primitive_nodes = get_important_nodes_labels_by_type(
        good_graph, "primitive", centrality_type, threshold=threshold
    )
    good_terminal_nodes = get_important_nodes_labels_by_type(
        good_graph, "terminal", centrality_type, threshold=threshold
    )
    nodes = good_primitive_nodes + good_terminal_nodes
    return nodes


def get_bad_primitive_and_terminals_by_ratio(
    good_graph, bad_graph, centrality_type, threshold: float
):
    # Extract bad fetures
    good_primitive_nodes = get_important_nodes_labels_by_type(
        good_graph, "primitive", centrality_type, threshold=threshold
    )
    good_terminal_nodes = get_important_nodes_labels_by_type(
        good_graph, "terminal", centrality_type, threshold=threshold
    )
    bad_primitive_nodes = get_important_nodes_labels_by_type(
        bad_graph, "primitive", centrality_type, threshold=threshold
    )
    bad_terminal_nodes = get_important_nodes_labels_by_type(
        bad_graph, "terminal", centrality_type, threshold=threshold
    )
    primitive_nodes = list(
        set(list(bad_terminal_nodes)) - set(list(good_terminal_nodes))
    )
    terminal_nodes = list(
        set(list(bad_primitive_nodes)) - set(list(good_primitive_nodes))
    )
    nodes = primitive_nodes + terminal_nodes
    return nodes


def get_important_nodes_labels_by_type(
    graph, node_type, centrality_type="betweenness", threshold=0.1
):
    """
    Identify and get the labels of important nodes of a specific type in a graph based on a specified centrality measure,
    considering the entire graph structure for centrality calculation.

    Parameters:
    - graph: A NetworkX graph object.
    - node_type: Type of nodes to consider ('primitive' or 'terminal').
    - centrality_type: Type of centrality measure to use ('degree', 'betweenness', 'closeness', 'eigenvector').
    - threshold: Centrality threshold to consider a node as important (range: 0 to 1, after normalization).

    Returns:
    - important_nodes_labels: A list of labels of the important nodes of the specified type.
    """

    # Calculate centrality for all nodes in the graph based on the specified centrality type
    if centrality_type == "degree":
        centrality = nx.degree_centrality(graph, weight="weight")
    elif centrality_type == "betweenness":
        centrality = nx.betweenness_centrality(graph, weight="weight")
    elif centrality_type == "closeness":
        centrality = nx.closeness_centrality(graph, weight="weight")
    elif centrality_type == "eigenvector":
        centrality = nx.eigenvector_centrality(
            graph, max_iter=1000, weight="weight"
        )  # max_iter may need adjustment
    else:
        raise ValueError(f"Unsupported centrality type: {centrality_type}")

    # Normalize centrality values for nodes of the specified type
    type_centrality = {
        node: val
        for node, val in centrality.items()
        if graph.nodes[node].get("type") == node_type
    }

    # please never use sum, because sum is influenced by the number of functions/terminals
    max_centrality = max(type_centrality.values(), default=1)
    normalized_centrality = {
        node: val / max_centrality for node, val in type_centrality.items()
    }

    # Identify important nodes of the specified type based on the normalized centrality threshold
    important_nodes = [
        node for node, val in normalized_centrality.items() if val >= threshold
    ]

    # Get the labels of important nodes
    important_nodes_labels = [
        graph.nodes[node].get("label", node) for node in important_nodes
    ]

    return important_nodes_labels
