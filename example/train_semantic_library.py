# Train the model
import editdistance
import numpy as np
from deap.gp import PrimitiveTree
from sklearn.model_selection import train_test_split

from evolutionary_forest.utility.mlp_library import (
    NeuralSemanticLibrary,
    filter_train_data_by_node_count,
    generate_synthetic_data,
)
from example.utils.primitive_sets import get_pset


def calculate_edit_distance(nl, test_data):
    edit_distance = []
    predicted_trees = []

    for tid in range(len(test_data)):
        # Convert the predicted tree and original tree to a list of tokens
        predicted_tree = nl.convert_to_primitive_tree(test_data[tid][1])
        original_tree = PrimitiveTree(test_data[tid][0])

        distance = editdistance.eval(
            [node.name for node in predicted_tree],
            [node.name for node in original_tree],
        )
        edit_distance.append(distance)
        predicted_trees.append(predicted_tree)
    avg_edit_distance = np.mean(edit_distance)
    return avg_edit_distance


def augment_data(train_data):
    augmented_data = train_data.copy()
    for gp_tree, target_semantics in train_data:
        augmented_data.append((gp_tree, -1 * target_semantics))
    return augmented_data


def evaluate_model(nl, test_data):
    test_data_aug = augment_data(test_data)
    edit_distance = calculate_edit_distance(nl, test_data_aug)
    print("Edit Distance: ", edit_distance)
    return edit_distance


if __name__ == "__main__":
    num_samples = 10000
    number_of_variables = 10
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, number_of_variables), scale=100)
    pset = get_pset(data)

    min_depth = 0
    max_depth = 5
    max_function_nodes = 5

    # Generate synthetic training data
    train_data = generate_synthetic_data(
        pset,
        input_data=data,
        num_samples=num_samples,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    print("Number of training samples: ", len(train_data))
    # Filter training data by node count
    train_data_all = filter_train_data_by_node_count(
        train_data, max_function_nodes=max_function_nodes
    )
    train_data, test_data = train_test_split(
        train_data_all, test_size=0.2, shuffle=True, random_state=0
    )

    nl = NeuralSemanticLibrary(
        data.shape[0],
        64,
        64,
        3,
        output_primitive_length=max_function_nodes,
        pset=pset,
        use_transformer=True,
    )

    nl.train(
        train_data,
        epochs=1000,
        lr=0.01,
        val_split=0.2,
        verbose=True,
        loss_weight=0.05,
        patience=5,
    )

    edit_distance = evaluate_model(nl, test_data)
    print("Test Edit Distance: ", edit_distance)
