import deap.base as base
import deap.creator as creator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap import gp
from deap.gp import PrimitiveSet, PrimitiveTree, Primitive
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from evolutionary_forest.probability_gp import genHalfAndHalf
from evolutionary_forest.utility.tree_parsing import mark_node_levels_recursive


class PrimitiveSetUtils:
    def __init__(self, pset):
        """
        Initialize the utility class with the DEAP PrimitiveSet.

        :param pset: DEAP PrimitiveSet used to decode GP trees.
        """
        self.pset = pset
        self.node_name_to_index = self.get_node_name_to_index()
        self.index_to_node_name = self.get_index_to_node_name()

    def get_node_name_to_index(self):
        """
        Create a mapping from node names to indices based on the PrimitiveSet.

        :return: Dictionary mapping node names to indices.
        """
        functions, terminals = self.extract_pset_elements()
        node_names = [func.name for func in functions] + [
            term.name for term in terminals
        ]
        return {
            name: idx + 1 for idx, name in enumerate(node_names)
        }  # Start indexing from 1

    def get_index_to_node_name(self):
        """
        Create a reverse mapping from indices to node names based on the PrimitiveSet.

        :return: Dictionary mapping indices to node names.
        """
        index_to_node_name = {
            0: ""
        }  # Index 0 is reserved for padding and mapped to an empty string or None
        index_to_node_name.update(
            {idx: name for name, idx in self.node_name_to_index.items()}
        )
        return index_to_node_name

    def decode_to_functions_and_terminals(self, indices):
        """
        Decode a list of indices into functions and terminals based on the PrimitiveSet.

        :param indices: List of integer indices corresponding to functions and terminals.
        :return: Tuple of lists (functions, terminals) where each list contains the decoded functions and terminals.
        """
        functions = []
        terminals = []

        for idx in indices:
            node_name = self.index_to_node_name.get(idx)
            if node_name is None:
                raise ValueError(f"Index '{idx}' not found in index_to_node_name")

            if node_name in self.pset.primitives:
                functions.append(self.pset.primitives[node_name])
            elif node_name in self.pset.terminals:
                terminals.append(self.pset.terminals[node_name])
            else:
                raise ValueError(f"Node name '{node_name}' not found in PrimitiveSet")

        return functions, terminals

    def extract_targets(self, train_data, max_length):
        """
        Extract targets from GP trees in the training data and pad them to have the same length.

        :param train_data: List of tuples (gp_tree, target_semantics) from generate_synthetic_data.
        :return: List of padded target tensors.
        """
        targets = []
        padding_index = 0  # Use 0 as the padding index

        # First pass: Extract tree indices and determine max length
        tree_indices_list = []
        for tree, _ in train_data:
            tree: creator.Individual
            tree = self.convert_gp_tree_to_node_names(tree)

            # Extract node names and convert to indices
            tree_indices = []
            for node in tree:
                node_name = node.name
                if node_name in self.node_name_to_index:
                    tree_indices.append(self.node_name_to_index[node_name])
                else:
                    raise ValueError(
                        f"Node name '{node_name}' not found in node_name_to_index"
                    )

            # Store the tree indices and update max_length
            tree_indices_list.append(tree_indices)

        # Second pass: Pad tree indices and convert to tensors
        for tree_indices in tree_indices_list:
            # Pad the indices with the padding index
            padded_indices = tree_indices + [padding_index] * (
                max_length - len(tree_indices)
            )

            # Convert to tensor and add to targets
            target_tensor = torch.tensor(padded_indices, dtype=torch.long)
            targets.append(target_tensor)

        return targets

    def extract_pset_elements(self):
        """
        Extract function and terminal names from the PrimitiveSet.

        :return: Tuple of lists (functions, terminals), where each list contains names of elements.
        """
        functions = self.pset.primitives[object]
        terminals = self.pset.terminals[object]
        return functions, terminals

    def convert_node_names_to_gp_tree(self, node_names):
        """
        Convert a list of node names into a DEAP GP tree.

        :param node_names: List of node names corresponding to functions and terminals.
        :return: DEAP GP tree constructed from the node names.
        """
        # Create a dictionary to map function names to their DEAP functions
        func_dict = {func.name: func for func in self.pset.primitives[object]}
        term_dict = {term.name: term for term in self.pset.terminals[object]}

        def build_tree(node_names):
            track = 0
            if not node_names:
                return None

            stack = []
            for idx, name in enumerate(node_names):
                if idx > 0 and track == 0:
                    break
                if name in func_dict:
                    func = func_dict[name]
                    arity = func.arity
                    if track != 0:
                        track -= 1
                    track += arity
                    stack.append(func_dict[name])
                elif name in term_dict:
                    track -= 1
                    terminal = term_dict[name]
                    if callable(terminal):
                        terminal = terminal()
                    stack.append(terminal)
                else:
                    raise ValueError(f"Node name '{name}' not found in PrimitiveSet")
            return stack

        def reorder_node_names_recursive(first, node_names):
            """
            Reorder a list of node names such that each function is followed by its arguments using recursion.

            :param node_names: List of node names corresponding to functions and terminals.
            :return: Reordered list of node names.
            """
            # If it's a function (Primitive), it will have arguments
            if isinstance(first, Primitive):
                args = []
                root_args = []
                for ix in range(first.arity):
                    a = node_names.pop(0)
                    # print(first.name, a.name)
                    root_args.append(a)

                for ix, a in enumerate(root_args):
                    nodes = reorder_node_names_recursive(a, node_names)
                    # print(first.name, [n.name for n in nodes])
                    args.extend(nodes)

                # Return the function followed by its arguments
                return [first] + args
            else:
                # If it's a terminal, just return it
                return [first]

        def pre_recursive(node_names):
            first = node_names.pop(0)
            return reorder_node_names_recursive(first, node_names)

        return pre_recursive(build_tree(node_names))

    def convert_gp_tree_to_node_names(self, gp_tree):
        level, _ = mark_node_levels_recursive(gp_tree, original_primitive=True)
        reordered_list = list(map(lambda x: x[0], sorted(level, key=lambda x: x[1])))
        return reordered_list


def get_max_arity(pset):
    """
    Extract the maximum arity of functions from the DEAP PrimitiveSet.

    :param pset: DEAP PrimitiveSet object containing functions and terminals.
    :return: Maximum arity of functions in the PrimitiveSet.
    """
    max_arity = 0

    # Iterate over functions in the PrimitiveSet
    for func in pset.primitives[object]:
        if hasattr(func, "arity"):
            max_arity = max(max_arity, func.arity)

    return max_arity


def calculate_terminals_needed(num_functions, pset):
    """
    Calculate the number of terminals needed to ensure a complete tree in GEP.

    :param num_functions: Number of primitive functions.
    :param max_arity: Maximum arity of the functions.
    :return: Number of terminals required.
    """
    max_arity = get_max_arity(pset)
    if num_functions < 0 or max_arity <= 0:
        raise ValueError(
            "Number of functions must be non-negative and maximum arity must be positive."
        )

    # Calculate the number of terminals required
    num_terminals = 1 + (num_functions * (max_arity - 1))

    return num_functions + num_terminals


class NeuralSemanticLibrary(nn.Module):
    def __init__(
        self,
        input_size=10,
        hidden_size=16,
        output_size=16,
        num_layers=1,
        dropout=0.0,
        output_primitive_length=3,
        pset=None,  # Add pset parameter
        batch_norm=True,  # Add batch_norm parameter
        residual=True,  # Add residual connection parameter
        padding_idx=0,  # Padding index for embedding
    ):
        super(NeuralSemanticLibrary, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_primitive_length = output_primitive_length
        self.batch_norm = batch_norm  # Store batch_norm flag
        self.residual = residual  # Store residual connection flag

        max_nodes = calculate_terminals_needed(output_primitive_length, pset)
        self.output_sequence_length = max_nodes

        self.pset_utils = (
            PrimitiveSetUtils(pset) if pset else None
        )  # Initialize PrimitiveSetUtils
        self.num_symbols = (
            len(self.pset_utils.node_name_to_index) + 1 if self.pset_utils else 0
        )  # Dynamically determine number of symbols (+1 for padding)
        self.num_terminals = (
            len(self.pset_utils.pset.terminals[object]) if self.pset_utils else 0
        )  # Store num_terminals as class attribute

        # Define MLP layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, self.output_sequence_length * output_size))
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()  # Initialize weights

        # Define the embedding layer with padding
        self.embedding = nn.Embedding(
            self.num_symbols, output_size, padding_idx=padding_idx
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure x has shape (batch_size, input_size)
        x = self.mlp[0](x)
        residual = x  # Save input for residual connection
        for layer in self.mlp[1:-1]:
            if isinstance(layer, nn.Linear) and self.residual:
                x = layer(x) + residual  # Apply residual connection
                residual = x  # Update residual for next layer
            else:
                x = layer(x)
        x = self.mlp[-1](x)

        # Reshape the output to (batch_size, output_sequence_length, output_size)
        x = x.view(-1, self.output_sequence_length, self.output_size)

        # Compute dot product with embedding vectors
        embedding_vectors = self.embedding.weight  # Shape: (num_symbols, output_size)
        # Compute dot product between each element of the sequence and the embedding vectors
        dot_products = torch.matmul(
            x, embedding_vectors.T
        )  # Shape: (batch_size, output_sequence_length, num_symbols)
        return dot_products

    def predict(self, semantics):
        number_of_terminals = self.output_sequence_length - self.output_primitive_length

        # Convert semantics to tensor
        semantics_tensor = torch.tensor(semantics, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension

        # Normalize the input semantics using the same scaler as in training
        semantics_tensor = torch.tensor(
            self.scaler.transform(semantics_tensor.numpy()), dtype=torch.float32
        )

        self.mlp.eval()

        # Perform forward pass
        with torch.no_grad():
            output_vector = self.forward(semantics_tensor)

            # Create a mask to ensure that the last four positions are terminals
            mask = torch.full_like(
                output_vector, -float("inf")
            )  # Initialize mask with a large negative value

            # Set the mask values for terminals and non-terminals
            mask[
                :,
                :-number_of_terminals,
            ] = 0
            mask[
                :, -number_of_terminals:, -self.num_terminals :
            ] = 0  # Allow terminals in last positions
            mask[:, :, self.embedding.padding_idx] = -float(
                "inf"
            )  # Mask out padding index

            # Apply the mask to the output vector
            masked_output_vector = torch.softmax(output_vector, dim=2) + mask
            masked_output_vector.squeeze(0).detach().numpy()
            # Get the index of the maximum value along the num_symbols dimension for each time step
            output_indices = (
                torch.argmax(masked_output_vector, dim=2).squeeze(0).numpy()
            )  # Shape: (output_sequence_length,)

        # Get the reverse mapping from indices to node names
        index_to_node_name = self.pset_utils.get_index_to_node_name()

        # Decode indices to node names
        tree_node_names = []
        for i, idx in enumerate(output_indices):
            node_name = index_to_node_name.get(idx, "Unknown")
            tree_node_names.append(node_name)

        return tree_node_names

    def train(
        self,
        train_data,
        batch_size=64,
        epochs=1000,
        lr=0.001,
        val_split=0.2,
        patience=20,
        verbose=False,
    ):
        """
        Train the neural network with optional validation and early stopping.

        :param train_data: List of tuples (gp_tree, target_semantics).
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        :param lr: Learning rate.
        :param val_split: Fraction of training data to use as validation.
        :param patience: Number of epochs to wait for improvement before early stopping.
        """
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Number of epochs for the first restart
            T_mult=2,  # Factor to increase the number of epochs between restarts
            eta_min=1e-6,  # Minimum learning rate
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.embedding.padding_idx)
        # criterion = nn.CrossEntropyLoss()

        # Extract targets and tensors
        tensors = [torch.tensor(d[1], dtype=torch.float32) for d in train_data]
        targets = self.pset_utils.extract_targets(
            train_data, self.output_sequence_length
        )
        targets = [torch.tensor(t, dtype=torch.long) for t in targets]

        # Normalize tensors
        scaler = StandardScaler()
        self.scaler = scaler
        tensors = torch.tensor(
            scaler.fit_transform(torch.stack(tensors).numpy()), dtype=torch.float32
        )

        # Split data into training and validation sets if val_split > 0
        if val_split > 0:
            train_tensors, val_tensors, train_targets, val_targets = train_test_split(
                tensors, targets, test_size=val_split, random_state=0
            )
            train_data = list(zip(train_tensors, train_targets))
            val_data = list(zip(val_tensors, val_targets))
        else:
            train_tensors = tensors
            train_targets = targets
            val_data = None

        # Stack tensors and targets to create a dataset
        stacked_tensors = train_tensors  # Tensors are already stacked and normalized
        stacked_targets = torch.stack(train_targets).view(
            -1, self.output_sequence_length
        )  # Shape: (num_samples, output_sequence_length)

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(stacked_tensors, stacked_targets)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            self.mlp.train()  # Set the model to training mode
            for batch_x, batch_y in dataloader:
                # Ensure batch_x has shape (batch_size, input_size)
                output = self.forward(batch_x)

                # Create a mask to ensure that the last four positions are terminals
                mask = torch.ones_like(
                    output
                )  # Shape: (batch_size, output_sequence_length, num_symbols)

                # Apply the mask to the output vector
                masked_output = output * mask

                # Flatten the targets and apply the mask
                batch_y = batch_y.view(
                    -1
                )  # Shape: (batch_size * output_sequence_length)
                # Flatten the masked output to match the targets
                masked_output = masked_output.view(
                    -1, self.num_symbols
                )  # Shape: (batch_size * output_sequence_length, num_symbols)

                # Calculate loss only for masked positions
                loss = criterion(masked_output, batch_y)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}")

            # Step the learning rate scheduler
            scheduler.step()

            # Validation
            if val_data:
                self.mlp.eval()  # Set the model to evaluation mode
                val_tensors, val_targets = zip(*val_data)
                val_tensors = torch.stack(val_tensors)
                val_targets = torch.stack(val_targets).view(
                    -1, self.output_sequence_length
                )

                val_output = self.forward(val_tensors)
                val_mask = torch.ones_like(val_output)
                val_masked_output = val_output * val_mask

                val_targets = val_targets.view(-1)
                val_masked_output = val_masked_output.view(-1, self.num_symbols)

                val_loss = criterion(val_masked_output, val_targets).item()
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Optionally save the model here
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(
                                "Early stopping due to no improvement in validation loss."
                            )
                        break

    def convert_to_primitive_tree(self, semantics):
        """
        Convert semantics to a GP tree and then to a PrimitiveTree.

        :param semantics: List of node names representing the GP tree.
        :return: PrimitiveTree constructed from the generated GP tree.
        """
        node_names = self.predict(semantics)
        gp_tree = self.pset_utils.convert_node_names_to_gp_tree(node_names)
        return PrimitiveTree(gp_tree)


def generate_synthetic_data(pset, num_samples=100, min_depth=2, max_depth=2):
    """
    Generate synthetic training data with random GP trees and their target semantics.

    :param pset: DEAP PrimitiveSet.
    :param num_samples: Number of samples to generate.
    :return: List of tuples (gp_tree, target_semantics).
    """
    # Create a random GP tree using DEAP's ramped half-and-half method
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    data = []
    for _ in range(num_samples):
        # Generate a random GP tree
        gp_tree = create_random_gp_tree(pset, min_depth=min_depth, max_depth=max_depth)

        # Generate synthetic semantics (e.g., applying the GP tree to a synthetic dataset)
        target_semantics = generate_target_semantics(gp_tree, pset)

        data.append((gp_tree, target_semantics))

    # sorting
    # all_semantics = np.argsort(np.median([s for _, s in data], axis=0))
    # for i, (gp_tree, target_semantics) in enumerate(data):
    #     data[i] = (gp_tree, target_semantics[all_semantics])

    return data


def create_random_gp_tree(pset, min_depth=2, max_depth=2):
    toolbox = base.Toolbox()
    # Initialize ramped half-and-half method
    toolbox.register(
        "individual",
        genHalfAndHalf,
        pset=pset,
        min_=min_depth,
        max_=max_depth,
    )

    return toolbox.individual()


def generate_target_semantics(gp_tree, pset):
    # Compile the GP tree into a callable function
    func = gp.compile(PrimitiveTree(gp_tree), pset)

    # Apply the compiled function to each row of the dataset
    target_semantics = func(*data.T)

    return target_semantics


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def sqrt(x):
    return np.sqrt(np.abs(x))


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def multiply(x, y):
    return x * y


def filter_train_data_by_node_count(train_data, max_nodes=7):
    """
    Filter out GP trees from train_data with more than a specified number of nodes.

    :param train_data: List of tuples (gp_tree, target_semantics), where gp_tree is a DEAP Individual.
    :param max_nodes: Maximum number of nodes allowed in the GP trees.
    :return: List of tuples (gp_tree, target_semantics) with GP trees having nodes count <= max_nodes.
    """
    filtered_train_data = [
        (
            tree,
            semantics / np.linalg.norm(semantics)
            if np.linalg.norm(semantics) > 0
            else semantics,
        )
        for tree, semantics in train_data
        if len(tree) <= max_nodes
    ]
    return filtered_train_data


if __name__ == "__main__":
    # np.random.seed(0)

    # ['subtract', 'add', 'ARG0', 'ARG1', 'subtract', 'ARG0', 'ARG1']
    # Example usage
    data = load_diabetes().data[:30]

    # Define DEAP PrimitiveSet
    pset = PrimitiveSet(
        "MAIN", data.shape[1]
    )  # Example, replace with your specific pset
    pset.addPrimitive(add, 2)
    pset.addPrimitive(subtract, 2)
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)
    pset.addPrimitive(sqrt, 1)
    pset.addPrimitive(multiply, 2)

    utils = PrimitiveSetUtils(pset)
    generated_tree = PrimitiveTree(
        utils.convert_node_names_to_gp_tree(
            ["subtract", "add", "subtract", "ARG0", "ARG1", "ARG0", "ARG1"]
        )
    )
    utils.convert_gp_tree_to_node_names(generated_tree)
    print([node.name for node in generated_tree])
    print(str(generated_tree))

    # Generate synthetic training data
    fix_depth = 2
    train_data = generate_synthetic_data(
        pset, num_samples=5000, min_depth=fix_depth, max_depth=fix_depth
    )

    # Filter training data by node count
    train_data = filter_train_data_by_node_count(train_data)

    # Split data into training and test sets
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=0)

    # Initialize the NeuralSemanticLibrary model
    nl = NeuralSemanticLibrary(
        data.shape[0],
        128,
        128,
        dropout=0.2,
        num_layers=3,
        pset=pset,
    )

    # Train the neural network
    nl.train(train_data, epochs=1000, lr=0.01, val_split=0.2, verbose=True)
    for tid in range(0, 5):
        print(f"Predicted Tree: {str(nl.convert_to_primitive_tree(test_data[tid][1]))}")
        print(f"Original Tree:  {str(PrimitiveTree(test_data[tid][0]))}")
