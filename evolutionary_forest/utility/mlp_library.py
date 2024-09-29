from collections import Counter

import deap.base as base
import deap.creator as creator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deap import gp
from deap.gp import PrimitiveTree, Primitive
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from evolutionary_forest.component.crossover.kan import KANLinear
from evolutionary_forest.model.causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)
from evolutionary_forest.probability_gp import genHalfAndHalf
from evolutionary_forest.utility.mlp_tools.info_nce import InfoNCELoss
from evolutionary_forest.utility.mlp_tools.mlp_utils import (
    get_max_length_excluding_padding,
)
from evolutionary_forest.utility.normalization_tool import normalize_vector
from evolutionary_forest.utility.retrieve_nn.quick_retrive import (
    retrieve_nearest_y,
    retrieve_nearest_y_skip_self,
)
from evolutionary_forest.utility.tree_parsing import mark_node_levels_recursive
from evolutionary_forest.utility.tree_utils.list_to_tree import (
    convert_node_list_to_tree,
    sort_son,
    convert_tree_to_node_list,
)
from evolutionary_forest.utils import reset_random


class IndependentLinearLayers(nn.Module):
    def __init__(self, input_dim, output_dim, num_positions):
        super(IndependentLinearLayers, self).__init__()
        self.num_positions = num_positions
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(num_positions)]
        )

    def forward(self, x):
        # x is expected to have shape [batch_size, num_positions, input_dim]
        outputs = []
        for i in range(x.shape[1]):
            # Apply the i-th linear layer to the i-th position
            outputs.append(self.linears[i](x[:, i, :]))

        # Stack the outputs along the position dimension
        return torch.stack(outputs, dim=1)


def compute_accuracy(val_output, val_targets, padding_idx):
    """
    Compute the accuracy of predictions, ignoring the padding index.

    :param val_output: Tensor of shape (batch_size, seq_len, num_symbols) containing the model outputs.
    :param val_targets: Tensor of shape (batch_size, seq_len) containing the true target values.
    :param padding_idx: Index used for padding in the target tensor.
    :return: Accuracy as a float.
    """
    # Flatten the tensors
    val_targets = val_targets.view(-1)
    val_output = val_output.view(-1, val_output.size(-1))

    # Get the predicted values by taking the argmax over the last dimension
    val_predictions = torch.argmax(val_output, dim=-1)

    # Create a mask that ignores the padding index
    mask = val_targets != padding_idx

    # Calculate the number of correct predictions
    correct_predictions = (val_predictions == val_targets) & mask
    correct_count = correct_predictions.sum().item()

    # Calculate the total number of valid (non-padding) targets
    valid_count = mask.sum().item()

    # Compute the accuracy
    accuracy = correct_count / valid_count if valid_count > 0 else 0.0

    return accuracy


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = d_model // 2

    def forward(self, x):
        seq_len = x.size(1)
        freqs = torch.arange(self.dim, dtype=torch.float32, device=x.device)
        freqs = freqs / self.dim
        freqs = 1.0 / (10000**freqs)
        angles = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        angles = torch.einsum("i,j->ij", angles, freqs)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        encoding = encoding.unsqueeze(0).expand_as(x)
        return x * encoding


def plot_similarity_matrices(
    cosine_similarity_features,
    cosine_similarity_target_features,
    title_features="Cosine Similarity (Features)",
    title_target="Cosine Similarity (Target Features)",
):
    """
    Visualize and compare cosine similarity matrices side by side.

    :param cosine_similarity_features: Cosine similarity matrix computed from features.
    :param cosine_similarity_target_features: Cosine similarity matrix computed from target_features.
    :param title_features: Title for the features similarity matrix plot.
    :param title_target: Title for the target features similarity matrix plot.
    """
    # Convert tensors to numpy arrays for plotting
    cosine_similarity_features = cosine_similarity_features.detach().cpu().numpy()
    cosine_similarity_target_features = (
        cosine_similarity_target_features.detach().cpu().numpy()
    )

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Plot the cosine similarity matrix for features
    sns.heatmap(
        cosine_similarity_features, ax=axes[0], cmap="viridis", cbar=True, annot=False
    )
    axes[0].set_title(title_features)
    axes[0].set_xlabel("Batch Index")
    axes[0].set_ylabel("Batch Index")

    # Plot the cosine similarity matrix for target features
    sns.heatmap(
        cosine_similarity_target_features,
        ax=axes[1],
        cmap="viridis",
        cbar=True,
        annot=False,
    )
    axes[1].set_title(title_target)
    axes[1].set_xlabel("Batch Index")
    axes[1].set_ylabel("Batch Index")

    plt.show()


class DataInputLayer(nn.Module):
    def __init__(self, b, num_weights):
        super(DataInputLayer, self).__init__()
        # Create a linear layer with multiple sets of weights (num_weights) of size b
        self.linear = nn.Linear(b, num_weights)

    def forward(self, x):
        # x should have shape [batch_size, a, b]
        # Apply the linear layer to each line of a (along the last dimension)
        x = self.linear(x)  # Shape: [batch_size, a, num_weights]

        # Apply max pooling along the dimension corresponding to a
        x, _ = torch.max(x, dim=1)  # Shape: [batch_size, num_weights]

        return x


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
            name: idx + 2 for idx, name in enumerate(node_names)
        }  # Start indexing from 1

    def get_index_to_node_name(self):
        """
        Create a reverse mapping from indices to node names based on the PrimitiveSet.

        :return: Dictionary mapping indices to node names.
        """
        index_to_node_name = {
            1: "<start>",
            0: "<end>",
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

    def extract_targets(self, train_data, max_length, sort_gp_tree=False):
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
            if sort_gp_tree:
                tree = PrimitiveTree(
                    convert_tree_to_node_list(
                        sort_son(convert_node_list_to_tree(tree, 0)[1])
                    )
                )
            tree = self.convert_gp_tree_to_node_names(tree)
            # self.convert_node_names_to_gp_tree([node.name for node in tree])
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

        # def reorder_node_names_recursive(first, node_names):
        #     """
        #     Reorder a list of node names such that each function is followed by its arguments using recursion.
        #
        #     :param node_names: List of node names corresponding to functions and terminals.
        #     :return: Reordered list of node names.
        #     """
        #     # If it's a function (Primitive), it will have arguments
        #     if isinstance(first, Primitive):
        #         args = []
        #         root_args = []
        #         for ix in range(first.arity):
        #             a = node_names.pop(0)
        #             # print(first.name, a.name)
        #             root_args.append(a)
        #
        #         for ix, a in enumerate(root_args):
        #             nodes = reorder_node_names_recursive(a, node_names)
        #             # print(first.name, [n.name for n in nodes])
        #             args.extend(nodes)
        #
        #         # Return the function followed by its arguments
        #         return [first] + args
        #     else:
        #         # If it's a terminal, just return it
        #         return [first]
        #
        # def pre_recursive(node_names):
        #     first = node_names.pop(0)
        #     return reorder_node_names_recursive(first, node_names)

        def reorder_node_names(node_names):
            """
            Reorder a list of node names such that each function is followed by its arguments using recursion.

            :param node_names: List of node names corresponding to functions and terminals.
            :return: Reordered list of node names.
            """
            if not node_names:
                return []

            first = node_names.pop(0)

            # If it's a function (Primitive), it will have arguments
            if isinstance(first, Primitive):
                args = []
                root_args = []
                for _ in range(first.arity):
                    a = node_names.pop(0)
                    root_args.append(a)

                for a in root_args:
                    nodes = reorder_node_names([a] + node_names)
                    args.extend(nodes)

                # Return the function followed by its arguments
                return [first] + args
            else:
                # If it's a terminal, just return it
                return [first]

        # return pre_recursive(build_tree(node_names))
        return reorder_node_names(build_tree(node_names))
        # [x.name for x in build_tree(node_names)]

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


info_nce_loss = InfoNCELoss(temperature=0.1)


class NeuralSemanticLibrary(nn.Module):
    def __init__(
        self,
        input_size=10,
        hidden_size=64,
        output_size=64,
        num_layers=1,
        dropout=0.1,
        output_primitive_length=3,
        pset=None,  # Add pset parameter
        batch_norm=True,  # Add batch_norm parameter
        residual=True,  # Add residual connection parameter
        padding_idx=0,  # Padding index for embedding
        num_heads=4,  # Add number of attention heads
        transformer_layers=1,  # Add number of transformer layers
        use_transformer=True,  # Flag to enable or disable transformer layer
        contrastive_loss_in_val=True,  # Add flag to enable contrastive loss in validation
        flatten_before_similarity=True,  # Contrastive loss is based on flatten embedding
        use_decoder_transformer="encoder-decoder",  # Flag to enable or disable decoder transformer
        contrastive_learning_stage="RAG",
        selective_retrain=True,
        retrieval_augmented_generation=True,
        causal_encoding=False,
        double_query=True,
        batch_sampling=1,
        augmented_k=1,
        numerical_token=False,
        independent_linear_layers=False,
        kd_tree_reconstruct=True,
        positional_embedding_over_kan=False,
        contrastive_margin=0,
        use_kan=False,
        prediction_mode="greedy",
        feature_fusion_strategy="concat~1",
        kv_cache_decoder=True,
        retrieval_data_augmentation=False,
        simple_data_augmentation=True,
        **params,
    ):
        super(NeuralSemanticLibrary, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_primitive_length = output_primitive_length
        self.batch_norm = batch_norm  # Store batch_norm flag
        self.residual = residual  # Store residual connection flag
        self.use_transformer = use_transformer  # Store use_transformer flag
        self.flatten_before_similarity = flatten_before_similarity  # Store flatten flag
        self.use_decoder_transformer = use_decoder_transformer
        self.use_kan = use_kan
        self.feature_fusion_strategy = feature_fusion_strategy

        max_nodes = calculate_terminals_needed(output_primitive_length, pset)
        self.output_sequence_length = max_nodes

        self.pset_utils = (
            PrimitiveSetUtils(pset) if pset else None
        )  # Initialize PrimitiveSetUtils
        self.num_symbols = (
            len(self.pset_utils.node_name_to_index) + 2 if self.pset_utils else 0
        )  # Dynamically determine number of symbols (+2 for padding and start tokens)
        self.num_terminals = (
            len(self.pset_utils.pset.terminals[object]) if self.pset_utils else 0
        )  # Store num_terminals as class attribute

        nn_sequential = self._create_layers(
            input_size, hidden_size, output_size, num_layers, dropout, kan=self.use_kan
        )
        self.mlp = nn_sequential
        # if self.use_kan:
        #     self.kan = self._create_layers(
        #         input_size, hidden_size, output_size, num_layers, dropout, kan=True
        #     )
        self._initialize_weights()  # Initialize weights

        if transformer_layers == 0:
            self.use_transformer = False

        # CNN Layer definition
        # if self.use_cnn:
        #     cnn_kernel_size = (3,)  # Kernel size for CNN layer
        #     cnn_stride = (1,)  # Stride for CNN layer
        #     cnn_padding = (1,)  # Padding for CNN layer
        #     self.cnn = nn.Conv1d(
        #         in_channels=1,
        #         out_channels=8,
        #         kernel_size=cnn_kernel_size,
        #         stride=cnn_stride,
        #         padding=cnn_padding,
        #     )

        if self.use_transformer:
            # Transformer Encoder for nearest `y`
            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=output_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer, num_layers=transformer_layers
            )

            # Transformer Decoder for combined `x` and encoded nearest `y`
            self.kv_cache_decoder = kv_cache_decoder
            cached_decoder = self.kv_cache_decoder

            if cached_decoder:
                transformer_decoder_layer = CausalTransformerDecoderLayer(
                    d_model=output_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size,
                    dropout=dropout,
                    activation="gelu",
                )
                self.transformer_decoder = CausalTransformerDecoder(
                    transformer_decoder_layer, num_layers=transformer_layers
                )
            else:
                transformer_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=output_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size,
                    dropout=dropout,
                    activation="gelu",
                )
                self.transformer_decoder = nn.TransformerDecoder(
                    transformer_decoder_layer, num_layers=transformer_layers
                )

        # Define a shared embedding layer for all positions
        self.embedding = nn.Embedding(
            self.num_symbols, output_size, padding_idx=padding_idx
        )
        self.independent_linear_layers = independent_linear_layers
        if self.independent_linear_layers:
            self.output_linear = IndependentLinearLayers(
                output_size, self.num_symbols, self.output_sequence_length
            )
        else:
            self.output_linear = nn.Linear(output_size, self.num_symbols)

        self.start_token_index = 1
        self.contrastive_loss_in_val = contrastive_loss_in_val
        self.contrastive_learning_stage = contrastive_learning_stage
        self.selective_retrain = selective_retrain

        # Positional embedding initialization
        self.positional_embedding = nn.Embedding(
            self.output_sequence_length * augmented_k, output_size
        )
        self.retrieval_augmented_generation = retrieval_augmented_generation
        self.causal_encoding = causal_encoding
        self.double_query = double_query
        self.batch_sampling = batch_sampling
        self.augmented_k = augmented_k
        self.numerical_token = numerical_token
        self.kd_tree_reconstruct = kd_tree_reconstruct
        self.positional_embedding_over_kan = positional_embedding_over_kan
        self.contrastive_margin = contrastive_margin
        self.trained = False
        self.prediction_mode = prediction_mode
        self.retrieval_data_augmentation = retrieval_data_augmentation
        self.simple_data_augmentation = simple_data_augmentation

    def _create_layers(
        self, input_size, hidden_size, output_size, num_layers, dropout, kan=False
    ):
        # Define MLP layers
        layers = []
        for _ in range(num_layers):
            if kan:
                layers.append(KANLinear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if not kan:
                layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        if self.feature_fusion_strategy.startswith("concat~"):
            head = int(self.feature_fusion_strategy.split("~")[1])
            layers.append(nn.Linear(hidden_size, head * output_size))
        else:
            layers.append(
                nn.Linear(hidden_size, self.output_sequence_length * output_size)
            )
        nn_sequential = nn.Sequential(*layers)
        return nn_sequential

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, batch_y=None, nearest_x=None, nearest_y=None):
        original_x = x
        x = self._process_mlp(original_x)
        if (
            self.contrastive_learning_stage == "MLP"
            and self.contrastive_loss_weight > 0
        ):
            nearest_x = self._process_mlp(-original_x)
        elif (
            self.contrastive_learning_stage == "RAG"
            and nearest_x is not None
            and self.contrastive_loss_weight > 0
        ):
            nearest_x = self._process_mlp(nearest_x)
        elif (
            self.contrastive_learning_stage == "MLP-RAG"
            and nearest_x is not None
            and self.contrastive_loss_weight > 0
        ):
            inverse_x = self._process_mlp(-original_x)
            nearest_x = self._process_mlp(nearest_x)
            nearest_x = (inverse_x, nearest_x)
        else:
            nearest_x = None
        # if self.use_kan:
        #     x += self._process_mlp(original_x, self.kan)
        # if self.use_cnn:
        #     x += self._process_cnn(x)
        dot_products, contrastive_context = self._forward_traditional_transformer(
            x, batch_y, nearest_y
        )
        return dot_products, nearest_x, contrastive_context

    def _process_cnn(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for CNN (batch, channels, seq_len)
        x = self.cnn(x)  # Apply CNN layer
        x = x.squeeze(1)  # Remove the channel dimension after CNN
        return x

    def _forward_traditional_transformer(self, x, batch_y, nearest_y):
        x_raw = x = self._reshape_output(x)
        if self.positional_embedding_over_kan:
            x = self._add_positional_embedding(x)

        nearest_y_encoded_by_transformer = None
        padding_mask = None
        if self.use_transformer and nearest_y is not None:
            if self.retrieval_augmented_generation:
                if self.use_decoder_transformer == "decoder":
                    nearest_y_embedded = self.embedding(nearest_y)
                    nearest_y_embedded = self._add_positional_embedding(
                        nearest_y_embedded
                    )
                    combined_features = (x, nearest_y_embedded)
                else:
                    nearest_y_encoded_by_transformer = self._encode_nearest_y(
                        nearest_y, x
                    )
                    views = self.augmented_k
                    if self.numerical_token:
                        views += 1
                    reshaped_tensor = nearest_y_encoded_by_transformer.view(
                        -1, self.output_sequence_length, views, self.output_size
                    )
                    pooled_tensor = torch.mean(reshaped_tensor, dim=2)
                    combined_features = self._combine_features(x, pooled_tensor)

                    # if self.feature_fusion_strategy.startswith("concat"):
                    #     padding_mask = torch.ones(
                    #         combined_features.shape[:2], dtype=torch.bool
                    #     )
                    #     padding_start = x.shape[1]
                    #     padding_mask[:, padding_start:] = nearest_y != 0

            else:
                combined_features = x

            if self.use_decoder_transformer is not None:
                output = self._decode(
                    combined_features,
                    batch_y,
                    self.use_decoder_transformer,
                    padding_mask,
                )
            else:
                output = combined_features
        else:
            output = x

        dot_products = self.output_linear(output)

        if self.contrastive_learning_stage == "Decoder":
            contrastive_context = output
        elif self.contrastive_learning_stage == "Encoder":
            contrastive_context = combined_features
        elif self.contrastive_learning_stage == "RAG":
            contrastive_context = nearest_y_encoded_by_transformer
        elif self.contrastive_learning_stage == "MLP":
            contrastive_context = x_raw
        elif self.contrastive_learning_stage == "MLP-RAG":
            contrastive_context = (x_raw, nearest_y_encoded_by_transformer)
        else:
            raise Exception

        return (dot_products, contrastive_context)

    def _process_mlp(self, x, mlp=None):
        """Pass the input through MLP layers with optional residual connections."""
        if mlp is None:
            mlp = self.mlp
        x = mlp[0](x)
        residual = x
        for layer in mlp[1:-1]:
            if isinstance(layer, (nn.Linear, KANLinear)) and self.residual:
                x = layer(x) + residual
                residual = x
            else:
                x = layer(x)
        x = mlp[-1](x)
        return x

    def _reshape_output(self, x):
        """Reshape the output to (batch_size, output_sequence_length, output_size)."""
        if self.feature_fusion_strategy.startswith("concat~"):
            head = int(self.feature_fusion_strategy.split("~")[1])
            return x.view(-1, head, self.output_size)
        else:
            return x.view(-1, self.output_sequence_length, self.output_size)

    def _add_positional_embedding(self, x):
        """Add positional embeddings to the input sequence."""
        batch_size, seq_length, _ = x.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        positional_embeds = self.positional_embedding(position_ids)
        return x + positional_embeds
        # return x

    def _add_positional_embedding_index(self, x, position):
        """Add positional embeddings to the input sequence."""
        position_ids = torch.full(
            (x.size(0), x.size(1)), position, dtype=torch.long, device=x.device
        )
        positional_embeds = self.positional_embedding(position_ids)
        return x + positional_embeds
        # return x

    def _encode_nearest_y(self, nearest_y, x):
        """Encode nearest `y` using the embedding layer and Transformer Encoder."""
        nearest_y_embedded = self.embedding(nearest_y)
        nearest_y_embedded = self._add_positional_embedding(nearest_y_embedded)

        # Generate padding mask where True indicates the padding token (0 index)
        padding_mask = nearest_y == 0

        if self.numerical_token:
            nearest_y_embedded = torch.cat([nearest_y_embedded, x], dim=1)
        if self.causal_encoding:
            # Generate the causal mask using generate_square_subsequent_mask
            seq_len = nearest_y_embedded.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
                nearest_y.device
            )
            # Generate causal mask
            nearest_y_encoded = self.transformer_encoder(
                nearest_y_embedded.permute(1, 0, 2),
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
            ).permute(1, 0, 2)
        else:
            nearest_y_encoded = self.transformer_encoder(
                nearest_y_embedded.permute(1, 0, 2), src_key_padding_mask=padding_mask
            ).permute(1, 0, 2)
            # np.array(nearest_y_encoded[0].detach())
            # padding_mask[0]
        return nearest_y_encoded

    def _combine_features(self, x, nearest_y_encoded):
        """Combine the MLP output `x` and encoded nearest `y`."""
        method = self.feature_fusion_strategy
        if method.startswith("concat"):
            # Concatenate along the last dimension (feature dimension)
            return torch.cat((x, nearest_y_encoded), dim=1)
        elif method == "add":
            return x + nearest_y_encoded
        else:
            raise ValueError("Unsupported combination method: use 'concat' or 'add'.")

    def _decode(self, combined_features, batch_y, decoder_mode, padding_mask):
        """Decode using Transformer Decoder."""
        if batch_y is not None:
            if decoder_mode == "decoder":
                return self._decode_training_decoder_only(combined_features, batch_y)
            else:
                return self._decode_training(combined_features, batch_y, padding_mask)
        else:
            if decoder_mode == "encoder-decoder":
                return self._decode_with_encoder_decoder(
                    combined_features, padding_mask
                )
            elif decoder_mode == "decoder":
                return self._decode_with_decoder_only(combined_features)
            else:
                raise ValueError(
                    "Unsupported decoder mode: use 'encoder-decoder' or 'decoder'."
                )

    # def sampling_batch(self, combined_features):
    #     batch_size = combined_features.size(0)
    #     best_results = None
    #     best_likelihoods = torch.full(
    #         (batch_size,), -float("inf"), device=combined_features.device
    #     )
    #     for _ in range(self.batch_sampling):
    #         results, likelihoods = self._decode_with_encoder_decoder(
    #             combined_features, use_sampling=True
    #         )
    #
    #         # Update best_results and best_likelihoods for each element in the batch
    #         update_mask = likelihoods > best_likelihoods
    #         best_likelihoods = torch.where(
    #             update_mask, likelihoods, best_likelihoods
    #         )
    #         if best_results is None:
    #             best_results = (
    #                 results.clone()
    #             )  # Initialize with the first set of results
    #         else:
    #             best_results[update_mask] = results[
    #                 update_mask
    #             ]  # Update only where likelihood improved
    #     return best_results

    def _decode_training(self, combined_features, batch_y, memory_key_padding_mask):
        """Decode in training mode with teacher forcing."""
        start_tokens = torch.full(
            (batch_y.size(0), 1),
            self.start_token_index,
            dtype=torch.long,
            device=combined_features.device,
        )
        length = get_max_length_excluding_padding(batch_y)
        batch_y = torch.cat([start_tokens, batch_y[:, : length - 1]], dim=1)

        tgt = self.embedding(batch_y)
        tgt = self._add_positional_embedding(tgt)
        tgt = tgt.permute(1, 0, 2)
        memory = combined_features.permute(1, 0, 2)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )
        if isinstance(self.transformer_decoder, CausalTransformerDecoder):
            # internal tgt
            output = self.transformer_decoder(tgt, memory, memory_key_padding_mask)
        else:
            output = self.transformer_decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return output.permute(1, 0, 2)

    def _decode_training_decoder_only(self, combined_features, batch_y):
        """
        Decode in training mode with teacher forcing in a decoder-only architecture.

        :param batch_y: The ground truth sequence used for teacher forcing.
        :return: The output sequence predicted by the decoder.
        """
        augmented_features, numerical_features = combined_features
        start_tokens = torch.full(
            (batch_y.size(0), 1),
            self.start_token_index,
            dtype=torch.long,
            device=batch_y.device,
        )
        batch_y = torch.cat([start_tokens, batch_y[:, :-1]], dim=1)

        tgt = self.embedding(batch_y)
        tgt = self._add_positional_embedding(tgt)
        tgt = torch.cat([augmented_features, tgt], dim=1)
        tgt = tgt.permute(1, 0, 2)  # Prepare for transformer decoder input
        numerical_features = numerical_features.permute(1, 0, 2)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )

        output = self.transformer_decoder(
            tgt,
            memory=numerical_features,
            tgt_mask=tgt_mask,
        )

        return output.permute(1, 0, 2)[:, -self.output_sequence_length :]

    def _decode_with_encoder_decoder(
        self, combined_features, memory_key_padding_mask, use_sampling=False
    ):
        batch_size = combined_features.size(0)
        tgt = self.embedding(
            torch.full(
                (batch_size, 1),
                self.start_token_index,
                dtype=torch.long,
                device=combined_features.device,
            )
        )
        tgt = self._add_positional_embedding_index(
            tgt, position=0
        )  # Add positional embedding for the first token
        tgt = tgt.permute(1, 0, 2)
        memory = combined_features.permute(1, 0, 2)

        # Initialize log likelihood
        log_likelihood = torch.zeros(batch_size, device=combined_features.device)

        cache = None
        for i in range(1, self.output_sequence_length + 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
                tgt.device
            )
            if isinstance(self.transformer_decoder, CausalTransformerDecoder):
                # internal tgt
                output, cache = self.transformer_decoder(tgt, memory, cache=cache)
            else:
                output = self.transformer_decoder(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            output = output.permute(1, 0, 2)

            if self.independent_linear_layers:
                last_token_logits = self.output_linear.linears[i - 1](output[:, -1:])
            else:
                last_token_logits = self.output_linear(output[:, -1:])

            if use_sampling:
                # Sample from the softmax probabilities
                probs = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
                selected_log_probs = torch.log(probs.squeeze(1).gather(1, next_token))
            else:
                # Greedy decoding: take the token with the highest probability
                next_token = last_token_logits.argmax(dim=-1)
                selected_log_probs = (
                    F.log_softmax(last_token_logits, dim=-1)
                    .gather(2, next_token.unsqueeze(-1))
                    .squeeze(-1)
                )

            # Accumulate the log likelihood
            log_likelihood += selected_log_probs.squeeze(1)

            if i == self.output_sequence_length:
                break

            next_token_embedding = self.embedding(next_token)
            next_token_embedding = self._add_positional_embedding_index(
                next_token_embedding, position=i
            )  # Add positional embedding for the new token
            next_token_embedding = next_token_embedding.permute(1, 0, 2)
            tgt = torch.cat([tgt, next_token_embedding], dim=0)

        likelihood = torch.exp(
            log_likelihood
        )  # Convert log likelihood to actual likelihood

        return output

    def _decode_with_decoder_only(self, combined_features):
        """
        Decode using the decoder-only architecture.

        :param combined_features: The prompt used as the initial sequence for decoding.
        :return: Generated output sequence.
        """
        combined_features, numerical_features = combined_features
        tgt = self.embedding(
            torch.full(
                (combined_features.size(0), 1),
                self.start_token_index,
                dtype=torch.long,
                device=combined_features.device,
            )
        )
        tgt = torch.cat([combined_features, tgt], dim=1)
        tgt = tgt.permute(1, 0, 2)  # Prepare for transformer decoder input
        numerical_features = numerical_features.permute(1, 0, 2)

        length = self.output_sequence_length * 2
        for i in range(combined_features.size(1), length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
                tgt.device
            )
            output = self.transformer_decoder(
                tgt, numerical_features, tgt_mask=tgt_mask
            )
            output = output.permute(1, 0, 2)
            last_token_logits = self.output_linear(output[:, -1:])
            next_token = last_token_logits.argmax(dim=-1)

            if i == length - 1:
                break

            next_token_embedding = self.embedding(next_token)
            next_token_embedding = self._add_positional_embedding_index(
                next_token_embedding, position=i
            )
            next_token_embedding = next_token_embedding.permute(1, 0, 2)
            tgt = torch.cat([tgt, next_token_embedding], dim=0)

        return output[:, -self.output_sequence_length :, :]

    def _compute_dot_product(self, output):
        """Compute the dot product with shared embedding vectors."""
        embedding_vectors = self.embedding.weight
        return torch.matmul(output, embedding_vectors.T)

    def contrastive_loss(self, features, target_features):
        """
        Compute contrastive loss to enforce that pairwise cosine distances between instances using features
        should correlate with pairwise cosine distances using target_features.

        :param features: The learned feature vectors with shape [batch_size, sequence_length, feature_dim].
        :param target_features: The original input features with shape [batch_size, input_dim].
        :param margin: Margin for contrastive loss.
        :return: Calculated contrastive loss.
        """
        # Flatten or mean features across sequence length before calculating similarity
        if self.flatten_before_similarity:
            features = features.reshape(features.size(0), -1)  # Flatten features
        else:
            features = features.mean(dim=1)  # Shape: [batch_size, feature_dim]

        # Normalize the feature vectors after mean or flatten
        features = F.normalize(
            features, p=2, dim=-1
        )  # Normalize across the feature_dim
        target_features = F.normalize(
            target_features, p=2, dim=-1
        )  # Normalize across the input_dim

        # Compute pairwise cosine similarity matrices
        cosine_similarity_features = torch.matmul(
            features, features.T
        )  # Shape: [batch_size, batch_size]
        cosine_similarity_target_features = torch.abs(
            torch.matmul(target_features, target_features.T)
        )  # Shape: [batch_size, batch_size]

        if self.contrastive_margin > 0:
            absolute_deviation = torch.abs(
                cosine_similarity_features - cosine_similarity_target_features
            )
            # Focus on very relevant and irrelevant pairs
            filter = (cosine_similarity_target_features <= self.contrastive_margin) | (
                cosine_similarity_target_features >= 1 - self.contrastive_margin
            )
            if torch.sum(filter) == 0:
                return 0
            margin_loss = torch.square(absolute_deviation[filter]).mean()
            return margin_loss
        else:
            loss = F.mse_loss(
                cosine_similarity_features, cosine_similarity_target_features
            )
            return loss

    def train(
        self,
        train_data,
        batch_size=64,
        epochs=1000,
        lr=0.001,
        val_split=0.2,
        patience=20,
        verbose=False,
        loss_weight=0,
        sort_gp_tree=False,
    ):
        self.contrastive_loss_weight = loss_weight

        optimizer, scheduler = self.setup_optimizer_and_scheduler(lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.embedding.padding_idx)

        tensors, targets = self.prepare_data(train_data, sort_gp_tree)
        # reverse each target
        # targets = [torch.flip(t, [0]) for t in targets]

        train_tensors, val_tensors, train_targets, val_targets = self.split_data(
            tensors, targets, val_split
        )
        if self.simple_data_augmentation:
            train_tensors = np.concatenate([train_tensors, -train_tensors], axis=0)
            train_targets = train_targets + train_targets
            val_tensors = np.concatenate([val_tensors, -val_tensors], axis=0)
            val_targets = val_targets + val_targets
        train_tensors = torch.tensor(train_tensors, dtype=torch.float32)

        # Need to check if the training data is enough
        if len(train_tensors) < batch_size:
            self.trained = False
            return
        self.trained = True
        self.batch_size = batch_size

        dataset, dataloader = self.create_dataloader_and_kd_tree(
            train_tensors, train_targets, batch_size
        )
        assert len(dataset) == len(
            train_tensors
        ), f"{len(dataset)} != {len(train_tensors)}"

        val_tensors = torch.tensor(val_tensors, dtype=torch.float32)
        val_data = self.prepare_validation_data(
            val_tensors, val_targets, train_tensors, train_targets
        )
        # train_tensors, val_tensors = self.normalize_data(train_tensors, val_tensors)

        # update kd-tree, not matter training or not
        self.reconstruct_kd_tree(targets, tensors)

        if val_data and self.selective_retrain:
            if self.should_skip_training(val_data, loss_weight, verbose):
                return self.previous_val_loss

        best_val_loss = self.execute_training(
            dataloader,
            criterion,
            optimizer,
            scheduler,
            val_data,
            loss_weight,
            patience,
            epochs,
            verbose,
        )

        self.previous_val_loss = best_val_loss
        return best_val_loss

    def reconstruct_kd_tree(self, targets, tensors):
        if self.kd_tree_reconstruct:
            # whole training data before split
            if self.simple_data_augmentation:
                tensors = np.concatenate([tensors, -tensors], axis=0)
                targets = targets + targets
            (
                augmented_tensors,
                augmented_targets,
            ) = self.augmentation_in_case_of_contrastive_learning(tensors, targets)

            self.whole_tensor = augmented_tensors
            self.whole_target = augmented_targets
            self.data_used_to_train_kd_tree = len(augmented_tensors)
            _, _, kd_tree = retrieve_nearest_y_skip_self(
                augmented_tensors, None, augmented_targets, k=self.augmented_k
            )
            self.kd_tree = kd_tree

    def setup_optimizer_and_scheduler(self, lr):
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        return optimizer, scheduler

    def prepare_data(self, train_data, sort_gp_tree=False):
        tensors = [torch.tensor(d[1], dtype=torch.float32) for d in train_data]
        targets = self.pset_utils.extract_targets(
            train_data, self.output_sequence_length, sort_gp_tree
        )
        targets = [torch.tensor(t, dtype=torch.long) for t in targets]
        tensors = torch.stack(tensors)
        return tensors, targets

    def split_data(self, tensors, targets, val_split):
        train_tensors, val_tensors, train_targets, val_targets = train_test_split(
            tensors.numpy(),
            targets,
            test_size=val_split,
            random_state=0,
            shuffle=True,
        )
        self.train_tensors, self.train_targets = train_tensors, train_targets
        return train_tensors, val_tensors, train_targets, val_targets

    def normalize_data(self, train_tensors, val_tensors):
        scaler = StandardScaler()
        scaler.fit(train_tensors)
        train_tensors = torch.tensor(
            scaler.transform(train_tensors), dtype=torch.float32
        )
        val_tensors = torch.tensor(scaler.transform(val_tensors), dtype=torch.float32)
        self.scaler = scaler
        return train_tensors, val_tensors

    def prepare_validation_data(
        self, val_tensors, val_targets, train_tensors, train_targets
    ):
        (
            augmented_tensors,
            augmented_targets,
        ) = self.augmentation_in_case_of_contrastive_learning(
            train_tensors, train_targets
        )
        nearest_x_val, nearest_y_val = retrieve_nearest_y(
            self.kd_tree,
            augmented_tensors,
            augmented_targets,
            val_tensors,
            k=self.augmented_k,
        )
        return list(zip(val_tensors, val_targets, nearest_x_val, nearest_y_val))

    def should_skip_training(self, val_data, loss_weight, verbose):
        # current_val_loss_backup = self.compute_val_loss(val_data, loss_weight, verbose)
        current_val_loss = self.validation_batch_mode(
            val_data, self.batch_size, loss_weight, verbose
        )
        if verbose:
            print(f"Initial Validation Loss: {current_val_loss}")
        if (
            hasattr(self, "previous_val_loss")
            and current_val_loss < self.previous_val_loss
        ):
            if verbose:
                print("Validation loss has not degraded. Skipping training.")
            return True
        return False

    def create_dataloader_and_kd_tree(self, train_tensors, train_targets, batch_size):
        stacked_tensors = train_tensors
        stacked_targets = torch.stack(train_targets).view(
            -1, self.output_sequence_length
        )

        (
            augmented_tensors,
            augmented_targets,
        ) = self.augmentation_in_case_of_contrastive_learning(
            stacked_tensors, stacked_targets
        )

        nearest_x_train, nearest_y_train, kd_tree = retrieve_nearest_y_skip_self(
            augmented_tensors, stacked_tensors, augmented_targets, k=self.augmented_k
        )
        self.kd_tree = kd_tree
        stacked_nearest_y = torch.stack(
            [torch.tensor(x) for x in nearest_y_train]
        ).view(-1, self.output_sequence_length * self.augmented_k)
        dataset = TensorDataset(
            stacked_tensors,
            stacked_targets,
            torch.tensor(nearest_x_train),
            stacked_nearest_y,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return dataset, dataloader

    def augmentation_in_case_of_contrastive_learning(
        self, stacked_tensors, stacked_targets
    ):
        if self.retrieval_data_augmentation:
            stacked_targets, stacked_tensors = self.inverse_augmentation(
                stacked_targets, stacked_tensors
            )
        return stacked_tensors, stacked_targets

    def inverse_augmentation(self, stacked_targets, stacked_tensors):
        # manually add negative samples
        stacked_tensors = torch.cat([stacked_tensors, -stacked_tensors], dim=0)
        if isinstance(stacked_targets, list):
            stacked_targets = stacked_targets + stacked_targets
        else:
            stacked_targets = torch.cat([stacked_targets, stacked_targets])
        return stacked_targets, stacked_tensors

    def execute_training(
        self,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        val_data,
        loss_weight,
        patience,
        epochs,
        verbose,
    ):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.epoch = epoch
            total_loss = 0
            self.training_mode()
            for batch_x, batch_y, nearest_x, nearest_y in dataloader:
                loss = self.train_single_batch(
                    batch_x,
                    batch_y,
                    nearest_x,
                    nearest_y,
                    criterion,
                    optimizer,
                    loss_weight,
                )
                total_loss += loss
                if self.prediction_mode == "chain-of-thought":
                    nearest_y = self.predict(
                        batch_x, mode="greedy", return_indices=True
                    )
                    loss = self.train_single_batch(
                        batch_x,
                        batch_y,
                        nearest_x,
                        nearest_y,
                        criterion,
                        optimizer,
                        loss_weight,
                    )
                    total_loss += loss

            avg_loss = total_loss / len(dataloader)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}")

            scheduler.step()
            best_val_loss, patience_counter = self.validate_and_early_stop(
                val_data, best_val_loss, patience_counter, loss_weight, verbose
            )
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping due to no improvement in validation loss.")
                break

        return best_val_loss

    def training_mode(self):
        self.mlp.train()
        self.transformer_encoder.train()
        self.transformer_decoder.train()
        self.embedding.train()
        self.output_linear.train()

    def train_single_batch(
        self, batch_x, batch_y, nearest_x, nearest_y, criterion, optimizer, loss_weight
    ):
        # if loss_weight > 0:
        #     batch_x, batch_y, nearest_y = self.batch_augmentation(
        #         batch_x, batch_y, nearest_y
        #     )

        output, retrieval_x, retrieval_y = self.forward(
            batch_x, batch_y=batch_y, nearest_x=nearest_x, nearest_y=nearest_y
        )
        if output.shape[:2] != batch_y.shape:
            max_len = get_max_length_excluding_padding(batch_y)
            batch_y = batch_y[:, :max_len].reshape(-1)
        else:
            batch_y = batch_y.view(-1)
        ce_loss = criterion(output.view(-1, self.num_symbols), batch_y)

        if loss_weight > 0 and retrieval_x is not None and retrieval_y is not None:
            if self.contrastive_learning_stage in ["RAG", "MLP"]:
                mask = self.get_mask(nearest_x)
                contrastive_loss = info_nce_loss(retrieval_x, retrieval_y, mask=mask)
            elif self.contrastive_learning_stage == "MLP-RAG":
                contrastive_loss = self.calculate_contrastive_loss_mlp_rag(
                    retrieval_x, retrieval_y
                )
            else:
                contrastive_loss = self.contrastive_loss(retrieval_y, batch_x)
            loss = ce_loss + loss_weight * contrastive_loss
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_mask(self, nearest_x):
        nearest_x_norm = F.normalize(nearest_x, dim=1)
        if self.contrastive_margin > 0:
            threshold = self.contrastive_margin
        else:
            threshold = 0.99
        mask = torch.abs(torch.matmul(nearest_x_norm, nearest_x_norm.T)) < threshold
        mask = mask.fill_diagonal_(True)
        # (torch.sum(mask == False) > 0).item()
        return mask

    def calculate_contrastive_loss_mlp_rag(self, retrieval_x, retrieval_y):
        inverse_x, retrieval_x = retrieval_x
        inverse_y, retrieval_y = retrieval_y
        contrastive_loss_inverse = info_nce_loss(inverse_x, inverse_y)
        contrastive_loss = info_nce_loss(retrieval_x, retrieval_y)
        contrastive_loss += contrastive_loss_inverse
        return contrastive_loss

    def validation_batch_mode(self, val_data, batch_size, loss_weight, verbose=False):
        num_data = len(val_data)
        final_loss = 0
        total_points = 0

        i = 0
        while i < num_data:
            if i + 2 * batch_size <= num_data:
                batch = val_data[i : i + batch_size]
            else:
                batch = val_data[i:]

            # Calculate loss for the current batch
            batch_loss = self.compute_val_loss(batch, loss_weight, verbose)
            final_loss += batch_loss * len(batch)
            total_points += len(batch)
            i += batch_size

            if len(batch) > batch_size:
                break

        # Calculate the weighted average of the losses
        final_loss /= total_points
        return final_loss

    def validate_and_early_stop(
        self, val_data, best_val_loss, patience_counter, loss_weight, verbose
    ):
        # current_val_loss = self.compute_val_loss(val_data, loss_weight, verbose)
        current_val_loss = self.validation_batch_mode(
            val_data, self.batch_size, loss_weight, verbose
        )
        if verbose:
            print(f"Validation Loss: {current_val_loss}")
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        return best_val_loss, patience_counter

    def compute_val_loss(self, val_data, loss_weight=0.5, verbose=False):
        """
        Compute the validation loss.

        :param val_data: Validation dataset in the form of a list of tuples (tensor, target).
        :param loss_weight: Weight to balance contrastive and ce loss. Set to 0 to ignore contrastive loss.
        :return: Computed validation loss.
        """
        self.eval_mode()
        val_tensors, val_targets, nearest_x_val, nearest_y_val = zip(*val_data)
        nearest_x_val = torch.tensor(nearest_x_val)
        val_tensors = torch.stack(val_tensors)
        val_targets = torch.stack(val_targets).view(-1, self.output_sequence_length)
        nearest_y_val = torch.stack([torch.tensor(x) for x in nearest_y_val]).view(
            -1, self.output_sequence_length * self.augmented_k
        )
        # if loss_weight > 0 and self.contrastive_loss_in_val:
        #     val_tensors, val_targets, nearest_y_val = self.batch_augmentation(
        #         val_tensors, val_targets, nearest_y_val
        #     )

        val_masked_output, retrieval_x, retrieval_y = self.forward(
            val_tensors, nearest_x=nearest_x_val, nearest_y=nearest_y_val
        )

        val_targets = val_targets.view(-1)
        val_masked_output = val_masked_output.view(-1, self.num_symbols)

        val_ce_loss = nn.CrossEntropyLoss(ignore_index=self.embedding.padding_idx)(
            val_masked_output, val_targets
        ).item()

        if (
            loss_weight > 0
            and self.contrastive_loss_in_val
            and retrieval_x is not None
            and retrieval_y is not None
        ):
            # Calculate contrastive loss for validation
            if self.contrastive_learning_stage in ["RAG", "MLP"]:
                mask = self.get_mask(nearest_x_val)
                val_contrastive_loss = info_nce_loss(
                    retrieval_x, retrieval_y, mask=mask
                )
            elif self.contrastive_learning_stage == "MLP-RAG":
                val_contrastive_loss = self.calculate_contrastive_loss_mlp_rag(
                    retrieval_x, retrieval_y
                )
            else:
                val_contrastive_loss = self.contrastive_loss(
                    retrieval_y, val_tensors
                ).item()

            # Combine the losses for validation loss
            val_loss = val_ce_loss + loss_weight * val_contrastive_loss
        else:
            val_loss = val_ce_loss

        if verbose:
            acc = compute_accuracy(
                val_masked_output, val_targets, self.embedding.padding_idx
            )
            print(f"Validation Loss: {val_loss}, Accuracy: {acc}")
        return val_loss

    def batch_augmentation(self, val_tensors, val_targets, nearest_y_val):
        tensor_shape = val_tensors.shape
        target_shape = val_targets.shape
        val_tensors = torch.stack([val_tensors, -1 * val_tensors])
        val_targets = torch.stack([val_targets, val_targets])
        nearest_y_val = torch.stack([nearest_y_val, nearest_y_val])
        val_tensors = val_tensors.view(-1, *tensor_shape[1:])
        val_targets = val_targets.view(-1, *target_shape[1:])
        nearest_y_val = nearest_y_val.view(-1, *target_shape[1:])
        return val_tensors, val_targets, nearest_y_val

    def eval_mode(self):
        self.mlp.eval()  # Set the model to evaluation mode
        # if self.use_kan:
        #     self.kan.eval()
        self.transformer_encoder.eval()
        self.transformer_decoder.eval()
        self.embedding.eval()
        self.output_linear.eval()

    def predict(self, semantics, mode="greedy", return_indices=False):
        nearest_y = self._retrieve_nearest_y_for_prediction(semantics)
        if len(semantics.shape) == 1:
            semantics_tensor = self._prepare_input_tensor(semantics)
        else:
            semantics_tensor = semantics

        output_vector = self._forward_pass(semantics_tensor, nearest_y)
        masked_output_vector = self._apply_mask(output_vector)

        if mode == "probability":
            output_indices, likelihood = self._probability_sampling(
                masked_output_vector
            )
        elif mode == "greedy":
            output_indices, likelihood = self._greedy_selection(masked_output_vector)
        elif mode == "chain-of-thought":
            output_indices, likelihood = self._greedy_selection(masked_output_vector)

            # input the output as input
            output_vector = self._forward_pass(semantics_tensor, output_indices)
            masked_output_vector = self._apply_mask(output_vector)
            output_indices, likelihood = self._greedy_selection(masked_output_vector)
        else:
            raise Exception(f"Invalid mode: {mode}")
        if return_indices:
            return output_indices
        # reverse the output_indices
        # output_indices = output_indices[::-1]
        tree_node_names = self._decode_indices_to_node_names(output_indices.numpy()[0])

        return tree_node_names, likelihood[0]

    def _retrieve_nearest_y_for_prediction(self, semantics):
        if self.kd_tree_reconstruct:
            assert self.data_used_to_train_kd_tree == len(
                self.whole_tensor
            ), f"Reconstructed data size {self.data_used_to_train_kd_tree} != {len(self.whole_tensor)}"
            assert self.data_used_to_train_kd_tree == len(
                self.whole_target
            ), f"Reconstructed data size (target) {self.data_used_to_train_kd_tree} != {len(self.whole_target)}"
        assert self.kd_tree.n == len(
            self.whole_target
        ), f"KD-Tree size {self.kd_tree.n} != {len(self.whole_target)}"
        nearest_x, nearest_y = retrieve_nearest_y(
            self.kd_tree,
            self.whole_tensor,
            self.whole_target,
            semantics.reshape(-1, self.input_size),
            k=self.augmented_k,
        )
        return torch.stack([torch.tensor(x) for x in nearest_y]).view(
            -1, self.output_sequence_length * self.augmented_k
        )

    def _prepare_input_tensor(self, semantics):
        return torch.tensor(semantics, dtype=torch.float32).unsqueeze(0)

    def _forward_pass(self, semantics_tensor, nearest_y):
        self.eval_mode()
        with torch.no_grad():
            output_vector, _, _ = self.forward(semantics_tensor, nearest_y=nearest_y)
        return output_vector

    def _apply_mask(self, output_vector):
        number_of_terminals = self.output_sequence_length - self.output_primitive_length
        mask = torch.full_like(output_vector, -float("inf"))
        mask[:, :-number_of_terminals] = 0
        mask[:, -number_of_terminals:, -self.num_terminals :] = 0
        # mask[:, number_of_terminals:] = 0
        # mask[:, :number_of_terminals, -self.num_terminals :] = 0
        mask[:, :, self.embedding.padding_idx] = -float("inf")
        mask[:, :, self.start_token_index] = -float("inf")
        return output_vector + mask

    def _probability_sampling(self, masked_output_vector):
        valid_indices = []
        likelihood = 1.0
        for i in range(masked_output_vector.size(1)):
            while True:
                probabilities = torch.softmax(masked_output_vector[:, i, :], dim=1)
                sampled_index = torch.multinomial(probabilities, num_samples=1).item()
                if masked_output_vector[0, i, sampled_index] != -float("inf"):
                    valid_indices.append(sampled_index)
                    likelihood *= probabilities[0, sampled_index].item()
                    break
        return np.array(valid_indices), likelihood

    def _greedy_selection(self, masked_output_vector):
        output_indices = torch.argmax(masked_output_vector, dim=2)
        probabilities = torch.softmax(
            masked_output_vector, dim=2
        )  # Softmax over the class dimension for all timesteps

        # Gather the probabilities corresponding to the output_indices for all samples in one go
        batch_indices = torch.arange(masked_output_vector.size(0)).unsqueeze(
            1
        )  # Batch indices (shape: [batch_size, 1])
        selected_probabilities = probabilities[
            batch_indices, torch.arange(output_indices.shape[1]), output_indices
        ]

        # Compute likelihood as the product of probabilities for each sample
        likelihoods = selected_probabilities.prod(
            dim=1
        )  # Product over time steps for each sample

        return output_indices, likelihoods.numpy()

    def _decode_indices_to_node_names(self, output_indices):
        index_to_node_name = self.pset_utils.get_index_to_node_name()
        return [index_to_node_name.get(idx, "Unknown") for idx in output_indices]

    def ensemble_predict(self, semantics):
        indices_list = []
        for _ in range(5):
            noise_semantics = semantics + np.random.normal(0, 0.01, semantics.shape)
            indices = self.predict(noise_semantics, mode="greedy", return_indices=True)
            indices_list.append(indices.numpy()[0])

        # get the most common indices
        indices = np.array(indices_list)
        indices = np.transpose(indices)
        indices = [
            Counter(indices[i]).most_common(1)[0][0] for i in range(len(indices))
        ]
        tree_node_names = self._decode_indices_to_node_names(indices)
        return tree_node_names

    def convert_to_primitive_tree(self, semantics, mode=None):
        """
        Convert semantics to a GP tree and then to a PrimitiveTree.

        :param semantics: List of node names representing the GP tree.
        :return: PrimitiveTree constructed from the generated GP tree.
        """
        if mode is None:
            mode = self.prediction_mode
        node_names_pos, likelihood_pos = self.predict(semantics, mode=mode)
        gp_tree = self.pset_utils.convert_node_names_to_gp_tree(node_names_pos)
        if self.double_query:
            node_names_neg, likelihood_neg = self.predict(-1 * semantics, mode=mode)
            if likelihood_neg > likelihood_pos:
                gp_tree = self.pset_utils.convert_node_names_to_gp_tree(node_names_neg)
        return PrimitiveTree(gp_tree)


def generate_synthetic_data(
    pset,
    input_data,
    num_samples=100,
    min_depth=2,
    max_depth=2,
    seed=0,
):
    """
    Generate synthetic training data with random GP trees and their target semantics.

    :param pset: DEAP PrimitiveSet.
    :param num_samples: Number of samples to generate.
    :return: List of tuples (gp_tree, target_semantics).
    """
    reset_random(seed)
    # Create a random GP tree using DEAP's ramped half-and-half method
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    data = []
    viewed = set()
    for _ in range(num_samples):
        # Generate a random GP tree
        gp_tree = create_random_gp_tree(pset, min_depth=min_depth, max_depth=max_depth)

        # Generate synthetic semantics (e.g., applying the GP tree to a synthetic dataset)
        target_semantics = normalize_vector(
            generate_target_semantics(gp_tree, input_data, pset)
        )
        if not isinstance(target_semantics, np.ndarray):
            # constant
            continue
        if tuple(target_semantics) in viewed:
            continue
        viewed.add(tuple(target_semantics))
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


def generate_target_semantics(gp_tree, data: np.ndarray, pset):
    # Compile the GP tree into a callable function
    func = gp.compile(PrimitiveTree(gp_tree), pset)

    # Apply the compiled function to each row of the dataset
    target_semantics = func(*data.T)

    return target_semantics


def filter_train_data_by_node_count(train_data, max_function_nodes=3):
    """
    Filter out GP trees from train_data with more than a specified number of nodes.

    :param train_data: List of tuples (gp_tree, target_semantics), where gp_tree is a DEAP Individual.
    :param max_function_nodes: Maximum number of nodes allowed in the GP trees.
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
        if len([primitive for primitive in tree if isinstance(primitive, Primitive)])
        <= max_function_nodes
    ]
    return filtered_train_data


def calculate_semantics_accuracy(nl, test_data, pset, x):
    cosine_similarities = []

    for tid in range(len(test_data)):
        # Convert the predicted tree and original tree to a list of tokens
        predicted_tree = nl.convert_to_primitive_tree(test_data[tid][1])
        original_tree = PrimitiveTree(test_data[tid][0])

        # Compile the trees to functions
        original_func = gp.compile(expr=original_tree, pset=pset)
        predicted_func = gp.compile(expr=predicted_tree, pset=pset)

        # Calculate the predicted and true values
        y_pred = predicted_func(*x.T)
        y_true = original_func(*x.T)

        # Normalize the vectors
        y_pred = normalize_vector(y_pred)
        y_true = normalize_vector(y_true)

        # Compute the cosine similarity
        similarity = abs(cosine_similarity([y_true], [y_pred])[0][0])
        cosine_similarities.append(similarity)

    # Calculate the average cosine similarity
    avg_cosine_similarity = np.mean(cosine_similarities)
    return avg_cosine_similarity


def calculate_token_accuracy(nl, test_data):
    """
    Calculate the token-level accuracy of the neural network's predictions.

    :param nl: The trained NeuralSemanticLibrary model.
    :param test_data: A list of tuples, where each tuple contains (original_tree, encoded_representation).
    :return: The token-level accuracy of the model's predictions.
    """
    total_tokens = 0
    correct_tokens = 0

    for tid in range(len(test_data)):
        # Convert the predicted tree and original tree to a list of tokens
        predicted_tree = nl.convert_to_primitive_tree(test_data[tid][1])
        original_tree = PrimitiveTree(test_data[tid][0])

        # Convert trees to lists of tokens
        predicted_tokens = list(predicted_tree)
        original_tokens = list(original_tree)
        print("Predicted tokens", str(predicted_tree))
        print("Original tokens", str(original_tree))

        # Count the total number of tokens in the original tree
        total_tokens += len(original_tokens)

        # Compare each token
        for pred_token, orig_token in zip(predicted_tokens, original_tokens):
            if pred_token == orig_token:
                correct_tokens += 1

    # Calculate token-level accuracy as a percentage
    token_accuracy = correct_tokens / total_tokens
    return token_accuracy
