import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import operator
from deap import gp
from functools import partial


###############################################################################
# 1. Define the EQL network components (as in Martius & Lampert)
###############################################################################

def identity(x):
    return x


def safe_div(x, y, theta=1e-4):
    return x / y if abs(y) > theta else 0.0


class EQLLayer(nn.Module):
    def __init__(self, in_features, out_unary, out_binary):
        """
        in_features: number of inputs.
        out_unary: number of unary units (must be divisible by 3).
        out_binary: number of binary units (each uses two linear outputs).
        """
        super(EQLLayer, self).__init__()
        self.out_unary = out_unary
        self.out_binary = out_binary
        self.total_units = out_unary + 2 * out_binary
        self.linear = nn.Linear(in_features, self.total_units)

    def forward(self, x):
        # Linear transformation
        z = self.linear(x)
        # Split the first out_unary neurons into three groups:
        group_size = self.out_unary // 3
        id_part = z[:, :group_size]
        sin_part = z[:, group_size:2 * group_size]
        cos_part = z[:, 2 * group_size: 3 * group_size]
        # Apply activations
        unary_out = torch.cat([id_part, torch.sin(sin_part), torch.cos(cos_part)], dim=1)
        # Binary units: take the remaining outputs in pairs and multiply.
        binary_inputs = z[:, self.out_unary:]
        binary_inputs = binary_inputs.view(-1, self.out_binary, 2)
        binary_out = torch.prod(binary_inputs, dim=2)
        return torch.cat([unary_out, binary_out], dim=1)


class DeepEquationLearner(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, use_division=False, division_threshold=0.1):
        """
        input_dim: number of input variables.
        hidden_layers: number of hidden EQL layers.
        output_dim: number of outputs.
        use_division: if True, the output layer produces numerator/denom pairs.
        division_threshold: initial threshold for safe division.
        """
        super(DeepEquationLearner, self).__init__()
        layers = []
        in_features = input_dim
        # Example hyperparameters: 9 unary units and 10 binary units per layer.
        out_unary = 9  # must be divisible by 3
        out_binary = 10
        for _ in range(hidden_layers):
            layers.append(EQLLayer(in_features, out_unary, out_binary))
            in_features = out_unary + out_binary  # dimension after one EQLLayer
        self.hidden = nn.Sequential(*layers)
        self.use_division = use_division
        self.division_threshold = division_threshold
        if use_division:
            # If using division, the output layer produces 2*output_dim values.
            self.output_linear = nn.Linear(in_features, 2 * output_dim)
        else:
            self.output_linear = nn.Linear(in_features, output_dim)

    def forward(self, x):
        h = self.hidden(x)
        out = self.output_linear(h)
        if self.use_division:
            half = out.shape[1] // 2
            num = out[:, :half]
            den = out[:, half:]
            results = []
            for i in range(num.shape[1]):
                numerator = num[:, i]
                denominator = den[:, i]
                safe_result = torch.where(denominator.abs() > self.division_threshold,
                                          numerator / denominator,
                                          torch.zeros_like(numerator))
                results.append(safe_result.unsqueeze(1))
            return torch.cat(results, dim=1)
        else:
            return out


###############################################################################
# 2. Training logic with three phases (following the paper)
###############################################################################

def train_eql_phases(model, optimizer, loss_fn, x_train, y_train, num_epochs=1000, lambda_reg=1e-3):
    """
    Trains the EQL model in three phases:
      Phase 1 (epochs < t1): No L1 regularization (λ = 0).
      Phase 2 (t1 <= epochs < t2): Apply L1 penalty.
      Phase 3 (epochs >= t2): Disable L1 penalty and clip small weights to zero.

    Also, if division units are used, updates the division threshold with curriculum:
         θ(t) = 1/√(t+1).
    """
    t1 = int(num_epochs * 0.25)  # e.g., first 25% epochs unregularized
    t2 = int(num_epochs * 0.95)  # final 5% epochs: refine by clipping weights
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        mse_loss = loss_fn(outputs, y_train)

        # Phase 1: no regularization
        if epoch < t1:
            loss = mse_loss
        # Phase 2: add L1 regularization on all parameters
        elif epoch < t2:
            l1_penalty = 0
            for param in model.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            loss = mse_loss + lambda_reg * l1_penalty
        # Phase 3: final phase, no L1 penalty; loss is MSE only.
        else:
            loss = mse_loss

        loss.backward()
        optimizer.step()

        # In Phase 3, enforce sparsity by clipping weights whose absolute value is below 0.001.
        if epoch >= t2:
            with torch.no_grad():
                for param in model.parameters():
                    param.data[torch.abs(param.data) < 0.001] = 0.0

        # If using division units, update the division threshold via curriculum.
        if model.use_division:
            model.division_threshold = 1.0 / math.sqrt(epoch + 1)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Total Loss: {loss.item():.6f} | MSE: {mse_loss.item():.6f}")
    print("Training completed.")
    return model


###############################################################################
# 3. Extraction of a symbolic expression from the pruned network
###############################################################################

# We represent an expression as a tree. For example, a linear combination is represented as:
#   ("add", term1, term2), and a term can be ("mul", weight, expr) if the weight is not 1.
def linear_expr_tree(weights, bias, in_trees, eps=1e-3):
    terms = []
    for w, expr in zip(weights, in_trees):
        if abs(w) > eps:
            # If the weight is nearly 1, we omit explicit multiplication.
            if abs(w - 1.0) < eps:
                terms.append(expr)
            else:
                terms.append(("mul", round(w, 3), expr))
    if abs(bias) > eps:
        terms.append(round(bias, 3))
    if not terms:
        return 0
    if len(terms) == 1:
        return terms[0]
    # Combine terms with addition (left-associative)
    tree = ("add", terms[0], terms[1])
    for term in terms[2:]:
        tree = ("add", tree, term)
    return tree


def extract_symbolic_expression_tree(model, input_vars, eps=1e-3):
    """
    Recursively extract a symbolic expression from the network.
    At the beginning, the input layer is represented by the variable names.
    Then each hidden layer is processed by reading the linear transformation,
    pruning near-zero weights, and wrapping the result with the corresponding activation.
    """
    # Start with input variables (e.g., ["x0", "x1"])
    current_trees = input_vars[:]

    # Process each hidden layer.
    for layer in model.hidden:
        # Get the weight matrix and bias from the layer's linear part.
        W = layer.linear.weight.detach().cpu().numpy()  # shape: (total_units, n_inputs)
        b = layer.linear.bias.detach().cpu().numpy()  # shape: (total_units,)
        in_trees = current_trees
        new_trees = []
        out_unary = layer.out_unary
        out_binary = layer.out_binary
        group_size = out_unary // 3

        # For the unary units:
        # Identity group.
        for i in range(0, group_size):
            tree = linear_expr_tree(W[i], b[i], in_trees, eps)
            new_trees.append(tree)
        # Sin group.
        for i in range(group_size, 2 * group_size):
            tree = linear_expr_tree(W[i], b[i], in_trees, eps)
            new_trees.append(("sin", tree))
        # Cos group.
        for i in range(2 * group_size, 3 * group_size):
            tree = linear_expr_tree(W[i], b[i], in_trees, eps)
            new_trees.append(("cos", tree))

        # For the binary units: each is computed from a pair of rows.
        for j in range(out_binary):
            idx1 = out_unary + 2 * j
            idx2 = out_unary + 2 * j + 1
            tree1 = linear_expr_tree(W[idx1], b[idx1], in_trees, eps)
            tree2 = linear_expr_tree(W[idx2], b[idx2], in_trees, eps)
            new_trees.append(("mul", tree1, tree2))

        current_trees = new_trees

    # Process the output layer.
    W_out = model.output_linear.weight.detach().cpu().numpy()
    b_out = model.output_linear.bias.detach().cpu().numpy()
    out_dim = model.output_linear.out_features
    output_trees = []
    if model.use_division:
        actual_out_dim = out_dim // 2
        for i in range(actual_out_dim):
            num_tree = linear_expr_tree(W_out[i], b_out[i], current_trees, eps)
            den_tree = linear_expr_tree(W_out[i + actual_out_dim], b_out[i + actual_out_dim], current_trees, eps)
            output_trees.append(("div", num_tree, den_tree))
    else:
        for i in range(out_dim):
            tree = linear_expr_tree(W_out[i], b_out[i], current_trees, eps)
            output_trees.append(tree)

    if len(output_trees) == 1:
        return output_trees[0]
    return output_trees


def tree_to_prefix_str(tree):
    """Convert an expression tree to a prefix string for DEAP GP."""
    if isinstance(tree, tuple):
        op = tree[0]
        args_str = ", ".join(tree_to_prefix_str(arg) for arg in tree[1:])
        return f"{op}({args_str})"
    else:
        return str(tree)


###############################################################################
# 4. Convert the extracted expression into a DEAP GP individual
###############################################################################

def one_const():
    return 1.0


def convert_to_deap(model, input_vars):
    """
    Extract the symbolic expression from the trained EQL model (using pruning)
    and convert it into a DEAP GP PrimitiveTree.
    """
    # Create a DEAP primitive set.
    pset = gp.PrimitiveSet("MAIN", len(input_vars))
    # Rename arguments (DEAP stores these as a list).
    pset.renameArguments(**{f"ARG{i}": var for i, var in enumerate(input_vars)})

    # Add basic primitives. (Names must match those in the extracted tree.)
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")

    def safe_div(x, y):
        return x / y if abs(y) > 1e-4 else 0.0

    pset.addPrimitive(safe_div, 2, name="div")
    pset.addPrimitive(math.sin, 1, name="sin")
    pset.addPrimitive(math.cos, 1, name="cos")
    pset.addEphemeralConstant("const", one_const)

    # Extract symbolic expression from the pruned network.
    expr_tree = extract_symbolic_expression_tree(model, input_vars)
    expr_str = tree_to_prefix_str(expr_tree)
    print("DEBUG: Extracted symbolic expression (prefix):", expr_str)

    try:
        gp_expr = gp.PrimitiveTree.from_string(expr_str, pset)
    except Exception as e:
        print("DEBUG: Error converting to DEAP GP tree:", e)
        raise e
    return gp_expr


###############################################################################
# 5. Example usage: Train EQL and convert to DEAP GP model
###############################################################################

if __name__ == "__main__":
    # For reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate synthetic training data.
    num_samples = 1000
    # Two inputs uniformly sampled in [-1,1]
    x_data = np.random.uniform(-1, 1, (num_samples, 2)).astype(np.float32)
    # Target function: y = sin(x0) + cos(x1)
    y_data = np.sin(x_data[:, 0:1]) + np.cos(x_data[:, 1:2])

    x_train = torch.tensor(x_data)
    y_train = torch.tensor(y_data)

    # Instantiate the Deep Equation Learner.
    input_dim = 2
    hidden_layers = 2
    output_dim = 1
    # For this example, we disable division units.
    model = DeepEquationLearner(input_dim, hidden_layers, output_dim,
                                use_division=False, division_threshold=0.1)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training of EQL model with phased loss...")
    model = train_eql_phases(model, optimizer, loss_fn, x_train, y_train, num_epochs=1000, lambda_reg=1e-3)

    # Evaluate final training loss.
    model.eval()
    with torch.no_grad():
        preds = model(x_train)
        final_loss = loss_fn(preds, y_train).item()
    print(f"Final training MSE: {final_loss:.6f}")

    # Convert the trained (and pruned) EQL model to a DEAP GP individual.
    input_vars = ["x0", "x1"]
    gp_expr = convert_to_deap(model, input_vars)
    print("DEAP GP expression:", gp_expr)
