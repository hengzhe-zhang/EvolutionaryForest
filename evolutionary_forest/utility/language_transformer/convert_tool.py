import random

import numpy as np
from deap import gp, base, creator, tools


def random_int():
    return random.randint(-1, 1)


def generate_primitive_trees(num_individuals, max_tree_depth=3):
    pset = gp.PrimitiveSet(
        "MAIN", arity=1
    )  # 'arity=1' means each function takes 1 argument

    # Add basic arithmetic operations
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)

    # Add an ephemeral constant and a variable
    # pset.addEphemeralConstant("rand101", random_int)
    # pset.renameArguments(ARG0="x")

    # Define fitness and individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_tree_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Generate population
    population = toolbox.population(n=num_individuals)

    return pset, population


def tree_to_tokens(tree):
    tokens = []

    for node in tree:
        if isinstance(node, gp.Primitive):
            tokens.append(node.name)
        elif isinstance(node, gp.Terminal):
            if isinstance(node.value, float):
                tokens.append(str(node.value))
            else:
                tokens.append(node.value)
        elif isinstance(node, gp.MetaEphemeral):
            tokens.append(str(node.value))
        else:
            raise TypeError("Unknown node type:", type(node))

    return tokens


def build_vocab(token_sequences):
    vocab = {}
    # Add a special token for unknown tokens
    vocab["<UNK>"] = 0
    index = 1  # Starting index for tokens

    for tokens in token_sequences:
        for token in tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1

    return vocab


def tokens_to_word_tokens(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]


def word_tokens_to_tree(word_tokens, vocab, pset):
    # Reverse the vocabulary to map indices to tokens
    inv_vocab = {v: k for k, v in vocab.items()}
    token_seq = [inv_vocab[ix] for ix in word_tokens]

    # Convert token sequence back into a GP tree
    tree = gp.PrimitiveTree([])
    for token in token_seq:
        if token in pset.mapping:
            tree.append(pset.mapping[token])
        else:
            # Assuming all other tokens are ephemeral constants
            tree.append(gp.Terminal(token, False, object))
    return tree


def generate_and_evaluate_trees(num_trees, num_points, x_range=(-1, 1)):
    pset, individuals = generate_primitive_trees(num_trees)

    # Sample random points within the specified range
    x_samples = np.random.uniform(low=x_range[0], high=x_range[1], size=(num_points,))

    return generate_training_data(individuals, pset, x_samples)


def generate_training_data(individuals, pset, x_samples):
    token_sequences = [tree_to_tokens(ind) for ind in individuals]
    vocab = build_vocab(token_sequences)
    # Initialize a dictionary to store unique gp_outputs and corresponding word token sequences
    unique_data = {}

    for ind, tokens in zip(individuals, token_sequences):
        # Compile the individual tree to a callable function
        func = gp.compile(expr=ind, pset=pset)

        # Evaluate the GP function on sampled points and convert to a NumPy array for easier manipulation
        gp_output = np.array([func(x) for x in x_samples])

        # Normalize the gp_output to have a 2-norm of 1, if it's not a zero vector
        norm = np.linalg.norm(gp_output, 2)  # Compute the 2-norm
        if norm > 0:
            gp_output_normalized = tuple(
                gp_output / norm
            )  # Normalize and convert back to tuple for hashability
        else:
            gp_output_normalized = tuple(
                gp_output
            )  # Keep original if norm is 0 to avoid division by 0

            # If this normalized gp_output hasn't been encountered before, store it and its corresponding tokens
        if gp_output_normalized not in unique_data:
            word_tokens = tokens_to_word_tokens(tokens, vocab)
            unique_data[gp_output_normalized] = word_tokens

    # Split the unique data back into separate lists for gp_outputs and word_token_sequences
    gp_outputs = list(unique_data.keys())
    word_token_sequences = list(unique_data.values())

    # Prepare the training data for the transformer
    training_data = {
        "word_token_sequences": word_token_sequences,
        "gp_outputs": gp_outputs,
        "vocab": vocab,
    }
    return training_data


if __name__ == "__main__":
    num_trees = 10
    num_points = 100
    print(generate_and_evaluate_trees(num_trees, num_points))
