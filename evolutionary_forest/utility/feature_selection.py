# Function to detect and delete constant variables from the primitive set
import numpy as np


def remove_constant_variables(pset, x):
    """Remove constant variables from a primitive set based on dataset x."""
    constant_indices = []
    seleceted_indices = []

    # Detect constant variables
    for i in range(x.shape[1]):
        if np.all(x[:, i] == x[0, i]):  # Check if all values in the column are the same
            constant_indices.append(i)  # Record the index of constant variable
        else:
            seleceted_indices.append(i)

    # Delete in reverse order to avoid index shifting
    for i in reversed(constant_indices):
        arg_name = f"ARG{i}"

        # 1. Remove from the terminals list
        terminal_to_remove = pset.mapping[arg_name]
        if terminal_to_remove in pset.terminals[pset.ins[i]]:
            pset.terminals[pset.ins[i]].remove(terminal_to_remove)

        # 2. Remove from the ins list
        del pset.ins[i]

        # 3. Remove from the arguments list
        pset.arguments.remove(arg_name)

        # 4. Remove from the mapping
        del pset.mapping[arg_name]

        # 5. Adjust the terms count
        pset.terms_count -= 1

    return np.array(
        seleceted_indices
    )  # Return the indices of removed variables for reporting
