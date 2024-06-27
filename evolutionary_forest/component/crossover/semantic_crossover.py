import numpy as np


def calculate_mutation_probabilities(beta, gamma):
    """
    Calculate mutation probabilities for features based on their coefficients.

    Parameters:
    - beta: numpy array of shape (n_features,), coefficients of the features
    - gamma: float, parameter controlling the amount of feedback from the weights

    Returns:
    - mutation_probabilities: numpy array of shape (n_features,), mutation probabilities for each feature
    """
    # Normalize the coefficients
    beta_normalized = np.abs(beta) / np.sum(np.abs(beta))

    # Calculate the softmax-normalized probabilities
    exp_terms = np.exp(1 - beta_normalized)
    s = exp_terms / np.sum(exp_terms)

    # Calculate the mutation probabilities
    m = len(beta)
    uniform_prob = 1 / m
    mutation_probabilities = gamma * s + (1 - gamma) * uniform_prob

    return mutation_probabilities


def resxo(phi_p1, phi_p2, beta_p1, y):
    """
    Perform Residual Fit Crossover (ResXO) between two parent feature representations.

    Parameters:
    - phi_p1: numpy array of shape (n_samples, n_features_p1), features of parent 1
    - phi_p2: numpy array of shape (n_samples, n_features_p2), features of parent 2
    - beta_p1: numpy array of shape (n_features_p1,), coefficients of the features in parent 1
    - y: numpy array of shape (n_samples,), target values

    Returns:
    - phi_c: numpy array of shape (n_samples, n_features_p1), features of the child representation
    """
    n_samples, n_features_p1 = phi_p1.shape

    # Step 1: Pick a feature from phi_p1 using probabilities (assuming uniform probability for simplicity)
    mutation_probabilities = calculate_mutation_probabilities(beta_p1, gamma)
    d = np.random.choice(len(mutation_probabilities), p=mutation_probabilities)
    phi_d = phi_p1[:, d]

    # Step 2: Calculate the residual of p1 without phi_d
    residual = y - (phi_p1 @ beta_p1) + (beta_p1[d] * phi_d)

    # Step 3: Choose the feature from phi_p2 most correlated with the residual
    correlations = np.array(
        [np.corrcoef(residual, phi_p2[:, j])[0, 1] for j in range(phi_p2.shape[1])]
    )
    best_feature_idx = np.argmax(np.abs(correlations))
    phi_star = phi_p2[:, best_feature_idx]

    # Step 4: Create the child representation by replacing phi_d with phi_star in phi_p1
    phi_c = phi_p1.copy()
    phi_c[:, d] = phi_star

    return phi_c, d, best_feature_idx


def stagexo(phi_p1, phi_p2, y):
    """
    Perform Forward Stagewise Crossover (StageXO) between two parent feature representations.

    Parameters:
    - phi_p1: numpy array of shape (n_samples, n_features_p1), features of parent 1
    - phi_p2: numpy array of shape (n_samples, n_features_p2), features of parent 2
    - y: numpy array of shape (n_samples,), target values

    Returns:
    - phi_c: numpy array of shape (n_samples, n_features_p1), features of the child representation
    """
    n_samples, n_features_p1 = phi_p1.shape

    # Step 1: Center means around zero for all features and set the initial residual equal to the target
    phi_p1 = phi_p1 - np.mean(phi_p1, axis=0)
    phi_p2 = phi_p2 - np.mean(phi_p2, axis=0)
    r = y - np.mean(y)

    # Step 2: Set phi_A to be all subprograms in phi_p1 and phi_p2
    phi_A = np.hstack((phi_p1, phi_p2))

    # Step 3: Initialize the child representation with no features
    phi_c = np.zeros((n_samples, 0))
    selected_indices_p1 = []
    selected_indices_p2 = []

    while phi_c.shape[1] < n_features_p1:
        # Step 3(a): Pick phi^* from phi_A which is most correlated with r
        correlations = np.array(
            [np.corrcoef(r, phi_A[:, j])[0, 1] for j in range(phi_A.shape[1])]
        )
        best_feature_idx = np.argmax(np.abs(correlations))
        phi_star = phi_A[:, best_feature_idx]

        # Step 3(b): Compute the least squares coefficient b for phi^* fit to r
        b = np.dot(phi_star, r) / np.dot(phi_star, phi_star)

        # Step 3(c): Update r = r - b * phi^*
        r = r - b * phi_star

        # Step 3(d): Add phi^* to phi_c
        phi_c = np.hstack((phi_c, phi_star.reshape(-1, 1)))

        # Step 3(e): Remove phi^* from phi_A and store selected index
        if best_feature_idx < phi_p1.shape[1]:
            selected_indices_p1.append(best_feature_idx)
        else:
            selected_indices_p2.append(best_feature_idx - phi_p1.shape[1])

        phi_A = np.delete(phi_A, best_feature_idx, axis=1)

    return phi_c, selected_indices_p1, selected_indices_p2


if __name__ == "__main__":
    # Assuming we have the following data (you can replace these with actual data)
    phi_p1 = np.random.rand(10, 10)  # 100 samples, 10 features for parent 1
    phi_p2 = np.random.rand(10, 15)  # 100 samples, 15 features for parent 2
    beta_p1 = np.random.rand(10)  # Coefficients for parent 1's features
    y = np.random.rand(10)  # Target values

    phi_c = resxo(phi_p1, phi_p2, beta_p1, y)
    print(phi_c)
    phi_c = stagexo(phi_p1, phi_p2, y)
    print(phi_c)
