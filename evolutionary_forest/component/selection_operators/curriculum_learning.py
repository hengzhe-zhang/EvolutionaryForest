import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans


def estimate_difficulty_tiers(case_values, difficulty_metric="variance", num_tiers=3):
    """
    Compute difficulty tiers using K-means (1D clustering) before selection loop.
    Returns tiered cases (Easy, Medium, Hard) without shuffling.
    """
    num_cases = case_values.shape[1]  # Number of test cases
    difficulty_scores = []

    for case_idx in range(num_cases):
        errors = case_values[:, case_idx]  # Extract all errors for the test case

        if difficulty_metric == "variance":
            difficulty = np.std(errors)  # Variance-based difficulty measure
        elif difficulty_metric == "median_error":
            difficulty = np.median(errors)  # Median error as difficulty measure
        elif difficulty_metric == "entropy":
            hist, _ = np.histogram(errors, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            difficulty = -np.sum(hist * np.log(hist))  # Shannon entropy
        else:
            raise ValueError("Unsupported difficulty metric. Choose 'variance', 'median_error', or 'entropy'.")

        difficulty_scores.append((case_idx, difficulty))

    # Convert difficulty scores to a NumPy array for K-means clustering
    difficulty_array = np.array([score[1] for score in difficulty_scores]).reshape(-1, 1)

    # Apply K-means clustering to split test cases into difficulty tiers
    kmeans = KMeans(n_clusters=num_tiers, n_init=10, random_state=42)
    tier_labels = kmeans.fit_predict(difficulty_array)

    # Group test cases into tiers based on clustering results
    tiered_cases = {i: [] for i in range(num_tiers)}
    for idx, label in enumerate(tier_labels):
        tiered_cases[label].append(difficulty_scores[idx][0])  # Store case index

    return tiered_cases  # Return tiered test case indices


def automatic_epsilon_lexicase_selection_CL(population, num_selected, difficulty_metric="variance"):
    """
    Automatic Epsilon Lexicase Selection with Median Absolute Deviation (MAD) and proper tracking of individuals.
    """
    selected = []
    case_values = np.array([ind.case_values for ind in population])

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    # Precompute difficulty tiers using K-means (but no shuffling yet)
    precomputed_tiers = estimate_difficulty_tiers(case_values, difficulty_metric)

    for _ in range(num_selected):
        # Shuffle **within** each tier before constructing ordered_cases
        shuffled_tiers = {tier: random.sample(cases, len(cases)) for tier, cases in precomputed_tiers.items()}

        # Construct ordered_cases: Medium first, then Easy + Hard mixed
        ordered_cases = shuffled_tiers.get(1, []) + shuffled_tiers.get(0, []) + shuffled_tiers.get(2, [])

        candidates = list(individual_map.keys())  # Track using indices

        for iteration, case_idx in enumerate(ordered_cases, start=1):
            # Compute threshold using Median Absolute Deviation (MAD)
            errors = case_values[candidates, case_idx]  # Extract candidate errors for the test case
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))  # Compute MAD
            epsilon = mad  # Automatic epsilon threshold

            # Select candidates within the epsilon range of the minimum error
            min_error = np.min(errors)
            candidates = [i for i in candidates if case_values[i, case_idx] <= min_error + epsilon]

            # Print number of iterated cases when a single candidate remains
            if len(candidates) == 1:
                print(f"Number of iterated cases before selection: {iteration}, difficulty_metric: {difficulty_metric}")
                break  # Stop once one candidate is left

        # Select one candidate randomly from remaining and store the individual
        selected.append(individual_map[random.choice(candidates)])

    return selected


def automatic_epsilon_lexicase_selection_CL(population, num_selected, difficulty_metric="variance"):
    """
    Automatic Epsilon Lexicase Selection with Median Absolute Deviation (MAD) and proper tracking of individuals.
    """
    selected = []
    case_values = np.array([ind.case_values for ind in population])

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    for _ in range(num_selected):
        # Implement mode selection for difficulty_metric
        if difficulty_metric == "random":
            ordered_cases = list(range(case_values.shape[1]))
            random.shuffle(ordered_cases)  # Shuffle test cases randomly
        else:
            # Only shuffle within each tier (No K-means for efficiency)
            ordered_cases = estimate_difficulty_tiers(case_values, difficulty_metric)

        candidates = list(individual_map.keys())  # Track using indices

        for iteration, case_idx in enumerate(ordered_cases, start=1):
            # Compute threshold using Median Absolute Deviation (MAD)
            errors = case_values[candidates, case_idx]  # Extract candidate errors for the test case
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))  # Compute MAD
            epsilon = mad  # Automatic epsilon threshold

            # Select candidates within the epsilon range of the minimum error
            min_error = np.min(errors)
            candidates = [i for i in candidates if case_values[i, case_idx] <= min_error + epsilon]

            # Print number of iterated cases when a single candidate remains
            if len(candidates) == 1:
                print(f"Number of iterated cases before selection: {iteration}, difficulty_metric: {difficulty_metric}")
                break  # Stop once one candidate is left

        # Select one candidate randomly from remaining and store the individual
        selected.append(individual_map[random.choice(candidates)])

    return selected


if __name__ == "__main__":
    # Define a simple individual structure
    class Individual:
        def __init__(self, id, case_values):
            self.id = id  # Unique identifier for tracking
            self.case_values = case_values  # List of test case errors

        def __repr__(self):
            return f"Ind{self.id}: {self.case_values}"


    test_cases = ["T1", "T2", "T3", "T4", "T5"]
    population = [Individual(i, [random.randint(0, 10) for _ in test_cases]) for i in range(10)]

    # Convert case_values into a 2D NumPy array
    case_values = np.array([ind.case_values for ind in population])

    # Apply Automatic Epsilon Lexicase Selection
    selected_individuals = automatic_epsilon_lexicase_selection_CL(population, num_selected=3,
                                                                   difficulty_metric="random")

    # Display results
    df_population = pd.DataFrame(
        [{"ID": ind.id, **{f"Test {j + 1}": val for j, val in enumerate(ind.case_values)}} for ind in population]
    )
    df_selected = pd.DataFrame(
        [{"ID": ind.id, **{f"Test {j + 1}": val for j, val in enumerate(ind.case_values)}} for ind in
         selected_individuals]
    )

    print("Population:")
    print(df_population)
    print("\nSelected Individuals:")
    print(df_selected)
