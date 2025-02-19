def estimate_difficulty_tiers(case_values, difficulty_metric="variance", num_tiers=3):
    """
    Compute difficulty tiers using different metrics: "variance", "median_error", or "entropy".
    """
    difficulty_scores = []

    for case, errors in case_values.items():
        if difficulty_metric == "variance":
            difficulty = np.std(errors)  # Variance-based difficulty measure
        elif difficulty_metric == "median_error":
            difficulty = np.median(errors)  # Median error as difficulty measure
        elif difficulty_metric == "entropy":
            hist, _ = np.histogram(errors, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            difficulty = -np.sum(hist * np.log(hist))  # Shannon entropy
        else:
            raise ValueError(
                "Unsupported difficulty metric. Choose 'variance', 'median_error', or 'entropy'."
            )

        difficulty_scores.append((case, difficulty))

    # Sort test cases by difficulty
    difficulty_scores.sort(key=lambda x: x[1])

    # Split into tiers (medium first, then easy/hard)
    num_cases = len(difficulty_scores)
    tier_size = max(1, num_cases // num_tiers)  # Ensure tier size is at least 1

    medium_tier = difficulty_scores[tier_size : num_cases - tier_size]  # Medium cases
    easy_tier = difficulty_scores[:tier_size]  # Easiest cases
    hard_tier = difficulty_scores[num_cases - tier_size :]  # Hardest cases

    # Shuffle within each tier
    random.shuffle(medium_tier)
    random.shuffle(easy_tier)
    random.shuffle(hard_tier)

    # Final ordering: Medium first, then Easy + Hard mixed
    final_order = [case for case, _ in medium_tier] + [
        case for case, _ in (easy_tier + hard_tier)
    ]

    return final_order


def automatic_epsilon_lexicase_selection(
    population, case_values, num_selected, difficulty_metric="variance"
):
    """
    Automatic Epsilon Lexicase Selection with Median Absolute Deviation (MAD) and proper tracking of individuals.
    """
    selected = []

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    # Compute adaptive case ordering with tier-based shuffling
    ordered_cases = estimate_difficulty_tiers(case_values, difficulty_metric)

    for _ in range(num_selected):
        candidates = list(individual_map.keys())  # Track using indices

        for case in ordered_cases:
            # Compute threshold using Median Absolute Deviation (MAD)
            errors = [case_values[case][i] for i in candidates]
            median_error = np.median(errors)
            mad = np.median(
                [abs(e - median_error) for e in errors]
            )  # Median absolute deviation
            epsilon = mad  # Automatic epsilon threshold

            # Select candidates within the epsilon range of the minimum error
            min_error = min(errors)
            candidates = [
                i for i in candidates if case_values[case][i] <= min_error + epsilon
            ]

            if len(candidates) == 1:
                break  # Select if only one remains

        # Select one candidate randomly from remaining and store the individual
        selected.append(individual_map[random.choice(candidates)])

    return selected


if __name__ == "__main__":
    import random
    import numpy as np
    import pandas as pd

    # Define a simple individual structure
    class Individual:
        def __init__(self, id, errors):
            self.id = id  # Unique identifier for tracking
            self.errors = errors  # Dictionary of test case errors

        def __repr__(self):
            return f"Ind{self.id}: {self.errors}"

    test_cases = ["T1", "T2", "T3", "T4", "T5"]
    population = [Individual(i, {}) for i in range(10)]

    # Generate case values: Dictionary {test_case: [errors for each individual]}
    case_values = {tc: [random.randint(0, 10) for _ in population] for tc in test_cases}

    # Assign errors to individuals from case values
    for i, ind in enumerate(population):
        ind.errors = {tc: case_values[tc][i] for tc in test_cases}

    # Apply RCL-Lexicase Selection using precomputed case values
    selected_individuals = automatic_epsilon_lexicase_selection(
        population, case_values, num_selected=3
    )

    # Display results

    df_population = pd.DataFrame(
        [{**{"ID": ind.id}, **ind.errors} for ind in population]
    )
    df_selected = pd.DataFrame(
        [{**{"ID": ind.id}, **ind.errors} for ind in selected_individuals]
    )

    print("Population:")
    print(df_population)
    print("\nSelected Individuals:")
    print(df_selected)
