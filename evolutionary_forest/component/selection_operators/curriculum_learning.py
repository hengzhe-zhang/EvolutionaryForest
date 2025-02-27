import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEBUG = False


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
            raise ValueError(
                "Unsupported difficulty metric. Choose 'variance', 'median_error', or 'entropy'."
            )

        difficulty_scores.append((case_idx, difficulty))

    # Convert difficulty scores to a NumPy array for K-means clustering
    difficulty_array = np.array([score[1] for score in difficulty_scores]).reshape(
        -1, 1
    )

    # Apply K-means clustering to split test cases into difficulty tiers
    kmeans = KMeans(n_clusters=num_tiers, n_init=10, random_state=42)
    tier_labels = kmeans.fit_predict(difficulty_array)

    # Group test cases into tiers based on clustering results
    tiered_cases = {i: [] for i in range(num_tiers)}
    for idx, label in enumerate(tier_labels):
        tiered_cases[label].append(difficulty_scores[idx][0])  # Store case index

    return tiered_cases  # Return tiered test case indices


class GenerationInfo:
    def __init__(
        self,
        current_gen=0,
        max_generations=0,
    ):
        self.current_gen = current_gen
        self.max_generations = max_generations


def automatic_epsilon_lexicase_selection_CL(
    population,
    num_selected,
    difficulty_metric="variance",
):
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
        shuffled_tiers = {
            tier: random.sample(cases, len(cases))
            for tier, cases in precomputed_tiers.items()
        }

        # Construct ordered_cases: Medium first, then Easy + Hard mixed
        ordered_cases = (
            shuffled_tiers.get(1, [])
            + shuffled_tiers.get(0, [])
            + shuffled_tiers.get(2, [])
        )

        candidates = list(individual_map.keys())  # Track using indices

        for iteration, case_idx in enumerate(ordered_cases, start=1):
            # Compute threshold using Median Absolute Deviation (MAD)
            errors = case_values[
                candidates, case_idx
            ]  # Extract candidate errors for the test case
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))  # Compute MAD
            epsilon = mad  # Automatic epsilon threshold

            # Select candidates within the epsilon range of the minimum error
            min_error = np.min(errors)
            candidates = [
                i for i in candidates if case_values[i, case_idx] <= min_error + epsilon
            ]

            # Print number of iterated cases when a single candidate remains
            if len(candidates) == 1:
                if DEBUG:
                    print(
                        f"Number of iterated cases before selection: {iteration}, difficulty_metric: {difficulty_metric}"
                    )
                break  # Stop once one candidate is left

        # Select one candidate randomly from remaining and store the individual
        selected.append(individual_map[random.choice(candidates)])

    return selected


def progressive_tier_lexicase_selection(
    population,
    num_selected,
    difficulty_metric="variance",
    generation_info=None,
):
    """
    Progressive Tier Lexicase Selection: Gradually shifts focus from easy to hard cases
    as evolution progresses.
    """
    selected = []
    case_values = np.array([ind.case_values for ind in population])

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    # Precompute difficulty tiers using K-means
    precomputed_tiers = estimate_difficulty_tiers(case_values, difficulty_metric)

    # Determine current generation ratio (0.0 to 1.0)
    generation_ratio = (
        min(1.0, generation_info.current_gen / generation_info.max_generations)
        if generation_info
        else 0.5
    )

    # Dynamic tier weights based on evolution progress
    tier_weights = {
        0: max(0.1, 0.7 - 0.6 * generation_ratio),  # Easy: 0.7 → 0.1
        1: 0.5,  # Medium: constant 0.5
        2: max(0.1, 0.1 + 0.6 * generation_ratio),  # Hard: 0.1 → 0.7
    }

    for _ in range(num_selected):
        # Create weighted tier sampling
        all_ordered_cases = []

        # Shuffle within each tier
        for tier, cases in precomputed_tiers.items():
            shuffled_cases = random.sample(cases, len(cases))
            # Calculate how many cases to take from this tier
            num_cases = max(1, int(len(shuffled_cases) * tier_weights[tier]))
            all_ordered_cases.extend(shuffled_cases[:num_cases])

        # Final shuffle of the selected cases
        random.shuffle(all_ordered_cases)

        candidates = list(individual_map.keys())

        for iteration, case_idx in enumerate(all_ordered_cases, start=1):
            # Compute threshold using MAD
            errors = case_values[candidates, case_idx]
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))
            epsilon = mad

            # Select candidates within epsilon range
            min_error = np.min(errors)
            candidates = [
                i for i in candidates if case_values[i, case_idx] <= min_error + epsilon
            ]

            if len(candidates) == 1:
                if DEBUG:
                    print(
                        f"Selected after {iteration} cases, gen_ratio: {generation_ratio}"
                    )
                break

        # Select one candidate randomly from remaining
        selected.append(individual_map[random.choice(candidates)])

    return selected


def dynamic_difficulty_lexicase_selection(
    population,
    num_selected,
    difficulty_metric="variance",
    generation_info=None,
):
    """
    Dynamic Difficulty Lexicase: Uses different sampling probabilities for the three
    difficulty tiers based on current generation.
    """
    selected = []
    case_values = np.array([ind.case_values for ind in population])

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    # Precompute difficulty tiers
    precomputed_tiers = estimate_difficulty_tiers(case_values, difficulty_metric)

    # Compute generation ratio
    gen_ratio = (
        min(1.0, generation_info.current_gen / generation_info.max_generations)
        if generation_info
        else 0.5
    )

    # Calculate sampling probabilities for each tier
    prob_easy = max(0.1, 0.8 - 0.7 * gen_ratio)  # 0.8 → 0.1
    prob_medium = 0.6  # Consistent focus on medium difficulty
    prob_hard = min(0.9, 0.2 + 0.7 * gen_ratio)  # 0.2 → 0.9

    tier_probs = {0: prob_easy, 1: prob_medium, 2: prob_hard}

    for _ in range(num_selected):
        candidates = list(individual_map.keys())
        remaining_cases = []

        # Build case order with probabilistic sampling
        for tier, cases in precomputed_tiers.items():
            for case_idx in random.sample(cases, len(cases)):
                # Only include cases with probability based on tier
                if random.random() < tier_probs[tier]:
                    remaining_cases.append(case_idx)

        # If no cases were selected, include all medium cases
        if not remaining_cases and 1 in precomputed_tiers:
            remaining_cases = precomputed_tiers[1].copy()
            random.shuffle(remaining_cases)

        for iteration, case_idx in enumerate(remaining_cases, start=1):
            # Compute adaptive epsilon using MAD
            errors = case_values[candidates, case_idx]
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))

            # Using a smaller epsilon for harder cases
            tier = next(
                (t for t, cases in precomputed_tiers.items() if case_idx in cases), 1
            )
            epsilon_factor = 1.2 if tier == 0 else (0.8 if tier == 2 else 1.0)
            epsilon = mad * epsilon_factor

            # Filter candidates
            min_error = np.min(errors)
            candidates = [
                i for i in candidates if case_values[i, case_idx] <= min_error + epsilon
            ]

            if len(candidates) == 1:
                if DEBUG:
                    print(f"Selected after {iteration} cases from tier {tier}")
                break

        selected.append(individual_map[random.choice(candidates)])

    return selected


def phase_based_curriculum_lexicase(
    population,
    num_selected,
    difficulty_metric="variance",
    generation_info=None,
    num_phases=3,
):
    """
    Phase-based Curriculum Lexicase: Divides evolution into distinct phases with
    different test case sampling strategies.
    """
    selected = []
    case_values = np.array([ind.case_values for ind in population])

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    # Precompute difficulty tiers
    precomputed_tiers = estimate_difficulty_tiers(case_values, difficulty_metric)

    # Determine current phase (0, 1, or 2) based on generation
    current_phase = 0
    if generation_info:
        phase_length = generation_info.max_generations / num_phases
        current_phase = min(
            num_phases - 1, int(generation_info.current_gen / phase_length)
        )

    # Define phase-specific sampling strategies
    sampling_strategies = [
        # Phase 0: Easy (70%), Medium (30%), Hard (0%)
        {0: 0.7, 1: 0.3, 2: 0.0},
        # Phase 1: Easy (30%), Medium (50%), Hard (20%)
        {0: 0.3, 1: 0.5, 2: 0.2},
        # Phase 2: Easy (10%), Medium (40%), Hard (50%)
        {0: 0.1, 1: 0.4, 2: 0.5},
    ]

    # Get current sampling probabilities
    tier_probs = sampling_strategies[current_phase]

    for _ in range(num_selected):
        ordered_cases = []

        # Sample cases according to current phase's probabilities
        for tier, cases in precomputed_tiers.items():
            if tier_probs[tier] > 0:
                tier_cases = random.sample(cases, len(cases))
                num_to_sample = max(1, int(len(tier_cases) * tier_probs[tier]))
                ordered_cases.extend(tier_cases[:num_to_sample])

        random.shuffle(ordered_cases)
        candidates = list(individual_map.keys())

        for iteration, case_idx in enumerate(ordered_cases, start=1):
            # Compute threshold using MAD
            errors = case_values[candidates, case_idx]
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))
            epsilon = mad

            # Filter candidates
            min_error = np.min(errors)
            candidates = [
                i for i in candidates if case_values[i, case_idx] <= min_error + epsilon
            ]

            if len(candidates) == 1:
                if DEBUG:
                    print(f"Selected after {iteration} cases in phase {current_phase}")
                break

        selected.append(individual_map[random.choice(candidates)])

    return selected


def adaptive_epsilon_curriculum_lexicase(
    population,
    num_selected,
    difficulty_metric="variance",
    generation_info=None,
):
    """
    Adaptive Epsilon Curriculum Lexicase: Adjusts epsilon value based on both
    difficulty tier and evolution progress.
    """
    selected = []
    case_values = np.array([ind.case_values for ind in population])

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    # Precompute difficulty tiers using K-means
    precomputed_tiers = estimate_difficulty_tiers(case_values, difficulty_metric)

    # Determine curriculum progress (0.0 to 1.0)
    progress = (
        min(1.0, generation_info.current_gen / generation_info.max_generations)
        if generation_info
        else 0.5
    )

    # Adjust tier ordering based on progress
    if progress < 0.3:
        # Early: Focus on easy, then medium, minimal hard cases
        tier_order = [0, 1, 2]
        tier_weights = {0: 0.6, 1: 0.3, 2: 0.1}
    elif progress < 0.7:
        # Middle: Focus on medium, then mix of easy and hard
        tier_order = [1, 0, 2]
        tier_weights = {0: 0.3, 1: 0.5, 2: 0.2}
    else:
        # Late: Focus on hard and medium, fewer easy cases
        tier_order = [1, 2, 0]
        tier_weights = {0: 0.1, 1: 0.4, 2: 0.5}

    # Dynamic epsilon adjustment based on progress
    # Early: larger epsilon, Late: smaller epsilon
    epsilon_scale = 1.5 - progress

    for _ in range(num_selected):
        ordered_cases = []

        # Build case order based on current progress
        for tier in tier_order:
            if tier in precomputed_tiers:
                tier_cases = random.sample(
                    precomputed_tiers[tier], len(precomputed_tiers[tier])
                )
                sample_size = max(1, int(len(tier_cases) * tier_weights[tier]))
                ordered_cases.extend(tier_cases[:sample_size])

        candidates = list(individual_map.keys())

        for iteration, case_idx in enumerate(ordered_cases, start=1):
            errors = case_values[candidates, case_idx]
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))

            # Find which tier this case belongs to
            case_tier = next(
                (t for t, cases in precomputed_tiers.items() if case_idx in cases), 1
            )

            # Adjust epsilon based on tier and progress
            if case_tier == 0:  # Easy
                tier_factor = 1.2  # Larger epsilon for easy cases
            elif case_tier == 2:  # Hard
                tier_factor = 0.8  # Smaller epsilon for hard cases
            else:  # Medium
                tier_factor = 1.0

            # Final epsilon calculation
            epsilon = mad * tier_factor * epsilon_scale

            # Filter candidates
            min_error = np.min(errors)
            candidates = [
                i for i in candidates if case_values[i, case_idx] <= min_error + epsilon
            ]

            if len(candidates) == 1:
                if DEBUG:
                    print(f"Selected after {iteration} cases, progress: {progress:.2f}")
                break

        selected.append(individual_map[random.choice(candidates)])

    return selected


def success_adaptive_curriculum_lexicase(
    population,
    num_selected,
    difficulty_metric="variance",
    generation_info=None,
    success_history=None,
):
    """
    Success-Adaptive Curriculum Lexicase: Adapts sampling based on which difficulty
    tiers have been most successful in producing good solutions.
    """
    if success_history is None:
        # Initialize with equal success counts for each tier
        success_history = {0: 10, 1: 10, 2: 10}

    selected = []
    case_values = np.array([ind.case_values for ind in population])

    # Assign unique IDs to individuals for tracking
    individual_map = {i: ind for i, ind in enumerate(population)}

    # Precompute difficulty tiers
    precomputed_tiers = estimate_difficulty_tiers(case_values, difficulty_metric)

    # Compute generation phase (early, middle, late)
    gen_phase = "early"
    if generation_info:
        if generation_info.current_gen > 0.7 * generation_info.max_generations:
            gen_phase = "late"
        elif generation_info.current_gen > 0.3 * generation_info.max_generations:
            gen_phase = "middle"

    # Calculate adaptive weights based on success history and current phase
    total_success = sum(success_history.values())
    base_weights = {
        0: success_history[0] / total_success,
        1: success_history[1] / total_success,
        2: success_history[2] / total_success,
    }

    # Apply phase-specific adjustments
    if gen_phase == "early":
        # Boost easy and medium cases in early phase
        tier_weights = {
            0: 0.5 + 0.5 * base_weights[0],
            1: 0.3 + 0.7 * base_weights[1],
            2: 0.1 * base_weights[2],
        }
    elif gen_phase == "middle":
        # Balance with slight preference for historically successful tiers
        tier_weights = {
            0: 0.3 + 0.7 * base_weights[0],
            1: 0.4 + 0.6 * base_weights[1],
            2: 0.2 + 0.8 * base_weights[2],
        }
    else:  # late phase
        # Emphasize hard and historically successful tiers
        tier_weights = {
            0: 0.1 + 0.9 * base_weights[0],
            1: 0.3 + 0.7 * base_weights[1],
            2: 0.6 + 0.4 * base_weights[2],
        }

    # Normalize weights
    weight_sum = sum(tier_weights.values())
    tier_weights = {k: v / weight_sum for k, v in tier_weights.items()}

    # Track which tiers contributed to selection for updating success history
    tier_contributions = {0: 0, 1: 0, 2: 0}

    for _ in range(num_selected):
        ordered_cases = []
        tier_usage = {0: [], 1: [], 2: []}  # Track which cases from each tier were used

        # Sample cases based on adaptive weights
        for tier, cases in precomputed_tiers.items():
            if tier_weights.get(tier, 0) > 0:
                shuffled_cases = random.sample(cases, len(cases))
                sample_size = max(1, int(len(shuffled_cases) * tier_weights[tier]))
                tier_sample = shuffled_cases[:sample_size]
                ordered_cases.extend(tier_sample)
                tier_usage[tier] = tier_sample

        random.shuffle(ordered_cases)
        candidates = list(individual_map.keys())
        case_usage = set()  # Track which cases actually contributed to filtering

        for iteration, case_idx in enumerate(ordered_cases, start=1):
            case_usage.add(case_idx)

            # Apply epsilon lexicase filtering
            errors = case_values[candidates, case_idx]
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))
            epsilon = mad

            # Filter candidates
            min_error = np.min(errors)
            new_candidates = [
                i for i in candidates if case_values[i, case_idx] <= min_error + epsilon
            ]

            # Only count as contributing if it actually reduced the candidate pool
            if len(new_candidates) < len(candidates):
                candidates = new_candidates

                # Find which tier this case belongs to and increment counter
                for tier, tier_cases in tier_usage.items():
                    if case_idx in tier_cases:
                        tier_contributions[tier] += 1
                        break

            if len(candidates) == 1:
                if DEBUG:
                    print(f"Selected after {iteration} cases in {gen_phase} phase")
                break

        selected.append(individual_map[random.choice(candidates)])

    # Update success history based on which tiers contributed to selection
    for tier, count in tier_contributions.items():
        success_history[tier] = 0.9 * success_history[tier] + 0.1 * count

    return selected, success_history


if __name__ == "__main__":
    # Define a simple individual structure
    class Individual:
        def __init__(self, id, case_values):
            self.id = id  # Unique identifier for tracking
            self.case_values = case_values  # List of test case errors

        def __repr__(self):
            return f"Ind{self.id}: {self.case_values}"

    test_cases = ["T1", "T2", "T3", "T4", "T5"]
    population = [
        Individual(i, [random.randint(0, 10) for _ in test_cases]) for i in range(10)
    ]

    # Convert case_values into a 2D NumPy array
    case_values = np.array([ind.case_values for ind in population])

    # Apply Automatic Epsilon Lexicase Selection
    selected_individuals = progressive_tier_lexicase_selection(
        population,
        num_selected=3,
        generation_info=GenerationInfo(current_gen=10, max_generations=100),
    )

    # Display results
    df_population = pd.DataFrame(
        [
            {
                "ID": ind.id,
                **{f"Test {j + 1}": val for j, val in enumerate(ind.case_values)},
            }
            for ind in population
        ]
    )
    df_selected = pd.DataFrame(
        [
            {
                "ID": ind.id,
                **{f"Test {j + 1}": val for j, val in enumerate(ind.case_values)},
            }
            for ind in selected_individuals
        ]
    )

    print("Population:")
    print(df_population)
    print("\nSelected Individuals:")
    print(df_selected)
