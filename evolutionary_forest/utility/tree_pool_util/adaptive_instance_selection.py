import numpy as np


def calculate_hardness(loss_matrix):
    return np.median(loss_matrix, axis=0)


def adaptive_hard_easy_selection(loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)
    mean_loss = np.mean(loss_matrix)
    median = np.median(hardness)
    if mean_loss > median:
        # Focus on harder instances
        selected_indices = np.argsort(hardness)[-selection_number:]
    else:
        # Focus on easier instances
        selected_indices = np.argsort(hardness)[:selection_number]

    return selected_indices


def adaptive_weighted_selection(loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)
    mean_loss = np.mean(loss_matrix)

    weight = mean_loss / (mean_loss + np.median(hardness))
    num_hard_instances = int(selection_number * weight)
    num_easy_instances = selection_number - num_hard_instances

    hard_indices = np.argsort(hardness)[-num_hard_instances:]
    easy_indices = np.argsort(hardness)[:num_easy_instances]

    selected_indices = np.concatenate((hard_indices, easy_indices))
    return selected_indices


def adaptive_progress_selection(loss_matrix, prev_loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]

    if prev_loss_matrix is None:
        selected_indices = np.random.choice(
            num_instances, selection_number, replace=False
        )
    else:
        improvement = np.mean(prev_loss_matrix - loss_matrix, axis=0)
        if np.mean(improvement) > 0:
            # Focus on instances with the least improvement
            selected_indices = np.argsort(improvement)[:selection_number]
        else:
            # Focus on instances with the most improvement
            selected_indices = np.argsort(improvement)[-selection_number:]

    return selected_indices


def adaptive_stability_selection(loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]
    stability = np.std(loss_matrix, axis=0)

    if np.mean(stability) > np.median(stability):
        # Focus on the least stable instances
        selected_indices = np.argsort(stability)[-selection_number:]
    else:
        # Focus on the most stable instances
        selected_indices = np.argsort(stability)[:selection_number]

    return selected_indices


def adaptive_ensemble_selection(loss_matrix, prev_loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]

    if prev_loss_matrix is None:
        selected_indices = np.random.choice(
            num_instances, selection_number, replace=False
        )
    else:
        hardness = calculate_hardness(loss_matrix)
        stability = np.std(loss_matrix, axis=0)
        improvement = np.mean(prev_loss_matrix - loss_matrix, axis=0)

        combined_score = hardness + stability - improvement

        selected_indices = np.argsort(combined_score)[-selection_number:]

    return selected_indices


def adaptive_diversity_selection(loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]
    diversity = np.var(loss_matrix, axis=0)  # Measure diversity as variance in losses

    if np.mean(diversity) > np.median(diversity):
        # Focus on more diverse instances
        selected_indices = np.argsort(diversity)[-selection_number:]
    else:
        # Focus on less diverse instances
        selected_indices = np.argsort(diversity)[:selection_number]

    return selected_indices


def adaptive_mixed_hardness_selection(loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)

    num_half = selection_number // 2
    hard_indices = np.argsort(hardness)[-num_half:]
    easy_indices = np.argsort(hardness)[: selection_number - num_half]

    selected_indices = np.concatenate((hard_indices, easy_indices))
    return selected_indices


def adaptive_confidence_selection(loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]
    mean_losses = np.mean(loss_matrix, axis=0)

    if np.mean(mean_losses) > np.median(mean_losses):
        # Focus on least confident (highest loss) instances
        selected_indices = np.argsort(mean_losses)[-selection_number:]
    else:
        # Focus on most confident (lowest loss) instances
        selected_indices = np.argsort(mean_losses)[:selection_number]

    return selected_indices


def adaptive_temporal_stability_selection(
    loss_matrix, prev_loss_matrix, selection_number
):
    num_instances = loss_matrix.shape[1]

    if prev_loss_matrix is None:
        selected_indices = np.random.choice(
            num_instances, selection_number, replace=False
        )
    else:
        temporal_stability = np.abs(
            np.mean(loss_matrix, axis=0) - np.mean(prev_loss_matrix, axis=0)
        )
        if np.mean(temporal_stability) > np.median(temporal_stability):
            # Focus on instances with the most change
            selected_indices = np.argsort(temporal_stability)[-selection_number:]
        else:
            # Focus on instances with the least change
            selected_indices = np.argsort(temporal_stability)[:selection_number]

    return selected_indices


def adaptive_hybrid_selection(loss_matrix, prev_loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]

    if prev_loss_matrix is None:
        selected_indices = np.random.choice(
            num_instances, selection_number, replace=False
        )
    else:
        hardness = calculate_hardness(loss_matrix)
        stability = np.std(loss_matrix, axis=0)
        diversity = np.var(loss_matrix, axis=0)

        combined_score = hardness + stability + diversity
        selected_indices = np.argsort(combined_score)[-selection_number:]

    return selected_indices


def curriculum_progression_selection(
    loss_matrix, current_generation, total_generations, selection_number
):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)

    progression_ratio = current_generation / total_generations

    num_easy_instances = int(selection_number * (1 - progression_ratio))
    num_hard_instances = selection_number - num_easy_instances

    easy_indices = (
        np.argsort(hardness)[:num_easy_instances]
        if num_easy_instances > 0
        else np.array([], dtype=int)
    )
    hard_indices = (
        np.argsort(hardness)[-num_hard_instances:]
        if num_hard_instances > 0
        else np.array([], dtype=int)
    )

    selected_indices = np.concatenate((easy_indices, hard_indices))
    return selected_indices


def curriculum_threshold_selection(
    loss_matrix, current_generation, total_generations, selection_number
):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)

    # Adjust the threshold over generations to include harder instances
    threshold = np.percentile(hardness, current_generation / total_generations * 100)
    selected_indices = np.where(hardness <= threshold)[0]

    if len(selected_indices) > selection_number:
        selected_indices = selected_indices[:selection_number]
    else:
        additional_indices = np.setdiff1d(np.arange(num_instances), selected_indices)
        selected_indices = np.concatenate(
            (
                selected_indices,
                additional_indices[: selection_number - len(selected_indices)],
            )
        )

    return selected_indices


def adaptive_curriculum_selection(
    loss_matrix, current_generation, total_generations, selection_number
):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)
    mean_loss = np.mean(loss_matrix)
    median_hardness = np.median(hardness)

    # Start with easy instances and progressively include harder ones based on performance
    progression_ratio = current_generation / total_generations

    if mean_loss <= median_hardness:
        # If performing well, include more hard instances
        num_hard_instances = int(selection_number * progression_ratio)
        num_easy_instances = selection_number - num_hard_instances
    else:
        # If not performing well, include more easy instances
        num_easy_instances = int(selection_number * (1 - progression_ratio))
        num_hard_instances = selection_number - num_easy_instances

    easy_indices = (
        np.argsort(hardness)[:num_easy_instances]
        if num_easy_instances > 0
        else np.array([], dtype=int)
    )
    hard_indices = (
        np.argsort(hardness)[-num_hard_instances:]
        if num_hard_instances > 0
        else np.array([], dtype=int)
    )

    selected_indices = np.concatenate((easy_indices, hard_indices))
    return selected_indices


def curriculum_mixing_selection(
    loss_matrix, current_generation, total_generations, selection_number
):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)

    # Proportion of hard instances increases over generations
    proportion_hard = current_generation / total_generations
    num_hard_instances = int(selection_number * proportion_hard)
    num_easy_instances = selection_number - num_hard_instances

    easy_indices = (
        np.argsort(hardness)[:num_easy_instances]
        if num_easy_instances > 0
        else np.array([], dtype=int)
    )
    hard_indices = (
        np.argsort(hardness)[-num_hard_instances:]
        if num_hard_instances > 0
        else np.array([], dtype=int)
    )

    selected_indices = np.concatenate((easy_indices, hard_indices))
    return selected_indices


from sklearn.cluster import KMeans


def curriculum_clustering_selection(
    loss_matrix, current_generation, total_generations, selection_number, num_clusters=3
):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)

    # Cluster instances based on hardness
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(hardness.reshape(-1, 1))

    # Sort clusters by their mean hardness
    cluster_hardness = [np.mean(hardness[clusters == i]) for i in range(num_clusters)]
    sorted_clusters = np.argsort(cluster_hardness)

    # Determine which clusters to select from based on the progression
    current_cluster = int(current_generation / total_generations * num_clusters)
    selected_clusters = sorted_clusters[: current_cluster + 1]

    selected_indices = np.hstack(
        [np.where(clusters == cluster)[0] for cluster in selected_clusters]
    )

    if len(selected_indices) > selection_number:
        selected_indices = selected_indices[:selection_number]
    else:
        additional_indices = np.setdiff1d(np.arange(num_instances), selected_indices)
        selected_indices = np.concatenate(
            (
                selected_indices,
                additional_indices[: selection_number - len(selected_indices)],
            )
        )

    return selected_indices


def adaptive_selection_strategy_controller(
    strategy,
    loss_matrix,
    instances,
    prev_loss_matrix=None,
    selection_number=10,
    current_generation=0,
    total_generations=0,
):
    if strategy == "adaptive_hard_easy":
        selected_indices = adaptive_hard_easy_selection(loss_matrix, selection_number)
    elif strategy == "adaptive_weighted":
        selected_indices = adaptive_weighted_selection(loss_matrix, selection_number)
    elif strategy == "adaptive_progress":
        selected_indices = adaptive_progress_selection(
            loss_matrix, prev_loss_matrix, selection_number
        )
    elif strategy == "adaptive_stability":
        selected_indices = adaptive_stability_selection(loss_matrix, selection_number)
    elif strategy == "adaptive_ensemble":
        selected_indices = adaptive_ensemble_selection(
            loss_matrix, prev_loss_matrix, selection_number
        )
    elif strategy == "adaptive_diversity":
        selected_indices = adaptive_diversity_selection(loss_matrix, selection_number)
    elif strategy == "adaptive_mixed_hardness":
        selected_indices = adaptive_mixed_hardness_selection(
            loss_matrix, selection_number
        )
    elif strategy == "adaptive_confidence":
        selected_indices = adaptive_confidence_selection(loss_matrix, selection_number)
    elif strategy == "adaptive_temporal_stability":
        selected_indices = adaptive_temporal_stability_selection(
            loss_matrix, prev_loss_matrix, selection_number
        )
    elif strategy == "adaptive_hybrid":
        selected_indices = adaptive_hybrid_selection(
            loss_matrix, prev_loss_matrix, selection_number
        )
    elif strategy == "curriculum_progression":
        selected_indices = curriculum_progression_selection(
            loss_matrix, current_generation, total_generations, selection_number
        )
    elif strategy == "curriculum_threshold":
        selected_indices = curriculum_threshold_selection(
            loss_matrix, current_generation, total_generations, selection_number
        )
    elif strategy == "adaptive_curriculum":
        selected_indices = adaptive_curriculum_selection(
            loss_matrix, current_generation, total_generations, selection_number
        )
    elif strategy == "curriculum_mixing":
        selected_indices = curriculum_mixing_selection(
            loss_matrix, current_generation, total_generations, selection_number
        )
    elif strategy == "curriculum_clustering":
        selected_indices = curriculum_clustering_selection(
            loss_matrix, current_generation, total_generations, selection_number
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    selected_instances = [instances[i] for i in selected_indices]
    return selected_instances
