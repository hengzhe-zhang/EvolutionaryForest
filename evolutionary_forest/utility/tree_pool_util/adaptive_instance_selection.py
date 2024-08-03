import numpy as np


def calculate_hardness(loss_matrix):
    return np.median(loss_matrix, axis=0)


def adaptive_hard_easy_selection(loss_matrix, selection_number):
    num_instances = loss_matrix.shape[1]
    hardness = calculate_hardness(loss_matrix)
    mean_loss = np.mean(loss_matrix)

    if mean_loss > np.median(hardness):
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


def adaptive_selection_strategy_controller(
    strategy,
    loss_matrix,
    instances,
    prev_loss_matrix=None,
    selection_number=10,
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
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    selected_instances = [instances[i] for i in selected_indices]
    return selected_instances
