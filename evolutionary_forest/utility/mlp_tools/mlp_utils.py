import torch


def get_max_length_excluding_padding(batch: torch.Tensor, pad_value=0):
    """
    Get the maximum length of sequences in the batch, excluding padding.

    Args:
        batch (list of torch.Tensor): List of sequences in the mini-batch.
        pad_value (int, optional): Value used for padding. Defaults to 0.

    Returns:
        int: Maximum sequence length excluding padding in the batch.
    """

    def get_effective_length(seq):
        # Find the length of the sequence excluding padding
        return (seq != pad_value).sum().item()

    # Compute the maximum length of non-padding parts of sequences
    return max(get_effective_length(seq) for seq in batch)
