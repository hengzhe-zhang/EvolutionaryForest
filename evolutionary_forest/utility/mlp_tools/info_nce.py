import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: tensor of shape (batch_size, embedding_dim) - embeddings from the first modality (e.g., symbolic)
        z_j: tensor of shape (batch_size, embedding_dim) - embeddings from the second modality (e.g., numeric)
        """
        batch_size = z_i.shape[0]
        z_j = z_j.mean(dim=1)

        # Normalize embeddings to have unit norm (L2 normalization)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Positive similarity (dot product between corresponding pairs)
        positive_sim = torch.sum(z_i * z_j, dim=1) / self.temperature

        # Compute the full similarity matrix (dot products between all pairs in the batch)
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        # For each row, apply softmax over all possible pairs (i.e., normalize)
        loss_i_to_j = -torch.log(
            torch.exp(positive_sim) / torch.exp(sim_matrix).sum(dim=1)
        )
        loss_j_to_i = -torch.log(
            torch.exp(positive_sim) / torch.exp(sim_matrix.T).sum(dim=1)
        )

        # Take the mean of the loss across the batch
        loss = (loss_i_to_j.mean() + loss_j_to_i.mean()) / 2
        return loss


if __name__ == "__main__":
    # Example usage
    batch_size = 10
    embedding_dim = 128

    # Randomly generated example embeddings for two modalities (symbolic and numeric)
    z_i = torch.randn(batch_size, embedding_dim)  # Embeddings from symbolic encoder
    z_j = torch.randn(batch_size, embedding_dim)  # Embeddings from numeric encoder

    # Instantiate and compute InfoNCE loss
    info_nce_loss = InfoNCELoss(temperature=0.1)
    loss = info_nce_loss(z_i, z_j)

    print(f"InfoNCE Loss: {loss.item()}")
