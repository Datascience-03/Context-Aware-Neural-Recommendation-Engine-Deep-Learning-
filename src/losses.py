import torch
import torch.nn as nn
import torch.nn.functional as F

class InBatchInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        q_norm = F.normalize(query_embeddings, p=2, dim=-1)
        k_norm = F.normalize(item_embeddings, p=2, dim=-1)
        similarity_matrix = torch.matmul(q_norm, k_norm.T) / self.temperature
        labels = torch.arange(query_embeddings.size(0), device=query_embeddings.device)
        return self.cross_entropy(similarity_matrix, labels)
