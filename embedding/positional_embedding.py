import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 context_length: int,
                 emb_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings= context_length,
                                      embedding_dim= emb_dim)
        
    def forward(self, x: torch.Tensor):
        bs, seq_len = x.shape
        # torch.arange creates a array pos number like [0, 1, 2, 3, etc.]
        return self.embedding(torch.arange(seq_len, device= x.device))