import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings= vocab_size,
                                      embedding_dim= emb_dim)
        
    def forward(self, x: torch.Tensor):
        return self.embedding(x)