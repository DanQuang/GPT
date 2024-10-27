import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, x: torch.Tensor):
        # x shape [bs, num_tokens, emb_dim]
        return self.layers(x)