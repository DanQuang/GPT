import torch
from torch import nn
from layers.transformers_layers import MultiHeadAttn, LayerNorm, FeedForward

class TransformersBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformersBlock, self).__init__()

        self.attn = MultiHeadAttn(cfg= cfg)
        self.ff = FeedForward(emb_dim= cfg['emb_dim'])
        self.norm1 = LayerNorm(emb_dim= cfg['emb_dim'])
        self.norm2 = LayerNorm(emb_dim= cfg['emb_dim'])
        self.dropout = nn.Dropout(p= cfg['dropout_rate'])

    def forward(self, x: torch.Tensor):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x