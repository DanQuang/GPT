import torch
import torch.nn as nn
from embedding.positional_embedding import PositionalEmbedding
from embedding.token_embedding import TokenEmbedding


class GPTEmbedding(nn.Module):
    def __init__(self, cfg):
        super(GPTEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size= cfg['vocab_size'],
                                      emb_dim= cfg['emb_dim'])
        self.pos_emb = PositionalEmbedding(context_length= cfg['context_length'],
                                      emb_dim= cfg['emb_dim'])
        self.dropout = nn.Dropout(p= cfg['dropout_rate'])
        
    def forward(self, x: torch.Tensor):
        token_emb = self.tok_emb(x)
        position_emb = self.pos_emb(x)
        x = token_emb + position_emb
        x = self.dropout(x)
        return x