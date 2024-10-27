import torch
import torch.nn as nn


class MultiHeadAttn(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttn, self).__init__()
        assert cfg['emb_dim'] % cfg['n_heads'] == 0, "d_out must be divisible by"

        self.d_head = cfg['emb_dim'] // cfg['n_heads']
        self.num_heads = cfg['n_heads']

        self.W_q = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias= cfg['qkv_bias'])
        self.W_k = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias= cfg['qkv_bias'])
        self.W_v = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias= cfg['qkv_bias'])

        self.out_proj = nn.Linear(cfg['emb_dim'], cfg['emb_dim'])
        self.dropout = nn.Dropout(p= cfg['dropout_rate'])
        self.register_buffer('mask', torch.triu(torch.ones(cfg['context_length'], cfg['context_length']), diagonal=1))

    def forward(self, x: torch.Tensor):

        b, num_tokens, d_in = x.shape
        # Shape: (batch_size, num_tokens, d_out)
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)

        # Shape: (b, n_heads, num_tokens, d_head)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # Shape: (batch_size, n_heads, num_tokens, num_tokens)
        attn_scores = q @ k.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / q.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_head*self.num_heads)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
        

    def split(self, x: torch.Tensor):
        b, num_tokens, d_out = x.shape
        # (b, num_tokens, d_out) -> (b, n_heads, num_tokens, d_head)
        x = x.view(b, num_tokens, self.d_head, self.num_heads).transpose(1, 2)
        return x