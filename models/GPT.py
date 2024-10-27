import torch
from torch import nn
from blocks.transformers_block import TransformersBlock
from embedding.gpt_embedding import GPTEmbedding
from layers.layer_norm import LayerNorm

GPT_CONFIG_SMALL = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "dropout_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 1024,         # Embedding dimension
    "n_heads": 16,          # Number of attention heads
    "n_layers": 24,         # Number of layers
    "dropout_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

GPT_CONFIG_LARGE = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1280, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 20,          # Number of attention heads
    "n_layers": 36,         # Number of layers
    "dropout_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class GPTModel(nn.Module):
    def __init__(self, model_name: str | None = None):
        super(GPTModel, self).__init__()
        if model_name == None or model_name == "small":
            self.cfg = GPT_CONFIG_SMALL
        elif model_name == "medium":
            self.cfg = GPT_CONFIG_MEDIUM
        elif model_name == "large":
            self.cfg = GPT_CONFIG_LARGE
        else:
            raise Exception("model_name GPT must be in (small, medium, large). None default to small.")
        
        self.embedding = GPTEmbedding(cfg= self.cfg)
        self.blocks = nn.Sequential(*[TransformersBlock(cfg=self.cfg) for _ in range(self.cfg['n_layers'])])
        
        self.final_norm = LayerNorm(emb_dim= self.cfg['emb_dim'])
        self.out_head = nn.Linear(in_features= self.cfg['emb_dim'],
                                  out_features= self.cfg['vocab_size'],
                                  bias= False)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, in_idx: torch.Tensor, labels: torch.Tensor = None):
        x = self.embedding(in_idx)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        # x shape: [bs, num_tokens, vocab_size]
        # labels shpe: [bs, num_tokens]
        if labels is not None:
            loss = self.loss_fn(logits.flatten(0, 1), labels.flatten())
            return logits, loss
        return logits