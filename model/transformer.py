import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_layers, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
