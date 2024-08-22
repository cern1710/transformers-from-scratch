import multihead_attention as mult
from torch import nn
import torch

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, dropout=0.0):
        super().__init__()

        # Self-attention layer
        self.self_attention = mult.MultiHeadAttention(input_dim, input_dim, num_heads)

        # Cross-attention layer
        self.cross_attention = mult.MultiHeadAttention(input_dim, input_dim, num_heads)

        # 2-Layer MLP
        self.linear_MLP = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, input_dim)
        )

        # Layers between main ones
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None):
        attn_output, _ = self.self_attention(x, x, x, mask=self_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention layer, where KV are encoder's output
        attn_output, _ = self.cross_attention(x, encoder_output,
                                           encoder_output, mask=cross_mask)
        x = self.norm2(x + self.dropout(attn_output))

        linear_output = self.linear_MLP(x)
        x = self.norm3(x + self.dropout(linear_output))
        return x