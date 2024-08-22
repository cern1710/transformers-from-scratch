import multihead_attention as mult
from torch import nn
import torch
import layer_norm as ln

class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, dropout=0.0):
        super().__init__()

        # Attention layer
        self.self_attention = mult.MultiHeadAttention(input_dim, input_dim, num_heads)

        # 2-Layer MLP
        self.linear_MLP = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, input_dim)
        )

        # Layers between main ones
        self.norm1 = ln.LayerNorm(input_dim)
        self.norm2 = ln.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Add residual connections to self-attention and MLP layers
        attn_output, _ = self.self_attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        linear_output = self.linear_MLP(x)
        x = self.norm2(x + self.dropout(linear_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, **block_args):
        super().__init__()
        self.layers =nn.ModuleList([EncoderBlock(**block_args)
                                    for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x: torch.Tensor, mask: torch.Tensor = None):
        attn_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attention(x, x, x, mask=mask)
            attn_maps.append(attn_map)
            x = layer(x)
        return attn_maps