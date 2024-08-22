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

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(**block_args)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(block_args['input_dim'])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, encoder_output,
                      self_mask=self_mask, cross_mask=cross_mask)
        return self.norm(x)

    def get_attention_maps(self, x: torch.Tensor, encoder_output: torch.Tensor,
                           self_mask: torch.Tensor = None,
                           cross_mask: torch.Tensor = None):
        attn_maps = []
        for layer in self.layers:
            self_attn_output, self_attn_map = layer.self_attention(
                x, x, x, mask=self_mask
            )
            x = layer.norm1(x + layer.dropout(self_attn_output))
            cross_attn_output, cross_attn_map = layer.cross_attention(
                x, encoder_output, encoder_output, mask=cross_mask
            )
            x = layer.norm2(x + layer.dropout(cross_attn_output))
            x = layer.norm3(x + layer.dropout(layer.linear_MLP(x)))
            attn_maps.append({
                'self_attn_map': self_attn_map,
                'cross_attn_map': cross_attn_map
            })
        return attn_maps