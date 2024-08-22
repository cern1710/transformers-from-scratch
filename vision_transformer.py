import torch
from torch import nn
import multihead_attention as mult
import layer_norm as ln
import positional_encoding as pos

class VisionTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_heads: int, num_layers: int, num_channels: int,
                 patch_size: int, num_patches: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.dropout = nn.DropOut(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = pos.PositionalEncoding(
            input_dim, max_len=num_patches+1
        )
        self._init_model()

    def _init_model(self):
        self.input_network = nn.Linear(
            self.num_channels * (self.patch_size**2), self.input_dim
        )
        self.transformer = nn.Sequential(
            *[mult.ViTAttention(self.input_dim, self.hidden_dim,
                                self.num_heads, dropout=self.dropout)
                                for _ in range(self.num_layers)])
        self.output_network = nn.Sequential(
            ln.LayerNorm(self.input_dim),
            ln.Linear(self.input_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor):
        return x