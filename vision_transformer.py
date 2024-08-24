import torch
from torch import nn, optim
import multihead_attention as mult
import layer_norm as ln
import positional_encoding as pe
import scheduler as sch

class VisionTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_heads: int, num_layers: int, num_channels: int,
                 patch_size: int, num_patches: int, learning_rate: float,
                 dropout: float = 0.0, warmup: int = 50, max_iter: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.max_iter = max_iter

        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = pe.PositionalEncoding(
            input_dim, max_len=num_patches+1
        )

        self._init_model(dropout)

    def _init_model(self, dropout: float):
        self.input_network = nn.Linear(
            self.num_channels * (self.patch_size**2), self.input_dim
        )
        self.transformer = nn.Sequential(
            *[mult.ViTAttention(self.input_dim, self.hidden_dim,
                                self.num_heads, dropout)
                                for _ in range(self.num_layers)])
        self.output_network = nn.Sequential(
            ln.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.num_classes)
        )

    def _split_image_to_patches(self, x: torch.Tensor):
        # batch size, channel num, (H, W) = resolution of original img
        B, C, H, W = x.shape
        P = self.patch_size

        x = x.reshape(B, C, (H // P), P, (W // P), P)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2)
        x = x.flatten(2, 4)

        return x

    def forward(self, x: torch.Tensor):
        x = self._split_image_to_patches(x)
        B, _, _ = x.shape
        x = self.input_network(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_embedding(x)

        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        cls = x[0]
        x = self.output_network(cls)
        return x

    def configure_optimizer(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = sch.CosineWarmupScheduler(
            optimizer=optimizer,
            warmup=self.warmup,
            max_iter=self.max_iter
            # milestones=[100, 150],
            # gamma=0.1
        )
        return [optimizer], [lr_scheduler]