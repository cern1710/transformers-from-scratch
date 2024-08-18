import torch
from torch import nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # [seqlen, hidden_dim] represents the positional encoding matrix
        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        scaling_factor = torch.exp(torch.arange(0, d_model, 2).float() *
                                   (-np.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(position * scaling_factor)
        PE[:, 1::2] = torch.cos(position * scaling_factor)
        PE = PE.unsqueeze(0)

        # Register PE buffer without saving it in the state dict
        self.register_buffer('PE', PE, persistent=False)

    def forward(self, x):
        x = x + self.PE[:, : x.size(1)]
        return x