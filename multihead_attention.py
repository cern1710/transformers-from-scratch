import torch
import self_attention as self_attention
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        """Multi-Head Attention layer constructor method.

        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of embedding space.
            num_heads (int): Number of attention heads.

        Attributes:
            hidden_dim (int): Hidden dimension.
            num_heads (int): Number of attention heads.
            head_dim (int): Dimensionality of each attention head.
            QVK_proj (nn.Linear): Linear layer projecting input features
                                  into QVK.
            O_proj (nn.Linear): Linear layer projecting concatenated outputs
                                of all attention heads to the hidden dimension.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.QVK_proj = nn.Linear(input_dim, 3 * hidden_dim)
        self.O_proj = nn.Linear(hidden_dim, hidden_dim)

        self._reset_params()

    def _reset_params(self):
        """Initialize weights and biases for projection layers."""
        nn.init.xavier_uniform_(self.QVK_proj.weight)
        self.QVK_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.O_proj.weight)
        self.O_proj.bias.data.fill_(0)

    def forward(self, x):
        return

