import torch
import self_attention
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

        self.QVK_proj = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.O_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._reset_params()

    def _reset_params(self):
        """Initialize weights and biases for projection layers."""
        nn.init.xavier_uniform_(self.QVK_proj.weight)
        nn.init.xavier_uniform_(self.O_proj.weight)

    def forward(self, x: torch.Tensor, mask: bool = False):
        """Computes Multi-Head attention by combining outputs of each head."""
        batch_size, seq_length, _ = x.size()
        QKV = self.QVK_proj(x)
        QKV = QKV.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        QKV = QKV.permute(0, 2, 1, 3)   # batch, head, seqlen, dim
        Q, K, V = QKV.chunk(3, dim=-1)

        attn_output, attn_weights = self_attention.scaled_dot_product(Q, K, V, mask=mask)
        attn_output = attn_output.permute(0, 2, 1, 3) # batch, seqlen, head, dim
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_dim)
        O = self.O_proj(attn_output)

        return O, attn_weights