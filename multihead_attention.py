import torch
import self_attention
import layer_norm as ln
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_heads: int, dropout: float = 0.0):
        """Multi-Head Attention layer constructor method.

        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of embedding space.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate. Default to 0.0.

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

        self.Q_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.KV_proj = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self.O_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._reset_params()

    def _reset_params(self):
        """Initialize weights and biases for projection layers."""
        nn.init.xavier_uniform_(self.Q_proj.weight)
        nn.init.xavier_uniform_(self.KV_proj.weight)
        nn.init.xavier_uniform_(self.O_proj.weight)

    def forward(self, Q: torch.Tensor,K: torch.Tensor,
                V: torch.Tensor, mask: torch.Tensor = None):
        """Computes Multi-Head attention by combining outputs of each head."""
        batch_size, seq_length, _ = Q.size()

        Q = self.Q_proj(Q)
        KV = self.KV_proj(K)
        K, V = KV.chunk(2, dim=-1)

        Q = Q.reshape(batch_size, seq_length, self.num_heads,
                      self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_length, self.num_heads,
                      self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_length, self.num_heads,
                      self.head_dim).permute(0, 2, 1, 3)

        attn_output, attn_weights = self_attention.scaled_dot_product(Q, K, V, mask=mask)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length,
                                                              self.hidden_dim)
        O = self.O_proj(attn_output)
        O = self.dropout(O)

        return O, attn_weights

class ViTAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_heads: int, dropout: float = 0.0):
        """Pre-LN attention block for the ViT model.

        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of embedding space.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate. Default to 0.0.
        """
        super().__init__()
        self.layer_norm1 = ln.LayerNorm(input_dim)
        self.attn = MultiHeadAttention(
            input_dim, hidden_dim, num_heads, dropout=dropout
        )
        self.layer_norm2 = ln.LayerNorm(input_dim)
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        input = self.layer_norm1(x)
        x = x + self.attn(input, input, input)[0]
        x = x + self.linear(self.layer_norm2(x))
        return x