import multihead_attention as mult
from torch import nn

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
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
