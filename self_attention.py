import torch
from typing import Tuple

def softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable Softmax using exp normalization trick.

    Based on https://cs231n.github.io/linear-classify/#softmax.
    """
    maxes = torch.max(x, dim=1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs

def scaled_dot_product(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                       mask: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention function using Scaled Dot Product Attention."""
    # Hidden dimensionality for queries Q and keys K
    sqrt_dk = torch.sqrt(torch.tensor(Q.size()[-1], dtype=torch.float32))
    attention = torch.matmul(Q, K.transpose(-2, -1)) / sqrt_dk
    if mask:
        attention = attention.masked_fill(mask == 0, -1e16)
    value_weights = softmax(attention)
    attention = torch.matmul(value_weights, V)
    return attention, value_weights