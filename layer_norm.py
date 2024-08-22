import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, norm_shape: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(norm_shape))
        self.beta = nn.Parameter(torch.zeros(norm_shape))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
        return (x_norm * self.gamma) + self.beta