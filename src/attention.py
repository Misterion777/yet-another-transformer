import torch
from torch import nn


class DotProductAttention(nn.Module):
    def __init__(self, k_dim: int):
        super().__init__()
        self.k_dim = torch.tensor(k_dim)

    def forward(self, query, key, value):
        mul = torch.matmul(query, key.T)
        mul = mul / torch.sqrt(self.k_dim)
        mul = torch.softmax(mul, 1)
        result = torch.matmul(mul, value)
        return result


class AttentionHead(nn.Module):
    def __init__(self, v_dim: int, k_dim: int, model_dim: int):
        super().__init__()
        self.q_linear = nn.Linear(k_dim, model_dim, bias=False)
        self.k_linear = nn.Linear(k_dim, model_dim, bias=False)
        self.v_linear = nn.Linear(v_dim, model_dim, bias=False)
        self.attn = DotProductAttention(k_dim)

    def forward(self, query, key, value):
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        return self.attn(q, k, v)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        v_dim: int,
        k_dim: int,
        model_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.ModuleList(
            [AttentionHead(v_dim, k_dim, model_dim) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * model_dim, model_dim, bias=False)

    def forward(self, query, key, value):
        result = [head(query, key, value) for head in self.attn_heads]
        result = torch.cat(result, dim=1)
        return self.proj(result)
